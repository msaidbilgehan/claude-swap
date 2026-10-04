"""The auto-switch instance registry: engines list themselves while they run."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from claude_swap import autoswitch
from claude_swap.autoswitch import (
    AutoSwitcherInstance,
    TickOutcome,
    instances_dir,
    register_instance,
    running_instances,
    unregister_instance,
)
from tests.test_autoswitch import EngineHarness


@pytest.fixture
def harness(temp_home: Path) -> EngineHarness:
    h = EngineHarness(temp_home)
    h.seed(1, "a@example.com")
    h.seed(2, "b@example.com")
    h.make_live("a@example.com", 1)
    return h


def _run_one_tick(harness: EngineHarness, monkeypatch, probe) -> int:
    """Run the real loop for one stubbed tick, calling ``probe`` inside it."""
    engine = harness.engine

    def tick():
        probe()
        engine.stop()
        return TickOutcome.NO_ACTION

    monkeypatch.setattr(engine, "tick", tick)
    monkeypatch.setattr(engine, "_next_delay", lambda _outcome: 0.0)
    return engine.run_loop()


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class TestEngineRegistration:
    def test_registered_while_running_and_removed_after(self, harness, monkeypatch):
        backup = harness.switcher.backup_dir
        seen: list[dict] = []

        def probe():
            path = instances_dir(backup) / f"{os.getpid()}.json"
            seen.append(json.loads(path.read_text()))

        assert _run_one_tick(harness, monkeypatch, probe) == 0
        record = seen[0]
        assert record["pid"] == os.getpid()
        assert record["dryRun"] is False
        assert record["schemaVersion"] == 1
        assert record["hostApp"]
        assert record["startedAt"].endswith("Z")
        assert not (instances_dir(backup) / f"{os.getpid()}.json").exists()

    def test_registry_files_are_private(self, harness, monkeypatch):
        backup = harness.switcher.backup_dir
        modes: list[int] = []

        def probe():
            path = instances_dir(backup) / f"{os.getpid()}.json"
            modes.append(path.stat().st_mode & 0o777)
            modes.append(path.parent.stat().st_mode & 0o777)

        _run_one_tick(harness, monkeypatch, probe)
        assert modes == [0o600, 0o700]

    def test_host_app_and_dry_run_are_recorded(self, harness, monkeypatch):
        harness.engine = harness._make_engine(dry_run=True, host_app="sakd")
        found: list[AutoSwitcherInstance] = []
        _run_one_tick(
            harness,
            monkeypatch,
            lambda: found.extend(running_instances(harness.switcher.backup_dir)),
        )
        assert [(i.host_app, i.dry_run) for i in found] == [("sakd", True)]

    def test_registry_failure_never_stops_the_engine(
        self, harness, monkeypatch, caplog
    ):
        backup = harness.switcher.backup_dir
        (backup / "autoswitch").write_text("not a directory")
        ticks: list[int] = []
        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            assert _run_one_tick(harness, monkeypatch, lambda: ticks.append(1)) == 0
        assert ticks == [1]
        assert "Could not register auto-switch engine" in caplog.text

    def test_unregister_failure_is_logged(self, harness, monkeypatch, caplog):
        def boom(_path, _token):
            raise OSError("read-only")

        monkeypatch.setattr(autoswitch, "unregister_instance", boom)
        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            assert _run_one_tick(harness, monkeypatch, lambda: None) == 0
        assert "Could not unregister auto-switch engine" in caplog.text

    def test_a_failing_loop_still_unregisters(self, harness, monkeypatch):
        backup = harness.switcher.backup_dir

        def explode():
            raise KeyboardInterrupt

        monkeypatch.setattr(harness.engine, "_run_loop", explode)
        with pytest.raises(KeyboardInterrupt):
            harness.engine.run_loop()
        assert list(instances_dir(backup).glob("*.json")) == []


class TestRunningInstances:
    def test_missing_directory(self, tmp_path):
        assert running_instances(tmp_path) == []

    def test_prunes_dead_processes(self, tmp_path):
        path = register_instance(tmp_path, host_app="cswap", dry_run=False, token="t")
        dead = _dead_pid()
        stale = instances_dir(tmp_path) / f"{dead}.json"
        record = json.loads(path.read_text())
        record["pid"] = dead
        stale.write_text(json.dumps(record))
        live = running_instances(tmp_path)
        assert [i.pid for i in live] == [os.getpid()]
        assert not stale.exists()
        assert path.exists()

    @pytest.mark.skipif(sys.platform != "linux", reason="/proc start ticks")
    def test_prunes_a_recycled_pid(self, tmp_path):
        path = register_instance(tmp_path, host_app="cswap", dry_run=False, token="t")
        record = json.loads(path.read_text())
        record["procStartTicks"] = "1"
        path.write_text(json.dumps(record))
        assert running_instances(tmp_path) == []
        assert not path.exists()

    def test_unreadable_entries_are_skipped(self, tmp_path, caplog):
        directory = instances_dir(tmp_path)
        directory.mkdir(parents=True)
        (directory / "1.json").write_text("{broken")
        (directory / "2.json").write_text(json.dumps({"pid": "two"}))
        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            assert running_instances(tmp_path) == []
        assert "unreadable auto-switch registry entry" in caplog.text

    def test_unknowable_start_time_counts_as_alive(self, tmp_path, monkeypatch):
        path = register_instance(tmp_path, host_app="cswap", dry_run=True, token="t")
        record = json.loads(path.read_text())
        record.pop("procStartTicks", None)
        record["procStartedAt"] = 12345
        path.write_text(json.dumps(record))
        monkeypatch.setattr(autoswitch, "process_started_at", lambda _pid: None)
        assert [i.pid for i in running_instances(tmp_path)] == [os.getpid()]

    def test_started_at_mismatch_is_pruned(self, tmp_path, monkeypatch):
        path = register_instance(tmp_path, host_app="cswap", dry_run=True, token="t")
        record = json.loads(path.read_text())
        record.pop("procStartTicks", None)
        record["procStartedAt"] = 12345
        path.write_text(json.dumps(record))
        monkeypatch.setattr(autoswitch, "process_started_at", lambda _pid: 99999)
        assert running_instances(tmp_path) == []


class TestUnregister:
    def test_keeps_a_newer_engines_record(self, tmp_path):
        path = register_instance(tmp_path, host_app="a", dry_run=False, token="old")
        register_instance(tmp_path, host_app="b", dry_run=False, token="new")
        unregister_instance(path, "old")
        assert path.exists()
        unregister_instance(path, "new")
        assert not path.exists()

    def test_missing_file_is_fine(self, tmp_path):
        unregister_instance(tmp_path / "nope.json", "t")
