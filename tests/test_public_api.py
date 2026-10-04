"""Pins the public embedding API in ``claude_swap.api``.

Every exported name, every signature and every payload/event shape below is
part of the contract the fork's ``sak`` releases promise to claude-SAK. A test
failing here means a breaking change: bump the ``+sak.N`` release and update
the consumer, never just the test.
"""

from __future__ import annotations

import builtins
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

from claude_swap import api
from claude_swap.autoswitch import AutoSwitchEngine
from claude_swap.mappings import MappingStore
from claude_swap.models import Platform
from tests.test_autoswitch import EngineHarness

EXPECTED_ALL = [
    "SCHEMA_VERSION",
    "AccountNotFoundError",
    "AllExhaustedEvent",
    "AutoSwitchEngine",
    "AutoSwitchEvent",
    "AutoSwitchSettings",
    "AutoSwitcherInstance",
    "ClaudeAccountSwitcher",
    "ClaudeSwitchError",
    "ConfigError",
    "ConfigWarningEvent",
    "CredentialError",
    "ErrorEvent",
    "LockError",
    "NoSwitchEvent",
    "PollEvent",
    "ProjectAccount",
    "QuarantineEvent",
    "SessionError",
    "SessionProfile",
    "SharedGrant",
    "SleepEvent",
    "SwitchError",
    "SwitchEvent",
    "UnquarantineEvent",
    "ValidationError",
    "accounts_json",
    "active_account",
    "create_engine",
    "inject_system_trust",
    "live_session_accounts",
    "open_switcher",
    "prepare_session_profile",
    "project_account",
    "running_autoswitchers",
    "set_rotation",
    "shared_grants",
    "switch_to",
    "version",
]

EXPECTED_SIGNATURES = {
    "open_switcher": "() -> 'ClaudeAccountSwitcher'",
    "accounts_json": (
        "(switcher: 'ClaudeAccountSwitcher', fetch: 'Collection[str]' = frozenset())"
        " -> 'dict[str, Any]'"
    ),
    "project_account": (
        "(switcher: 'ClaudeAccountSwitcher', path: 'str | os.PathLike[str]')"
        " -> 'ProjectAccount'"
    ),
    "active_account": "(switcher: 'ClaudeAccountSwitcher') -> 'str | None'",
    "switch_to": (
        "(switcher: 'ClaudeAccountSwitcher', identifier: 'str') -> 'dict[str, Any]'"
    ),
    "set_rotation": (
        "(switcher: 'ClaudeAccountSwitcher', identifier: 'str', enabled: 'bool')"
        " -> 'None'"
    ),
    "live_session_accounts": (
        "(switcher: 'ClaudeAccountSwitcher') -> 'dict[str, tuple[int, ...]]'"
    ),
    "create_engine": (
        "(switcher: 'ClaudeAccountSwitcher', "
        "on_event: 'Callable[[AutoSwitchEvent], None]', *, dry_run: 'bool')"
        " -> 'AutoSwitchEngine'"
    ),
    "prepare_session_profile": (
        "(switcher: 'ClaudeAccountSwitcher', identifier: 'str', *, "
        "share_history: 'bool' = True) -> 'SessionProfile'"
    ),
    "running_autoswitchers": (
        "(switcher: 'ClaudeAccountSwitcher') -> 'list[AutoSwitcherInstance]'"
    ),
    "shared_grants": (
        "(switcher: 'ClaudeAccountSwitcher', "
        "profile_dirs: 'Collection[str | os.PathLike[str]]') -> 'list[SharedGrant]'"
    ),
    "version": "() -> 'str'",
    "inject_system_trust": "() -> 'bool'",
}

ACCOUNT_ROW_KEYS = {
    "number",
    "email",
    "organizationName",
    "organizationUuid",
    "active",
    "usageStatus",
    "usage",
}


@pytest.fixture
def no_stdin(monkeypatch):
    """Fail any attempt to read stdin while the test runs."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("the public API must never read stdin")

    monkeypatch.setattr(builtins, "input", refuse)

    class _NoStdin:
        def read(self, *_a):
            refuse()

        def readline(self, *_a):
            refuse()

        def fileno(self):
            refuse()

    monkeypatch.setattr(sys, "stdin", _NoStdin())


@pytest.fixture
def seeded(temp_home: Path, no_stdin) -> EngineHarness:
    h = EngineHarness(temp_home)
    h.seed(1, "a@example.com")
    h.seed(2, "b@example.com")
    h.make_live("a@example.com", 1)
    return h


def _assert_silent(capfd) -> None:
    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


class TestSurface:
    def test_all_is_pinned(self):
        assert api.__all__ == EXPECTED_ALL

    def test_every_exported_name_exists(self):
        for name in api.__all__:
            assert hasattr(api, name), name

    @pytest.mark.parametrize("name", sorted(EXPECTED_SIGNATURES))
    def test_signature_is_pinned(self, name):
        assert str(inspect.signature(getattr(api, name))) == EXPECTED_SIGNATURES[name]

    def test_schema_version_is_one(self):
        assert api.SCHEMA_VERSION == 1

    def test_exceptions_share_the_base(self):
        for name in (
            "AccountNotFoundError",
            "ConfigError",
            "CredentialError",
            "LockError",
            "SessionError",
            "SwitchError",
            "ValidationError",
        ):
            assert issubclass(getattr(api, name), api.ClaudeSwitchError)

    def test_version_matches_distribution_metadata(self):
        from importlib.metadata import version

        assert api.version() == version("claude-swap")

    def test_inject_system_trust_reports_success(self, monkeypatch):
        import truststore

        calls = []
        monkeypatch.setattr(truststore, "inject_into_ssl", lambda: calls.append(1))
        assert api.inject_system_trust() is True
        assert calls == [1]

    def test_inject_system_trust_reports_failure(self, monkeypatch):
        import truststore

        def boom():
            raise RuntimeError("no native trust here")

        monkeypatch.setattr(truststore, "inject_into_ssl", boom)
        assert api.inject_system_trust() is False


class TestOpenSwitcher:
    def test_opens_on_the_default_profile(self, temp_home, no_stdin, capfd):
        switcher = api.open_switcher()
        assert isinstance(switcher, api.ClaudeAccountSwitcher)
        _assert_silent(capfd)

    def test_accepts_config_dir_naming_the_default_profile(
        self, temp_home, monkeypatch, no_stdin
    ):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(temp_home / ".claude"))
        assert isinstance(api.open_switcher(), api.ClaudeAccountSwitcher)

    def test_refuses_a_hand_made_profile(self, temp_home, monkeypatch):
        other = temp_home / ".claude-work"
        other.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(other))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.open_switcher()

    def test_refuses_a_missing_profile(self, temp_home, monkeypatch):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(temp_home / "nowhere"))
        with pytest.raises(api.ConfigError):
            api.open_switcher()


class TestAccounts:
    def test_empty_store_yields_an_empty_payload(self, temp_home, no_stdin, capfd):
        switcher = api.open_switcher()
        capfd.readouterr()
        payload = api.accounts_json(switcher)
        assert payload == {
            "schemaVersion": 1,
            "activeAccountNumber": None,
            "accounts": [],
        }
        _assert_silent(capfd)

    def test_payload_shape(self, seeded, capfd):
        capfd.readouterr()
        payload = api.accounts_json(seeded.switcher)
        _assert_silent(capfd)
        assert payload["schemaVersion"] == 1
        rows = payload["accounts"]
        assert [row["number"] for row in rows] == [1, 2]
        for row in rows:
            assert ACCOUNT_ROW_KEYS <= set(row)
        assert [row["active"] for row in rows] == [True, False]
        json.dumps(payload)  # JSON-serialisable as is

    def test_cache_only_by_default(self, seeded, monkeypatch):
        seen = []
        original = seeded.switcher._collect_usage_entries

        def spy(accounts_info, fetch=None, **kwargs):
            seen.append(fetch)
            return original(accounts_info, fetch=fetch, **kwargs)

        monkeypatch.setattr(seeded.switcher, "_collect_usage_entries", spy)
        api.accounts_json(seeded.switcher)
        api.accounts_json(seeded.switcher, fetch={"2"})
        assert seen == [set(), {"2"}]

    def test_active_account(self, seeded, capfd):
        capfd.readouterr()
        assert api.active_account(seeded.switcher) == "1"
        _assert_silent(capfd)

    def test_active_account_is_none_without_a_live_login(self, temp_home, no_stdin):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        assert api.active_account(h.switcher) is None


class TestProjectAccount:
    def test_unmapped_directory(self, seeded, tmp_path):
        assert api.project_account(seeded.switcher, tmp_path) == (None, None)

    def test_mapped_parent_wins(self, seeded, tmp_path):
        project = tmp_path / "proj"
        (project / "sub").mkdir(parents=True)
        MappingStore(seeded.switcher.backup_dir).set(
            project, "b@example.com", ""
        )
        result = api.project_account(seeded.switcher, project / "sub")
        assert result == api.ProjectAccount(slot="2", email="b@example.com")
        assert result.slot == "2"

    def test_mapping_to_a_removed_account(self, seeded, tmp_path):
        MappingStore(seeded.switcher.backup_dir).set(tmp_path, "gone@example.com", "")
        assert api.project_account(seeded.switcher, str(tmp_path)) == (
            None,
            "gone@example.com",
        )


class TestSwitchAndRotation:
    def test_switch_returns_the_json_payload(self, seeded, capfd):
        capfd.readouterr()
        payload = api.switch_to(seeded.switcher, "2")
        _assert_silent(capfd)
        assert payload["schemaVersion"] == 1
        assert payload["switched"] is True
        assert payload["to"]["number"] == 2
        assert payload["to"]["email"] == "b@example.com"
        assert api.active_account(seeded.switcher) == "2"

    def test_switch_to_the_active_account_is_a_noop(self, seeded):
        payload = api.switch_to(seeded.switcher, "1")
        assert payload["switched"] is False

    def test_unknown_account_raises(self, seeded):
        with pytest.raises(api.AccountNotFoundError):
            api.switch_to(seeded.switcher, "9")

    def test_ambiguous_email_raises_instead_of_prompting(self, seeded):
        seeded.seed(3, "b@example.com")
        data = seeded.switcher._get_sequence_data()
        data["accounts"]["3"]["organizationUuid"] = "org-3"
        data["accounts"]["3"]["organizationName"] = "Org Three"
        seeded.switcher._write_json(seeded.switcher.sequence_file, data)
        with pytest.raises(api.ConfigError):
            api.switch_to(seeded.switcher, "b@example.com")

    def test_set_rotation_is_silent_and_idempotent(self, seeded, capfd):
        capfd.readouterr()
        api.set_rotation(seeded.switcher, "2", enabled=False)
        api.set_rotation(seeded.switcher, "2", enabled=False)
        _assert_silent(capfd)
        rows = {row["number"]: row for row in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[2].get("disabled") is True
        api.set_rotation(seeded.switcher, "b@example.com", enabled=True)
        _assert_silent(capfd)
        rows = {row["number"]: row for row in api.accounts_json(seeded.switcher)["accounts"]}
        assert not rows[2].get("disabled")


class TestLiveSessions:
    def test_no_sessions(self, seeded):
        assert api.live_session_accounts(seeded.switcher) == {}

    def test_live_session_profile_is_reported(self, seeded):
        from claude_swap.session import session_dir_for

        profile = session_dir_for(seeded.switcher.backup_dir, "2", "b@example.com")
        (profile / "sessions").mkdir(parents=True)
        (profile / "sessions" / f"{os.getpid()}.json").write_text(
            json.dumps({"pid": os.getpid(), "sessionId": "s", "cwd": "/"})
        )
        assert api.live_session_accounts(seeded.switcher) == {"2": (os.getpid(),)}


class TestEngine:
    def test_uses_saved_settings(self, seeded):
        from claude_swap.settings import AutoSwitchSettings, save_settings

        save_settings(
            seeded.switcher.backup_dir,
            AutoSwitchSettings(threshold=80.0, strategy="consume-first"),
        )
        events = []
        engine = api.create_engine(seeded.switcher, events.append, dry_run=True)
        assert isinstance(engine, AutoSwitchEngine)
        assert engine.settings.threshold == 80.0
        assert engine.settings.strategy == "consume-first"
        assert engine.dry_run is True
        assert engine.on_event == events.append

    def test_stop_before_run_returns_at_once(self, seeded):
        engine = api.create_engine(seeded.switcher, lambda _e: None, dry_run=True)
        engine.stop()
        assert engine.run_loop() == 0


EVENT_SHAPES = [
    (
        api.PollEvent(
            active={"number": 1, "email": "a@example.com"},
            headroom={"1": 40.0},
            threshold=90.0,
        ),
        "poll",
        {"active", "headroomPct", "threshold"},
    ),
    (
        api.SwitchEvent(trigger="proactive", from_ref=None, to_ref=None),
        "switch",
        {"trigger", "from", "to", "warnings", "dryRun"},
    ),
    (api.NoSwitchEvent(reason="cooldown"), "no-switch", {"reason", "detail"}),
    (
        api.QuarantineEvent(number="2", email="b@example.com", reason="dead"),
        "account-quarantined",
        {"number", "email", "reason"},
    ),
    (
        api.UnquarantineEvent(number="2", email="b@example.com"),
        "account-unquarantined",
        {"number", "email", "reason"},
    ),
    (api.AllExhaustedEvent(earliest_reset_at=None), "all-exhausted", {"earliestResetAt"}),
    (api.SleepEvent(seconds=600.0, until="2026-01-01T00:00:00Z"), "sleep", {"seconds", "until"}),
    (api.ErrorEvent(message="boom"), "error", {"message", "transient"}),
    (api.ConfigWarningEvent(message="inert"), "config-warning", {"message"}),
]


@pytest.mark.parametrize(("event", "kind", "fields"), EVENT_SHAPES)
def test_event_payload_shapes(event, kind, fields):
    assert isinstance(event, api.AutoSwitchEvent)
    payload = event.to_json()
    assert payload["schemaVersion"] == 1
    assert payload["event"] == kind
    assert isinstance(payload["ts"], str)
    assert set(payload) == {"schemaVersion", "event", "ts", *fields}


class TestPrepareSessionProfile:
    def test_refuses_the_active_default_login(self, seeded, monkeypatch):
        from claude_swap.session import SessionManager

        def must_not_run(*_a, **_k):
            raise AssertionError("setup_session must not run for the active account")

        monkeypatch.setattr(SessionManager, "setup_session", must_not_run)
        with pytest.raises(api.SessionError, match="active default login"):
            api.prepare_session_profile(seeded.switcher, "1")
        with pytest.raises(api.SessionError):
            api.prepare_session_profile(seeded.switcher, "a@example.com")

    def test_delegates_for_another_account(self, seeded, monkeypatch, capfd):
        from claude_swap.session import SessionManager, session_dir_for

        calls = []
        profile = session_dir_for(seeded.switcher.backup_dir, "2", "b@example.com")

        def fake_setup(self, identifier, share, share_history=False):
            calls.append((identifier, share, share_history))
            print("Bootstrapping session profile")  # must be captured
            return profile, "2", "b@example.com"

        monkeypatch.setattr(SessionManager, "setup_session", fake_setup)
        capfd.readouterr()
        result = api.prepare_session_profile(seeded.switcher, "2", share_history=False)
        _assert_silent(capfd)
        assert calls == [("2", True, False)]
        assert result == api.SessionProfile(
            config_dir=str(profile),
            slot="2",
            email="b@example.com",
            strip_env=(
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_OAUTH_TOKEN",
                "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
                "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
            ),
        )
        assert isinstance(result.config_dir, str)

    def test_shares_history_by_default(self, seeded, monkeypatch):
        from claude_swap.session import SessionManager

        calls = []

        def fake_setup(self, identifier, share, share_history=False):
            calls.append(share_history)
            return Path("/x"), "2", "b@example.com"

        monkeypatch.setattr(SessionManager, "setup_session", fake_setup)
        api.prepare_session_profile(seeded.switcher, "2")
        assert calls == [True]

    def test_unknown_account(self, seeded):
        with pytest.raises(api.AccountNotFoundError):
            api.prepare_session_profile(seeded.switcher, "7")

    def test_refuses_a_foreign_config_dir(self, seeded, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
        with pytest.raises(api.ConfigError):
            api.prepare_session_profile(seeded.switcher, "2")


class TestRunningAutoswitchers:
    def test_empty(self, seeded):
        assert api.running_autoswitchers(seeded.switcher) == []

    def test_lists_a_running_engine(self, seeded, monkeypatch):
        from claude_swap.autoswitch import TickOutcome

        seen = []
        engine = api.create_engine(seeded.switcher, lambda _e: None, dry_run=True)

        def tick():
            seen.extend(api.running_autoswitchers(seeded.switcher))
            engine.stop()
            return TickOutcome.NO_ACTION

        monkeypatch.setattr(engine, "tick", tick)
        monkeypatch.setattr(engine, "_next_delay", lambda _o: 0.0)
        assert engine.run_loop() == 0
        assert len(seen) == 1
        assert seen[0].pid == os.getpid()
        assert seen[0].dry_run is True
        assert api.running_autoswitchers(seeded.switcher) == []


def _write_login(directory: Path, access: str, refresh: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": access, "refreshToken": refresh}})
    )


class TestSharedGrants:
    def test_reports_a_copied_login(self, seeded, temp_home, capfd):
        work = temp_home / ".claude-work"
        _write_login(work, access="sk-rotated", refresh="rt-2")
        capfd.readouterr()
        grants = api.shared_grants(seeded.switcher, [work])
        _assert_silent(capfd)
        assert grants == [
            api.SharedGrant(slot="2", email="b@example.com", directory=str(work))
        ]
        assert set(grants[0]._fields) == {"slot", "email", "directory"}

    def test_unrelated_and_missing_profiles_match_nothing(self, seeded, temp_home):
        other = temp_home / ".claude-other"
        _write_login(other, access="sk-x", refresh="rt-someone-else")
        assert api.shared_grants(seeded.switcher, [other, temp_home / "gone"]) == []

    def test_expands_the_home_directory(self, seeded, temp_home):
        _write_login(temp_home / ".claude-zeker", access="sk-1b", refresh="rt-1")
        grants = api.shared_grants(seeded.switcher, ["~/.claude-zeker"])
        assert grants == [
            api.SharedGrant(
                slot="1",
                email="a@example.com",
                directory=str(temp_home / ".claude-zeker"),
            )
        ]

    def test_several_profiles(self, seeded, temp_home):
        _write_login(temp_home / "p1", access="a", refresh="rt-1")
        _write_login(temp_home / "p2", access="b", refresh="rt-2")
        grants = api.shared_grants(seeded.switcher, [temp_home / "p2", temp_home / "p1"])
        assert [(g.slot, Path(g.directory).name) for g in grants] == [
            ("1", "p1"),
            ("2", "p2"),
        ]

    def test_no_profiles(self, seeded):
        assert api.shared_grants(seeded.switcher, []) == []


def test_platform_is_forced_to_the_file_backend(seeded):
    # The seeded fixtures rely on the Linux file backend, as the engine tests do.
    assert seeded.switcher.platform == Platform.LINUX
