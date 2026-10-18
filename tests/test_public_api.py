"""Pins the public embedding API in ``claude_swap.api``.

Every exported name, every signature and every payload/event shape below is
part of the contract the fork's ``sak`` releases promise to claude-SAK. A test
failing here means a breaking change: bump the ``+sak.N`` release and update
the consumer, never just the test.
"""

from __future__ import annotations

import builtins
import contextlib
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from claude_swap import api
from claude_swap.autoswitch import AutoSwitchEngine
from claude_swap.mappings import MappingStore
from claude_swap.models import Platform
from claude_swap.oauth import RefreshOutcome
from tests.test_autoswitch import EngineHarness

EXPECTED_ALL = [
    "SCHEMA_VERSION",
    "AccessTokenInfo",
    "AccountNotFoundError",
    "AddedAccount",
    "AllExhaustedEvent",
    "AutoSwitchEngine",
    "AutoSwitchEvent",
    "AutoSwitchSettings",
    "AutoSwitcherInstance",
    "ClaudeAccountSwitcher",
    "ClaudeSwitchError",
    "ConfigError",
    "ConfigWarningEvent",
    "CopiedLogin",
    "CredentialError",
    "DefaultLoginInfo",
    "ErrorEvent",
    "HandedOver",
    "LockError",
    "LoginNotAddedError",
    "LoginSource",
    "MovedLogin",
    "NoSwitchEvent",
    "OwnLoginError",
    "PollEvent",
    "ProjectAccount",
    "ProjectMapping",
    "QuarantineEvent",
    "SessionError",
    "SessionProfile",
    "SharedGrant",
    "SleepEvent",
    "SwitchError",
    "SwitchEvent",
    "UnquarantineEvent",
    "ValidationError",
    "access_token",
    "accounts_json",
    "active_account",
    "add_current_login",
    "add_login_copy",
    "add_profile_login",
    "add_setup_token",
    "check_login_source",
    "create_engine",
    "default_login",
    "forget_access_login",
    "forget_default_access_login",
    "forget_profile_login",
    "hand_over_login",
    "import_login",
    "inject_system_trust",
    "keep_access_login",
    "live_session_accounts",
    "map_project",
    "open_switcher",
    "prepare_session_profile",
    "project_account",
    "project_mappings",
    "remove_account",
    "running_autoswitchers",
    "set_alias",
    "set_rotation",
    "shared_grants",
    "switch_to",
    "unmap_project",
    "use_access_login",
    "version",
]

EXPECTED_SIGNATURES = {
    "open_switcher": "() -> 'ClaudeAccountSwitcher'",
    "accounts_json": (
        "(switcher: 'ClaudeAccountSwitcher', "
        "fetch: 'Collection[str] | None' = frozenset()) -> 'dict[str, Any]'"
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
        "share_history: 'bool' = True, "
        "extra_env: 'Mapping[str, str] | None' = None) -> 'SessionProfile'"
    ),
    "running_autoswitchers": (
        "(switcher: 'ClaudeAccountSwitcher') -> 'list[AutoSwitcherInstance]'"
    ),
    "shared_grants": (
        "(switcher: 'ClaudeAccountSwitcher', "
        "profile_dirs: 'Collection[str | os.PathLike[str]]') -> 'list[SharedGrant]'"
    ),
    "add_current_login": (
        "(switcher: 'ClaudeAccountSwitcher', *, alias: 'str | None' = None)"
        " -> 'AddedAccount'"
    ),
    "add_profile_login": (
        "(switcher: 'ClaudeAccountSwitcher', config_dir: 'str | os.PathLike[str]', "
        "*, alias: 'str | None' = None) -> 'AddedAccount'"
    ),
    "forget_profile_login": "(config_dir: 'str | os.PathLike[str]') -> 'None'",
    "check_login_source": (
        "(switcher: 'ClaudeAccountSwitcher', source: 'str | os.PathLike[str]')"
        " -> 'LoginSource'"
    ),
    "import_login": (
        "(switcher: 'ClaudeAccountSwitcher', source: 'str | os.PathLike[str]', "
        "config_dir: 'str | os.PathLike[str]', *, alias: 'str | None' = None, "
        "claude: 'str | os.PathLike[str] | None' = None, "
        "extra_env: 'Mapping[str, str] | None' = None) -> 'MovedLogin'"
    ),
    "add_login_copy": (
        "(switcher: 'ClaudeAccountSwitcher', config_dir: 'str | os.PathLike[str]', *, "
        "credentials: 'str | None' = None, "
        "source: 'str | os.PathLike[str] | None' = None, "
        "alias: 'str | None' = None, account: 'str | None' = None) -> 'CopiedLogin'"
    ),
    "add_setup_token": (
        "(switcher: 'ClaudeAccountSwitcher', token: 'str', *, "
        "email: 'str | None' = None, alias: 'str | None' = None, "
        "account: 'str | None' = None) -> 'AddedAccount'"
    ),
    "hand_over_login": (
        "(switcher: 'ClaudeAccountSwitcher', identifier: 'str', "
        "destination: 'str | os.PathLike[str]') -> 'HandedOver'"
    ),
    "remove_account": (
        "(switcher: 'ClaudeAccountSwitcher', identifier: 'str') -> 'tuple[str, str]'"
    ),
    "set_alias": (
        "(switcher: 'ClaudeAccountSwitcher', identifier: 'str', alias: 'str | None')"
        " -> 'str | None'"
    ),
    "map_project": (
        "(switcher: 'ClaudeAccountSwitcher', path: 'str | os.PathLike[str]', "
        "identifier: 'str') -> 'ProjectAccount'"
    ),
    "unmap_project": (
        "(switcher: 'ClaudeAccountSwitcher', path: 'str | os.PathLike[str]')"
        " -> 'bool'"
    ),
    "project_mappings": (
        "(switcher: 'ClaudeAccountSwitcher') -> 'list[ProjectMapping]'"
    ),
    "version": "() -> 'str'",
    "inject_system_trust": "() -> 'bool'",
    "access_token": (
        "(switcher: 'ClaudeAccountSwitcher', identifier: 'str', *, "
        "renew_within: 'float' = 0.0) -> 'AccessTokenInfo'"
    ),
    "keep_access_login": (
        "(config_dir: 'str | os.PathLike[str]', *, access_token: 'str', "
        "expires_at: 'str', scopes: 'Sequence[str]', "
        "subscription_type: 'str | None', rate_limit_tier: 'str | None', "
        "email: 'str', organization_uuid: 'str | None', "
        "organization: 'str | None', account_uuid: 'str | None', "
        "share_history: 'bool' = True) -> 'None'"
    ),
    "forget_access_login": "(config_dir: 'str | os.PathLike[str]') -> 'None'",
    "default_login": "() -> 'DefaultLoginInfo'",
    "use_access_login": (
        "(config_dir: 'str | os.PathLike[str]', *, replace_own: 'bool' = False)"
        " -> 'bool'"
    ),
    "forget_default_access_login": "() -> 'bool'",
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
        api.accounts_json(seeded.switcher, fetch=None)
        assert seen == [set(), {"2"}, None]

    def test_none_fetches_what_is_due_including_a_disabled_account(
        self, seeded, capfd
    ):
        from unittest.mock import patch

        from claude_swap.usage_store import FetchRecord

        api.set_rotation(seeded.switcher, "2", enabled=False)
        usage = {"five_hour": {"pct": 10.0}, "seven_day": {"pct": 20.0}}
        fetched: list[list[str]] = []

        def run(infos, entries=None):
            fetched.append(sorted(str(info[0]) for info in infos))
            return {str(info[0]): FetchRecord(usage=usage) for info in infos}

        capfd.readouterr()
        with patch.object(seeded.switcher, "_run_usage_fetches", side_effect=run):
            api.accounts_json(seeded.switcher)  # the default stays cache-only
            assert fetched == []
            payload = api.accounts_json(seeded.switcher, None)  # never fetched
            assert fetched == [["1", "2"]]
            # Past the serve TTL: the active account's shorter plan is due, the
            # disabled account's is not, so it is not fetched yet.
            seeded.clock.advance(200)
            api.accounts_json(seeded.switcher, None)
            assert fetched == [["1", "2"], ["1"]]
            seeded.clock.advance(800)  # past every poll plan
            api.accounts_json(seeded.switcher, None)
            assert fetched == [["1", "2"], ["1"], ["1", "2"]]
        _assert_silent(capfd)
        rows = {row["number"]: row for row in payload["accounts"]}
        assert rows[2]["disabled"] is True
        assert rows[2]["usageStatus"] == "ok"
        assert rows[2]["usage"]["sevenDay"]["pct"] == 20.0

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

    def test_extra_env_reaches_claude_auth_status(self, seeded, monkeypatch):
        import subprocess

        from claude_swap import session as session_mod

        profile = session_mod.session_dir_for(
            seeded.switcher.backup_dir, "2", "b@example.com"
        )
        profile.mkdir(parents=True)
        probes = []

        def auth_status(cmd, env=None, **_kwargs):
            probes.append((cmd[1:], env))
            status = {
                "loggedIn": True,
                "authMethod": "claude.ai",
                "email": "b@example.com",
            }
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps(status), stderr=""
            )

        monkeypatch.setattr(session_mod.subprocess, "run", auth_status)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-ignored")
        result = api.prepare_session_profile(
            seeded.switcher,
            "2",
            share_history=False,
            extra_env={
                "DISABLE_TELEMETRY": "1",
                "CLAUDE_CONFIG_DIR": "/somewhere/else",
                "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-not-here",
            },
        )
        assert result.config_dir == str(profile)
        ((argv, env),) = probes
        assert argv == ["auth", "status", "--json"]
        assert env["DISABLE_TELEMETRY"] == "1"
        assert env["CLAUDE_CONFIG_DIR"] == str(profile)
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
        assert "ANTHROPIC_API_KEY" not in env

    def test_extra_env_no_environment_holds_is_refused(self, seeded, monkeypatch):
        from claude_swap.session import SessionManager

        def must_not_run(*_a, **_k):
            raise AssertionError("nothing may run for a refused extra_env")

        monkeypatch.setattr(SessionManager, "setup_session", must_not_run)
        with pytest.raises(api.ValidationError, match="extra_env"):
            api.prepare_session_profile(seeded.switcher, "2", extra_env={"A=B": "1"})

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


def _live_login(home: Path, email: str, uuid: str) -> None:
    (home / ".claude" / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {"accessToken": f"sk-{uuid}", "refreshToken": f"rt-{uuid}"},
    }))
    (home / ".claude.json").write_text(json.dumps({
        "oauthAccount": {"emailAddress": email, "accountUuid": uuid},
    }))


class TestAccountManagement:
    """The silent, non-interactive account management surface (sak.2)."""

    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        # Whose token is this? Unknown offline: add_account registers anyway.
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile", lambda *_a, **_k: None
        )

    def test_add_registers_a_new_login_in_the_next_slot(self, seeded, capfd):
        _live_login(seeded.temp_home, "c@example.com", "uuid-3")
        capfd.readouterr()
        added = api.add_current_login(seeded.switcher, alias="work")
        _assert_silent(capfd)
        assert added == api.AddedAccount(slot="3", email="c@example.com", created=True)
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[3]["email"] == "c@example.com"
        assert rows[3].get("alias") == "work"

    def test_add_of_a_managed_login_refreshes_it(self, seeded, capfd):
        capfd.readouterr()
        added = api.add_current_login(seeded.switcher)
        _assert_silent(capfd)
        assert added == api.AddedAccount(slot="1", email="a@example.com", created=False)

    def test_add_without_a_live_login_raises(self, temp_home, no_stdin):
        h = EngineHarness(temp_home)
        with pytest.raises(api.ConfigError):
            api.add_current_login(h.switcher)

    def test_add_refuses_a_foreign_profile(self, seeded, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "hand-made"))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.add_current_login(seeded.switcher)

    def test_remove_drops_the_account_and_its_mappings(self, seeded, tmp_path, capfd):
        api.map_project(seeded.switcher, tmp_path, "2")
        capfd.readouterr()
        assert api.remove_account(seeded.switcher, "b@example.com") == (
            "2",
            "b@example.com",
        )
        _assert_silent(capfd)
        numbers = [r["number"] for r in api.accounts_json(seeded.switcher)["accounts"]]
        assert numbers == [1]
        assert api.project_mappings(seeded.switcher) == []

    def test_remove_unknown_raises(self, seeded):
        with pytest.raises(api.AccountNotFoundError):
            api.remove_account(seeded.switcher, "9")

    def test_remove_ambiguous_email_raises_instead_of_prompting(self, seeded):
        data = seeded.switcher._get_sequence_data()
        data["accounts"]["2"]["email"] = "a@example.com"
        data["accounts"]["2"]["organizationUuid"] = "org-2"
        seeded.switcher._write_json(seeded.switcher.sequence_file, data)
        with pytest.raises(api.ConfigError):
            api.remove_account(seeded.switcher, "a@example.com")

    def test_set_and_clear_an_alias(self, seeded, capfd):
        capfd.readouterr()
        assert api.set_alias(seeded.switcher, "2", "Side") == "side"
        assert api.set_alias(seeded.switcher, "side", None) is None
        _assert_silent(capfd)
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert not rows[2].get("alias")

    def test_taken_alias_raises(self, seeded):
        api.set_alias(seeded.switcher, "1", "main")
        with pytest.raises(api.ConfigError, match="already used"):
            api.set_alias(seeded.switcher, "2", "main")
        with pytest.raises(api.ValidationError):
            api.set_alias(seeded.switcher, "2", "not an alias!")

    def test_map_list_and_unmap(self, seeded, tmp_path, capfd):
        project = tmp_path / "api"
        project.mkdir()
        capfd.readouterr()
        assert api.map_project(seeded.switcher, project, "b@example.com") == (
            api.ProjectAccount(slot="2", email="b@example.com")
        )
        assert api.project_account(seeded.switcher, project / "sub").slot == "2"
        (mapping,) = api.project_mappings(seeded.switcher)
        assert mapping.slot == "2"
        assert mapping.email == "b@example.com"
        assert Path(mapping.path) == project.resolve()
        assert api.unmap_project(seeded.switcher, project) is True
        assert api.unmap_project(seeded.switcher, project) is False
        _assert_silent(capfd)
        assert api.project_mappings(seeded.switcher) == []

    def test_mapping_to_a_removed_account_has_no_slot(self, seeded, tmp_path):
        MappingStore(seeded.switcher.backup_dir).set(tmp_path, "gone@example.com", "")
        (mapping,) = api.project_mappings(seeded.switcher)
        assert mapping.slot is None
        assert mapping.email == "gone@example.com"


def _profile_login(
    directory: Path,
    email: str,
    uuid: str,
    *,
    credentials: str | None = None,
) -> Path:
    """A ``CLAUDE_CONFIG_DIR`` profile as ``claude auth login`` leaves it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ".credentials.json").write_text(
        credentials
        or json.dumps({
            "claudeAiOauth": {"accessToken": f"sk-{uuid}", "refreshToken": f"rt-{uuid}"}
        })
    )
    (directory / ".claude.json").write_text(json.dumps({
        "numStartups": 1,
        "oauthAccount": {
            "emailAddress": email,
            "accountUuid": uuid,
            "organizationUuid": None,
            "organizationName": None,
        },
    }))
    return directory


class TestProfileLogin:
    """Adding an account from a separate profile, never the default one (sak.3)."""

    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile", lambda *_a, **_k: None
        )

    @staticmethod
    def _default_profile(home: Path) -> tuple[str, str]:
        return (
            (home / ".claude" / ".credentials.json").read_text(),
            (home / ".claude.json").read_text(),
        )

    def test_a_new_account_takes_the_next_slot_and_the_default_stays(
        self, seeded, tmp_path, capfd
    ):
        before = self._default_profile(seeded.temp_home)
        profile = _profile_login(tmp_path / "login", "c@example.com", "uuid-3")
        capfd.readouterr()
        added = api.add_profile_login(seeded.switcher, profile, alias="Work")
        _assert_silent(capfd)
        assert added == api.AddedAccount(slot="3", email="c@example.com", created=True)
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[3].get("alias") == "work"
        assert rows[1]["active"] is True
        assert rows[3]["active"] is False
        assert api.active_account(seeded.switcher) == "1"
        assert self._default_profile(seeded.temp_home) == before
        stored = seeded.switcher._read_account_credentials("3", "c@example.com")
        assert stored == (profile / ".credentials.json").read_text()
        config = json.loads(seeded.switcher.read_account_config("3", "c@example.com"))
        assert set(config) == {"oauthAccount"}
        assert config["oauthAccount"]["emailAddress"] == "c@example.com"

    def test_a_managed_account_gets_the_login_and_keeps_its_alias(
        self, seeded, tmp_path
    ):
        api.set_alias(seeded.switcher, "2", "side")
        profile = _profile_login(tmp_path / "login", "b@example.com", "uuid-2b")
        added = api.add_profile_login(seeded.switcher, profile, alias="other")
        assert added == api.AddedAccount(slot="2", email="b@example.com", created=False)
        stored = seeded.switcher._read_account_credentials("2", "b@example.com")
        assert stored == (profile / ".credentials.json").read_text()
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[2].get("alias") == "side"
        assert api.active_account(seeded.switcher) == "1"

    def test_the_default_login_gets_no_second_login(self, seeded, tmp_path):
        before = seeded.switcher._read_account_credentials("1", "a@example.com")
        profile = _profile_login(tmp_path / "login", "a@example.com", "uuid-1b")
        added = api.add_profile_login(seeded.switcher, profile)
        assert added == api.AddedAccount(slot="1", email="a@example.com", created=False)
        assert seeded.switcher._read_account_credentials("1", "a@example.com") == before

    def test_an_unmanaged_default_login_is_registered_from_the_live_login(
        self, temp_home, no_stdin, tmp_path
    ):
        h = EngineHarness(temp_home)
        _live_login(temp_home, "c@example.com", "uuid-3")
        profile = _profile_login(tmp_path / "login", "c@example.com", "uuid-3b")
        added = api.add_profile_login(h.switcher, profile, alias="home")
        assert added == api.AddedAccount(slot="1", email="c@example.com", created=True)
        stored = h.switcher._read_account_credentials("1", "c@example.com")
        assert "rt-uuid-3" in stored
        assert "rt-uuid-3b" not in stored

    def test_a_profile_without_a_login_is_refused(self, seeded, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(api.ConfigError, match="log in first"):
            api.add_profile_login(seeded.switcher, empty)
        no_credential = _profile_login(tmp_path / "half", "c@example.com", "uuid-3")
        (no_credential / ".credentials.json").unlink()
        with pytest.raises(api.CredentialError, match="log in first"):
            api.add_profile_login(seeded.switcher, no_credential)
        rows = api.accounts_json(seeded.switcher)["accounts"]
        assert [r["number"] for r in rows] == [1, 2]

    def test_an_api_key_and_a_taken_alias_are_refused(self, seeded, tmp_path):
        key = _profile_login(
            tmp_path / "key", "c@example.com", "uuid-3", credentials="sk-ant-api03-x"
        )
        with pytest.raises(api.ValidationError, match="API key"):
            api.add_profile_login(seeded.switcher, key)
        api.set_alias(seeded.switcher, "1", "main")
        profile = _profile_login(tmp_path / "login", "c@example.com", "uuid-3")
        with pytest.raises(api.ValidationError, match="already used"):
            api.add_profile_login(seeded.switcher, profile, alias="main")
        rows = api.accounts_json(seeded.switcher)["accounts"]
        assert [r["number"] for r in rows] == [1, 2]

    def test_refuses_a_foreign_profile(self, seeded, monkeypatch, tmp_path):
        profile = _profile_login(tmp_path / "login", "c@example.com", "uuid-3")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "hand-made"))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.add_profile_login(seeded.switcher, profile)

    def test_forget_removes_the_profiles_credential(self, tmp_path, capfd):
        profile = _profile_login(tmp_path / "login", "c@example.com", "uuid-3")
        capfd.readouterr()
        api.forget_profile_login(profile)
        api.forget_profile_login(profile)  # already gone is fine
        _assert_silent(capfd)
        assert not (profile / ".credentials.json").exists()

    def test_forget_removes_the_keychain_entry_named_after_the_profile(
        self, tmp_path, monkeypatch
    ):
        from claude_swap import macos_keychain, session

        removed: list[tuple[str, str]] = []
        monkeypatch.setattr(
            session.Platform, "detect", classmethod(lambda _cls: Platform.MACOS)
        )
        monkeypatch.setattr(
            macos_keychain,
            "delete_password",
            lambda service, account: removed.append((service, account)),
        )
        profile = _profile_login(tmp_path / "login", "c@example.com", "uuid-3")
        api.forget_profile_login(str(profile))
        assert [service for service, _ in removed] == [
            session.keychain_service_name(str(profile))
        ]

        def locked(_service, _account):
            raise macos_keychain.KeychainError("the keychain is locked")

        monkeypatch.setattr(macos_keychain, "delete_password", locked)
        with pytest.raises(api.CredentialError, match="keychain"):
            api.forget_profile_login(str(profile))


# -- moving a login in, adding a setup-token (sak.7) --------------------------------

_SCOPES = ["user:profile", "user:inference", "user:sessions:claude_code"]

_FAKE_CLAUDE = '''
import json, os, sys, time
from pathlib import Path

here = Path(__file__).parent
plan = json.loads((here / "plan.json").read_text())
seen = {
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "env": {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("CLAUDE", "ANTHROPIC", "DISABLE_"))
    },
}
(here / "seen.json").write_text(json.dumps(seen))
mode = plan["mode"]
if mode == "hang":
    time.sleep(30)
if mode == "fail":
    print("Login failed: Request failed with status code 400", file=sys.stderr)
    sys.exit(1)
profile = Path(os.environ["CLAUDE_CONFIG_DIR"])
(profile / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
    "accessToken": "sk-ant-oat01-moved",
    "refreshToken": "sk-ant-ort01-moved-" + plan["uuid"],
    "expiresAt": 4102444800000,
    "scopes": os.environ["CLAUDE_CODE_OAUTH_SCOPES"].split(),
}}))
(profile / ".claude.json").write_text(json.dumps({"oauthAccount": {
    "emailAddress": plan["email"],
    "accountUuid": plan["uuid"],
    "organizationUuid": plan.get("org"),
    "organizationName": None,
}}))
if mode == "stored-then-fail":
    print("Login failed: could not save the settings", file=sys.stderr)
    sys.exit(1)
print("Login successful.")
'''


class FakeClaude:
    """A ``claude`` that plays ``claude auth login`` with a refresh token."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.path = directory / "claude"
        self.path.write_text(f"#!{sys.executable}\n{_FAKE_CLAUDE}")
        self.path.chmod(0o755)
        self.plan("ok")

    def plan(self, mode: str, email: str = "c@example.com", uuid: str = "uuid-3",
             org: str | None = None) -> None:
        (self.directory / "plan.json").write_text(
            json.dumps({"mode": mode, "email": email, "uuid": uuid, "org": org})
        )

    @property
    def seen(self) -> dict | None:
        path = self.directory / "seen.json"
        return json.loads(path.read_text()) if path.exists() else None


def _login_file(path: Path, refresh: str = "sk-ant-ort01-source", **extra) -> Path:
    entry = {
        "accessToken": "sk-ant-oat01-source",
        "refreshToken": refresh,
        "expiresAt": 4102444800000,
        "scopes": _SCOPES,
        **extra,
    }
    entry = {k: v for k, v in entry.items() if v is not None}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"claudeAiOauth": entry}))
    return path


@pytest.fixture
def claude(tmp_path: Path) -> FakeClaude:
    return FakeClaude(tmp_path / "bin")


@pytest.fixture
def target(tmp_path: Path) -> Path:
    directory = tmp_path / "login-profile"
    directory.mkdir()
    return directory


class TestLoginSource:
    """What a move would take, refusing what it would refuse (sak.7)."""

    def test_a_credentials_file(self, seeded, tmp_path, capfd):
        source = _login_file(
            tmp_path / "copy" / ".credentials.json",
            refreshTokenExpiresAt=4102444800000,
        )
        before = source.read_text()
        capfd.readouterr()
        found = api.check_login_source(seeded.switcher, source)
        _assert_silent(capfd)
        assert found == api.LoginSource(
            path=str(source),
            kind="file",
            email=None,
            slot=None,
            expires_at="2100-01-01T00:00:00Z",
        )
        assert set(found._fields) == {"path", "kind", "email", "slot", "expires_at"}
        assert source.read_text() == before

    def test_a_profile_says_whose_login_it_is(self, seeded, temp_home):
        work = _profile_login(
            temp_home / ".claude-work", "b@example.com", "uuid-2b",
            credentials=(_login_file(temp_home / "tmp.json").read_text()),
        )
        found = api.check_login_source(seeded.switcher, "~/.claude-work")
        assert found.path == str(work)
        assert (found.kind, found.email, found.slot) == ("profile", "b@example.com", "2")

    def test_a_new_account_in_a_profile_has_no_slot(self, seeded, temp_home):
        _profile_login(
            temp_home / ".claude-new", "c@example.com", "uuid-3",
            credentials=(_login_file(temp_home / "tmp.json").read_text()),
        )
        found = api.check_login_source(seeded.switcher, temp_home / ".claude-new")
        assert (found.email, found.slot) == ("c@example.com", None)

    @pytest.mark.parametrize(
        ("content", "error", "match"),
        [
            ("sk-ant-api03-key", api.ValidationError, "API key"),
            ('{"other": 1}', api.ValidationError, "no Claude login"),
            ("not json", api.ValidationError, "no Claude login"),
            ("[1, 2]", api.ValidationError, "no Claude login"),
            (
                json.dumps({"claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-t", "scopes": ["user:inference"],
                }}),
                api.ValidationError,
                "setup-token",
            ),
            (
                json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-t"}}),
                api.ValidationError,
                "logged out or emptied",
            ),
            (
                json.dumps({"claudeAiOauth": {
                    "refreshToken": "rt", "scopes": _SCOPES,
                    "refreshTokenExpiresAt": 1_000,
                }}),
                api.ValidationError,
                "expired on 1970-01-01T00:00:01Z",
            ),
            (
                json.dumps({"claudeAiOauth": {"refreshToken": "rt"}}),
                api.ValidationError,
                "scopes",
            ),
            (
                json.dumps({"claudeAiOauth": {"refreshToken": "rt", "scopes": [""]}}),
                api.ValidationError,
                "scopes",
            ),
        ],
    )
    def test_refuses_what_cannot_move(self, seeded, tmp_path, content, error, match):
        source = tmp_path / "creds.json"
        source.write_text(content)
        with pytest.raises(error, match=match):
            api.check_login_source(seeded.switcher, source)

    def test_refuses_a_missing_source_and_an_empty_name(self, seeded, tmp_path):
        with pytest.raises(api.ConfigError, match="does not exist"):
            api.check_login_source(seeded.switcher, tmp_path / "gone.json")
        with pytest.raises(api.ConfigError, match="Name a credentials file"):
            api.check_login_source(seeded.switcher, "  ")

    def test_refuses_a_profile_without_a_login(self, seeded, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(api.CredentialError, match="No Claude login found"):
            api.check_login_source(seeded.switcher, empty)

    def test_refuses_the_default_profile_and_claude_swaps_store(
        self, seeded, temp_home
    ):
        with pytest.raises(api.ConfigError, match="default profile"):
            api.check_login_source(seeded.switcher, temp_home / ".claude")
        with pytest.raises(api.ConfigError, match="default profile"):
            api.check_login_source(
                seeded.switcher, temp_home / ".claude" / ".credentials.json"
            )
        with pytest.raises(api.ConfigError, match="claude-swap's own store"):
            api.check_login_source(seeded.switcher, seeded.switcher.backup_dir)

    def test_refuses_a_login_that_is_managed_already(self, seeded, tmp_path):
        stored = seeded.switcher.read_account_credentials("2", "b@example.com")
        copy = tmp_path / "copy.json"
        copy.write_text(stored.replace('"refreshToken"', '"scopes": ["x"], "refreshToken"'))
        with pytest.raises(api.ConfigError, match="account 2 \\(b@example.com\\)"):
            api.check_login_source(seeded.switcher, copy)
        live = tmp_path / "live.json"
        live.write_text(json.dumps({"claudeAiOauth": {
            "accessToken": "sk-other", "refreshToken": "rt-live", "scopes": _SCOPES,
        }}))
        with pytest.raises(api.ConfigError, match="default profile uses now"):
            api.check_login_source(seeded.switcher, live)

    def test_refuses_a_profile_of_the_default_logins_account(self, seeded, temp_home):
        _profile_login(
            temp_home / ".claude-a", "a@example.com", "uuid-1",
            credentials=(_login_file(temp_home / "tmp.json").read_text()),
        )
        with pytest.raises(api.ConfigError, match="nothing to move"):
            api.check_login_source(seeded.switcher, temp_home / ".claude-a")


class TestImportLogin:
    """The one-way move: Claude Code turns the login into a new one (sak.7)."""

    def test_moves_a_file_into_a_new_account(
        self, seeded, tmp_path, temp_home, claude, target, monkeypatch, capfd, caplog
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-ignored")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-ignored")
        monkeypatch.setenv("CLAUDECODE", "1")
        default_before = (temp_home / ".claude" / ".credentials.json").read_text()
        source = _login_file(tmp_path / "copy" / ".credentials.json")
        capfd.readouterr()
        moved = api.import_login(
            seeded.switcher, source, target, alias="Work", claude=claude.path
        )
        _assert_silent(capfd)
        assert moved == api.MovedLogin(
            account=api.AddedAccount(slot="3", email="c@example.com", created=True),
            source=api.LoginSource(
                path=str(source), kind="file", email=None, slot=None, expires_at=None
            ),
            default_login=False,
            source_removed=True,
            source_error=None,
        )
        assert not source.exists()
        seen = claude.seen
        assert seen["argv"] == ["auth", "login"]
        assert seen["cwd"] == str(target)
        assert seen["env"]["CLAUDE_CONFIG_DIR"] == str(target)
        assert seen["env"]["CLAUDE_CODE_OAUTH_REFRESH_TOKEN"] == "sk-ant-ort01-source"
        assert seen["env"]["CLAUDE_CODE_OAUTH_SCOPES"] == " ".join(_SCOPES)
        for name in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDECODE"):
            assert name not in seen["env"]
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[3]["email"] == "c@example.com"
        assert rows[3].get("alias") == "work"
        assert api.active_account(seeded.switcher) == "1"
        stored = seeded.switcher.read_account_credentials("3", "c@example.com")
        assert "sk-ant-ort01-moved-uuid-3" in stored
        assert (temp_home / ".claude" / ".credentials.json").read_text() == default_before
        assert "sk-ant-ort01-source" not in caplog.text

    def test_a_managed_account_gets_the_moved_login(
        self, seeded, temp_home, claude, target
    ):
        api.set_alias(seeded.switcher, "2", "side")
        work = _profile_login(
            temp_home / ".claude-work", "b@example.com", "uuid-2",
            credentials=_login_file(temp_home / "tmp.json").read_text(),
        )
        claude.plan("ok", email="b@example.com", uuid="uuid-2")
        moved = api.import_login(
            seeded.switcher, work, target, alias="side", claude=claude.path
        )
        assert moved.account == api.AddedAccount(
            slot="2", email="b@example.com", created=False
        )
        assert moved.source.slot == "2"
        assert moved.source_removed is True
        assert not (work / ".credentials.json").exists()
        assert (work / ".claude.json").exists()
        stored = seeded.switcher.read_account_credentials("2", "b@example.com")
        assert "sk-ant-ort01-moved-uuid-2" in stored
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[2].get("alias") == "side"

    def test_the_default_logins_account_keeps_its_live_login(
        self, seeded, tmp_path, claude, target
    ):
        before = seeded.switcher.read_account_credentials("1", "a@example.com")
        claude.plan("ok", email="a@example.com", uuid="uuid-1")
        source = _login_file(tmp_path / "a.json")
        moved = api.import_login(seeded.switcher, source, target, claude=claude.path)
        assert moved.default_login is True
        assert moved.account == api.AddedAccount(
            slot="1", email="a@example.com", created=False
        )
        assert seeded.switcher.read_account_credentials("1", "a@example.com") == before

    def test_extra_env_reaches_claude_under_its_own_variables(
        self, seeded, tmp_path, claude, target
    ):
        source = _login_file(tmp_path / "copy.json")
        api.import_login(
            seeded.switcher,
            source,
            target,
            claude=claude.path,
            extra_env={
                "DISABLE_TELEMETRY": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CONFIG_DIR": "/somewhere/else",
                "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "sk-ant-ort01-other",
                "CLAUDE_CODE_OAUTH_SCOPES": "user:inference",
                "CLAUDE_SECURESTORAGE_CONFIG_DIR": "/somewhere/else",
                "ANTHROPIC_API_KEY": "sk-ant-api03-not-here",
                "CLAUDECODE": "1",
            },
        )
        env = claude.seen["env"]
        assert env["DISABLE_TELEMETRY"] == "1"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert env["CLAUDE_CONFIG_DIR"] == str(target)
        assert env["CLAUDE_CODE_OAUTH_REFRESH_TOKEN"] == "sk-ant-ort01-source"
        assert env["CLAUDE_CODE_OAUTH_SCOPES"] == " ".join(_SCOPES)
        for name in ("CLAUDE_SECURESTORAGE_CONFIG_DIR", "ANTHROPIC_API_KEY", "CLAUDECODE"):
            assert name not in env

    @pytest.mark.parametrize(
        "extra_env",
        [{"A=B": "1"}, {"": "1"}, {"DISABLE_TELEMETRY": "1\x00"}, {"X": 1}],
    )
    def test_extra_env_no_environment_holds_is_refused(
        self, seeded, tmp_path, claude, target, extra_env
    ):
        source = _login_file(tmp_path / "copy.json")
        with pytest.raises(api.ValidationError, match="extra_env"):
            api.import_login(
                seeded.switcher, source, target, claude=claude.path, extra_env=extra_env
            )
        assert claude.seen is None
        assert source.exists()

    def test_a_failed_move_changes_nothing(
        self, seeded, tmp_path, claude, target, caplog
    ):
        claude.plan("fail")
        source = _login_file(tmp_path / "copy.json")
        before = source.read_text()
        with pytest.raises(api.CredentialError, match="status code 400") as raised:
            api.import_login(seeded.switcher, source, target, claude=claude.path)
        assert not isinstance(raised.value, api.LoginNotAddedError)
        assert "sk-ant-ort01-source" not in str(raised.value)
        assert "sk-ant-ort01-source" not in caplog.text
        assert source.read_text() == before
        assert [r["number"] for r in api.accounts_json(seeded.switcher)["accounts"]] == [
            1,
            2,
        ]

    def test_a_login_stored_before_claude_failed_is_registered(
        self, seeded, tmp_path, claude, target
    ):
        claude.plan("stored-then-fail")
        source = _login_file(tmp_path / "copy.json")
        moved = api.import_login(seeded.switcher, source, target, claude=claude.path)
        assert moved.account.slot == "3"
        assert moved.source_removed is True

    def test_a_hung_claude_times_out(
        self, seeded, tmp_path, claude, target, monkeypatch
    ):
        monkeypatch.setattr(api, "LOGIN_MOVE_TIMEOUT", 0.5)
        claude.plan("hang")
        source = _login_file(tmp_path / "copy.json")
        with pytest.raises(api.CredentialError, match="did not finish"):
            api.import_login(seeded.switcher, source, target, claude=claude.path)
        assert source.exists()

    def test_a_failed_registration_keeps_the_moved_login(
        self, seeded, tmp_path, claude, target, monkeypatch
    ):
        def refuse(*_a, **_k):
            raise api.CredentialError("the keychain is locked")

        monkeypatch.setattr(seeded.switcher, "add_account_from_profile", refuse)
        source = _login_file(tmp_path / "copy.json")
        with pytest.raises(api.LoginNotAddedError, match="keychain is locked") as raised:
            api.import_login(seeded.switcher, source, target, claude=claude.path)
        assert raised.value.config_dir == str(target)
        assert isinstance(raised.value, api.CredentialError)
        assert "sk-ant-ort01-moved" in (target / ".credentials.json").read_text()

    def test_a_taken_alias_is_refused_before_anything_moves(
        self, seeded, tmp_path, claude, target
    ):
        api.set_alias(seeded.switcher, "1", "main")
        source = _login_file(tmp_path / "copy.json")
        with pytest.raises(api.ValidationError, match="already used by account 1"):
            api.import_login(
                seeded.switcher, source, target, alias="main", claude=claude.path
            )
        with pytest.raises(api.ValidationError):
            api.import_login(
                seeded.switcher, source, target, alias="not ok!", claude=claude.path
            )
        assert claude.seen is None
        assert source.exists()

    def test_the_target_must_be_an_empty_profile(
        self, seeded, tmp_path, temp_home, claude
    ):
        source = _login_file(tmp_path / "copy.json")
        with pytest.raises(api.ConfigError, match="is missing"):
            api.import_login(
                seeded.switcher, source, tmp_path / "nowhere", claude=claude.path
            )
        busy = _profile_login(tmp_path / "busy", "d@example.com", "uuid-4")
        with pytest.raises(api.ConfigError, match="holds a login already"):
            api.import_login(seeded.switcher, source, busy, claude=claude.path)
        with pytest.raises(api.ConfigError, match="default profile"):
            api.import_login(
                seeded.switcher, source, temp_home / ".claude", claude=claude.path
            )
        assert claude.seen is None

    def test_a_missing_claude_is_reported(
        self, seeded, tmp_path, target, monkeypatch
    ):
        monkeypatch.setattr(api.shutil, "which", lambda _name: None)
        source = _login_file(tmp_path / "copy.json")
        with pytest.raises(api.SessionError, match="not found on PATH"):
            api.import_login(seeded.switcher, source, target)
        assert source.exists()

    def test_refuses_a_foreign_profile(self, seeded, tmp_path, claude, target, monkeypatch):
        source = _login_file(tmp_path / "copy.json")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "hand-made"))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.import_login(seeded.switcher, source, target, claude=claude.path)
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.check_login_source(seeded.switcher, source)

    def test_an_old_copy_that_cannot_go_is_reported(
        self, seeded, tmp_path, claude, target, monkeypatch
    ):
        source = _login_file(tmp_path / "copy.json")
        real_unlink = Path.unlink

        def stuck(self, missing_ok=False):
            if self == source:
                raise PermissionError("read-only")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", stuck)
        moved = api.import_login(seeded.switcher, source, target, claude=claude.path)
        assert moved.account.slot == "3"
        assert moved.source_removed is False
        assert moved.source_error == "read-only"


class TestSetupToken:
    """Registering a setup-token from `claude setup-token` (sak.7)."""

    def test_adds_a_new_account(self, seeded, capfd, caplog):
        capfd.readouterr()
        added = api.add_setup_token(
            seeded.switcher, "  sk-ant-oat01-secret-token\n", alias="Server"
        )
        _assert_silent(capfd)
        assert added == api.AddedAccount(
            slot="3", email="setup-token-3@token.local", created=True
        )
        stored = seeded.switcher.read_account_credentials("3", added.email)
        assert json.loads(stored) == {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-secret-token",
                "scopes": ["user:inference"],
            }
        }
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[3]["setupToken"] is True
        assert rows[3]["setupTokenAddedAt"].endswith("Z")
        assert rows[3].get("alias") == "server"
        assert "setupToken" not in rows[1] and "setupToken" not in rows[2]
        assert "secret-token" not in caplog.text

    def test_the_same_email_gets_the_new_token(self, seeded):
        first = api.add_setup_token(
            seeded.switcher, "sk-ant-oat01-one", email="ci@example.com", alias="ci"
        )
        again = api.add_setup_token(
            seeded.switcher, "sk-ant-oat01-two", email="ci@example.com", alias="other"
        )
        assert first == api.AddedAccount(slot="3", email="ci@example.com", created=True)
        assert again == api.AddedAccount(slot="3", email="ci@example.com", created=False)
        stored = seeded.switcher.read_account_credentials("3", "ci@example.com")
        assert "sk-ant-oat01-two" in stored
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[3].get("alias") == "ci"

    @pytest.mark.parametrize(
        ("token", "match"),
        [
            ("", "empty"),
            ("   ", "empty"),
            ("sk-ant-api03-key", "API key"),
            ("sk-ant-ort01-refresh", "not a setup-token"),
            ("hello", "not a setup-token"),
            ("sk-ant-oat01-a b", "not a setup-token"),
        ],
    )
    def test_refuses_what_is_not_a_setup_token(self, seeded, token, match):
        with pytest.raises(api.ValidationError, match=match):
            api.add_setup_token(seeded.switcher, token)
        assert [r["number"] for r in api.accounts_json(seeded.switcher)["accounts"]] == [
            1,
            2,
        ]

    def test_refuses_a_login_accounts_email_and_a_taken_alias(self, seeded):
        with pytest.raises(api.ValidationError, match="account 2, which is not"):
            api.add_setup_token(
                seeded.switcher, "sk-ant-oat01-x", email="b@example.com"
            )
        api.set_alias(seeded.switcher, "1", "main")
        with pytest.raises(api.ValidationError, match="already used by account 1"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-x", alias="main")
        with pytest.raises(api.ValidationError, match="Invalid email"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-x", email="nope")
        assert [r["number"] for r in api.accounts_json(seeded.switcher)["accounts"]] == [
            1,
            2,
        ]

    def test_refuses_a_foreign_profile(self, seeded, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "hand-made"))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-x")


_FRESH = 4102444800000
"""2100-01-01, in epoch milliseconds."""


def _copy_text(
    refresh: str | None = "sk-ant-ort01-copy",
    access: str | None = "sk-ant-oat01-copy",
    **extra,
) -> str:
    entry = {
        "accessToken": access,
        "refreshToken": refresh,
        "expiresAt": _FRESH,
        "scopes": _SCOPES,
        **extra,
    }
    return json.dumps({"claudeAiOauth": {k: v for k, v in entry.items() if v is not None}})


def _owner(num: int, email: str, org: str | None, name: str | None = None) -> dict:
    return {
        "uuid": f"uuid-{num}",
        "email": email,
        "organizationUuid": org,
        "organizationName": name,
    }


def _give_org(seeded, num: int, org: str) -> None:
    data = seeded.switcher._get_sequence_data()
    data["accounts"][str(num)]["organizationUuid"] = org
    seeded.switcher._write_json(seeded.switcher.sequence_file, data)


def _numbers(seeded) -> list[int]:
    return [r["number"] for r in api.accounts_json(seeded.switcher)["accounts"]]


class _Anthropic:
    """Anthropic's side of a copy: whose each access token is, and renewals."""

    def __init__(self) -> None:
        owner = _owner(3, "c@example.com", "org-3", "Acme")
        self.owners: dict[str, dict | None] = {
            "sk-ant-oat01-copy": owner,
            "sk-ant-oat01-renewed": owner,
        }
        self.asked: list[str] = []
        self.renewals: list[str] = []
        self.renewal: str | None = None

    def profile(self, token: str) -> dict | None:
        self.asked.append(token)
        return self.owners.get(token)

    def refresh(self, credentials: str, timeout_s: float = 10.0) -> RefreshOutcome:
        self.renewals.append(credentials)
        if self.renewal is not None:
            return RefreshOutcome(None, self.renewal)
        data = json.loads(credentials)
        data["claudeAiOauth"].update(
            accessToken="sk-ant-oat01-renewed",
            refreshToken="sk-ant-ort01-renewed",
            expiresAt=_FRESH,
        )
        return RefreshOutcome(json.dumps(data), None)


class TestLoginCopy:
    """Copying a login in as it is, the source left alone (sak.8)."""

    @pytest.fixture
    def anthropic(self, monkeypatch) -> _Anthropic:
        fake = _Anthropic()
        monkeypatch.setattr("claude_swap.oauth.fetch_oauth_profile", fake.profile)
        monkeypatch.setattr("claude_swap.oauth.try_refresh_oauth_credentials", fake.refresh)
        return fake

    def test_copies_a_file_into_a_new_account(
        self, seeded, tmp_path, temp_home, target, anthropic, capfd, caplog
    ):
        default_before = (temp_home / ".claude" / ".credentials.json").read_text()
        source = tmp_path / "copy" / ".credentials.json"
        source.parent.mkdir()
        source.write_text(_copy_text() + "\n")
        before = source.read_text()
        capfd.readouterr()
        copied = api.add_login_copy(seeded.switcher, target, source=source, alias="Work")
        _assert_silent(capfd)
        assert copied == api.CopiedLogin(
            account=api.AddedAccount(slot="3", email="c@example.com", created=True),
            default_login=False,
            renewed=False,
        )
        assert api.CopiedLogin._fields == ("account", "default_login", "renewed")
        assert source.read_text() == before
        assert anthropic.renewals == []
        assert seeded.switcher.read_account_credentials("3", "c@example.com") == (
            _copy_text()
        )
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[3]["email"] == "c@example.com"
        assert rows[3]["organizationUuid"] == "org-3"
        assert rows[3]["organizationName"] == "Acme"
        assert rows[3].get("alias") == "work"
        assert api.active_account(seeded.switcher) == "1"
        assert (temp_home / ".claude" / ".credentials.json").read_text() == default_before
        kept = target / ".credentials.json"
        assert kept.read_text() == _copy_text()
        if os.name != "nt":
            assert kept.stat().st_mode & 0o777 == 0o600
        assert json.loads((target / ".claude.json").read_text()) == {
            "oauthAccount": {
                "accountUuid": "uuid-3",
                "emailAddress": "c@example.com",
                "organizationUuid": "org-3",
                "organizationName": "Acme",
            }
        }
        assert "sk-ant-ort01-copy" not in caplog.text
        assert "sk-ant-oat01-copy" not in caplog.text

    def test_copies_text_as_given(self, seeded, target, anthropic):
        copied = api.add_login_copy(
            seeded.switcher, target, credentials="\n  " + _copy_text() + "\n"
        )
        assert copied.account == api.AddedAccount(
            slot="3", email="c@example.com", created=True
        )
        assert copied.renewed is False
        assert seeded.switcher.read_account_credentials("3", "c@example.com") == (
            _copy_text()
        )
        assert anthropic.asked[0] == "sk-ant-oat01-copy"

    @pytest.mark.parametrize("lapsed", [{"expiresAt": 1_000}, {"access": None}])
    def test_a_lapsed_access_token_is_renewed_first(
        self, seeded, target, anthropic, caplog, lapsed
    ):
        text = _copy_text(**lapsed)
        copied = api.add_login_copy(seeded.switcher, target, credentials=text)
        assert copied.renewed is True
        assert copied.account.slot == "3"
        assert anthropic.renewals == [text]
        assert anthropic.asked[0] == "sk-ant-oat01-renewed"
        stored = seeded.switcher.read_account_credentials("3", "c@example.com")
        assert "sk-ant-ort01-renewed" in stored
        assert "sk-ant-ort01-renewed" in (target / ".credentials.json").read_text()
        assert "sk-ant-ort01-renewed" not in caplog.text

    @pytest.mark.parametrize(
        ("error", "match"),
        [
            ("invalid_grant", "no longer works"),
            ("transient", "could not be renewed now \\(transient\\)"),
        ],
    )
    def test_a_login_that_cannot_be_renewed_is_refused(
        self, seeded, target, anthropic, error, match
    ):
        anthropic.renewal = error
        with pytest.raises(api.CredentialError, match=match) as raised:
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(expiresAt=1_000)
            )
        assert not isinstance(raised.value, api.LoginNotAddedError)
        assert anthropic.asked == []
        assert _numbers(seeded) == [1, 2]

    @pytest.mark.parametrize(
        "owner",
        [
            None,
            _owner(3, None, "org-3"),
            _owner(3, "c@example.com", None),
        ],
    )
    def test_a_login_anthropic_cannot_place_is_refused(
        self, seeded, target, anthropic, owner
    ):
        anthropic.owners["sk-ant-oat01-copy"] = owner
        with pytest.raises(api.CredentialError, match="did not say whose login") as raised:
            api.add_login_copy(seeded.switcher, target, credentials=_copy_text())
        assert not isinstance(raised.value, api.LoginNotAddedError)
        assert _numbers(seeded) == [1, 2]

    def test_a_renewed_copy_anthropic_cannot_place_is_kept(
        self, seeded, target, anthropic
    ):
        anthropic.owners["sk-ant-oat01-renewed"] = None
        with pytest.raises(api.LoginNotAddedError, match="did not say whose") as raised:
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(expiresAt=1_000)
            )
        assert raised.value.config_dir == str(target)
        assert "sk-ant-ort01-renewed" in (target / ".credentials.json").read_text()
        assert _numbers(seeded) == [1, 2]

    def test_a_renewed_copy_that_cannot_be_registered_is_kept(
        self, seeded, target, anthropic, monkeypatch
    ):
        def refuse(*_a, **_k):
            raise api.CredentialError("the keychain is locked")

        monkeypatch.setattr(seeded.switcher, "add_account_from_profile", refuse)
        with pytest.raises(api.LoginNotAddedError, match="keychain is locked") as raised:
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(expiresAt=1_000)
            )
        assert raised.value.config_dir == str(target)
        assert "sk-ant-ort01-renewed" in (target / ".credentials.json").read_text()

    def test_a_copy_that_cannot_be_registered_changes_nothing(
        self, seeded, target, anthropic, monkeypatch
    ):
        def refuse(*_a, **_k):
            raise api.CredentialError("the keychain is locked")

        monkeypatch.setattr(seeded.switcher, "add_account_from_profile", refuse)
        with pytest.raises(api.CredentialError, match="keychain is locked") as raised:
            api.add_login_copy(seeded.switcher, target, credentials=_copy_text())
        assert not isinstance(raised.value, api.LoginNotAddedError)
        assert anthropic.renewals == []

    def test_an_account_managed_already_gets_the_copy(self, seeded, target, anthropic):
        _give_org(seeded, 2, "org-2")
        api.set_alias(seeded.switcher, "2", "side")
        anthropic.owners["sk-ant-oat01-copy"] = _owner(2, "b@example.com", "org-2")
        copied = api.add_login_copy(
            seeded.switcher, target, credentials=_copy_text(), alias="other"
        )
        assert copied.account == api.AddedAccount(
            slot="2", email="b@example.com", created=False
        )
        assert copied.default_login is False
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == (
            _copy_text()
        )
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[2].get("alias") == "side"
        assert _numbers(seeded) == [1, 2]

    def test_the_default_logins_account_keeps_its_live_login(
        self, seeded, temp_home, target, anthropic
    ):
        _give_org(seeded, 1, "org-1")
        (temp_home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {
                "emailAddress": "a@example.com",
                "accountUuid": "uuid-1",
                "organizationUuid": "org-1",
            },
        }))
        before = seeded.switcher.read_account_credentials("1", "a@example.com")
        anthropic.owners["sk-ant-oat01-copy"] = _owner(1, "a@example.com", "org-1")
        copied = api.add_login_copy(seeded.switcher, target, credentials=_copy_text())
        assert copied.default_login is True
        assert copied.account == api.AddedAccount(
            slot="1", email="a@example.com", created=False
        )
        assert seeded.switcher.read_account_credentials("1", "a@example.com") == before

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("sk-ant-api03-key", "API key"),
            ("sk-ant-oat01-bare", "bare token"),
            ('{"other": 1}', "no Claude login"),
            ("not json", "no Claude login"),
            (
                json.dumps({"claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-t", "scopes": ["user:inference"],
                }}),
                "setup-token",
            ),
            (_copy_text(refresh=None), "logged out or emptied"),
            (
                _copy_text(refreshTokenExpiresAt=1_000),
                "holds a login that expired on 1970-01-01T00:00:01Z",
            ),
            (_copy_text(scopes=None), "needs to use it"),
        ],
    )
    def test_refuses_what_cannot_be_copied(
        self, seeded, target, anthropic, content, match
    ):
        with pytest.raises(api.ValidationError, match=match) as raised:
            api.add_login_copy(seeded.switcher, target, credentials=content)
        assert str(raised.value).startswith("The text given ")
        assert anthropic.asked == []
        assert anthropic.renewals == []
        assert not (target / ".credentials.json").exists()
        assert _numbers(seeded) == [1, 2]

    def test_refuses_what_is_not_one_credentials_file(
        self, seeded, tmp_path, temp_home, target, anthropic
    ):
        with pytest.raises(api.ConfigError, match="either the credentials or"):
            api.add_login_copy(seeded.switcher, target)
        with pytest.raises(api.ConfigError, match="either the credentials or"):
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(), source=tmp_path
            )
        with pytest.raises(api.ConfigError, match="does not exist"):
            api.add_login_copy(seeded.switcher, target, source=tmp_path / "gone.json")
        with pytest.raises(api.ConfigError, match="Name a credentials file"):
            api.add_login_copy(seeded.switcher, target, source="  ")
        folder = tmp_path / "folder"
        folder.mkdir()
        with pytest.raises(api.ConfigError, match="not a file"):
            api.add_login_copy(seeded.switcher, target, source=folder)
        with pytest.raises(api.ConfigError, match="default profile"):
            api.add_login_copy(
                seeded.switcher,
                target,
                source=temp_home / ".claude" / ".credentials.json",
            )
        with pytest.raises(api.ValidationError, match="too big"):
            api.add_login_copy(
                seeded.switcher, target, credentials="x" * ((1 << 20) + 1)
            )
        assert anthropic.asked == []

    def test_refuses_a_login_that_is_managed_already(self, seeded, target, anthropic):
        stored = seeded.switcher.read_account_credentials("2", "b@example.com")
        copy = stored.replace('"refreshToken"', '"scopes": ["x"], "refreshToken"')
        with pytest.raises(api.ConfigError, match="account 2 \\(b@example.com\\)"):
            api.add_login_copy(seeded.switcher, target, credentials=copy)
        live = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-other", "refreshToken": "rt-live", "scopes": _SCOPES,
        }})
        with pytest.raises(api.ConfigError, match="default profile uses now"):
            api.add_login_copy(seeded.switcher, target, credentials=live)
        assert anthropic.asked == []

    def test_a_taken_alias_is_refused_before_anything_happens(
        self, seeded, target, anthropic
    ):
        api.set_alias(seeded.switcher, "1", "main")
        with pytest.raises(api.ValidationError, match="already used by account 1"):
            api.add_login_copy(
                seeded.switcher,
                target,
                credentials=_copy_text(expiresAt=1_000),
                alias="main",
            )
        assert anthropic.renewals == []
        assert anthropic.asked == []
        assert not (target / ".credentials.json").exists()

    def test_the_target_must_be_an_empty_profile(
        self, seeded, tmp_path, anthropic
    ):
        with pytest.raises(api.ConfigError, match="profile to copy the login into"):
            api.add_login_copy(
                seeded.switcher, tmp_path / "nowhere", credentials=_copy_text()
            )
        busy = _profile_login(tmp_path / "busy", "d@example.com", "uuid-4")
        with pytest.raises(api.ConfigError, match="holds a login already"):
            api.add_login_copy(seeded.switcher, busy, credentials=_copy_text())
        assert anthropic.asked == []

    def test_refuses_a_foreign_profile(
        self, seeded, tmp_path, target, anthropic, monkeypatch
    ):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "hand-made"))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.add_login_copy(seeded.switcher, target, credentials=_copy_text())


class TestUpdateSetupToken:
    """A setup-token account getting a new setup-token in place (sak.9)."""

    def _token_account(self, seeded) -> api.AddedAccount:
        return api.add_setup_token(
            seeded.switcher, "sk-ant-oat01-one", email="ci@example.com", alias="ci"
        )

    def _added_at(self, seeded, slot: str) -> str:
        return seeded.switcher._get_sequence_data()["accounts"][slot][
            "setupTokenAddedAt"
        ]

    def test_the_account_gets_the_new_token(self, seeded, capfd, caplog, monkeypatch):
        added = self._token_account(seeded)
        monkeypatch.setattr(api, "get_timestamp", lambda: "2030-01-01T00:00:00Z")
        capfd.readouterr()
        again = api.add_setup_token(seeded.switcher, " sk-ant-oat01-two\n", account="ci")
        _assert_silent(capfd)
        assert again == api.AddedAccount(
            slot=added.slot, email="ci@example.com", created=False
        )
        stored = seeded.switcher.read_account_credentials(added.slot, "ci@example.com")
        assert json.loads(stored) == {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-two",
                "scopes": ["user:inference"],
            }
        }
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[3].get("alias") == "ci"
        assert rows[3]["setupTokenAddedAt"] == "2030-01-01T00:00:00Z"
        assert _numbers(seeded) == [1, 2, 3]
        assert "sk-ant-oat01-two" not in caplog.text

    def test_the_default_login_runs_on_the_new_token_too(self, seeded, capfd):
        added = self._token_account(seeded)
        api.switch_to(seeded.switcher, "ci")
        capfd.readouterr()
        api.add_setup_token(seeded.switcher, "sk-ant-oat01-two", account="ci")
        _assert_silent(capfd)
        live = json.loads(seeded.switcher._read_credentials())
        assert live["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-two"
        assert api.active_account(seeded.switcher) == added.slot
        # switching away backs the live login up into the account: the new one
        api.switch_to(seeded.switcher, "1")
        stored = seeded.switcher.read_account_credentials(added.slot, "ci@example.com")
        assert json.loads(stored)["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-two"

    def test_the_same_email_again_goes_live_too(self, seeded):
        self._token_account(seeded)
        api.switch_to(seeded.switcher, "ci")
        api.add_setup_token(seeded.switcher, "sk-ant-oat01-two", email="ci@example.com")
        live = json.loads(seeded.switcher._read_credentials())
        assert live["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-two"

    def test_the_old_token_stays_when_the_default_login_cannot_take_it(
        self, seeded, monkeypatch
    ):
        added = self._token_account(seeded)
        api.switch_to(seeded.switcher, "ci")
        before = self._added_at(seeded, added.slot)

        def refuse(*_args, **_kwargs):
            raise api.SwitchError("the keychain refused the write")

        monkeypatch.setattr(seeded.switcher, "switch_to", refuse)
        with pytest.raises(api.SwitchError, match="keychain refused"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-two", account="ci")
        stored = seeded.switcher.read_account_credentials(added.slot, "ci@example.com")
        assert json.loads(stored)["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-one"
        assert self._added_at(seeded, added.slot) == before

    def test_a_placeholder_email_works_by_slot(self, seeded):
        added = api.add_setup_token(seeded.switcher, "sk-ant-oat01-one")
        again = api.add_setup_token(seeded.switcher, "sk-ant-oat01-two", account="3")
        assert again == api.AddedAccount(
            slot="3", email=added.email, created=False
        )
        assert "sk-ant-oat01-two" in seeded.switcher.read_account_credentials(
            "3", added.email
        )

    def test_refuses_the_token_the_account_has_already(self, seeded):
        added = self._token_account(seeded)
        before = self._added_at(seeded, added.slot)
        with pytest.raises(api.ValidationError, match="has this setup-token already"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-one", account="3")
        assert self._added_at(seeded, added.slot) == before

    def test_refuses_an_account_that_runs_on_a_login(self, seeded):
        before = seeded.switcher.read_account_credentials("2", "b@example.com")
        with pytest.raises(api.ValidationError, match="runs on a login, not a"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-x", account="2")
        _give_org(seeded, 2, "org-2")
        with pytest.raises(api.ValidationError, match="runs on a login, not a"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-x", account="2")
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == before

    def test_refuses_what_does_not_go_with_an_account(self, seeded):
        self._token_account(seeded)
        with pytest.raises(api.ConfigError, match="not both"):
            api.add_setup_token(
                seeded.switcher, "sk-ant-oat01-x", account="3", email="d@example.com"
            )
        with pytest.raises(api.ConfigError, match="not both"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-x", account="3", alias="x")
        with pytest.raises(api.AccountNotFoundError):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-x", account="9")
        with pytest.raises(api.ValidationError, match="not a setup-token"):
            api.add_setup_token(seeded.switcher, "hello", account="3")
        assert _numbers(seeded) == [1, 2, 3]

    def test_an_unreadable_stored_login_is_not_taken_for_missing(
        self, seeded, monkeypatch
    ):
        self._token_account(seeded)
        monkeypatch.setattr(
            seeded.switcher, "_read_account_credentials_ex", lambda *_a: ("", True)
        )
        with pytest.raises(api.CredentialReadError, match="locked or busy"):
            api.add_setup_token(seeded.switcher, "sk-ant-oat01-two", account="3")


class TestUpdateLoginCopy:
    """A login account getting a copy of its own login in place (sak.9)."""

    @pytest.fixture
    def anthropic(self, monkeypatch) -> _Anthropic:
        fake = _Anthropic()
        monkeypatch.setattr("claude_swap.oauth.fetch_oauth_profile", fake.profile)
        monkeypatch.setattr("claude_swap.oauth.try_refresh_oauth_credentials", fake.refresh)
        return fake

    @pytest.fixture
    def owned(self, seeded, anthropic) -> _Anthropic:
        """Account 2 has its organization, and the copy is its login."""
        _give_org(seeded, 2, "org-2")
        api.set_alias(seeded.switcher, "2", "side")
        owner = _owner(2, "b@example.com", "org-2", "Bee")
        anthropic.owners["sk-ant-oat01-copy"] = owner
        anthropic.owners["sk-ant-oat01-renewed"] = owner
        return anthropic

    def test_the_account_gets_the_copy(self, seeded, target, owned, capfd, caplog):
        capfd.readouterr()
        copied = api.add_login_copy(
            seeded.switcher, target, credentials=_copy_text(), account="side"
        )
        _assert_silent(capfd)
        assert copied == api.CopiedLogin(
            account=api.AddedAccount(slot="2", email="b@example.com", created=False),
            default_login=False,
            renewed=False,
        )
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == (
            _copy_text()
        )
        rows = {r["number"]: r for r in api.accounts_json(seeded.switcher)["accounts"]}
        assert rows[2].get("alias") == "side"
        assert _numbers(seeded) == [1, 2]
        assert "sk-ant-ort01-copy" not in caplog.text

    def test_a_file_works_and_stays_as_it_is(self, seeded, tmp_path, target, owned):
        source = tmp_path / "work" / ".credentials.json"
        source.parent.mkdir()
        source.write_text(_copy_text())
        before = source.read_text()
        copied = api.add_login_copy(seeded.switcher, target, source=source, account="2")
        assert copied.account.slot == "2"
        assert source.read_text() == before

    def test_a_lapsed_copy_is_renewed_first(self, seeded, target, owned):
        copied = api.add_login_copy(
            seeded.switcher, target, credentials=_copy_text(expiresAt=1_000), account="2"
        )
        assert copied.renewed is True
        stored = seeded.switcher.read_account_credentials("2", "b@example.com")
        assert "sk-ant-ort01-renewed" in stored

    def test_another_accounts_login_is_refused_and_changes_nothing(
        self, seeded, target, anthropic
    ):
        _give_org(seeded, 2, "org-2")
        before = seeded.switcher.read_account_credentials("2", "b@example.com")
        with pytest.raises(
            api.ValidationError,
            match=(
                r"^The text given is the login of c@example.com, not of account 2 "
                r"\(b@example.com\); nothing changed\.$"
            ),
        ) as raised:
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(), account="2"
            )
        assert not isinstance(raised.value, api.LoginNotAddedError)
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == before
        assert _numbers(seeded) == [1, 2]
        assert anthropic.renewals == []

    def test_the_same_email_in_another_organization_is_named(
        self, seeded, target, anthropic
    ):
        _give_org(seeded, 2, "org-2")
        anthropic.owners["sk-ant-oat01-copy"] = _owner(2, "b@example.com", "org-x", "Ex")
        with pytest.raises(
            api.ValidationError, match=r"b@example.com in another organization \(Ex\)"
        ):
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(), account="2"
            )
        assert _numbers(seeded) == [1, 2]

    def test_a_renewed_copy_of_another_login_is_kept(self, seeded, target, anthropic):
        _give_org(seeded, 2, "org-2")
        with pytest.raises(api.LoginNotAddedError, match="not of account 2") as raised:
            api.add_login_copy(
                seeded.switcher,
                target,
                credentials=_copy_text(expiresAt=1_000),
                account="2",
            )
        assert "nothing changed" not in str(raised.value)
        assert raised.value.config_dir == str(target)
        assert "sk-ant-ort01-renewed" in (target / ".credentials.json").read_text()
        assert _numbers(seeded) == [1, 2]

    def test_refuses_the_login_the_account_has_already(self, seeded, target, owned):
        stored = seeded.switcher.read_account_credentials("2", "b@example.com")
        copy = stored.replace('"refreshToken"', '"scopes": ["x"], "refreshToken"')
        with pytest.raises(api.ConfigError, match="the login account 2 .* has already"):
            api.add_login_copy(seeded.switcher, target, credentials=copy, account="2")
        assert owned.asked == []

    def test_refuses_the_default_logins_account(
        self, seeded, temp_home, target, anthropic
    ):
        with pytest.raises(api.ConfigError, match="default profile is logged in to"):
            api.add_login_copy(
                seeded.switcher,
                target,
                credentials=_copy_text(expiresAt=1_000),
                account="1",
            )
        assert anthropic.renewals == []
        assert anthropic.asked == []
        assert not (target / ".credentials.json").exists()

    def test_refuses_a_setup_token_account(self, seeded, target, anthropic):
        api.add_setup_token(seeded.switcher, "sk-ant-oat01-one", email="ci@example.com")
        with pytest.raises(api.ValidationError, match="takes a new setup-token"):
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(), account="3"
            )
        assert anthropic.asked == []

    def test_refuses_what_does_not_go_with_an_account(self, seeded, target, anthropic):
        with pytest.raises(api.ConfigError, match="keeps its alias"):
            api.add_login_copy(
                seeded.switcher,
                target,
                credentials=_copy_text(),
                account="2",
                alias="x",
            )
        with pytest.raises(api.AccountNotFoundError):
            api.add_login_copy(
                seeded.switcher, target, credentials=_copy_text(), account="9"
            )
        assert anthropic.asked == []
        assert not (target / ".credentials.json").exists()


def _login(refresh: str = "rt-2", **extra) -> dict:
    """A stored login as Claude Code keeps it, with what a move needs."""
    entry = {
        "accessToken": "sk-2",
        "refreshToken": refresh,
        "expiresAt": 1_900_000_000_000,
        "scopes": _SCOPES,
        **extra,
    }
    return {"claudeAiOauth": {k: v for k, v in entry.items() if v is not None}}


def _store_login(seeded, num: int = 2, email: str = "b@example.com", **login) -> dict:
    stored = _login(**login)
    seeded.switcher._write_account_credentials(str(num), email, json.dumps(stored))
    return stored


class TestHandOverLogin:
    """The other half of the one-way move (sak.11): an account's login leaves
    claude-swap for a credentials file, and the account goes."""

    @pytest.fixture
    def stored(self, seeded) -> dict:
        return _store_login(seeded)

    def test_writes_the_login_and_removes_the_account(
        self, seeded, stored, tmp_path, capfd
    ):
        api.map_project(seeded.switcher, tmp_path, "2")
        api.set_alias(seeded.switcher, "2", "work")
        destination = tmp_path / "login.json"
        capfd.readouterr()
        handed = api.hand_over_login(seeded.switcher, "work", destination)
        _assert_silent(capfd)
        assert handed == api.HandedOver(
            slot="2",
            email="b@example.com",
            organization_uuid=None,
            organization=None,
            alias="work",
            expires_at=None,
        )
        assert handed.removal_error is None
        assert json.loads(destination.read_text()) == stored
        assert destination.stat().st_mode & 0o777 == 0o600
        assert _numbers(seeded) == [1]
        assert api.project_mappings(seeded.switcher) == []

    def test_carries_the_logins_expiry_and_organization(self, seeded, tmp_path):
        _store_login(seeded, refreshTokenExpiresAt=1_900_000_000_000)
        _give_org(seeded, 2, "org-2")
        data = seeded.switcher._get_sequence_data()
        data["accounts"]["2"]["organizationName"] = "Team"
        seeded.switcher._write_json(seeded.switcher.sequence_file, data)
        handed = api.hand_over_login(seeded.switcher, "2", tmp_path / "login.json")
        assert handed.organization_uuid == "org-2"
        assert handed.organization == "Team"
        assert handed.expires_at == "2030-03-17T17:46:40Z"

    def test_the_file_is_on_the_disk_before_the_account_goes(
        self, seeded, stored, tmp_path, monkeypatch
    ):
        destination = tmp_path / "login.json"
        steps: list[str] = []
        monkeypatch.setattr(api, "sync_file", lambda fd: steps.append("file"))
        monkeypatch.setattr(api, "sync_folder", lambda folder: steps.append("folder"))
        remove = seeded.switcher.remove_account

        def removing(*a, **k):
            steps.append("remove")
            assert json.loads(destination.read_text()) == stored
            return remove(*a, **k)

        monkeypatch.setattr(seeded.switcher, "remove_account", removing)
        api.hand_over_login(seeded.switcher, "2", destination)
        assert steps == ["file", "folder", "remove"]

    def test_refuses_the_default_logins_account(self, seeded, tmp_path):
        destination = tmp_path / "login.json"
        with pytest.raises(api.ConfigError, match="switch to another account first"):
            api.hand_over_login(seeded.switcher, "1", destination)
        assert not destination.exists()
        assert _numbers(seeded) == [1, 2]

    def test_refuses_an_account_a_live_session_uses(self, seeded, stored, tmp_path):
        from claude_swap.session import session_dir_for

        profile = session_dir_for(seeded.switcher.backup_dir, "2", "b@example.com")
        (profile / "sessions").mkdir(parents=True)
        (profile / "sessions" / f"{os.getpid()}.json").write_text(
            json.dumps({"pid": os.getpid(), "sessionId": "s", "cwd": "/"})
        )
        destination = tmp_path / "login.json"
        with pytest.raises(api.SessionError, match="end it first"):
            api.hand_over_login(seeded.switcher, "2", destination)
        assert not destination.exists()
        assert _numbers(seeded) == [1, 2]

    @pytest.mark.parametrize(
        ("stored", "match"),
        [
            ("sk-ant-api03-key", "API key"),
            (
                json.dumps({
                    "claudeAiOauth": {
                        "accessToken": "sk-ant-oat01-x",
                        "scopes": ["user:inference"],
                    }
                }),
                "setup-token",
            ),
        ],
    )
    def test_refuses_what_is_not_a_login(self, seeded, tmp_path, stored, match):
        seeded.switcher._write_account_credentials("2", "b@example.com", stored)
        destination = tmp_path / "login.json"
        with pytest.raises(api.ValidationError, match=match):
            api.hand_over_login(seeded.switcher, "2", destination)
        assert not destination.exists()
        assert _numbers(seeded) == [1, 2]

    @pytest.mark.parametrize(
        ("login", "match"),
        [
            ({"scopes": None}, "scopes"),
            ({"refresh": ""}, "no refresh token"),
            ({"refreshTokenExpiresAt": 1_000_000_000_000}, "expired"),
        ],
    )
    def test_refuses_a_login_the_other_computer_could_not_take(
        self, seeded, tmp_path, login, match
    ):
        before = _store_login(seeded, **login)
        destination = tmp_path / "login.json"
        with pytest.raises(api.ValidationError, match=match):
            api.hand_over_login(seeded.switcher, "2", destination)
        assert not destination.exists()
        assert _numbers(seeded) == [1, 2]
        stored = seeded.switcher.read_account_credentials("2", "b@example.com")
        assert json.loads(stored) == before

    def test_refuses_an_existing_file_and_a_place_it_must_not_go(
        self, seeded, stored, temp_home, tmp_path
    ):
        existing = tmp_path / "login.json"
        existing.write_text("keep")
        with pytest.raises(api.ConfigError, match="exists already"):
            api.hand_over_login(seeded.switcher, "2", existing)
        assert existing.read_text() == "keep"
        for place, match in (
            (temp_home / ".claude" / "login.json", "default profile"),
            (Path(seeded.switcher.backup_dir) / "login.json", "own store"),
            (tmp_path / "missing" / "login.json", "not a folder"),
            ("", "Name the file"),
        ):
            with pytest.raises(api.ConfigError, match=match):
                api.hand_over_login(seeded.switcher, "2", place)
        assert _numbers(seeded) == [1, 2]

    def test_a_failed_removal_takes_the_file_back(
        self, seeded, stored, tmp_path, monkeypatch
    ):
        def fail(*_a, **_k):
            raise api.LockError("busy")

        monkeypatch.setattr(seeded.switcher, "remove_account", fail)
        destination = tmp_path / "login.json"
        with pytest.raises(api.LockError, match="busy"):
            api.hand_over_login(seeded.switcher, "2", destination)
        assert not destination.exists()
        assert _numbers(seeded) == [1, 2]

    def test_a_removal_that_fails_after_the_login_went_keeps_the_file(
        self, seeded, stored, tmp_path, monkeypatch, capfd
    ):
        sequence = seeded.switcher.sequence_file
        write = seeded.switcher._write_json

        def full_disk(target, data):
            if target == sequence:
                raise OSError(28, "No space left on device")
            return write(target, data)

        monkeypatch.setattr(seeded.switcher, "_write_json", full_disk)
        destination = tmp_path / "login.json"
        capfd.readouterr()
        handed = api.hand_over_login(seeded.switcher, "2", destination)
        _assert_silent(capfd)
        assert "No space left on device" in (handed.removal_error or "")
        assert json.loads(destination.read_text()) == stored
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == ""
        monkeypatch.setattr(seeded.switcher, "_write_json", write)
        assert _numbers(seeded) == [1, 2]  # listed still, without its login

    def test_a_removal_whose_outcome_cannot_be_read_keeps_the_file(
        self, seeded, stored, tmp_path, monkeypatch
    ):
        removing = {"started": False}

        def fail(*_a, **_k):
            removing["started"] = True
            raise api.CredentialError("the keychain went away")

        monkeypatch.setattr(seeded.switcher, "remove_account", fail)
        read = seeded.switcher._read_account_credentials_ex

        def unreadable_after_the_removal(slot, email):
            return ("", True) if removing["started"] else read(slot, email)

        monkeypatch.setattr(
            seeded.switcher, "_read_account_credentials_ex", unreadable_after_the_removal
        )
        destination = tmp_path / "login.json"
        handed = api.hand_over_login(seeded.switcher, "2", destination)
        assert handed.removal_error == "the keychain went away"
        assert json.loads(destination.read_text()) == stored

    def test_an_interrupt_after_the_login_went_keeps_the_file(
        self, seeded, stored, tmp_path, monkeypatch
    ):
        def interrupted(slot, assume_yes=False):
            seeded.switcher._delete_account_credentials(slot, "b@example.com")
            raise KeyboardInterrupt

        monkeypatch.setattr(seeded.switcher, "remove_account", interrupted)
        destination = tmp_path / "login.json"
        with pytest.raises(KeyboardInterrupt):
            api.hand_over_login(seeded.switcher, "2", destination)
        assert json.loads(destination.read_text()) == stored

    def test_hands_over_the_login_its_session_profile_renewed(
        self, seeded, stored, tmp_path
    ):
        from claude_swap.session import session_dir_for

        profile = session_dir_for(seeded.switcher.backup_dir, "2", "b@example.com")
        profile.mkdir(parents=True)
        renewed = _login(refresh="rt-2-renewed", expiresAt=1_950_000_000_000)
        (profile / ".credentials.json").write_text(json.dumps(renewed))
        destination = tmp_path / "login.json"
        api.hand_over_login(seeded.switcher, "2", destination)
        assert json.loads(destination.read_text()) == renewed
        assert _numbers(seeded) == [1]

    def test_hands_over_a_renewal_claude_swap_set_aside(
        self, seeded, stored, tmp_path
    ):
        from claude_swap.oauth import credential_fingerprint

        successor = _login(refresh="rt-2-successor", expiresAt=1_950_000_000_000)
        seeded.switcher._store._write_unclaimed_credential(
            json.dumps(successor),
            {
                "reason": "consume-gate-unpersisted",
                "configSlot": "2",
                "consumedFp": credential_fingerprint(json.dumps(stored)),
                "fingerprint": credential_fingerprint(json.dumps(successor)),
            },
        )
        destination = tmp_path / "login.json"
        api.hand_over_login(seeded.switcher, "2", destination)
        assert json.loads(destination.read_text()) == successor
        assert seeded.switcher.list_unclaimed_credentials() == {}

    def test_waits_for_a_renewal_of_the_account(
        self, seeded, stored, tmp_path, monkeypatch
    ):
        from claude_swap.locking import FileLock

        monkeypatch.setattr(api, "HAND_OVER_LOCK_TIMEOUT", 0.2)
        destination = tmp_path / "login.json"
        with (
            FileLock(seeded.switcher.consume_lock_file("2")),
            pytest.raises(api.LockError, match="being renewed"),
        ):
            api.hand_over_login(seeded.switcher, "2", destination)
        assert not destination.exists()
        assert _numbers(seeded) == [1, 2]
        api.hand_over_login(seeded.switcher, "2", destination)
        assert _numbers(seeded) == [1]

    def test_unknown_account_raises(self, seeded, tmp_path):
        with pytest.raises(api.AccountNotFoundError):
            api.hand_over_login(seeded.switcher, "9", tmp_path / "login.json")

    def test_refuses_a_foreign_profile(self, seeded, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "hand-made"))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.hand_over_login(seeded.switcher, "2", tmp_path / "login.json")


# -- an access token for another computer (sak.12) ----------------------------------


def _in(*, minutes: float = 0, hours: float = 0) -> int:
    """Epoch milliseconds that far from now."""
    return int((time.time() + minutes * 60 + hours * 3600) * 1000)


def _token_login(access: str, refresh: str | None, expires: int, **extra) -> dict:
    """A login as Claude Code keeps it, with what an access token carries."""
    entry = {
        "accessToken": access,
        "refreshToken": refresh,
        "expiresAt": expires,
        "scopes": _SCOPES,
        "subscriptionType": "max",
        "rateLimitTier": "default_claude_max_20x",
        **extra,
    }
    return {"claudeAiOauth": entry}


class _TokenEndpoint:
    """Anthropic's token endpoint: a refresh token renews its login once."""

    def __init__(self) -> None:
        self.posted: list[str] = []
        self.error: str | None = None
        self.during = None

    def __call__(self, credentials: str, timeout_s: float = 10.0) -> RefreshOutcome:
        data = json.loads(credentials)
        entry = data["claudeAiOauth"]
        refresh = entry.get("refreshToken")
        if not refresh:
            return RefreshOutcome(None, "no_refresh_token")
        self.posted.append(refresh)
        if self.during is not None:
            self.during()
        if self.error is not None:
            return RefreshOutcome(None, self.error)
        renewed = {
            **entry,
            "accessToken": f"sk-ant-oat01-renewed-{len(self.posted)}",
            "refreshToken": f"{refresh}-next",
            "expiresAt": _in(hours=8),
        }
        return RefreshOutcome(json.dumps({**data, "claudeAiOauth": renewed}), None)


@pytest.fixture
def endpoint(monkeypatch) -> _TokenEndpoint:
    fake = _TokenEndpoint()
    monkeypatch.setattr("claude_swap.oauth.try_refresh_oauth_credentials", fake)
    return fake


def _stored(seeded, num: int = 2, email: str = "b@example.com") -> dict:
    return json.loads(seeded.switcher.read_account_credentials(str(num), email))


def _keep_stored(seeded, login: dict, num: int = 2, email: str = "b@example.com"):
    seeded.switcher._write_account_credentials(str(num), email, json.dumps(login))


def _session_profile(seeded, num: int = 2, email: str = "b@example.com") -> Path:
    from claude_swap.session import session_dir_for

    profile = session_dir_for(seeded.switcher.backup_dir, str(num), email)
    profile.mkdir(parents=True, exist_ok=True)
    return profile


def _run_session_in(profile: Path) -> None:
    (profile / "sessions").mkdir(parents=True, exist_ok=True)
    (profile / "sessions" / f"{os.getpid()}.json").write_text(
        json.dumps({"pid": os.getpid(), "sessionId": "s", "cwd": "/"})
    )


def _live(seeded) -> dict:
    return json.loads(
        (seeded.temp_home / ".claude" / ".credentials.json").read_text()
    )


def _make_live(seeded, login: dict) -> None:
    (seeded.temp_home / ".claude" / ".credentials.json").write_text(json.dumps(login))


def _unlist(seeded, slot: str) -> None:
    """Take an account off claude-swap's list and leave its stored login, as
    a removal that finds the macOS Keychain locked leaves it."""
    data = seeded.switcher._get_sequence_data()
    del data["accounts"][slot]
    data["sequence"] = [n for n in data["sequence"] if str(n) != slot]
    seeded.switcher._write_json(seeded.switcher.sequence_file, data)


def _profile_of_account_one(_token: str) -> dict:
    """What Anthropic answers when asked whose a token of account 1 is."""
    return {
        "uuid": "uuid-1",
        "email": "a@example.com",
        "organizationUuid": None,
        "organizationName": None,
    }


class TestAccessToken:
    """An account's newest access token, renewed first when it is due (sak.12)."""

    def test_hands_out_a_login_that_is_not_due_as_it_is(
        self, seeded, endpoint, capfd
    ):
        expires = _in(hours=3)
        _keep_stored(seeded, _token_login("sk-ant-oat01-b", "rt-2", expires))
        data = seeded.switcher._get_sequence_data()
        data["accounts"]["2"]["organizationUuid"] = "org-2"
        data["accounts"]["2"]["organizationName"] = "Team"
        seeded.switcher._write_json(seeded.switcher.sequence_file, data)
        capfd.readouterr()
        info = api.access_token(seeded.switcher, "b@example.com", renew_within=900)
        _assert_silent(capfd)
        assert info == api.AccessTokenInfo(
            slot="2",
            email="b@example.com",
            organization_uuid="org-2",
            organization="Team",
            account_uuid="uuid-2",
            access_token="sk-ant-oat01-b",
            expires_at=api._instant(expires),
            scopes=tuple(_SCOPES),
            subscription_type="max",
            rate_limit_tier="default_claude_max_20x",
            renewed=False,
        )
        assert endpoint.posted == []
        assert "sk-ant-oat01-b" not in repr(info)
        assert "sk-ant-oat01-b" not in str(info)

    def test_the_expiry_is_iso_utc_with_milliseconds(self, seeded, endpoint):
        _keep_stored(seeded, _token_login("sk-a", "rt-2", 1_900_000_000_123))
        info = api.access_token(seeded.switcher, "2")
        assert info.expires_at == "2030-03-17T17:46:40.123Z"

    def test_renews_a_login_that_is_due_and_keeps_the_new_one(
        self, seeded, endpoint, capfd
    ):
        _keep_stored(seeded, _token_login("sk-old", "rt-2", _in(minutes=10)))
        capfd.readouterr()
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        _assert_silent(capfd)
        assert endpoint.posted == ["rt-2"]
        assert info.renewed is True
        assert info.access_token == "sk-ant-oat01-renewed-1"
        stored = _stored(seeded)["claudeAiOauth"]
        assert stored["accessToken"] == "sk-ant-oat01-renewed-1"
        assert stored["refreshToken"] == "rt-2-next"
        assert info.expires_at == api._instant(stored["expiresAt"])
        # the next call has nothing to renew
        again = api.access_token(seeded.switcher, "2", renew_within=900)
        assert again.renewed is False
        assert again.access_token == info.access_token
        assert endpoint.posted == ["rt-2"]

    def test_renew_within_zero_renews_only_a_lapsed_token(self, seeded, endpoint):
        _keep_stored(seeded, _token_login("sk-soon", "rt-2", _in(minutes=2)))
        assert api.access_token(seeded.switcher, "2").renewed is False
        _keep_stored(seeded, _token_login("sk-gone", "rt-2", _in(minutes=-1)))
        assert api.access_token(seeded.switcher, "2").renewed is True
        assert endpoint.posted == ["rt-2"]

    def test_takes_in_what_its_cswap_run_profile_renewed(self, seeded, endpoint):
        _keep_stored(seeded, _token_login("sk-old", "rt-2", _in(minutes=-30)))
        newer = _token_login("sk-ant-oat01-profile", "rt-2b", _in(hours=7))
        profile = _session_profile(seeded)
        (profile / ".credentials.json").write_text(json.dumps(newer))
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.access_token == "sk-ant-oat01-profile"
        assert info.renewed is False
        assert endpoint.posted == []
        assert _stored(seeded) == newer

    def test_never_renews_a_login_a_running_session_holds(self, seeded, endpoint):
        login = _token_login("sk-due", "rt-2", _in(minutes=10))
        _keep_stored(seeded, login)
        profile = _session_profile(seeded)
        (profile / ".credentials.json").write_text(json.dumps(login))
        _run_session_in(profile)
        with pytest.raises(api.SessionError, match="renews it itself"):
            api.access_token(seeded.switcher, "2", renew_within=900)
        assert endpoint.posted == []
        assert _stored(seeded) == login
        # not due yet: handed out as it is, nothing renewed
        assert api.access_token(seeded.switcher, "2").access_token == "sk-due"
        assert endpoint.posted == []

    def test_a_running_sessions_newer_login_is_handed_out_as_it_is(
        self, seeded, endpoint
    ):
        stored = _token_login("sk-old", "rt-2", _in(minutes=-30))
        _keep_stored(seeded, stored)
        profile = _session_profile(seeded)
        newer = _token_login("sk-ant-oat01-live", "rt-2b", _in(hours=7))
        (profile / ".credentials.json").write_text(json.dumps(newer))
        _run_session_in(profile)
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.access_token == "sk-ant-oat01-live"
        assert endpoint.posted == []
        assert _stored(seeded) == stored  # the running session owns the newer one

    def test_a_session_on_another_login_does_not_hold_this_one(
        self, seeded, endpoint
    ):
        _keep_stored(seeded, _token_login("sk-due", "rt-2", _in(minutes=10)))
        profile = _session_profile(seeded)
        (profile / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "c@example.com"}})
        )
        _run_session_in(profile)
        assert api.access_token(seeded.switcher, "2", renew_within=900).renewed
        assert endpoint.posted == ["rt-2"]

    def test_takes_in_a_renewal_claude_swap_set_aside(self, seeded, endpoint):
        from claude_swap.oauth import credential_fingerprint

        stored = _token_login("sk-old", "rt-2", _in(minutes=-30))
        _keep_stored(seeded, stored)
        successor = _token_login("sk-ant-oat01-aside", "rt-2-next", _in(hours=7))
        seeded.switcher._store._write_unclaimed_credential(
            json.dumps(successor),
            {
                "reason": "consume-gate-persist-failed",
                "configSlot": "2",
                "consumedFp": credential_fingerprint(json.dumps(stored)),
                "fingerprint": credential_fingerprint(json.dumps(successor)),
            },
        )
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.access_token == "sk-ant-oat01-aside"
        assert endpoint.posted == []
        assert seeded.switcher.list_unclaimed_credentials() == {}

    @pytest.mark.parametrize(
        ("error", "kind", "match"),
        [
            ("invalid_grant", api.CredentialError, "no longer works"),
            ("transient", api.CredentialError, "could not be renewed now"),
            ("invalid_client", api.CredentialError, "invalid_client"),
        ],
    )
    def test_a_renewal_that_fails_says_why(self, seeded, endpoint, error, kind, match):
        login = _token_login("sk-due", "rt-2", _in(minutes=10))
        _keep_stored(seeded, login)
        endpoint.error = error
        with pytest.raises(kind, match=match):
            api.access_token(seeded.switcher, "2", renew_within=900)
        assert _stored(seeded) == login

    def test_waits_for_a_renewal_or_a_switch_under_way(
        self, seeded, endpoint, monkeypatch
    ):
        from claude_swap.locking import FileLock

        _keep_stored(seeded, _token_login("sk-due", "rt-2", _in(minutes=10)))
        monkeypatch.setattr(api, "ACCESS_TOKEN_LOCK_TIMEOUT", 0.2)
        for held in (seeded.switcher.consume_lock_file("2"), seeded.switcher.lock_file):
            with (
                FileLock(held),
                pytest.raises(api.LockError, match="being renewed or switched"),
            ):
                api.access_token(seeded.switcher, "2", renew_within=900)
        assert endpoint.posted == []
        assert api.access_token(seeded.switcher, "2", renew_within=900).renewed

    def test_a_switch_to_the_account_waits_for_its_renewal(self, seeded, endpoint):
        sw = seeded.switcher
        live = _token_login("sk-a", "rt-1", _in(hours=3))
        _make_live(seeded, live)
        _keep_stored(seeded, live, 1, "a@example.com")
        # 12 minutes left: renewed with renew_within=900, while the auto-switch
        # engine (10 minutes) calls it fresh and may switch to it meanwhile
        _keep_stored(seeded, _token_login("sk-b", "rt-2", _in(minutes=12)))
        switch = threading.Thread(target=lambda: sw.switch_to("2", json_output=True))

        def a_switch_lands() -> None:
            switch.start()
            switch.join(timeout=0.5)
            assert switch.is_alive()  # it waits for claude-swap's lock

        endpoint.during = a_switch_lands
        info = api.access_token(sw, "2", renew_within=900)
        switch.join(timeout=15)
        assert info.renewed is True
        assert sw.current_account_number() == "2"
        # the switch put the renewed login in the default profile, not the spent one
        assert _live(seeded)["claudeAiOauth"]["refreshToken"] == "rt-2-next"
        assert _stored(seeded)["claudeAiOauth"]["refreshToken"] == "rt-2-next"
        assert endpoint.posted == ["rt-2"]

    def test_refuses_while_claude_code_keeps_its_login_elsewhere(
        self, seeded, endpoint, monkeypatch
    ):
        _keep_stored(seeded, _token_login("sk-b", "rt-2", _in(hours=3)))
        _make_live(seeded, _token_login("sk-a", "rt-1", _in(hours=3)))
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", "/elsewhere")
        for slot in ("1", "2"):
            with pytest.raises(api.ConfigError, match="CLAUDE_SECURESTORAGE_CONFIG_DIR"):
                api.access_token(seeded.switcher, slot)
        assert endpoint.posted == []

    def test_a_renewal_set_aside_that_cannot_be_read_says_so(
        self, seeded, endpoint, monkeypatch
    ):
        _keep_stored(seeded, _token_login("sk-due", "rt-2", _in(minutes=10)))

        def unreadable(*_a):
            raise api.CredentialReadError("the unclaimed manifest is unreadable")

        monkeypatch.setattr(seeded.switcher, "_adopt_stashed_successor", unreadable)
        with pytest.raises(api.CredentialReadError, match="set aside"):
            api.access_token(seeded.switcher, "2", renew_within=900)
        assert endpoint.posted == []

    def test_a_renewal_that_cannot_be_stored_is_set_aside_and_taken_in(
        self, seeded, endpoint, monkeypatch
    ):
        from claude_swap.oauth import credential_fingerprint

        old = _token_login("sk-due", "rt-2", _in(minutes=10))
        _keep_stored(seeded, old)

        def full_disk(*_a):
            raise OSError(28, "No space left on device")

        with monkeypatch.context() as patched:
            patched.setattr(seeded.switcher, "_write_account_credentials", full_disk)
            info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.renewed is True
        assert info.access_token == "sk-ant-oat01-renewed-1"
        assert _stored(seeded) == old
        (aside,) = seeded.switcher.list_unclaimed_credentials().values()
        assert aside["configSlot"] == "2"
        assert aside["consumedFp"] == credential_fingerprint(json.dumps(old))
        # the next call takes it in instead of renewing the spent login again
        again = api.access_token(seeded.switcher, "2", renew_within=900)
        assert again.access_token == "sk-ant-oat01-renewed-1"
        assert again.renewed is False
        assert endpoint.posted == ["rt-2"]
        assert _stored(seeded)["claudeAiOauth"]["refreshToken"] == "rt-2-next"
        assert seeded.switcher.list_unclaimed_credentials() == {}

    def test_a_login_added_again_meanwhile_is_kept(self, seeded, endpoint):
        _keep_stored(seeded, _token_login("sk-due", "rt-2", _in(minutes=10)))
        added = _token_login("sk-added", "rt-added", _in(hours=8))
        # adding an account does not wait for claude-swap's lock
        endpoint.during = lambda: _keep_stored(seeded, added)
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.renewed is True
        assert info.access_token == "sk-ant-oat01-renewed-1"
        assert _stored(seeded) == added
        (aside,) = seeded.switcher.list_unclaimed_credentials().values()
        assert aside["reason"] == "consume-gate-cas-conflict"
        again = api.access_token(seeded.switcher, "2", renew_within=900)
        assert (again.access_token, again.renewed) == ("sk-added", False)
        assert seeded.switcher.list_unclaimed_credentials() == {}  # retired
        assert endpoint.posted == ["rt-2"]

    def test_a_login_removed_meanwhile_is_not_brought_back(self, seeded, endpoint):
        _keep_stored(seeded, _token_login("sk-due", "rt-2", _in(minutes=10)))
        endpoint.during = lambda: seeded.switcher._delete_account_credentials(
            "2", "b@example.com"
        )
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.renewed is True
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == ""
        (aside,) = seeded.switcher.list_unclaimed_credentials().values()
        assert aside["reason"] == "consume-gate-slot-removed"

    def test_a_removal_that_leaves_the_stored_login_behind_gets_no_renewal(
        self, seeded, endpoint
    ):
        due = _token_login("sk-due", "rt-2", _in(minutes=10))
        _keep_stored(seeded, due)
        endpoint.during = lambda: _unlist(seeded, "2")
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.renewed is True
        assert _stored(seeded) == due  # left as the removal left it: spent
        (aside,) = seeded.switcher.list_unclaimed_credentials().values()
        assert aside["reason"] == "consume-gate-slot-removed"

    def test_a_session_on_an_older_login_does_not_hold_the_new_one(
        self, seeded, endpoint
    ):
        profile = _session_profile(seeded)
        (profile / ".credentials.json").write_text(
            json.dumps(_token_login("sk-old", "rt-old", _in(minutes=-30)))
        )
        _run_session_in(profile)
        # the account was logged in again meanwhile: a login of its own
        _keep_stored(seeded, _token_login("sk-new", "rt-new", _in(minutes=10)))
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.renewed is True
        assert endpoint.posted == ["rt-new"]

    def test_a_session_whose_login_cannot_be_read_holds_it(
        self, seeded, endpoint, macos, monkeypatch
    ):
        from claude_swap import macos_keychain

        _keep_stored(seeded, _token_login("sk-due", "rt-2", _in(minutes=10)))
        _run_session_in(_session_profile(seeded))

        def locked(_service, _account):
            raise macos_keychain.KeychainError("the keychain is locked")

        monkeypatch.setattr(macos_keychain, "get_password", locked)
        with pytest.raises(api.SessionError, match="renews it itself"):
            api.access_token(seeded.switcher, "2", renew_within=900)
        assert endpoint.posted == []

    def test_refuses_a_login_without_scopes(self, seeded, endpoint):
        login = _token_login("sk-b", "rt-2", _in(hours=3))
        del login["claudeAiOauth"]["scopes"]
        _keep_stored(seeded, login)
        with pytest.raises(api.ValidationError, match="no scopes"):
            api.access_token(seeded.switcher, "2")
        assert endpoint.posted == []

    @pytest.mark.parametrize(
        ("stored", "kind", "match"),
        [
            ("sk-ant-api03-key", api.ValidationError, "API key"),
            (
                json.dumps({
                    "claudeAiOauth": {
                        "accessToken": "sk-ant-oat01-x",
                        "scopes": ["user:inference"],
                    }
                }),
                api.ValidationError,
                "setup-token",
            ),
            ("", api.CredentialError, "has no stored login"),
        ],
    )
    def test_refuses_an_account_with_no_access_token_to_hand_out(
        self, seeded, endpoint, stored, kind, match
    ):
        if stored:
            seeded.switcher._write_account_credentials("2", "b@example.com", stored)
        else:
            seeded.switcher._delete_account_credentials("2", "b@example.com")
        with pytest.raises(kind, match=match):
            api.access_token(seeded.switcher, "2", renew_within=900)
        assert endpoint.posted == []

    @pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "900"])
    def test_refuses_a_renew_within_that_is_no_duration(self, seeded, value):
        with pytest.raises(api.ValidationError, match="renew_within"):
            api.access_token(seeded.switcher, "2", renew_within=value)

    def test_unknown_account_raises(self, seeded):
        with pytest.raises(api.AccountNotFoundError):
            api.access_token(seeded.switcher, "9")

    def test_refuses_a_foreign_profile(self, seeded, monkeypatch, tmp_path):
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "hand-made"))
        with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
            api.access_token(seeded.switcher, "2")


class TestAccessTokenOfTheDefaultLogin:
    """The account Claude Code's default profile runs on: its live login."""

    def test_hands_out_the_default_profiles_login(self, seeded, endpoint, capfd):
        live = _token_login("sk-ant-oat01-live", "rt-1", _in(hours=3))
        _make_live(seeded, live)
        capfd.readouterr()
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        _assert_silent(capfd)
        assert info.slot == "1"
        assert info.access_token == "sk-ant-oat01-live"
        assert info.renewed is False
        assert endpoint.posted == []

    def test_renews_it_under_claude_codes_locks_and_writes_it_back(
        self, seeded, endpoint
    ):
        from claude_swap import claude_locks

        live = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, live)
        _keep_stored(seeded, live, 1, "a@example.com")
        held: list[bool] = []
        endpoint.during = lambda: held.append(
            claude_locks.oauth_refresh_lock_dir().is_dir()
            and claude_locks.credentials_lock_dir().is_dir()
            and seeded.switcher.consume_lock_file("1").exists()
        )
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert endpoint.posted == ["rt-1"]
        assert held == [True]
        assert info.renewed is True
        assert info.access_token == "sk-ant-oat01-renewed-1"
        assert _live(seeded)["claudeAiOauth"]["refreshToken"] == "rt-1-next"
        stored = _stored(seeded, 1, "a@example.com")
        assert stored["claudeAiOauth"]["refreshToken"] == "rt-1-next"
        assert not claude_locks.oauth_refresh_lock_dir().exists()
        assert not claude_locks.credentials_lock_dir().exists()

    def test_a_renewal_claude_code_made_is_confirmed_and_taken(
        self, seeded, endpoint, monkeypatch
    ):
        _keep_stored(
            seeded, _token_login("sk-1", "rt-1", _in(hours=-1)), 1, "a@example.com"
        )
        fresh = _token_login("sk-ant-oat01-cc", "rt-1-cc", _in(hours=8))
        _make_live(seeded, fresh)
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile", _profile_of_account_one
        )
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert info.access_token == "sk-ant-oat01-cc"
        assert info.renewed is False
        assert endpoint.posted == []
        assert _stored(seeded, 1, "a@example.com") == fresh

    def test_a_renewal_claude_code_makes_meanwhile_is_confirmed_first(
        self, seeded, endpoint, monkeypatch
    ):
        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")
        fresh = _token_login("sk-ant-oat01-cc", "rt-1-cc", _in(hours=8))
        confirm = api._confirm_live

        def claude_code_renews(*args) -> None:
            confirm(*args)
            _make_live(seeded, fresh)  # after claude-swap looked, before its locks

        monkeypatch.setattr(api, "_confirm_live", claude_code_renews)
        with pytest.raises(api.CredentialError, match="not confirmed"):
            api.access_token(seeded.switcher, "1", renew_within=900)
        assert endpoint.posted == []
        assert _live(seeded) == fresh
        monkeypatch.setattr(api, "_confirm_live", confirm)
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile", _profile_of_account_one
        )
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert info.access_token == "sk-ant-oat01-cc"
        assert info.renewed is False
        assert endpoint.posted == []

    def test_another_accounts_login_is_never_handed_out_under_its_name(
        self, seeded, endpoint, monkeypatch
    ):
        _keep_stored(
            seeded, _token_login("sk-a", "rt-1", _in(hours=3)), 1, "a@example.com"
        )
        _make_live(seeded, _token_login("sk-ant-oat01-z", "rt-z", _in(hours=3)))
        with pytest.raises(api.CredentialError, match="not confirmed"):
            api.access_token(seeded.switcher, "1")
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile",
            lambda token: {
                "uuid": "uuid-9",
                "email": "z@example.com",
                "organizationUuid": "",
                "organizationName": None,
            },
        )
        with pytest.raises(api.CredentialError, match="another account's login"):
            api.access_token(seeded.switcher, "1")
        assert endpoint.posted == []

    def test_a_switch_waits_until_its_login_is_read(
        self, seeded, endpoint, monkeypatch
    ):
        from claude_swap.locking import FileLock

        sw = seeded.switcher
        live = _token_login("sk-ant-oat01-a", "rt-1", _in(hours=3))
        _make_live(seeded, live)
        _keep_stored(seeded, live, 1, "a@example.com")
        _keep_stored(seeded, _token_login("sk-ant-oat01-b", "rt-2", _in(hours=3)))
        read = sw._read_active_credentials
        switch = threading.Thread(target=lambda: sw.switch_to("2", json_output=True))
        landed: list[bool] = []

        def read_while_a_switch_lands():
            if not landed and not FileLock(sw.lock_file, timeout=0).acquire():
                # access_token holds claude-swap's lock: a switch comes now
                landed.append(True)
                switch.start()
                switch.join(timeout=0.5)
                assert switch.is_alive()  # it waits for that lock
            return read()

        monkeypatch.setattr(sw, "_read_active_credentials", read_while_a_switch_lands)
        info = api.access_token(sw, "1")
        switch.join(timeout=15)
        assert landed
        assert (info.slot, info.access_token) == ("1", "sk-ant-oat01-a")
        assert sw.current_account_number() == "2"  # it went ahead afterwards
        assert _live(seeded)["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-b"

    def test_is_not_renewed_while_its_session_holds_the_same_login(
        self, seeded, endpoint
    ):
        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")
        profile = _session_profile(seeded, 1, "a@example.com")
        (profile / ".credentials.json").write_text(json.dumps(due))
        _run_session_in(profile)
        with pytest.raises(api.SessionError, match="renews it itself"):
            api.access_token(seeded.switcher, "1", renew_within=900)
        assert endpoint.posted == []
        assert api.access_token(seeded.switcher, "1").access_token == "sk-due"
        # a session on another login of the account does not hold this one
        (profile / ".credentials.json").write_text(
            json.dumps(_token_login("sk-o", "rt-other", _in(hours=1)))
        )
        assert api.access_token(seeded.switcher, "1", renew_within=900).renewed
        assert endpoint.posted == ["rt-1"]

    def test_a_renewal_neither_store_takes_is_set_aside_and_taken_in(
        self, seeded, endpoint, monkeypatch
    ):
        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")

        def full_disk(*_a):
            raise OSError(28, "No space left on device")

        with monkeypatch.context() as patched:
            patched.setattr(seeded.switcher, "_write_account_credentials", full_disk)
            patched.setattr(seeded.switcher, "_write_credentials", full_disk)
            info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert info.renewed is True
        assert _live(seeded) == due
        assert len(seeded.switcher.list_unclaimed_credentials()) == 1
        from claude_swap import claude_locks

        write = seeded.switcher._write_credentials
        locked: list[bool] = []

        def write_under_claude_codes_locks(credentials: str) -> None:
            locked.append(
                claude_locks.oauth_refresh_lock_dir().is_dir()
                and claude_locks.credentials_lock_dir().is_dir()
            )
            write(credentials)

        monkeypatch.setattr(
            seeded.switcher, "_write_credentials", write_under_claude_codes_locks
        )
        again = api.access_token(seeded.switcher, "1", renew_within=900)
        assert locked == [True]  # put back where Claude Code reads it, under its lock
        assert again.access_token == info.access_token
        assert again.renewed is False
        assert endpoint.posted == ["rt-1"]
        assert _live(seeded)["claudeAiOauth"]["refreshToken"] == "rt-1-next"
        stored = _stored(seeded, 1, "a@example.com")
        assert stored["claudeAiOauth"]["refreshToken"] == "rt-1-next"
        assert seeded.switcher.list_unclaimed_credentials() == {}

    @pytest.mark.parametrize("removal", ["whole", "its stored login left behind"])
    def test_an_account_removed_during_its_renewal_gets_no_stored_login_back(
        self, seeded, endpoint, removal
    ):
        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")
        # removing an account does not wait for claude-swap's lock
        if removal == "whole":
            endpoint.during = lambda: api.remove_account(seeded.switcher, "1")
        else:
            endpoint.during = lambda: _unlist(seeded, "1")
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert info.renewed is True
        if removal == "whole":
            assert seeded.switcher.read_account_credentials("1", "a@example.com") == ""
        else:
            assert _stored(seeded, 1, "a@example.com") == due  # spent, not renewed
        assert "1" not in seeded.switcher._get_sequence_data()["accounts"]
        # Claude Code's default profile, still logged in, holds the renewal
        assert _live(seeded)["claudeAiOauth"]["refreshToken"] == "rt-1-next"
        assert seeded.switcher.list_unclaimed_credentials() == {}

    def test_a_renewal_of_a_removed_account_the_default_profile_refuses_is_set_aside(
        self, seeded, endpoint, monkeypatch
    ):
        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")
        endpoint.during = lambda: api.remove_account(seeded.switcher, "1")

        def full_disk(*_a):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(seeded.switcher, "_write_credentials", full_disk)
        assert api.access_token(seeded.switcher, "1", renew_within=900).renewed
        assert seeded.switcher.read_account_credentials("1", "a@example.com") == ""
        (aside,) = seeded.switcher.list_unclaimed_credentials().values()
        assert aside["reason"] == "consume-gate-slot-removed"

    def test_a_login_added_again_during_the_renewal_is_kept(self, seeded, endpoint):
        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")
        added = _token_login("sk-added", "rt-added", _in(hours=8))
        # adding an account does not wait for claude-swap's lock either
        endpoint.during = lambda: _keep_stored(seeded, added, 1, "a@example.com")
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert info.renewed is True
        assert _stored(seeded, 1, "a@example.com") == added
        assert _live(seeded)["claudeAiOauth"]["refreshToken"] == "rt-1-next"
        assert seeded.switcher.list_unclaimed_credentials() == {}

    @pytest.mark.parametrize("moved_on", ["a new login", "an unreadable profile"])
    def test_a_set_aside_renewal_never_goes_over_what_came_in_between(
        self, seeded, endpoint, monkeypatch, moved_on
    ):
        """A renewal claude-swap set aside goes back into the default profile
        only while that profile still holds the login it renewed, read again
        under Claude Code's locks: a ``/login`` that lands after claude-swap
        read the profile, before those locks, stays. The renewal is the
        account's stored login then, and is handed out as one."""
        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")

        def full_disk(*_a):
            raise OSError(28, "No space left on device")

        with monkeypatch.context() as patched:
            patched.setattr(seeded.switcher, "_write_account_credentials", full_disk)
            patched.setattr(seeded.switcher, "_write_credentials", full_disk)
            first = api.access_token(seeded.switcher, "1", renew_within=900)
        assert len(seeded.switcher.list_unclaimed_credentials()) == 1
        relogin = _token_login("sk-ant-oat01-relogin", "rt-relogin", _in(hours=8))
        locks = api._claude_code_locks

        @contextlib.contextmanager
        def it_lands_first(slot):
            if moved_on == "a new login":
                _make_live(seeded, relogin)
            else:
                from claude_swap.credentials import ActiveCredentials

                unreadable = ActiveCredentials(json.dumps(due), False, True)
                monkeypatch.setattr(
                    seeded.switcher, "_read_active_credentials", lambda: unreadable
                )
            with locks(slot):
                yield

        monkeypatch.setattr(api, "_claude_code_locks", it_lands_first)
        again = api.access_token(seeded.switcher, "1", renew_within=900)
        assert _live(seeded) == (relogin if moved_on == "a new login" else due)
        assert (again.access_token, again.renewed) == (first.access_token, False)
        stored = _stored(seeded, 1, "a@example.com")
        assert stored["claudeAiOauth"]["refreshToken"] == "rt-1-next"
        assert seeded.switcher.list_unclaimed_credentials() == {}
        assert endpoint.posted == ["rt-1"]

    def test_a_set_aside_renewal_that_is_due_is_renewed_as_the_stored_login(
        self, seeded, endpoint, monkeypatch
    ):
        from claude_swap.oauth import credential_fingerprint

        due = _token_login("sk-due", "rt-1", _in(minutes=-10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")
        successor = _token_login("sk-ant-oat01-aside", "rt-aside", _in(minutes=5))
        seeded.switcher._store._write_unclaimed_credential(
            json.dumps(successor),
            {
                "reason": "consume-gate-persist-failed",
                "configSlot": "1",
                "consumedFp": credential_fingerprint(json.dumps(due)),
                "fingerprint": credential_fingerprint(json.dumps(successor)),
            },
        )
        relogin = _token_login("sk-ant-oat01-relogin", "rt-relogin", _in(hours=8))
        locks = api._claude_code_locks

        @contextlib.contextmanager
        def a_login_lands_first(slot):
            _make_live(seeded, relogin)
            with locks(slot):
                yield

        monkeypatch.setattr(api, "_claude_code_locks", a_login_lands_first)
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert endpoint.posted == ["rt-aside"]
        assert (info.access_token, info.renewed) == ("sk-ant-oat01-renewed-1", True)
        assert _live(seeded) == relogin
        stored = _stored(seeded, 1, "a@example.com")
        assert stored["claudeAiOauth"]["refreshToken"] == "rt-aside-next"
        assert seeded.switcher.list_unclaimed_credentials() == {}

    def test_a_lapsed_login_it_cannot_check_says_what_helps(self, seeded, endpoint):
        _keep_stored(
            seeded, _token_login("sk-s", "rt-s", _in(hours=-3)), 1, "a@example.com"
        )
        _make_live(seeded, _token_login("sk-l", "rt-l", _in(minutes=-5)))
        with pytest.raises(
            api.CredentialError, match="Use Claude Code on this computer once"
        ):
            api.access_token(seeded.switcher, "1")
        assert endpoint.posted == []

    def test_an_unconfirmed_login_is_not_renewed(self, seeded, endpoint):
        _keep_stored(
            seeded, _token_login("sk-1", "rt-1", _in(hours=-1)), 1, "a@example.com"
        )
        live = _token_login("sk-due", "rt-elsewhere", _in(minutes=10))
        _make_live(seeded, live)
        with pytest.raises(api.CredentialError, match="not confirmed"):
            api.access_token(seeded.switcher, "1", renew_within=900)
        assert endpoint.posted == []
        assert _live(seeded) == live

    def test_a_login_anthropic_confirms_is_renewed(
        self, seeded, endpoint, monkeypatch
    ):
        _keep_stored(
            seeded, _token_login("sk-1", "rt-1", _in(hours=-1)), 1, "a@example.com"
        )
        _make_live(seeded, _token_login("sk-due", "rt-cc", _in(minutes=10)))
        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile",
            lambda token: {
                "uuid": "uuid-1",
                "email": "a@example.com",
                "organizationUuid": None,
                "organizationName": None,
            },
        )
        info = api.access_token(seeded.switcher, "1", renew_within=900)
        assert endpoint.posted == ["rt-cc"]
        assert info.renewed is True
        assert _live(seeded)["claudeAiOauth"]["refreshToken"] == "rt-cc-next"

    def test_an_unreadable_default_login_is_refused(
        self, seeded, endpoint, monkeypatch
    ):
        from claude_swap.credentials import ActiveCredentials

        _make_live(seeded, _token_login("sk-live", "rt-1", _in(hours=3)))
        for verdict in (
            ActiveCredentials(None, False),
            ActiveCredentials("x", False, True),
            ActiveCredentials("", True, True),
        ):
            monkeypatch.setattr(
                seeded.switcher, "_read_active_credentials", lambda v=verdict: v
            )
            with pytest.raises(api.CredentialError, match="cannot be read now"):
                api.access_token(seeded.switcher, "1")

    def test_a_logged_out_default_profile_has_no_login(self, seeded, endpoint):
        (seeded.temp_home / ".claude" / ".credentials.json").unlink()
        with pytest.raises(api.CredentialError, match="has no login now"):
            api.access_token(seeded.switcher, "1")

    def test_waits_for_a_renewal_under_way(self, seeded, endpoint, monkeypatch):
        from claude_swap.locking import FileLock

        due = _token_login("sk-due", "rt-1", _in(minutes=10))
        _make_live(seeded, due)
        _keep_stored(seeded, due, 1, "a@example.com")
        monkeypatch.setattr(api, "ACCESS_TOKEN_LOCK_TIMEOUT", 0.2)
        with (
            FileLock(seeded.switcher.consume_lock_file("1")),
            pytest.raises(api.LockError, match="being renewed"),
        ):
            api.access_token(seeded.switcher, "1", renew_within=900)
        assert endpoint.posted == []
        assert api.access_token(seeded.switcher, "1", renew_within=900).renewed


_KEPT = {
    "access_token": "sk-ant-oat01-kept",
    "expires_at": "2026-10-16T21:47:12.345Z",
    "scopes": ("user:profile", "user:inference"),
    "subscription_type": "max",
    "rate_limit_tier": "default_claude_max_20x",
    "email": "b@example.com",
    "organization_uuid": "org-2",
    "organization": "Team",
    "account_uuid": "uuid-2",
}


@pytest.fixture
def macos(monkeypatch, block_real_keychain):
    """Claude Code's macOS stores: the profile's Keychain item comes first."""
    from claude_swap import session

    monkeypatch.setattr(
        session.Platform, "detect", classmethod(lambda _cls: Platform.MACOS)
    )
    return block_real_keychain


def _item_name(profile: Path) -> tuple[str, str]:
    from claude_swap import macos_keychain, session

    return session.keychain_service_name(str(profile)), (
        macos_keychain.keychain_account_name()
    )


def _kept_login(profile: Path) -> dict:
    return json.loads((profile / ".credentials.json").read_text())


def _another_spelling(folder: Path, tmp_path: Path, spelling: str) -> str:
    """``folder`` named another way that reaches the same folder; skips the
    test where this disk has no such spelling."""
    if spelling == "slash":
        return str(folder) + os.sep
    if spelling == "case":
        place = folder.parent / folder.name.upper()
    elif spelling == "firmlink":
        place = Path("/System/Volumes/Data" + str(folder.resolve()))
    elif spelling == "symlink":
        place = tmp_path / f"link-to-{folder.name}"
        try:
            place.symlink_to(folder, target_is_directory=True)
        except OSError:
            pytest.skip("this system cannot make a symlink here")
    else:  # "dotdot"
        place = folder / "projects" / ".." / ".." / folder.name
    if not os.path.isdir(place) or not os.path.samefile(place, folder):
        pytest.skip(f"this disk has no {spelling} spelling of {folder}")
    return str(place)


class TestKeepAccessLogin:
    """A profile that runs on another computer's access token (sak.12)."""

    @pytest.fixture
    def profile(self, temp_home, no_stdin, tmp_path) -> Path:
        return tmp_path / "synced" / "b"

    def test_makes_a_profile_that_runs_on_the_token(self, temp_home, profile, capfd):
        (temp_home / ".claude" / "settings.json").write_text("{}")
        (temp_home / ".claude.json").write_text(json.dumps({"theme": "light"}))
        capfd.readouterr()
        api.keep_access_login(profile, **_KEPT)
        _assert_silent(capfd)
        assert _kept_login(profile) == {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-kept",
                "refreshToken": None,
                "expiresAt": 1_792_187_232_345,
                "scopes": ["user:profile", "user:inference"],
                "subscriptionType": "max",
                "rateLimitTier": "default_claude_max_20x",
            }
        }
        assert (profile / ".credentials.json").stat().st_mode & 0o777 == 0o600
        config = json.loads((profile / ".claude.json").read_text())
        assert config["oauthAccount"] == {
            "accountUuid": "uuid-2",
            "emailAddress": "b@example.com",
            "organizationUuid": "org-2",
            "organizationName": "Team",
        }
        assert config["hasCompletedOnboarding"] is True
        assert config["theme"] == "light"
        if sys.platform != "win32":
            assert (profile / "settings.json").is_symlink()
            assert profile.stat().st_mode & 0o777 == 0o700

    def test_the_expiry_round_trips_with_access_token(
        self, seeded, endpoint, tmp_path
    ):
        _keep_stored(seeded, _token_login("sk-a", "rt-2", 1_900_000_000_123))
        info = api.access_token(seeded.switcher, "2")
        profile = tmp_path / "synced" / "b"
        api.keep_access_login(
            profile,
            access_token=info.access_token,
            expires_at=info.expires_at,
            scopes=info.scopes,
            subscription_type=info.subscription_type,
            rate_limit_tier=info.rate_limit_tier,
            email=info.email,
            organization_uuid=info.organization_uuid,
            organization=info.organization,
            account_uuid=info.account_uuid,
        )
        assert _kept_login(profile)["claudeAiOauth"]["expiresAt"] == 1_900_000_000_123

    def test_replaces_the_token_and_keeps_what_claude_code_keeps_besides(
        self, profile
    ):
        profile.mkdir(parents=True)
        (profile / ".credentials.json").write_text(json.dumps({
            **_token_login("sk-old", None, 1),
            "mcpOAuth": {"server": {"accessToken": "mcp"}},
        }))
        api.keep_access_login(profile, **_KEPT)
        kept = _kept_login(profile)
        assert kept["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-kept"
        assert kept["mcpOAuth"] == {"server": {"accessToken": "mcp"}}

    def test_moves_the_keychain_item_into_the_file(self, macos, profile):
        profile.mkdir(parents=True)
        (profile / ".credentials.json").write_text("stale")
        before = (profile / ".credentials.json").stat().st_mtime_ns
        macos.set_password(
            *_item_name(profile),
            json.dumps({
                **_token_login("sk-old", None, 1),
                "mcpOAuth": {"server": {"accessToken": "mcp"}},
            }),
        )
        api.keep_access_login(profile, **_KEPT)
        assert macos.get_password(*_item_name(profile)) is None
        kept = _kept_login(profile)
        assert kept["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-kept"
        assert kept["mcpOAuth"] == {"server": {"accessToken": "mcp"}}
        assert (profile / ".credentials.json").stat().st_mtime_ns > before

    def test_an_unreadable_keychain_item_changes_nothing(
        self, macos, profile, monkeypatch
    ):
        from claude_swap import macos_keychain

        def locked(_service, _account):
            raise macos_keychain.KeychainError("the keychain is locked")

        monkeypatch.setattr(macos_keychain, "get_password", locked)
        with pytest.raises(api.CredentialReadError, match="cannot be read now"):
            api.keep_access_login(profile, **_KEPT)
        assert not (profile / ".credentials.json").exists()

    def test_refuses_a_profile_someone_logged_in_to(self, macos, profile):
        profile.mkdir(parents=True)
        own = json.dumps(_token_login("sk-own", "rt-own", _in(hours=8)))
        (profile / ".credentials.json").write_text(own)
        with pytest.raises(api.ConfigError, match="login of its own"):
            api.keep_access_login(profile, **_KEPT)
        assert (profile / ".credentials.json").read_text() == own
        (profile / ".credentials.json").unlink()
        macos.set_password(*_item_name(profile), own)
        with pytest.raises(api.ConfigError, match="login of its own"):
            api.keep_access_login(profile, **_KEPT)
        assert macos.get_password(*_item_name(profile)) == own
        assert not (profile / ".credentials.json").exists()

    def test_keeps_what_claude_code_recorded_about_the_same_account(self, profile):
        profile.mkdir(parents=True)
        (profile / ".claude.json").write_text(json.dumps({
            "projects": {"/x": {}},
            "theme": "dark",
            "oauthAccount": {
                "emailAddress": "b@example.com",
                "organizationUuid": "org-2",
                "displayName": "Bee",
            },
        }))
        api.keep_access_login(profile, **_KEPT)
        config = json.loads((profile / ".claude.json").read_text())
        assert config["projects"] == {"/x": {}}
        assert config["oauthAccount"]["displayName"] == "Bee"
        assert config["oauthAccount"]["accountUuid"] == "uuid-2"
        api.keep_access_login(profile, **{**_KEPT, "email": "c@example.com"})
        config = json.loads((profile / ".claude.json").read_text())
        assert "displayName" not in config["oauthAccount"]
        assert config["oauthAccount"]["emailAddress"] == "c@example.com"

    def test_refuses_places_that_are_not_its_own(self, temp_home, no_stdin, tmp_path):
        from claude_swap.paths import get_backup_root

        a_file = tmp_path / "file"
        a_file.write_text("x")
        for place, match in (
            (temp_home / ".claude", "default profile"),
            (temp_home / ".claude" / "inner", "default profile"),
            (temp_home, "default profile"),
            (get_backup_root() / "synced", "own store"),
            ("relative/b", "full path"),
            ("", "Name the profile folder"),
            (a_file, "not a folder"),
        ):
            with pytest.raises(api.ConfigError, match=match):
                api.keep_access_login(place, **_KEPT)
            with pytest.raises(api.ConfigError, match=match):
                api.forget_access_login(place)

    @pytest.mark.parametrize(
        "spelling", ["case", "firmlink", "symlink", "dotdot", "slash"]
    )
    def test_refuses_the_default_profile_by_any_spelling(
        self, temp_home, no_stdin, tmp_path, spelling
    ):
        claude = temp_home / ".claude"
        transcript = claude / "projects" / "-x" / "session.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text("{}\n")
        (claude / "history.jsonl").write_text('{"display":"hi"}\n')
        place = _another_spelling(claude, tmp_path, spelling)
        for where in (place, os.path.join(place, "synced")):
            with pytest.raises(api.ConfigError, match="default profile"):
                api.keep_access_login(where, **_KEPT)
            with pytest.raises(api.ConfigError, match="default profile"):
                api.forget_access_login(where)
        # refused before anything was written or merged
        assert transcript.read_text() == "{}\n"
        assert (claude / "history.jsonl").read_text() == '{"display":"hi"}\n'
        assert not (claude / ".credentials.json").exists()
        assert not (claude / "synced").exists()

    @pytest.mark.parametrize("spelling", ["case", "symlink"])
    def test_refuses_claude_swaps_store_by_another_spelling(
        self, temp_home, no_stdin, tmp_path, spelling
    ):
        from claude_swap.paths import get_backup_root

        store = get_backup_root()
        store.mkdir(parents=True)
        place = _another_spelling(store, tmp_path, spelling)
        with pytest.raises(api.ConfigError, match="own store"):
            api.keep_access_login(os.path.join(place, "sessions", "b"), **_KEPT)
        assert not (store / "sessions").exists()

    def test_leaves_no_copy_of_the_token_behind(self, profile, monkeypatch):
        profile.mkdir(parents=True)
        stray = profile / "..credentials.json.k3x9q2.tmp"
        stray.write_text(json.dumps(_token_login("sk-ant-oat01-old", None, 1)))
        mine = profile / "notes.tmp"
        mine.write_text("mine")
        synced: list[Path] = []
        monkeypatch.setattr(api, "sync_folder", lambda folder: synced.append(Path(folder)))
        api.keep_access_login(profile, **_KEPT)
        assert not stray.exists()  # a dead write's copy is swept
        assert mine.read_text() == "mine"
        assert profile in synced  # the rename itself is on the disk

        def failing_disk(_fd):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(api, "sync_file", failing_disk)
        with pytest.raises(api.CredentialWriteError):
            api.keep_access_login(profile, **{**_KEPT, "access_token": "sk-ant-oat01-n"})
        leftovers = sorted(p.name for p in profile.iterdir() if p.name.endswith(".tmp"))
        assert leftovers == ["notes.tmp"]
        assert _kept_login(profile)["claudeAiOauth"]["accessToken"] == (
            "sk-ant-oat01-kept"
        )

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"access_token": "sk-ant-api03-key"}, "API key"),
            ({"access_token": "sk-ant-ort01-refresh"}, "refresh token"),
            ({"access_token": ""}, "access token"),
            ({"access_token": "two tokens"}, "access token"),
            ({"expires_at": "2026-10-16T21:47:00"}, "zone"),
            ({"expires_at": "soon"}, "zone"),
            ({"scopes": ()}, "scopes"),
            ({"scopes": "user:inference"}, "scopes"),
            ({"scopes": ("user inference",)}, "scopes"),
            ({"email": " "}, "email"),
        ],
    )
    def test_refuses_values_that_are_not_what_they_should_be(
        self, profile, change, match
    ):
        with pytest.raises(api.ValidationError, match=match):
            api.keep_access_login(profile, **{**_KEPT, **change})
        assert not profile.exists()

    def test_waits_for_claude_codes_lock(self, profile, monkeypatch):
        from claude_swap import claude_locks

        monkeypatch.setattr(claude_locks, "DEFAULT_TIMEOUT_S", 0.3)
        lock = claude_locks.storage_write_lock_dir(profile)
        lock.mkdir(parents=True)
        with pytest.raises(api.LockError, match="right now"):
            api.keep_access_login(profile, **_KEPT)
        assert not (profile / ".credentials.json").exists()
        stale = time.time() - claude_locks.STORAGE_WRITE_STALENESS_S - 5
        os.utime(lock, (stale, stale))
        api.keep_access_login(profile, **_KEPT)
        assert _kept_login(profile)["claudeAiOauth"]["accessToken"] == (
            "sk-ant-oat01-kept"
        )
        assert not lock.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="history is linked on POSIX")
    def test_shares_history_with_the_default_profile(self, temp_home, profile):
        (temp_home / ".claude" / "projects").mkdir()
        api.keep_access_login(profile, **_KEPT)
        assert (profile / "projects").is_symlink()
        other = profile.parent / "c"
        api.keep_access_login(other, **_KEPT, share_history=False)
        assert not (other / "projects").exists()


class TestForgetAccessLogin:
    @pytest.fixture
    def profile(self, temp_home, no_stdin, tmp_path) -> Path:
        profile = tmp_path / "synced" / "b"
        api.keep_access_login(profile, **_KEPT)
        return profile

    def test_removes_the_kept_token_and_leaves_the_folder(self, profile, capfd):
        capfd.readouterr()
        api.forget_access_login(profile)
        api.forget_access_login(profile)  # gone already is fine
        api.forget_access_login(profile.parent / "never-made")
        _assert_silent(capfd)
        assert not (profile / ".credentials.json").exists()
        assert (profile / ".claude.json").exists()

    def test_removes_the_keychain_item_too(self, macos, profile):
        macos.set_password(
            *_item_name(profile), json.dumps(_token_login("sk-x", None, 1))
        )
        api.forget_access_login(profile)
        assert macos.get_password(*_item_name(profile)) is None
        assert not (profile / ".credentials.json").exists()

    def test_refuses_a_login_someone_made_there(self, profile):
        own = json.dumps(_token_login("sk-own", "rt-own", _in(hours=8)))
        (profile / ".credentials.json").write_text(own)
        with pytest.raises(api.ConfigError, match="login of its own"):
            api.forget_access_login(profile)
        assert (profile / ".credentials.json").read_text() == own

    def test_keeps_what_claude_code_keeps_besides(self, profile):
        mcp = {"server": {"accessToken": "mcp"}}
        (profile / ".credentials.json").write_text(
            json.dumps({**_kept_login(profile), "mcpOAuth": mcp})
        )
        api.forget_access_login(profile)
        assert _kept_login(profile) == {"mcpOAuth": mcp}
        assert (profile / ".credentials.json").stat().st_mode & 0o777 == 0o600

    def test_keeps_it_when_claude_code_moved_it_into_the_keychain_item(
        self, macos, profile
    ):
        mcp = {"server": {"accessToken": "mcp"}}
        macos.set_password(
            *_item_name(profile), json.dumps({**_kept_login(profile), "mcpOAuth": mcp})
        )
        (profile / ".credentials.json").unlink()
        api.forget_access_login(profile)
        assert macos.get_password(*_item_name(profile)) is None
        assert _kept_login(profile) == {"mcpOAuth": mcp}

    def test_a_login_that_stays_is_an_error(self, macos, profile, monkeypatch):
        from claude_swap import macos_keychain

        def locked(_service, _account):
            raise macos_keychain.KeychainError("the keychain is locked")

        monkeypatch.setattr(macos_keychain, "delete_password", locked)
        with pytest.raises(api.CredentialError, match="keychain"):
            api.forget_access_login(profile)


# -- Claude Code's default login on a device (sak.13) -------------------------------


@pytest.fixture
def linux(monkeypatch):
    """Claude Code's stores off macOS: the default profile's file only."""
    from claude_swap import session

    monkeypatch.setattr(
        session.Platform, "detect", classmethod(lambda _cls: Platform.LINUX)
    )


@pytest.fixture
def kept(temp_home, no_stdin, tmp_path) -> Path:
    """A profile token sync keeps account b signed in with (sak.12)."""
    profile = tmp_path / "synced" / "b"
    api.keep_access_login(profile, **_KEPT)
    return profile


def _default_item() -> tuple[str, str]:
    """Claude Code's Keychain item of the default profile."""
    from claude_swap import macos_keychain

    return "Claude Code-credentials", macos_keychain.keychain_account_name()


def _managed_item() -> tuple[str, str]:
    """Claude Code's Keychain item of a managed API key."""
    from claude_swap import macos_keychain

    return "Claude Code", macos_keychain.keychain_account_name()


def _default_file(home: Path) -> Path:
    return home / ".claude" / ".credentials.json"


def _home_config(home: Path) -> dict:
    return json.loads((home / ".claude.json").read_text())


_KEPT_ACCOUNT = {
    "accountUuid": "uuid-2",
    "emailAddress": "b@example.com",
    "organizationUuid": "org-2",
    "organizationName": "Team",
}


def _everything(temp_home: Path, keychain) -> tuple:
    """What the default profile holds, to show nothing changed."""
    files = tuple(
        path.read_text() if path.exists() else None
        for path in (_default_file(temp_home), temp_home / ".claude.json")
    )
    return dict(keychain.data) if keychain is not None else None, files


class TestDefaultLogin:
    """What Claude Code's default profile is signed in with (sak.13)."""

    def test_nothing_there_is_none(self, temp_home, no_stdin, macos, capfd):
        capfd.readouterr()
        assert api.default_login() == api.DefaultLoginInfo(
            kind="none", email=None, organization_uuid=None, expires_at=None
        )
        _assert_silent(capfd)

    @pytest.mark.parametrize(
        "held",
        [
            "",  # worker-1: the item is there and empty
            "  \n",
            "{}",
            "null",
            "not json",
            json.dumps({"mcpOAuth": {"server": {"accessToken": "mcp"}}}),
            json.dumps({"claudeAiOauth": {"accessToken": "", "refreshToken": ""}}),
        ],
    )
    def test_a_keychain_item_without_a_login_is_none(
        self, temp_home, no_stdin, macos, held
    ):
        macos.set_password(*_default_item(), held)
        (temp_home / ".claude.json").write_text(json.dumps({"theme": "dark"}))
        assert api.default_login().kind == "none"

    @pytest.mark.parametrize("held", ["", "{}", json.dumps({"mcpOAuth": {}})])
    def test_a_file_without_a_login_is_none(self, temp_home, no_stdin, linux, held):
        _default_file(temp_home).write_text(held)
        assert api.default_login().kind == "none"

    def test_the_kept_login_reads_as_access_with_its_account(self, kept, macos):
        api.use_access_login(kept)
        assert api.default_login() == api.DefaultLoginInfo(
            kind="access",
            email="b@example.com",
            organization_uuid="org-2",
            expires_at="2026-10-16T21:47:12.345Z",
        )

    @pytest.mark.parametrize("refresh", [None, "", "missing"])
    def test_an_access_token_alone_reads_as_access(
        self, temp_home, no_stdin, linux, refresh
    ):
        login = _token_login("sk-ant-oat01-a", refresh, 1_900_000_000_000)
        if refresh == "missing":
            del login["claudeAiOauth"]["refreshToken"]
        _default_file(temp_home).write_text(json.dumps(login))
        info = api.default_login()
        assert info.kind == "access"
        assert info.expires_at == "2030-03-17T17:46:40.000Z"
        assert info.email is None  # ~/.claude.json records no account

    def test_a_login_of_its_own_reads_as_own(self, temp_home, no_stdin, linux):
        own = _token_login(
            "sk-ant-oat01-own", "rt-own", _in(hours=8),
            refreshTokenExpiresAt=1_900_000_000_000,
        )
        _default_file(temp_home).write_text(json.dumps(own))
        (temp_home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {"emailAddress": "me@example.com", "organizationUuid": "o"},
        }))
        assert api.default_login() == api.DefaultLoginInfo(
            kind="own",
            email="me@example.com",
            organization_uuid="o",
            expires_at="2030-03-17T17:46:40Z",
            own_kind="login",
        )

    @pytest.mark.parametrize(
        "token",
        [
            {"accessToken": "sk-ant-oat01-t", "scopes": ["user:inference"]},
            {"accessToken": "sk-ant-oat01-t", "scopes": _SCOPES},
            {"accessToken": "sk-ant-oat01-t"},
            {
                "accessToken": "sk-ant-oat01-t", "refreshToken": None,
                "expiresAt": None, "scopes": ["user:inference"],
            },
        ],
    )
    def test_a_setup_token_reads_as_own(self, temp_home, no_stdin, linux, token):
        """Every shape claude-swap ever stored a setup-token in: one of its
        own accounts is active."""
        _default_file(temp_home).write_text(json.dumps({"claudeAiOauth": token}))
        assert api.default_login().kind == "own"

    def test_a_login_of_its_own_behind_the_item_reads_as_own(
        self, temp_home, no_stdin, macos
    ):
        macos.set_password(
            *_default_item(), json.dumps(_token_login("sk-lent", None, _in(hours=1)))
        )
        _default_file(temp_home).write_text(
            json.dumps(_token_login("sk-own", "rt-own", _in(hours=1)))
        )
        assert api.default_login().kind == "own"

    def test_the_item_comes_before_the_file(self, temp_home, no_stdin, macos):
        macos.set_password(
            *_default_item(), json.dumps(_token_login("sk-a", None, 1_900_000_000_000))
        )
        _default_file(temp_home).write_text(json.dumps(_token_login("sk-b", None, 1)))
        assert api.default_login().expires_at == "2030-03-17T17:46:40.000Z"
        macos.set_password(*_default_item(), "")  # an empty item hides nothing
        assert api.default_login().expires_at == "1970-01-01T00:00:00.001Z"

    def test_an_api_key_reads_as_api_key(self, temp_home, no_stdin, macos):
        (temp_home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {"emailAddress": "me@example.com"},
        }))
        macos.set_password(*_managed_item(), "sk-ant-api03-key")
        assert api.default_login() == api.DefaultLoginInfo(
            kind="api_key",
            email=None,
            organization_uuid=None,
            expires_at=None,
            own_kind="api_key",
        )
        macos.delete_password(*_managed_item())
        (temp_home / ".claude.json").write_text(json.dumps({
            "primaryApiKey": "sk-ant-api03-key",
        }))
        _default_file(temp_home).write_text(json.dumps(_token_login("sk", None, 1)))
        assert api.default_login().kind == "api_key"

    def test_a_locked_keychain_is_an_error(
        self, temp_home, no_stdin, macos, monkeypatch
    ):
        from claude_swap import macos_keychain

        def locked(_service, _account):
            raise macos_keychain.KeychainError("the keychain is locked")

        monkeypatch.setattr(macos_keychain, "get_password", locked)
        with pytest.raises(api.CredentialReadError, match="cannot be read now"):
            api.default_login()

    def test_refuses_another_profile_or_secure_store(
        self, temp_home, no_stdin, monkeypatch, tmp_path
    ):
        other = temp_home / ".claude-work"
        other.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(other))
        for call in (
            api.default_login,
            api.forget_default_access_login,
            lambda: api.use_access_login(tmp_path / "synced" / "b"),
        ):
            with pytest.raises(api.ConfigError, match="CLAUDE_CONFIG_DIR"):
                call()
        monkeypatch.delenv("CLAUDE_CONFIG_DIR")
        monkeypatch.setenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", str(other))
        for call in (
            api.default_login,
            api.forget_default_access_login,
            lambda: api.use_access_login(tmp_path / "synced" / "b"),
        ):
            with pytest.raises(api.ConfigError, match="CLAUDE_SECURESTORAGE"):
                call()

    def test_the_error_is_a_config_error(self):
        assert issubclass(api.OwnLoginError, api.ConfigError)


class TestUseAccessLogin:
    """Claude Code's default login signed in with a kept access token (sak.13)."""

    def test_signs_an_empty_keychain_item_in_in_place(
        self, kept, macos, temp_home, capfd, caplog
    ):
        macos.set_password(*_default_item(), "")  # worker-1's empty item
        (temp_home / ".claude.json").write_text(
            json.dumps({"theme": "light", "projects": {"/x": {}}})
        )
        capfd.readouterr()
        assert api.use_access_login(kept) is True
        _assert_silent(capfd)
        held = json.loads(macos.get_password(*_default_item()))
        assert held == {"claudeAiOauth": _kept_login(kept)["claudeAiOauth"]}
        assert held["claudeAiOauth"]["refreshToken"] is None
        assert not _default_file(temp_home).exists()  # no file is made
        assert _home_config(temp_home) == {
            "theme": "light",
            "projects": {"/x": {}},
            "oauthAccount": _KEPT_ACCOUNT,
            "hasCompletedOnboarding": True,
        }
        assert "sk-ant-oat01-kept" not in caplog.text

    def test_that_very_login_again_changes_nothing(self, kept, macos, temp_home):
        assert api.use_access_login(kept) is True
        before = _everything(temp_home, macos)
        assert api.use_access_login(kept) is False
        assert _everything(temp_home, macos) == before

    def test_keeps_what_claude_code_keeps_besides(self, kept, macos):
        mcp = {"server": {"accessToken": "mcp"}}
        macos.set_password(*_default_item(), json.dumps({
            **_token_login("sk-old", None, 1),
            "mcpOAuth": mcp,
            "pluginSecrets": {"p": "s"},
            "trustedDeviceToken": "another-logins-enrollment",
        }))
        api.use_access_login(kept)
        held = json.loads(macos.get_password(*_default_item()))
        assert held["mcpOAuth"] == mcp
        assert held["pluginSecrets"] == {"p": "s"}
        assert "trustedDeviceToken" not in held
        assert held["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-kept"

    def test_an_existing_file_is_written_too(self, kept, macos, temp_home):
        old = json.dumps(_token_login("sk-old", None, 1))
        macos.set_password(*_default_item(), old)
        _default_file(temp_home).write_text(old)
        api.use_access_login(kept)
        assert json.loads(_default_file(temp_home).read_text()) == json.loads(
            macos.get_password(*_default_item())
        )

    def test_off_macos_the_file_holds_it(self, kept, linux, temp_home):
        api.use_access_login(kept)
        held = json.loads(_default_file(temp_home).read_text())
        assert held["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-kept"
        if sys.platform != "win32":
            assert _default_file(temp_home).stat().st_mode & 0o777 == 0o600

    def test_a_newer_token_follows(self, kept, linux, temp_home):
        api.use_access_login(kept)
        api.keep_access_login(kept, **{**_KEPT, "access_token": "sk-ant-oat01-newer"})
        assert api.use_access_login(kept) is True
        held = json.loads(_default_file(temp_home).read_text())
        assert held["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-newer"

    def test_another_accounts_login_replaces_it_and_its_account(
        self, kept, linux, temp_home, tmp_path
    ):
        (temp_home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {
                "emailAddress": "b@example.com",
                "organizationUuid": "org-2",
                "displayName": "Bee",
            },
        }))
        api.use_access_login(kept)
        assert _home_config(temp_home)["oauthAccount"] == {
            **_KEPT_ACCOUNT, "displayName": "Bee",
        }
        other = tmp_path / "synced" / "c"
        api.keep_access_login(other, **{
            **_KEPT,
            "access_token": "sk-ant-oat01-c",
            "email": "c@example.com",
            "organization_uuid": None,
            "organization": None,
            "account_uuid": "uuid-3",
        })
        assert api.use_access_login(other) is True
        assert _home_config(temp_home)["oauthAccount"] == {
            "accountUuid": "uuid-3", "emailAddress": "c@example.com",
        }
        assert api.default_login().email == "c@example.com"

    @pytest.mark.parametrize(
        "where",
        [
            "item", "file", "setup-token", "early-setup-token", "managed-key",
            "config-key",
        ],
    )
    def test_never_touches_a_login_of_its_own(self, kept, macos, temp_home, where):
        """``forget_default_access_login`` refuses too, as nothing else is
        left there (TestLeftoverAccess: an access token left beside one)."""
        own = json.dumps(_token_login("sk-own", "rt-own", _in(hours=8)))
        if where == "item":
            macos.set_password(*_default_item(), own)
        elif where == "file":
            _default_file(temp_home).write_text(own)
        elif where == "setup-token":
            macos.set_password(*_default_item(), json.dumps({"claudeAiOauth": {
                "accessToken": "sk-ant-oat01-t", "scopes": ["user:inference"],
            }}))
        elif where == "early-setup-token":
            macos.set_password(
                *_default_item(),
                json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-t"}}),
            )
        elif where == "managed-key":
            macos.set_password(*_managed_item(), "sk-ant-api03-key")
        else:
            (temp_home / ".claude.json").write_text(
                json.dumps({"primaryApiKey": "sk-ant-api03-key"})
            )
        before = _everything(temp_home, macos)
        with pytest.raises(api.OwnLoginError, match="left as it is"):
            api.use_access_login(kept)
        with pytest.raises(api.OwnLoginError, match="left as it is"):
            api.forget_default_access_login()
        assert _everything(temp_home, macos) == before

    @pytest.mark.parametrize(
        "case", ["missing", "empty", "own", "no-account", "no-scopes"]
    )
    def test_refuses_a_profile_without_a_kept_login(
        self, temp_home, no_stdin, macos, tmp_path, case
    ):
        profile = tmp_path / "synced" / "b"
        if case != "missing":
            profile.mkdir(parents=True)
        if case in ("own", "no-account", "no-scopes"):
            login = _token_login("sk-x", "rt-x" if case == "own" else None, 1)
            if case == "no-scopes":
                del login["claudeAiOauth"]["scopes"]
            (profile / ".credentials.json").write_text(json.dumps(login))
        if case not in ("missing", "no-account"):
            (profile / ".claude.json").write_text(
                json.dumps({"oauthAccount": _KEPT_ACCOUNT})
            )
        before = _everything(temp_home, macos)
        with pytest.raises(api.ConfigError):
            api.use_access_login(profile)
        assert _everything(temp_home, macos) == before
        assert profile.exists() is (case != "missing")

    def test_refuses_places_that_are_not_a_profile_of_its_own(
        self, temp_home, no_stdin, linux
    ):
        from claude_swap.paths import get_backup_root

        for place, match in (
            (temp_home / ".claude", "default profile"),
            (temp_home, "default profile"),
            (get_backup_root() / "synced", "own store"),
            ("relative/b", "full path"),
        ):
            with pytest.raises(api.ConfigError, match=match):
                api.use_access_login(place)
        assert not _default_file(temp_home).exists()

    def test_an_unreadable_keychain_changes_nothing(
        self, kept, macos, temp_home, monkeypatch
    ):
        from claude_swap import macos_keychain

        def locked(service, account):
            if service == "Claude Code-credentials":
                raise macos_keychain.KeychainError("the keychain is locked")
            return macos.get_password(service, account)

        monkeypatch.setattr(macos_keychain, "get_password", locked)
        before = _everything(temp_home, macos)
        with pytest.raises(api.CredentialReadError, match="cannot be read now"):
            api.use_access_login(kept)
        assert _everything(temp_home, macos) == before

    def test_never_writes_over_a_claude_json_it_cannot_read(
        self, kept, linux, temp_home
    ):
        (temp_home / ".claude.json").write_text('{"projects": ')
        with pytest.raises(api.ConfigError, match="Repair or move"):
            api.use_access_login(kept)
        assert (temp_home / ".claude.json").read_text() == '{"projects": '
        assert not _default_file(temp_home).exists()

    def test_waits_for_claude_codes_locks(self, kept, linux, temp_home, monkeypatch):
        from claude_swap import claude_locks

        monkeypatch.setattr(claude_locks, "DEFAULT_TIMEOUT_S", 0.3)
        for lock in (
            temp_home / ".claude" / ".oauth_refresh.lock",
            temp_home / ".claude.lock",
            temp_home / ".claude" / ".storage-write.lock",
            temp_home / ".claude.json.lock",
            kept / ".storage-write.lock",
        ):
            lock.mkdir()
            with pytest.raises(api.LockError, match="right now"):
                api.use_access_login(kept)
            lock.rmdir()
            assert not _default_file(temp_home).exists()
        assert api.use_access_login(kept) is True

    def test_waits_for_a_switch(self, kept, linux, temp_home, monkeypatch):
        from claude_swap.locking import FileLock
        from claude_swap.paths import get_backup_root

        monkeypatch.setattr(api, "DEFAULT_LOGIN_LOCK_TIMEOUT", 0.2)
        switch = FileLock(get_backup_root() / ".lock")
        assert switch.acquire()
        try:
            with pytest.raises(api.LockError, match="switching"):
                api.use_access_login(kept)
            with pytest.raises(api.LockError, match="switching"):
                api.forget_default_access_login()
        finally:
            switch.release()
        assert not _default_file(temp_home).exists()

    def test_the_login_is_put_back_when_its_account_cannot_be_recorded(
        self, kept, macos, temp_home, monkeypatch
    ):
        old = json.dumps(_token_login("sk-old", None, 1))
        macos.set_password(*_default_item(), old)

        def full_disk(_config):
            raise api.CredentialWriteError("No space left on device")

        monkeypatch.setattr(api, "_write_default_config", full_disk)
        with pytest.raises(api.CredentialWriteError, match="No space"):
            api.use_access_login(kept)
        assert macos.get_password(*_default_item()) == old
        assert not _default_file(temp_home).exists()

    def test_never_shows_the_token(self, kept, macos, temp_home, caplog, capfd):
        import logging

        caplog.set_level(logging.DEBUG, logger="claude-swap")
        capfd.readouterr()
        api.use_access_login(kept)
        info = api.default_login()
        api.forget_default_access_login()
        out, err = capfd.readouterr()
        assert "sk-ant-oat01-kept" not in caplog.text + out + err + repr(info)


class TestForgetDefaultAccessLogin:
    """The access-only default login removed again (sak.13)."""

    def test_removes_the_login_and_its_account(self, kept, macos, temp_home, capfd):
        (temp_home / ".claude.json").write_text(json.dumps({"theme": "light"}))
        api.use_access_login(kept)
        capfd.readouterr()
        assert api.forget_default_access_login() is True
        _assert_silent(capfd)
        assert macos.get_password(*_default_item()) is None
        assert _home_config(temp_home) == {
            "theme": "light", "hasCompletedOnboarding": True,
        }
        assert api.default_login().kind == "none"
        assert api.forget_default_access_login() is False

    def test_keeps_what_claude_code_keeps_besides(self, kept, macos):
        mcp = {"server": {"accessToken": "mcp"}}
        macos.set_password(*_default_item(), json.dumps({"mcpOAuth": mcp}))
        api.use_access_login(kept)
        assert api.forget_default_access_login() is True
        assert json.loads(macos.get_password(*_default_item())) == {"mcpOAuth": mcp}

    def test_the_file_goes_when_nothing_else_is_left(self, kept, linux, temp_home):
        api.use_access_login(kept)
        assert api.forget_default_access_login() is True
        assert not _default_file(temp_home).exists()
        mcp = {"server": {"accessToken": "mcp"}}
        _default_file(temp_home).write_text(json.dumps({"mcpOAuth": mcp}))
        api.use_access_login(kept)
        api.forget_default_access_login()
        assert json.loads(_default_file(temp_home).read_text()) == {"mcpOAuth": mcp}
        if sys.platform != "win32":
            assert _default_file(temp_home).stat().st_mode & 0o777 == 0o600

    def test_an_empty_item_is_no_login(self, temp_home, no_stdin, macos):
        macos.set_password(*_default_item(), "")
        assert api.forget_default_access_login() is False
        assert macos.get_password(*_default_item()) == ""

    def test_a_login_that_stays_is_an_error_and_a_retry_ends_it(
        self, kept, macos, temp_home, monkeypatch
    ):
        from claude_swap import macos_keychain

        api.use_access_login(kept)

        def locked(_service, _account):
            raise macos_keychain.KeychainError("the keychain is locked")

        monkeypatch.setattr(macos_keychain, "delete_password", locked)
        with pytest.raises(api.CredentialWriteError, match="Keychain"):
            api.forget_default_access_login()
        assert "oauthAccount" not in _home_config(temp_home)
        monkeypatch.setattr(macos_keychain, "delete_password", macos.delete_password)
        assert api.forget_default_access_login() is True
        assert macos.get_password(*_default_item()) is None


def _lend_live(seeded, email: str = "b@example.com", num: int = 2) -> None:
    """Make the default login an access token another computer lent for an
    account claude-swap also has here (sak.13)."""
    _make_live(seeded, _token_login("sk-ant-oat01-lent", None, _in(hours=1)))
    (seeded.temp_home / ".claude.json").write_text(json.dumps({
        "oauthAccount": {"emailAddress": email, "accountUuid": f"uuid-{num}"},
    }))


class TestALentDefaultLogin:
    """The facade with an access token another computer lent in the default
    profile (sak.13): it is no login of this computer's own, so the
    account's stored login stays its login here."""

    def test_its_account_is_not_the_active_one(self, seeded, linux):
        _lend_live(seeded)
        assert api.active_account(seeded.switcher) is None
        rows = api.accounts_json(seeded.switcher)["accounts"]
        assert [row["active"] for row in rows] == [False, False]

    def test_access_token_hands_out_the_stored_login(self, seeded, linux, endpoint):
        _keep_stored(seeded, _token_login("sk-ant-oat01-b", "rt-2", _in(hours=3)))
        _lend_live(seeded)
        info = api.access_token(seeded.switcher, "2", renew_within=900)
        assert info.access_token == "sk-ant-oat01-b"
        assert info.renewed is False
        assert endpoint.posted == []

    def test_it_is_never_added(self, seeded, linux):
        _lend_live(seeded)
        with pytest.raises(api.ValidationError, match="claude /login"):
            api.add_current_login(seeded.switcher)
        assert _numbers(seeded) == [1, 2]

    def test_a_profile_login_of_its_account_is_stored(self, seeded, linux, target):
        _lend_live(seeded)
        (target / ".credentials.json").write_text(_copy_text())
        (target / ".claude.json").write_text(json.dumps({
            "oauthAccount": {"emailAddress": "b@example.com", "accountUuid": "uuid-2"},
        }))
        added = api.add_profile_login(seeded.switcher, target)
        assert added == api.AddedAccount(slot="2", email="b@example.com", created=False)
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == (
            _copy_text()
        )

    def test_a_copy_for_its_account_is_taken(self, seeded, linux, target, monkeypatch):
        anthropic = _Anthropic()
        monkeypatch.setattr("claude_swap.oauth.fetch_oauth_profile", anthropic.profile)
        _give_org(seeded, 2, "org-2")
        anthropic.owners["sk-ant-oat01-copy"] = _owner(2, "b@example.com", "org-2")
        _lend_live(seeded)
        config = _home_config(seeded.temp_home)
        config["oauthAccount"]["organizationUuid"] = "org-2"
        (seeded.temp_home / ".claude.json").write_text(json.dumps(config))
        copied = api.add_login_copy(
            seeded.switcher, target, credentials=_copy_text(), account="2"
        )
        assert copied.default_login is False
        assert copied.account == api.AddedAccount(
            slot="2", email="b@example.com", created=False
        )

    def test_its_account_may_be_handed_over(self, seeded, linux, tmp_path):
        stored = _store_login(seeded)
        _lend_live(seeded)
        handed = api.hand_over_login(seeded.switcher, "2", tmp_path / "b.json")
        assert handed.slot == "2"
        assert json.loads((tmp_path / "b.json").read_text()) == stored
        assert _numbers(seeded) == [1]

    def test_a_session_profile_may_be_made_for_it(self, seeded, linux, monkeypatch):
        from claude_swap.session import SessionManager

        calls = []

        def fake_setup(self, identifier, share, share_history=False):
            calls.append(identifier)
            return Path("/x"), "2", "b@example.com"

        monkeypatch.setattr(SessionManager, "setup_session", fake_setup)
        _lend_live(seeded)
        api.prepare_session_profile(seeded.switcher, "2")
        assert calls == ["2"]

    def test_a_switch_puts_the_accounts_own_login_in(self, linux, seeded, kept):
        _lend_live(seeded)
        payload = api.switch_to(seeded.switcher, "2")
        assert payload["switched"] is True
        assert _live(seeded)["claudeAiOauth"]["refreshToken"] == "rt-2"
        assert api.default_login().kind == "own"
        with pytest.raises(api.OwnLoginError):
            api.use_access_login(kept)
        assert _stored(seeded)["claudeAiOauth"]["refreshToken"] == "rt-2"


class TestClaudeSwapsOwnLogins:
    """What makes the default login one of this computer's own besides a
    refresh token (sak.13): a setup-token, an access token without its
    expiry, an access token alone of one of claude-swap's accounts here. It
    reads as "own" and is never replaced or removed."""

    @staticmethod
    def _left_alone(home: Path, kept: Path, keychain=None) -> None:
        before = _everything(home, keychain)
        with pytest.raises(api.OwnLoginError, match="left as it is"):
            api.use_access_login(kept)
        with pytest.raises(api.OwnLoginError, match="left as it is"):
            api.forget_default_access_login()
        assert _everything(home, keychain) == before

    def test_an_active_setup_token_account(self, linux, seeded, kept):
        added = api.add_setup_token(
            seeded.switcher, "sk-ant-oat01-setup", email="ci@example.com"
        )
        assert api.switch_to(seeded.switcher, added.slot)["switched"] is True
        assert _live(seeded)["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-setup"
        info = api.default_login()
        assert info.kind == "own"
        assert info.email == "ci@example.com"
        self._left_alone(seeded.temp_home, kept)

    def test_an_access_token_alone_of_one_of_its_accounts(self, linux, seeded, kept):
        _lend_live(seeded)
        info = api.default_login()
        assert (info.kind, info.email) == ("own", "b@example.com")
        assert info.expires_at is not None
        assert info.leftover_access is False  # it is the login there, no leftover
        self._left_alone(seeded.temp_home, kept)
        # matched as current_account_number() matches: email and organization
        config = _home_config(seeded.temp_home)
        config["oauthAccount"]["organizationUuid"] = "org-elsewhere"
        (seeded.temp_home / ".claude.json").write_text(json.dumps(config))
        assert api.default_login().kind == "access"

    def test_in_the_keychain_item_too(self, macos, seeded, kept):
        _lend_live(seeded)
        _default_file(seeded.temp_home).unlink()
        lent = _token_login("sk-ant-oat01-lent", None, _in(hours=1))
        macos.set_password(*_default_item(), json.dumps(lent))
        assert api.default_login().kind == "own"
        self._left_alone(seeded.temp_home, kept, macos)

    @pytest.mark.parametrize("expires", ["missing", None, "1900000000000", True])
    def test_an_access_token_without_its_expiry(
        self, temp_home, no_stdin, linux, kept, expires
    ):
        login = _token_login("sk-ant-oat01-a", None, 0)
        if expires == "missing":
            del login["claudeAiOauth"]["expiresAt"]
        else:
            login["claudeAiOauth"]["expiresAt"] = expires
        _default_file(temp_home).write_text(json.dumps(login))
        assert api.default_login().kind == "own"
        self._left_alone(temp_home, kept)

    def test_a_kept_login_of_one_of_its_accounts_is_not_put_in(
        self, linux, seeded, kept
    ):
        _give_org(seeded, 2, "org-2")  # account 2 is the account kept
        _default_file(seeded.temp_home).unlink()  # logged out here
        (seeded.temp_home / ".claude.json").write_text("{}")
        before = _everything(seeded.temp_home, None)
        with pytest.raises(api.ConfigError, match="account 2 of claude") as refused:
            api.use_access_login(kept)
        assert not isinstance(refused.value, api.OwnLoginError)
        assert "cswap switch 2" in str(refused.value)
        assert _everything(seeded.temp_home, None) == before

    def test_an_account_list_it_cannot_read_is_an_error(self, linux, seeded):
        _lend_live(seeded)
        seeded.switcher.sequence_file.write_text('{"accounts": ')
        with pytest.raises(api.ConfigError, match="account list"):
            api.default_login()
        seeded.make_live("a@example.com", 1)  # a refresh token needs no list
        assert api.default_login().kind == "own"


# -- sak.13 review fixes ----------------------------------------------------------


def _read_degraded(monkeypatch, seeded) -> None:
    """The Keychain read failed this pass and the file covered for it: the
    bytes served may be an older generation of the live login."""
    from claude_swap.credentials import ActiveCredentials

    live = (seeded.temp_home / ".claude" / ".credentials.json").read_text()
    degraded = ActiveCredentials(live, False, True)
    monkeypatch.setattr(seeded.switcher, "_read_active_credentials", lambda: degraded)


class TestADefaultLoginThatCannotBeReadNow:
    """A login of the account the default profile is logged in to is neither
    stored nor dropped while that live login cannot be read cleanly: whether
    the account's stored login follows it cannot be told then."""

    @pytest.fixture
    def anthropic(self, monkeypatch) -> _Anthropic:
        fake = _Anthropic()
        monkeypatch.setattr("claude_swap.oauth.fetch_oauth_profile", fake.profile)
        monkeypatch.setattr(
            "claude_swap.oauth.try_refresh_oauth_credentials", fake.refresh
        )
        return fake

    @staticmethod
    def _stored_one(seeded) -> str:
        return seeded.switcher.read_account_credentials("1", "a@example.com")

    def test_add_profile_login(self, seeded, target, monkeypatch):
        (target / ".credentials.json").write_text(_copy_text())
        (target / ".claude.json").write_text(json.dumps({
            "oauthAccount": {"emailAddress": "a@example.com", "accountUuid": "uuid-1"},
        }))
        before = self._stored_one(seeded)
        _read_degraded(monkeypatch, seeded)
        with pytest.raises(api.CredentialReadError, match="cannot be read now"):
            api.add_profile_login(seeded.switcher, target)
        assert self._stored_one(seeded) == before
        assert (target / ".credentials.json").read_text() == _copy_text()

    def test_import_login(self, seeded, tmp_path, claude, target, monkeypatch):
        before = self._stored_one(seeded)
        claude.plan("ok", email="a@example.com", uuid="uuid-1")
        source = _login_file(tmp_path / "a.json")
        _read_degraded(monkeypatch, seeded)
        with pytest.raises(api.LoginNotAddedError, match="cannot be read") as raised:
            api.import_login(seeded.switcher, source, target, claude=claude.path)
        assert raised.value.config_dir == str(target)
        assert "sk-ant-ort01-moved" in (target / ".credentials.json").read_text()
        assert self._stored_one(seeded) == before
        assert _numbers(seeded) == [1, 2]

    def test_add_login_copy(self, seeded, temp_home, target, anthropic, monkeypatch):
        _give_org(seeded, 1, "org-1")
        (temp_home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {
                "emailAddress": "a@example.com",
                "accountUuid": "uuid-1",
                "organizationUuid": "org-1",
            },
        }))
        anthropic.owners["sk-ant-oat01-copy"] = _owner(1, "a@example.com", "org-1")
        before = self._stored_one(seeded)
        _read_degraded(monkeypatch, seeded)
        with pytest.raises(api.CredentialReadError, match="cannot be read now"):
            api.add_login_copy(seeded.switcher, target, credentials=_copy_text())
        assert self._stored_one(seeded) == before
        assert _numbers(seeded) == [1, 2]


def _own_file_in(layout: str, entry: dict) -> tuple[str, str]:
    """A ``.credentials.json`` with ``entry`` as its login, written in
    ``layout``, and the text of that login in it."""
    if layout == "compact":  # as Claude Code writes it
        login = json.dumps(entry, separators=(",", ":"))
        return login, '{"claudeAiOauth":' + login + "}"
    if layout == "spaced":
        login = json.dumps(entry)
        return login, '{"claudeAiOauth": ' + login + "}"
    if layout == "indented":
        login = json.dumps(entry, indent=4)
        return login, '{\n  "claudeAiOauth": ' + login + "\n}\n"
    login = json.dumps(entry, ensure_ascii=False, separators=(" ,\t", " :\r\n"))
    return login, (
        '\t{ "trustedDeviceToken" : "tdt-own" ,\r\n "claudeAiOauth"\t:'
        + login
        + ' , "mcpOAuth":{"mine":{"accessToken":"m"}} }\n'
    )


class TestLeftoverAccess:
    """An access token alone left in one of Claude Code's stores next to a
    login of the computer's own or an API key: ``leftover_access`` says so,
    and ``forget_default_access_login`` removes only that token, leaving the
    own login and ``~/.claude.json`` as they are."""

    def test_a_stale_file_next_to_an_own_keychain_login(self, kept, macos, temp_home):
        own = json.dumps(_token_login("sk-own", "rt-own", _in(hours=8)))
        macos.set_password(*_default_item(), own)
        mcp = {"server": {"accessToken": "mcp"}}
        _default_file(temp_home).write_text(json.dumps({
            **_token_login("sk-ant-oat01-old", None, _in(hours=1)), "mcpOAuth": mcp,
        }))
        config = {"oauthAccount": {"emailAddress": "me@example.com"}, "theme": "dark"}
        (temp_home / ".claude.json").write_text(json.dumps(config))
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("own", True)
        with pytest.raises(api.OwnLoginError, match="left as it is"):
            api.use_access_login(kept)
        assert api.forget_default_access_login() is True
        assert macos.get_password(*_default_item()) == own
        assert json.loads(_default_file(temp_home).read_text()) == {"mcpOAuth": mcp}
        assert _home_config(temp_home) == config
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("own", False)
        with pytest.raises(api.OwnLoginError, match="left as it is"):
            api.forget_default_access_login()

    def test_a_keychain_item_hiding_an_own_login_in_the_file(
        self, kept, macos, temp_home
    ):
        own = json.dumps(_token_login("sk-own", "rt-own", _in(hours=8)))
        _default_file(temp_home).write_text(own)
        lent = _token_login("sk-ant-oat01-lent", None, _in(hours=1))
        macos.set_password(*_default_item(), json.dumps(lent))
        (temp_home / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "me@example.com"}})
        )
        config = (temp_home / ".claude.json").read_text()
        assert api.default_login().leftover_access is True
        assert api.forget_default_access_login() is True
        assert macos.get_password(*_default_item()) is None  # the file is read again
        assert _default_file(temp_home).read_text() == own
        assert (temp_home / ".claude.json").read_text() == config
        assert api.default_login().leftover_access is False

    def test_what_the_item_keeps_besides_moves_next_to_the_own_login(
        self, kept, macos, temp_home
    ):
        """Claude Code reads the item while it holds JSON, so a leftover item
        that keeps an MCP server's login besides goes whole; what it keeps
        moves into the file, and the lent login's device token goes too."""
        own = _token_login("sk-own", "rt-own", _in(hours=8))
        _default_file(temp_home).write_text(json.dumps(own, separators=(",", ":")))
        mcp = {"server|1": {"accessToken": "mcp", "expiresAt": 1}}
        plugins = {"plugin": {"key": "secret"}}
        macos.set_password(*_default_item(), json.dumps({
            **_token_login("sk-ant-oat01-lent", None, _in(hours=1)),
            "mcpOAuth": mcp,
            "pluginSecrets": plugins,
            "trustedDeviceToken": "tdt-lent",
        }))
        (temp_home / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "me@example.com"}})
        )
        config = (temp_home / ".claude.json").read_text()
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("own", True)
        assert api.forget_default_access_login() is True
        assert macos.get_password(*_default_item()) is None
        text = _default_file(temp_home).read_text()
        assert json.loads(text) == {**own, "mcpOAuth": mcp, "pluginSecrets": plugins}
        assert "sk-ant-oat01-lent" not in text and "tdt-lent" not in text
        if sys.platform != "win32":
            assert _default_file(temp_home).stat().st_mode & 0o777 == 0o600
        assert (temp_home / ".claude.json").read_text() == config
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("own", False)
        with pytest.raises(api.OwnLoginError, match="left as it is"):
            api.forget_default_access_login()

    @pytest.mark.parametrize("layout", ["compact", "spaced", "indented", "unusual"])
    def test_the_own_login_stays_byte_for_byte(self, kept, macos, temp_home, layout):
        entry = _token_login("sk-own", "rt-own", _in(hours=8), note='"}{,:[" \\ é\t')[
            "claudeAiOauth"
        ]
        login, written = _own_file_in(layout, entry)
        _default_file(temp_home).write_bytes(written.encode())
        mcp = {"server": {"accessToken": "mcp"}}
        macos.set_password(*_default_item(), json.dumps(
            {**_token_login("sk-ant-oat01-lent", None, _in(hours=1)), "mcpOAuth": mcp}
        ))
        before = json.loads(written)
        assert api.forget_default_access_login() is True
        text = _default_file(temp_home).read_bytes().decode()
        assert login in text
        assert json.loads(text) == {
            **before, "mcpOAuth": {**before.get("mcpOAuth", {}), **mcp}
        }

    def test_the_files_own_keys_win(self, kept, macos, temp_home):
        own = {
            **_token_login("sk-own", "rt-own", _in(hours=8)),
            "trustedDeviceToken": "tdt-own",
            "mcpOAuth": {"shared": {"accessToken": "file"}},
            "mcpXaaIdp": "file",
        }
        _default_file(temp_home).write_text(json.dumps(own))
        more = {"accessToken": "o"}
        macos.set_password(*_default_item(), json.dumps({
            **_token_login("sk-ant-oat01-lent", None, _in(hours=1)),
            "trustedDeviceToken": "tdt-lent",
            "mcpOAuth": {"shared": {"accessToken": "item"}, "more": more},
            "mcpXaaIdp": {"idp": "item"},
            "laterKey": [1, 2],
        }))
        assert api.forget_default_access_login() is True
        assert json.loads(_default_file(temp_home).read_text()) == {
            **own,
            "mcpOAuth": {"shared": {"accessToken": "file"}, "more": more},
            "laterKey": [1, 2],
        }
        assert macos.get_password(*_default_item()) is None

    def test_a_failed_write_of_the_own_file_changes_nothing(
        self, kept, macos, temp_home, monkeypatch
    ):
        _default_file(temp_home).write_text(
            json.dumps(_token_login("sk-own", "rt-own", _in(hours=8)))
        )
        macos.set_password(*_default_item(), json.dumps({
            **_token_login("sk-ant-oat01-lent", None, _in(hours=1)),
            "mcpOAuth": {"server": {"accessToken": "mcp"}},
        }))
        (temp_home / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "me@example.com"}})
        )
        before = _everything(temp_home, macos)

        def full_disk(source, target):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(api, "replace_with_retry", full_disk)
        with pytest.raises(api.CredentialWriteError, match="No space"):
            api.forget_default_access_login()
        assert _everything(temp_home, macos) == before
        assert [
            path.name
            for path in (temp_home / ".claude").iterdir()
            if "credentials" in path.name
        ] == [".credentials.json"]  # no copy of the file is left behind
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("own", True)

    def test_a_failed_removal_of_the_item_is_finished_by_a_retry(
        self, kept, macos, temp_home, monkeypatch
    ):
        from claude_swap import macos_keychain

        own = _token_login("sk-own", "rt-own", _in(hours=8))
        _default_file(temp_home).write_text(json.dumps(own))
        mcp = {"server": {"accessToken": "mcp"}}
        item = json.dumps(
            {**_token_login("sk-ant-oat01-lent", None, _in(hours=1)), "mcpOAuth": mcp}
        )
        macos.set_password(*_default_item(), item)

        def busy(service, account):
            raise macos_keychain.KeychainError("security timed out")

        monkeypatch.setattr(macos_keychain, "delete_password", busy)
        with pytest.raises(api.CredentialWriteError, match="Keychain"):
            api.forget_default_access_login()
        assert macos.get_password(*_default_item()) == item  # the token stays there
        moved = {**own, "mcpOAuth": mcp}
        assert json.loads(_default_file(temp_home).read_text()) == moved
        assert api.default_login().leftover_access is True
        monkeypatch.setattr(macos_keychain, "delete_password", macos.delete_password)
        assert api.forget_default_access_login() is True
        assert macos.get_password(*_default_item()) is None
        assert json.loads(_default_file(temp_home).read_text()) == moved
        assert api.default_login().leftover_access is False

    def test_an_api_key_with_an_access_token_left(self, kept, linux, temp_home):
        (temp_home / ".claude.json").write_text(
            json.dumps({"primaryApiKey": "sk-ant-api03-key"})
        )
        lent = _token_login("sk-ant-oat01-lent", None, _in(hours=1))
        _default_file(temp_home).write_text(json.dumps(lent))
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("api_key", True)
        assert api.forget_default_access_login() is True
        assert not _default_file(temp_home).exists()
        assert _home_config(temp_home) == {"primaryApiKey": "sk-ant-api03-key"}

    def test_the_file_one_failed_keychain_write_left_is_removed(
        self, kept, macos, temp_home, monkeypatch
    ):
        """One failed Keychain write puts the login in the file, later writes
        keep the file, and the user's own login then goes into the Keychain
        item only (``claude /login`` in a terminal)."""
        from claude_swap import macos_keychain

        failed: list[str] = []

        def flaky_set(service, account, password):
            if service == "Claude Code-credentials" and not failed:
                failed.append(service)
                raise macos_keychain.KeychainError("security timed out")
            return macos.set_password(service, account, password)

        monkeypatch.setattr(macos_keychain, "set_password", flaky_set)
        api.use_access_login(kept)
        assert _default_file(temp_home).exists()
        api.keep_access_login(kept, **{**_KEPT, "access_token": "sk-ant-oat01-newer"})
        api.use_access_login(kept)  # the item takes it again; the file is rewritten
        own = json.dumps(_token_login("sk-own", "rt-own", _in(hours=8)))
        macos.set_password(*_default_item(), own)
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("own", True)
        assert api.forget_default_access_login() is True
        assert not _default_file(temp_home).exists()
        assert macos.get_password(*_default_item()) == own


class TestALegacyStoreOnLinux:
    """Linux/WSL: a legacy ``~/.claude-swap-backup`` not moved yet is moved
    first, as opening claude-swap moves it, so no lock or read of the
    default-login functions leaves the new store beside it."""

    @staticmethod
    def _legacy(home: Path) -> Path:
        legacy = home / ".claude-swap-backup"
        legacy.mkdir()
        (legacy / "sequence.json").write_text(json.dumps({
            "activeAccountNumber": 1,
            "sequence": [1],
            "accounts": {"1": {"email": "a@example.com", "organizationUuid": ""}},
        }))
        return legacy

    @staticmethod
    def _new_store(home: Path) -> Path:
        return home / ".local" / "share" / "claude-swap"

    def test_forget_moves_it_first(self, temp_home, no_stdin, linux):
        from claude_swap.switcher import ClaudeAccountSwitcher

        legacy = self._legacy(temp_home)
        assert api.forget_default_access_login() is False
        assert not legacy.exists()
        moved = json.loads((self._new_store(temp_home) / "sequence.json").read_text())
        assert moved["sequence"] == [1]
        ClaudeAccountSwitcher()  # opens: no "both paths exist" refusal

    def test_its_accounts_count_before_it_is_moved(self, temp_home, no_stdin, linux):
        self._legacy(temp_home)
        lent = _token_login("sk-ant-oat01-lent", None, _in(hours=1))
        _default_file(temp_home).write_text(json.dumps(lent))
        (temp_home / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "a@example.com"}})
        )
        assert api.default_login().kind == "own"

    def test_a_store_it_cannot_move_is_a_config_error(
        self, temp_home, no_stdin, linux, kept
    ):
        legacy = self._legacy(temp_home)
        new = self._new_store(temp_home)
        new.mkdir(parents=True)
        (new / "sequence.json").write_text("{}")  # both hold accounts: never merged
        calls = (api.forget_default_access_login, lambda: api.use_access_login(kept))
        for call in calls:
            with pytest.raises(api.ConfigError, match="cannot be used now"):
                call()
        assert (legacy / "sequence.json").exists()
        assert sorted(p.name for p in new.iterdir()) == ["sequence.json"]


class TestATransientKeychainFailure:
    """The default-login functions read the Keychain with claude-swap's
    bounded retry, as its active read does."""

    def test_a_read_that_fails_once_is_tried_again(
        self, temp_home, no_stdin, macos, monkeypatch
    ):
        from claude_swap import macos_keychain

        monkeypatch.setattr(api, "_ACTIVE_READ_RETRY_DELAY", 0)
        tries: list[str] = []

        def flaky_get(service, account):
            tries.append(service)
            if service == "Claude Code-credentials" and tries.count(service) == 1:
                raise macos_keychain.KeychainError("security timed out")
            return macos.get_password(service, account)

        monkeypatch.setattr(macos_keychain, "get_password", flaky_get)
        own = _token_login("sk-own", "rt-own", _in(hours=8))
        macos.set_password(*_default_item(), json.dumps(own))
        assert api.default_login().kind == "own"
        assert tries.count("Claude Code-credentials") == 2

    def test_a_keychain_that_stays_locked_is_an_error(
        self, temp_home, no_stdin, macos, monkeypatch, kept
    ):
        from claude_swap import macos_keychain

        monkeypatch.setattr(api, "_ACTIVE_READ_RETRY_DELAY", 0)
        tries: list[str] = []

        def locked(service, account):
            tries.append(service)
            raise macos_keychain.KeychainError("the keychain is locked")

        monkeypatch.setattr(macos_keychain, "get_password", locked)
        with pytest.raises(api.CredentialReadError, match="cannot be read now"):
            api.default_login()
        assert tries == ["Claude Code-credentials"] * 2
        tries.clear()
        with pytest.raises(api.CredentialReadError):
            api.use_access_login(kept)
        assert len(tries) == 2  # the profile's own item, then nothing more


# -- a login of this computer's own replaced, with the user's OK (sak.14) ---------


def _own_login(
    access: str = "sk-ant-oat01-own", refresh: str = "sk-ant-ort01-own", **extra
) -> dict:
    """A login someone made on this computer (``claude /login``)."""
    return _token_login(access, refresh, _in(hours=8), **extra)


def _anywhere(home: Path, keychain, secret: str) -> list[str]:
    """Where ``secret`` is still kept: a file under ``home`` (claude-swap's
    base64 ``.enc`` files read decoded), or a Keychain item."""
    import base64
    import binascii

    places = []
    for path in home.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        if path.suffix == ".enc":
            try:
                text = base64.b64decode(text.strip()).decode("utf-8", errors="ignore")
            except binascii.Error:
                pass
        if secret in text:
            places.append(str(path.relative_to(home)))
    if keychain is not None:
        places += [
            service for (service, _), value in keychain.data.items() if secret in value
        ]
    return places


_OWN_SECRETS = ("sk-ant-oat01-own", "sk-ant-ort01-own")


class TestOwnKind:
    """What a replace would do to the computer's own login or key
    (``DefaultLoginInfo.own_kind``, sak.14)."""

    def test_nothing_of_its_own_is_none(self, kept, linux):
        assert api.default_login().own_kind is None
        api.use_access_login(kept)
        assert api.default_login().own_kind is None

    @pytest.mark.parametrize(
        "entry",
        [
            {"accessToken": "sk-ant-oat01-own", "refreshToken": "sk-ant-ort01-own"},
            {"accessToken": "sk-ant-oat01-own", "scopes": ["user:inference"]},
            {
                "accessToken": "sk-ant-oat01-own",
                "refreshToken": None,
                "scopes": _SCOPES,
            },
        ],
    )
    def test_a_login_made_here(self, temp_home, no_stdin, linux, entry):
        _default_file(temp_home).write_text(json.dumps({"claudeAiOauth": entry}))
        (temp_home / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "me@example.com"}})
        )
        info = api.default_login()
        assert (info.kind, info.own_kind) == ("own", "login")

    def test_claude_swaps_active_account(self, linux, seeded):
        info = api.default_login()
        assert (info.kind, info.own_kind) == ("own", "account")
        assert info.email == "a@example.com"

    def test_an_access_token_alone_of_one_of_its_accounts(self, linux, seeded):
        _lend_live(seeded)
        assert api.default_login().own_kind == "account"

    def test_an_api_key_also_next_to_a_login(self, temp_home, no_stdin, macos):
        macos.set_password(*_default_item(), json.dumps(_own_login()))
        macos.set_password(*_managed_item(), "sk-ant-api03-key")
        info = api.default_login()
        assert (info.kind, info.own_kind) == ("own", "api_key")

    def test_an_account_list_it_cannot_read(self, linux, seeded, kept):
        """Without the list a login of its own reads as one made here, the
        stronger warning; a replace waits for the list."""
        seeded.switcher.sequence_file.write_text('{"accounts": ')
        info = api.default_login()
        assert (info.kind, info.own_kind) == ("own", "login")
        before = _everything(seeded.temp_home, None)
        with pytest.raises(api.ConfigError, match="could not be parsed"):
            api.use_access_login(kept, replace_own=True)
        assert _everything(seeded.temp_home, None) == before


class TestReplaceAnOwnLogin:
    """``use_access_login(..., replace_own=True)`` with a login made on this
    computer (sak.14): it goes from every store Claude Code reads, and the
    kept access login takes its place."""

    def test_in_the_keychain_item(self, kept, macos, temp_home, capfd):
        mcp = {"server": {"accessToken": "mcp"}}
        macos.set_password(*_default_item(), json.dumps({
            **_own_login(), "mcpOAuth": mcp, "trustedDeviceToken": "tdt-own",
        }))
        (temp_home / ".claude.json").write_text(json.dumps({
            "theme": "dark",
            "oauthAccount": {"emailAddress": "me@example.com", "displayName": "Me"},
        }))
        capfd.readouterr()
        assert api.use_access_login(kept, replace_own=True) is True
        _assert_silent(capfd)
        assert json.loads(macos.get_password(*_default_item())) == {
            "claudeAiOauth": _kept_login(kept)["claudeAiOauth"], "mcpOAuth": mcp,
        }
        assert not _default_file(temp_home).exists()  # no file is made
        assert _home_config(temp_home) == {
            "theme": "dark",
            "oauthAccount": _KEPT_ACCOUNT,
            "hasCompletedOnboarding": True,
        }
        assert api.default_login() == api.DefaultLoginInfo(
            kind="access",
            email="b@example.com",
            organization_uuid="org-2",
            expires_at="2026-10-16T21:47:12.345Z",
        )
        for secret in (*_OWN_SECRETS, "tdt-own"):
            assert _anywhere(temp_home, macos, secret) == []
        assert api.use_access_login(kept, replace_own=True) is False

    def test_in_the_file(self, kept, linux, temp_home):
        _default_file(temp_home).write_text(json.dumps(_own_login()))
        assert api.use_access_login(kept, replace_own=True) is True
        assert json.loads(_default_file(temp_home).read_text()) == {
            "claudeAiOauth": _kept_login(kept)["claudeAiOauth"]
        }
        if sys.platform != "win32":
            assert _default_file(temp_home).stat().st_mode & 0o777 == 0o600
        for secret in _OWN_SECRETS:
            assert _anywhere(temp_home, None, secret) == []

    def test_in_both_stores(self, kept, macos, temp_home):
        macos.set_password(*_default_item(), json.dumps(_own_login()))
        _default_file(temp_home).write_text(
            json.dumps(_own_login("sk-ant-oat01-old", "sk-ant-ort01-old"))
        )
        assert api.use_access_login(kept, replace_own=True) is True
        held = json.loads(macos.get_password(*_default_item()))
        assert held["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-kept"
        # running sessions read the file's new time and take it again
        assert json.loads(_default_file(temp_home).read_text()) == held
        for secret in (*_OWN_SECRETS, "sk-ant-ort01-old"):
            assert _anywhere(temp_home, macos, secret) == []

    def test_hidden_behind_an_item_without_a_login(self, kept, macos, temp_home):
        mcp = {"server": {"accessToken": "mcp"}}
        macos.set_password(*_default_item(), json.dumps({"mcpOAuth": mcp}))
        _default_file(temp_home).write_text(json.dumps(_own_login()))
        assert api.default_login().own_kind == "login"
        assert api.use_access_login(kept, replace_own=True) is True
        assert json.loads(macos.get_password(*_default_item()))["mcpOAuth"] == mcp
        for secret in _OWN_SECRETS:
            assert _anywhere(temp_home, macos, secret) == []
        assert api.default_login().kind == "access"

    def test_beside_the_very_login_it_puts_in(self, kept, macos, temp_home):
        api.use_access_login(kept)
        item = macos.get_password(*_default_item())
        mcp = {"server": {"accessToken": "mcp"}}
        _default_file(temp_home).write_text(
            json.dumps({**_own_login(), "mcpOAuth": mcp, "trustedDeviceToken": "t"})
        )
        info = api.default_login()
        assert (info.kind, info.own_kind) == ("own", "login")
        assert info.leftover_access is True
        assert api.use_access_login(kept, replace_own=True) is True
        assert macos.get_password(*_default_item()) == item
        assert json.loads(_default_file(temp_home).read_text()) == {"mcpOAuth": mcp}
        info = api.default_login()
        assert (info.kind, info.leftover_access) == ("access", False)

    @pytest.mark.parametrize(
        "entry",
        [
            {"accessToken": "sk-ant-oat01-own", "scopes": ["user:inference"]},
            {"accessToken": "sk-ant-oat01-own"},  # a setup-token stored early
            {
                "accessToken": "sk-ant-oat01-own",
                "refreshToken": None,
                "scopes": _SCOPES,
            },
        ],
    )
    def test_its_other_logins(self, kept, linux, temp_home, entry):
        _default_file(temp_home).write_text(json.dumps({"claudeAiOauth": entry}))
        assert api.use_access_login(kept, replace_own=True) is True
        assert _anywhere(temp_home, None, "sk-ant-oat01-own") == []
        assert api.default_login().kind == "access"

    @pytest.mark.parametrize("where", ["managed-key", "config-key", "beside-a-login"])
    def test_an_api_key_is_never_replaced(self, kept, macos, temp_home, where):
        if where == "config-key":
            (temp_home / ".claude.json").write_text(
                json.dumps({"primaryApiKey": "sk-ant-api03-key"})
            )
        else:
            macos.set_password(*_managed_item(), "sk-ant-api03-key")
        if where == "beside-a-login":
            macos.set_password(*_default_item(), json.dumps(_own_login()))
        before = _everything(temp_home, macos)
        with pytest.raises(api.OwnLoginError, match="uses an API key of its own"):
            api.use_access_login(kept, replace_own=True)
        assert _everything(temp_home, macos) == before

    def test_a_store_the_write_did_not_reach_loses_it_too(
        self, kept, macos, temp_home, monkeypatch
    ):
        """The Keychain item takes the new login while the file beside it
        cannot be written now: the own login still goes from the file."""
        from claude_swap.credentials import CredentialStore

        macos.set_password(*_default_item(), json.dumps(_own_login()))
        _default_file(temp_home).write_text(
            json.dumps(_own_login("sk-ant-oat01-old", "sk-ant-ort01-old"))
        )

        def full_disk(_store, _credentials):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(
            CredentialStore, "_write_active_credentials_file", full_disk
        )
        assert api.use_access_login(kept, replace_own=True) is True
        assert not _default_file(temp_home).exists()
        held = json.loads(macos.get_password(*_default_item()))
        assert held["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-kept"
        assert api.default_login().kind == "access"

    def test_an_own_login_that_stays_puts_both_stores_back(
        self, kept, macos, temp_home, monkeypatch
    ):
        """A Keychain that takes no write: the new login lands in the file,
        and the item Claude Code reads first would keep the own login."""
        from claude_swap import macos_keychain

        macos.set_password(*_default_item(), json.dumps(_own_login()))
        (temp_home / ".claude.json").write_text(
            json.dumps({"oauthAccount": {"emailAddress": "me@example.com"}})
        )
        before = _everything(temp_home, macos)

        def locked(service, *_args):
            raise macos_keychain.KeychainError(f"{service}: the keychain is locked")

        monkeypatch.setattr(macos_keychain, "set_password", locked)
        monkeypatch.setattr(macos_keychain, "delete_password", locked)
        with pytest.raises(api.CredentialWriteError, match="Keychain item"):
            api.use_access_login(kept, replace_own=True)
        assert _everything(temp_home, macos) == before
        assert api.default_login().own_kind == "login"

    def test_it_is_put_back_when_the_account_cannot_be_recorded(
        self, kept, macos, temp_home, monkeypatch
    ):
        macos.set_password(*_default_item(), json.dumps(_own_login()))
        _default_file(temp_home).write_text(json.dumps(_own_login()))
        before = _everything(temp_home, macos)

        def full_disk(_config):
            raise api.CredentialWriteError("No space left on device")

        monkeypatch.setattr(api, "_write_default_config", full_disk)
        with pytest.raises(api.CredentialWriteError, match="No space"):
            api.use_access_login(kept, replace_own=True)
        assert _everything(temp_home, macos) == before

    def test_waits_for_a_switch(self, kept, linux, temp_home, monkeypatch):
        from claude_swap.locking import FileLock
        from claude_swap.paths import get_backup_root

        _default_file(temp_home).write_text(json.dumps(_own_login()))
        before = _everything(temp_home, None)
        monkeypatch.setattr(api, "DEFAULT_LOGIN_LOCK_TIMEOUT", 0.2)
        switch = FileLock(get_backup_root() / ".lock")
        assert switch.acquire()
        try:
            with pytest.raises(api.LockError, match="switching"):
                api.use_access_login(kept, replace_own=True)
        finally:
            switch.release()
        assert _everything(temp_home, None) == before

    def test_never_shows_a_token(self, kept, macos, temp_home, caplog, capfd):
        import logging

        caplog.set_level(logging.DEBUG, logger="claude-swap")
        macos.set_password(*_default_item(), json.dumps(_own_login()))
        capfd.readouterr()
        api.use_access_login(kept, replace_own=True)
        info = api.default_login()
        out, err = capfd.readouterr()
        seen = caplog.text + out + err + repr(info)
        for secret in ("sk-ant-oat01-kept", *_OWN_SECRETS):
            assert secret not in seen

    def test_with_nothing_of_its_own_it_is_a_plain_switch(self, kept, macos, temp_home):
        macos.set_password(*_default_item(), "")  # worker-1's empty item
        assert api.use_access_login(kept, replace_own=True) is True
        assert json.loads(macos.get_password(*_default_item())) == {
            "claudeAiOauth": _kept_login(kept)["claudeAiOauth"]
        }
        assert api.use_access_login(kept, replace_own=True) is False

    def test_refuses_what_it_refuses_without_it(
        self, kept, temp_home, linux, monkeypatch, tmp_path
    ):
        _default_file(temp_home).write_text(json.dumps(_own_login()))
        before = _everything(temp_home, None)
        other = temp_home / ".claude-work"
        other.mkdir()
        for name, match in (
            ("CLAUDE_CONFIG_DIR", "CLAUDE_CONFIG_DIR"),
            ("CLAUDE_SECURESTORAGE_CONFIG_DIR", "CLAUDE_SECURESTORAGE"),
        ):
            monkeypatch.setenv(name, str(other))
            with pytest.raises(api.ConfigError, match=match):
                api.use_access_login(kept, replace_own=True)
            monkeypatch.delenv(name)
        for place, match in (
            (temp_home / ".claude", "default profile"),
            ("relative/b", "full path"),
            (tmp_path / "nowhere", "does not exist"),
        ):
            with pytest.raises(api.ConfigError, match=match):
                api.use_access_login(place, replace_own=True)
        assert _everything(temp_home, None) == before


class TestReplaceClaudeSwapsActiveAccount:
    """``use_access_login(..., replace_own=True)`` while one of claude-swap's
    accounts is active (sak.14): its live login is kept in claude-swap's
    store as a switch away keeps it, the account stays, and the kept access
    login takes its place."""

    @staticmethod
    def _stash(seeded) -> dict[str, str]:
        store = seeded.switcher._store
        return {
            entry: store._read_unclaimed_credential(entry)[0]
            for entry in store._list_unclaimed_credentials()
        }

    @classmethod
    def _left(cls, seeded, stashed: dict[str, str]) -> tuple:
        """What claude-swap keeps of account 1, and what it set aside."""
        switcher = seeded.switcher
        return (
            switcher.read_account_credentials("1", "a@example.com"),
            switcher._read_account_config("1", "a@example.com"),
            switcher._get_sequence_data()["accounts"]["1"]["uuid"],
            sorted(
                text for entry, text in cls._stash(seeded).items()
                if entry not in stashed
            ),
        )

    @pytest.mark.parametrize(
        ("case", "kind"),
        [
            ("its-bytes", "own-bytes"),
            ("its-lineage", "own-family"),
            ("renewed", "own-rotated"),
            ("renewed-no-uuid", "own-rotated"),
            ("unresolved", "unresolved"),
            ("another-account", "foreign"),
            ("another-accounts-very-login", "foreign-synced"),
            ("no-account-here", "alien"),
        ],
    )
    def test_its_login_is_kept_as_a_switch_away_keeps_it(
        self, linux, seeded, kept, monkeypatch, case, kind
    ):
        from claude_swap.switcher import ClaudeAccountSwitcher

        kinds: list[str | None] = []
        keep = ClaudeAccountSwitcher._back_up_replaced_login

        def spy(switcher, live, provenance):
            kinds.append(keep(switcher, live, provenance))
            return kinds[-1]

        monkeypatch.setattr(ClaudeAccountSwitcher, "_back_up_replaced_login", spy)
        stored = json.dumps(
            _token_login("sk-ant-oat01-a1", "sk-ant-ort01-a", _in(hours=1))
        )
        live = {
            "its-bytes": stored,
            "its-lineage": json.dumps(
                _token_login("sk-ant-oat01-a2", "sk-ant-ort01-a", _in(hours=8))
            ),
        }.get(case, json.dumps(_own_login("sk-ant-oat01-a3", "sk-ant-ort01-a3")))
        owner = {
            "renewed": _owner(1, "a@example.com", None),
            "renewed-no-uuid": _owner(1, "a@example.com", None),
            "another-account": _owner(2, "b@example.com", None),
            "another-accounts-very-login": _owner(2, "b@example.com", None),
            "no-account-here": _owner(9, "z@example.com", "org-z"),
        }.get(case)
        monkeypatch.setattr("claude_swap.oauth.fetch_oauth_profile", lambda _t: owner)
        config = json.dumps({
            "oauthAccount": {"emailAddress": "a@example.com", "accountUuid": "uuid-1"},
            "theme": "dark",
        })

        def arrange() -> None:
            _keep_stored(seeded, json.loads(stored), num=1, email="a@example.com")
            seeded.switcher._write_account_config("1", "a@example.com", "{}")
            if case == "another-accounts-very-login":
                _keep_stored(seeded, json.loads(live))  # account 2 holds it
            data = seeded.switcher._get_sequence_data()
            uuid = "" if case == "renewed-no-uuid" else "uuid-1"
            data["accounts"]["1"]["uuid"] = uuid
            seeded.switcher._write_json(seeded.switcher.sequence_file, data)
            _make_live(seeded, json.loads(live))
            (seeded.temp_home / ".claude.json").write_text(config)

        arrange()
        assert api.default_login().own_kind == "account"
        stashed = self._stash(seeded)
        assert api.use_access_login(kept, replace_own=True) is True
        assert kinds == [kind]
        replaced = self._left(seeded, stashed)
        assert api.default_login().kind == "access"
        assert api.active_account(seeded.switcher) is None
        arrange()
        stashed = self._stash(seeded)
        assert api.switch_to(seeded.switcher, "2")["switched"] is True
        assert self._left(seeded, stashed) == replaced

    def test_the_account_stays_and_a_switch_goes_back_to_it(self, linux, seeded, kept):
        live = _own_login("sk-ant-oat01-a2", "sk-ant-ort01-a2")
        _make_live(seeded, live)
        assert api.use_access_login(kept, replace_own=True) is True
        assert _numbers(seeded) == [1, 2]
        stored = seeded.switcher.read_account_credentials("1", "a@example.com")
        assert json.loads(stored) == live
        assert api.switch_to(seeded.switcher, "1")["switched"] is True
        assert _live(seeded) == live
        info = api.default_login()
        assert (info.kind, info.own_kind) == ("own", "account")
        assert info.email == "a@example.com"

    def test_in_the_keychain_item(self, macos, seeded, kept):
        live = _own_login("sk-ant-oat01-a2", "sk-ant-ort01-a2")
        _default_file(seeded.temp_home).unlink()
        macos.set_password(*_default_item(), json.dumps(live))
        assert api.use_access_login(kept, replace_own=True) is True
        held = json.loads(macos.get_password(*_default_item()))
        assert held["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-kept"
        stored = api.open_switcher().read_account_credentials("1", "a@example.com")
        assert json.loads(stored) == live

    def test_an_access_token_alone_of_one_of_its_accounts(self, linux, seeded, kept):
        _lend_live(seeded)  # account 2's access, lent by another computer
        stored = seeded.switcher.read_account_credentials("2", "b@example.com")
        assert api.use_access_login(kept, replace_own=True) is True
        assert seeded.switcher.read_account_credentials("2", "b@example.com") == stored
        assert seeded.switcher.list_unclaimed_credentials() == {}
        assert api.default_login().kind == "access"

    def test_an_active_setup_token_account(self, linux, seeded, kept):
        added = api.add_setup_token(
            seeded.switcher, "sk-ant-oat01-setup", email="ci@example.com"
        )
        api.switch_to(seeded.switcher, added.slot)
        assert api.use_access_login(kept, replace_own=True) is True
        assert "sk-ant-oat01-setup" in seeded.switcher.read_account_credentials(
            added.slot, "ci@example.com"
        )
        assert api.default_login().kind == "access"
        assert api.switch_to(seeded.switcher, added.slot)["switched"] is True
        assert _live(seeded)["claudeAiOauth"]["accessToken"] == "sk-ant-oat01-setup"

    def test_a_replace_that_cannot_finish_leaves_the_account_as_it_was(
        self, linux, seeded, kept, monkeypatch
    ):
        live = _own_login("sk-ant-oat01-a2", "sk-ant-ort01-a2")
        _make_live(seeded, live)

        def full_disk(_config):
            raise api.CredentialWriteError("No space left on device")

        monkeypatch.setattr(api, "_write_default_config", full_disk)
        with pytest.raises(api.CredentialWriteError, match="No space"):
            api.use_access_login(kept, replace_own=True)
        assert _live(seeded) == live
        stored = seeded.switcher.read_account_credentials("1", "a@example.com")
        assert json.loads(stored) == live
        assert api.active_account(seeded.switcher) == "1"

    def test_whose_login_it_is_is_asked_before_any_lock(
        self, linux, seeded, kept, monkeypatch
    ):
        """Telling whose a renewed login is takes a network call, which never
        runs while claude-swap's or Claude Code's locks are held."""
        from claude_swap.locking import FileLock
        from claude_swap.paths import get_backup_root

        home = seeded.temp_home
        claude_code_locks = (
            home / ".claude" / ".oauth_refresh.lock",
            home / ".claude.lock",
            home / ".claude" / ".storage-write.lock",
            home / ".claude.json.lock",
        )
        held: list[bool] = []

        def owner(_token: str) -> dict:
            switching = FileLock(get_backup_root() / ".lock")
            free = switching.acquire(timeout=0)
            if free:
                switching.release()
            held.append(not free or any(lock.exists() for lock in claude_code_locks))
            return _owner(1, "a@example.com", None)

        monkeypatch.setattr("claude_swap.oauth.fetch_oauth_profile", owner)
        live = _own_login("sk-ant-oat01-a3", "sk-ant-ort01-a3")  # a renewal
        _make_live(seeded, live)
        assert api.use_access_login(kept, replace_own=True) is True
        assert held and not any(held)
        stored = seeded.switcher.read_account_credentials("1", "a@example.com")
        assert json.loads(stored) == live

    def test_a_safety_copy_that_cannot_be_written_changes_nothing(
        self, linux, seeded, kept, monkeypatch
    ):
        """A live login proven to be no login of the account's goes into a
        safety copy before it is replaced, as a switch keeps it; when that
        copy cannot be written, nothing is replaced."""
        from claude_swap.credentials import CredentialStore

        monkeypatch.setattr(
            "claude_swap.oauth.fetch_oauth_profile",
            lambda _token: _owner(9, "z@example.com", "org-z"),
        )
        _make_live(seeded, _own_login("sk-ant-oat01-z", "sk-ant-ort01-z"))
        stored = seeded.switcher.read_account_credentials("1", "a@example.com")
        before = _everything(seeded.temp_home, None)

        def full_disk(_store, _credentials, _context):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(CredentialStore, "_write_unclaimed_credential", full_disk)
        with pytest.raises(api.CredentialWriteError, match="No space") as refused:
            api.use_access_login(kept, replace_own=True)
        assert "left as it is" in str(refused.value)
        assert _everything(seeded.temp_home, None) == before
        assert seeded.switcher.read_account_credentials("1", "a@example.com") == stored
        assert seeded.switcher.list_unclaimed_credentials() == {}
        assert api.active_account(seeded.switcher) == "1"
