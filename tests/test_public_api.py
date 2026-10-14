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
from claude_swap.oauth import RefreshOutcome
from tests.test_autoswitch import EngineHarness

EXPECTED_ALL = [
    "SCHEMA_VERSION",
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
    "ErrorEvent",
    "LockError",
    "LoginNotAddedError",
    "LoginSource",
    "MovedLogin",
    "NoSwitchEvent",
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
    "accounts_json",
    "active_account",
    "add_current_login",
    "add_login_copy",
    "add_profile_login",
    "add_setup_token",
    "check_login_source",
    "create_engine",
    "forget_profile_login",
    "import_login",
    "inject_system_trust",
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
        "share_history: 'bool' = True) -> 'SessionProfile'"
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
        "claude: 'str | os.PathLike[str] | None' = None) -> 'MovedLogin'"
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
    "env": {k: v for k, v in os.environ.items() if k.startswith(("CLAUDE", "ANTHROPIC"))},
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
