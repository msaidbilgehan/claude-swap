"""Pins ``CredentialStore.early_reprobe``, claude-SAK's Keychain re-probe.

``api.open_switcher`` turns it on: a long-running process re-probes the macOS
Keychain after a failure anyway, so its store does it as soon as the Keychain
answers, and the active read asks the Keychain during a failure cooldown
instead of waiting it out. Without it, one idle slot's ``security`` timeout
made the active slot read as "no credentials" for up to a minute. A switcher
built directly (the CLI, TUI and menu bar) keeps upstream's routing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claude_swap import api, credentials, macos_keychain
from claude_swap.credentials import (
    CLAUDE_CODE_KEYCHAIN_SERVICE,
    SECURITY_SERVICE,
    ActiveCredentials,
    CredentialStore,
)
from claude_swap.models import Platform
from claude_swap.paths import get_credentials_path
from claude_swap.switcher import ClaudeAccountSwitcher

EMAILS = {1: "a@example.com", 2: "b@example.com", 3: "c@example.com"}


def _login(token: str) -> str:
    return json.dumps({
        "claudeAiOauth": {
            "accessToken": token,
            "refreshToken": f"rt-{token}",
            "expiresAt": 9_999_999_999_999,
        }
    })


def _active_item() -> tuple[str, str]:
    return CLAUDE_CODE_KEYCHAIN_SERVICE, macos_keychain.keychain_account_name()


def _backup_item(num: int) -> tuple[str, str]:
    return SECURITY_SERVICE, f"account-{num}-{EMAILS[num]}"


class _Keychain:
    """The in-memory Keychain, with items whose next reads time out."""

    def __init__(self, fake) -> None:
        self._fake = fake
        self.data: dict[tuple[str, str], str] = fake.data
        self.timeouts: dict[tuple[str, str], int] = {}
        self.reads: list[tuple[str, str]] = []

    def get_password(self, service: str, account: str) -> str | None:
        item = (service, account)
        self.reads.append(item)
        if self.timeouts.get(item, 0):
            self.timeouts[item] -= 1
            raise macos_keychain.KeychainError(
                "security find-generic-password timed out after 5.0s"
            )
        return self._fake.get_password(service, account)


@pytest.fixture
def keychain(block_real_keychain, monkeypatch) -> _Keychain:
    fake = _Keychain(block_real_keychain)
    monkeypatch.setattr(macos_keychain, "get_password", fake.get_password)
    monkeypatch.setattr(credentials, "_ACTIVE_READ_RETRY_DELAY", 0.0)
    return fake


def _macos_store(*, early_reprobe: bool) -> CredentialStore:
    switcher = ClaudeAccountSwitcher()
    switcher.platform = Platform.MACOS
    switcher._store.early_reprobe = early_reprobe
    return switcher._store


def _start_cooldown(store: CredentialStore, keychain: _Keychain) -> None:
    """An unrelated item's read times out, as slot 1's backup did."""
    keychain.timeouts[("svc", "acct")] = 1
    with pytest.raises(macos_keychain.KeychainError):
        store._kc_call(macos_keychain.get_password, "svc", "acct")
    assert store._use_keychain() is False, "premise: the cooldown started"


def _seed_three_accounts(
    switcher: ClaudeAccountSwitcher, temp_home: Path, keychain: _Keychain
) -> None:
    """Slots 1-3 with their backups in the Keychain; slot 3 is the live login.

    The field shape: no ``.enc`` backups and no plaintext credentials file, so
    every login, the active one too, lives only in the Keychain, and the
    active slot is read after the others.
    """
    switcher.platform = Platform.MACOS
    switcher._setup_directories()
    switcher._init_sequence_file()
    data = switcher._get_sequence_data()
    for num, email in EMAILS.items():
        switcher._write_account_credentials(str(num), email, _login(f"at-{num}"))
        switcher._write_account_config(
            str(num),
            email,
            json.dumps({
                "oauthAccount": {"emailAddress": email, "accountUuid": f"uuid-{num}"},
            }),
        )
        data["accounts"][str(num)] = {
            "email": email,
            "uuid": f"uuid-{num}",
            "organizationUuid": "",
            "organizationName": "",
            "added": "2024-01-01T00:00:00Z",
        }
        data["sequence"].append(num)
    data["activeAccountNumber"] = 3
    switcher._write_json(switcher.sequence_file, data)
    keychain.data[_active_item()] = _login("at-3")
    (temp_home / ".claude.json").write_text(json.dumps({
        "oauthAccount": {"emailAddress": EMAILS[3], "accountUuid": "uuid-3"},
    }))
    assert keychain.data[_backup_item(1)] == _login("at-1"), "premise"
    assert not get_credentials_path().exists(), "premise"


def _statuses(payload: dict) -> dict[int, str]:
    return {row["number"]: row["usageStatus"] for row in payload["accounts"]}


class TestOpenSwitcher:
    def test_the_embedding_switcher_reprobes_early(self, temp_home):
        assert api.open_switcher()._store.early_reprobe is True

    def test_a_switcher_built_directly_keeps_upstream_routing(self, temp_home):
        assert ClaudeAccountSwitcher()._store.early_reprobe is False


class TestTheFieldCase:
    """Through ``api.accounts_json``, the way claude-SAK lists accounts."""

    @pytest.mark.parametrize("failing", [1, 2], ids=["slot-1", "slot-2"])
    def test_an_idle_backup_timeout_does_not_hide_the_active_login(
        self, temp_home, keychain, failing
    ):
        """Slot 1 is the measured case: slot 2's readable backup cleared the
        failure, and the active slot 3 then read as "no credentials" until
        the cooldown ran out. Slot 2 fails right before the active read, with
        no answer in between: the active read asks the Keychain itself."""
        switcher = api.open_switcher()
        _seed_three_accounts(switcher, temp_home, keychain)
        keychain.timeouts[_backup_item(failing)] = 1

        first = _statuses(api.accounts_json(switcher))
        second = _statuses(api.accounts_json(switcher))

        login_problems = {"no_credentials", "keychain_unavailable"}
        assert first[3] not in login_problems, first
        assert second[3] not in login_problems, second
        assert second[1] not in login_problems and second[2] not in login_problems


class TestEarlyReprobe:
    def test_an_answer_ends_a_failure_cooldown(self, temp_home, keychain):
        store = _macos_store(early_reprobe=True)
        _start_cooldown(store, keychain)

        store._kc_call(macos_keychain.get_password, "svc", "acct")

        assert store._use_keychain() is True
        assert store._keychain_disabled_until == 0.0

    @pytest.mark.parametrize("residual_cleared", [True, False])
    def test_an_answer_never_ends_a_pin(self, temp_home, keychain, residual_cleared):
        """A write that fell back to the file pinned file mode so a re-probe
        can never read a residual Keychain item over it."""
        store = _macos_store(early_reprobe=True)
        store._pin_file_mode(residual_cleared=residual_cleared)
        keychain.data[_active_item()] = _login("residual")

        store._kc_call(macos_keychain.get_password, "svc", "acct")
        store._read_active_credentials()

        assert store._use_keychain() is False
        assert _active_item() not in keychain.reads

    def test_the_active_read_asks_the_keychain_during_a_cooldown(
        self, temp_home, keychain
    ):
        store = _macos_store(early_reprobe=True)
        keychain.data[_active_item()] = _login("live")
        _start_cooldown(store, keychain)

        assert store._read_active_credentials() == ActiveCredentials(
            _login("live"), False, False
        )
        assert store._use_keychain() is True, "the answer ended the cooldown"

    def test_an_absent_active_item_still_reads_as_no_login(
        self, temp_home, keychain
    ):
        """The Keychain answered "not found", so the slot really is empty."""
        store = _macos_store(early_reprobe=True)
        _start_cooldown(store, keychain)

        assert store._read_active_credentials() == ActiveCredentials("", False, False)

    def test_a_failed_reprobe_reads_as_keychain_unavailable(
        self, temp_home, keychain
    ):
        store = _macos_store(early_reprobe=True)
        keychain.data[_active_item()] = _login("live")
        _start_cooldown(store, keychain)
        keychain.timeouts[_active_item()] = 2  # both bounded attempts

        assert store._read_active_credentials() == ActiveCredentials("", True, True)
        assert store._active_read_failed is True
        assert store._use_keychain() is False, "the failure re-armed the cooldown"

    def test_a_write_after_the_reprobe_stays_in_the_keychain(
        self, temp_home, keychain
    ):
        """Routing follows the Keychain that answered. Left in the cooldown, a
        refresh persisted in the same pass would go to the plaintext file and
        delete the item Claude Code reads first."""
        store = _macos_store(early_reprobe=True)
        keychain.data[_active_item()] = _login("old")
        _start_cooldown(store, keychain)
        assert store._read_active_credentials().value == _login("old")

        store._write_oauth_credentials(_login("new"))

        assert keychain.data[_active_item()] == _login("new")
        assert store._last_active_credentials_backend == "keychain"
        assert not get_credentials_path().exists()


class TestUpstreamRoutingIsKept:
    def test_without_it_the_cooldown_holds_and_the_active_read_waits(
        self, temp_home, keychain
    ):
        """Upstream's one-command guarantee: once an op failed, a later success
        does not route back to the Keychain, and the active read does not ask
        it until the cooldown lapses."""
        store = _macos_store(early_reprobe=False)
        keychain.data[_active_item()] = _login("live")
        _start_cooldown(store, keychain)

        store._kc_call(macos_keychain.get_password, "svc", "acct")
        store._read_active_credentials()

        assert store._use_keychain() is False
        assert _active_item() not in keychain.reads
