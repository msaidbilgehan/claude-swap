"""Stable Python API for embedding claude-swap in another program.

The CLI, TUI and menu bar are claude-swap's own frontends; this module is the
surface for everything else (it exists for claude-SAK, which drives claude-swap
from a long-running daemon). It wraps the internals those frontends use and
adds three promises the internals do not make:

* **Silent.** Nothing here writes to stdout or stderr. Human-oriented output
  the wrapped code produces is captured and sent to the ``claude-swap`` logger
  at DEBUG level instead.
* **Non-interactive.** Nothing here reads stdin. Ambiguous identifiers raise
  :class:`ConfigError` instead of prompting.
* **Honest errors.** claude-swap's own exceptions (:class:`ClaudeSwitchError`
  and its subclasses, re-exported here) propagate unchanged, so callers can
  translate them at their boundary.

Everything in ``__all__`` is covered by ``tests/test_public_api.py``; a change
to a name, a signature or a payload shape is a breaking change of the fork's
``sak`` releases. The JSON payloads keep the CLI's ``schemaVersion: 1``
contract (additive: consumers ignore unknown fields and event kinds).

Thread safety: output capture swaps the process-wide ``sys.stdout`` and
``sys.stderr`` for the duration of a call, serialised by a module lock. Callers
that print from other threads while a call runs may see that output captured.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, NamedTuple, cast

from claude_swap import macos_keychain, oauth
from claude_swap.autoswitch import (
    AllExhaustedEvent,
    AutoSwitcherInstance,
    AutoSwitchEngine,
    AutoSwitchEvent,
    ConfigWarningEvent,
    ErrorEvent,
    NoSwitchEvent,
    PollEvent,
    QuarantineEvent,
    SleepEvent,
    SwitchEvent,
    UnquarantineEvent,
    running_instances,
)
from claude_swap.claude_locks import (
    CONFIG_STALENESS_S,
    CREDENTIALS_STALENESS_S,
    STORAGE_WRITE_STALENESS_S,
    claude_config_lock,
    claude_credentials_lock,
    proper_lockfile,
    storage_write_lock_dir,
)
from claude_swap.credentials import (
    _ACTIVE_READ_ATTEMPTS,
    _ACTIVE_READ_RETRY_DELAY,
    ACCOUNT_CREDENTIAL_KEYS,
    CLAUDE_CODE_KEYCHAIN_SERVICE,
    CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE,
    CredentialStore,
    looks_like_api_key,
)
from claude_swap.exceptions import (
    AccountNotFoundError,
    ClaudeCodeLockTimeout,
    ClaudeSwitchError,
    ConfigError,
    CredentialError,
    CredentialReadError,
    CredentialWriteError,
    LockError,
    MigrationError,
    SessionError,
    SwitchError,
    ValidationError,
)
from claude_swap.fsutil import replace_with_retry, sync_file, sync_folder
from claude_swap.json_output import SCHEMA_VERSION
from claude_swap.locking import FileLock
from claude_swap.models import Platform, get_timestamp, normalize_alias
from claude_swap.oauth import (
    credential_fingerprint,
    is_setup_token,
    login_expires_at_iso,
)
from claude_swap.paths import (
    get_backup_root,
    get_claude_config_home,
    get_default_claude_config_home,
    get_default_global_config_path,
    get_legacy_backup_root,
    migrate_legacy_backup_dir,
)
from claude_swap.session import (
    AUTH_OVERRIDE_ENV_VARS,
    SessionManager,
    child_env,
    forget_config_dir_login,
    keychain_service_name,
    profile_is_quiescent,
    read_config_dir_credentials,
    session_identity_drifted,
)
from claude_swap.settings import AutoSwitchSettings, atomic_write_json, load_settings
from claude_swap.switcher import ClaudeAccountSwitcher

__all__ = [
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

_logger = logging.getLogger("claude-swap")
_output_lock = threading.RLock()


class AddedAccount(NamedTuple):
    """The account :func:`add_current_login` or :func:`add_profile_login`
    registered or refreshed.

    ``created`` is False when the login was already managed and its stored
    copy was only refreshed in place (``cswap add`` again).
    """

    slot: str
    email: str
    created: bool


class ProjectMapping(NamedTuple):
    """One directory mapping (``cswap map``): the normalised directory, the
    account's slot (``None`` when the account was removed) and its email."""

    path: str
    slot: str | None
    email: str


class ProjectAccount(NamedTuple):
    """The account a directory mapping (``cswap map``) resolves to.

    ``(None, None)``: no mapping covers the directory. ``(None, email)``: a
    mapping exists but its account was removed. ``(slot, email)``: resolved.
    """

    slot: str | None
    email: str | None


class SessionProfile(NamedTuple):
    """A ready session-mode profile (``cswap run``) for one account.

    Launch Claude Code with ``CLAUDE_CONFIG_DIR`` set to ``config_dir``
    exactly as given (the macOS Keychain item name is hashed from that string)
    and with every variable in ``strip_env`` removed from the environment,
    because each of them would override the profile's login.
    """

    config_dir: str
    slot: str
    email: str
    strip_env: tuple[str, ...]


class SharedGrant(NamedTuple):
    """A managed account whose stored login is the login of a Claude profile.

    Two copies of one OAuth grant drift apart when either side refreshes, so
    such an account must not also be used through claude-swap. Carries no
    token material.
    """

    slot: str
    email: str
    directory: str


class LoginSource(NamedTuple):
    """A Claude login that :func:`import_login` can move in (sak.7): a Claude
    Code credentials file, or a Claude profile folder. Carries no token
    material.
    """

    path: str
    """The file or the profile folder, as it is read (``~`` expanded)."""
    kind: str
    """``file`` or ``profile``."""
    email: str | None
    """The account, when the profile records it; a bare file does not."""
    slot: str | None
    """The managed account with that email, whose stored login the moved one
    replaces; ``None`` for a new account, or when the email is unknown."""
    expires_at: str | None
    """When the login itself lapses (ISO 8601 UTC), when Claude Code recorded
    it."""


class MovedLogin(NamedTuple):
    """What :func:`import_login` did."""

    account: AddedAccount
    """The account the moved login was registered as (``created`` False: an
    account already managed got it as its stored login)."""
    source: LoginSource
    default_login: bool
    """The login is the account Claude Code's default profile is logged in
    to: that account keeps following the live login, and the moved one was
    not stored."""
    source_removed: bool
    """The old copy (the file, or the profile's login) is gone."""
    source_error: str | None
    """Why the old copy could not be removed; it no longer works anyway."""


class CopiedLogin(NamedTuple):
    """What :func:`add_login_copy` did (sak.8)."""

    account: AddedAccount
    """The account the copy was registered as (``created`` False: an account
    already managed got it as its stored login)."""
    default_login: bool
    """The login is the account Claude Code's default profile is logged in
    to: that account keeps following the live login, and the copy was not
    stored."""
    renewed: bool
    """The copy's access token had lapsed and was renewed first: the copy it
    came from no longer works."""


class HandedOver(NamedTuple):
    """What :func:`hand_over_login` did (sak.11): the account whose login left
    claude-swap for a credentials file. Carries no token material."""

    slot: str
    """The slot the account had; it is free now."""
    email: str
    organization_uuid: str | None
    organization: str | None
    """The organization's name, when claude-swap recorded one."""
    alias: str | None
    expires_at: str | None
    """When the login itself lapses (ISO 8601 UTC), when Claude Code recorded
    it."""
    removal_error: str | None = None
    """``None`` when the account went with its login. Otherwise the login
    left (the file is its only copy here: deliver it all the same), but
    removing the rest of the account failed with this error: claude-swap
    may still list the account, without a login, until it is removed."""


class LoginNotAddedError(CredentialError):
    """Claude Code moved the login into ``config_dir`` (or a copy was renewed
    there), but it could not be registered: that profile holds the only
    working copy now. Fix the cause and import from ``config_dir``; do not
    discard it."""

    def __init__(self, message: str, config_dir: str) -> None:
        super().__init__(message)
        self.config_dir = config_dir


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Capture stdout/stderr written by wrapped code and log it at DEBUG."""
    with _output_lock:
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                yield
        finally:
            for stream, text in (("stdout", out.getvalue()), ("stderr", err.getvalue())):
                if text.strip():
                    _logger.debug("api: captured %s: %s", stream, text.strip())


def _refuse_foreign_profile() -> None:
    """Refuse to run against any profile but the default one.

    claude-swap's live store (the active credential and ``.claude.json``)
    follows ``CLAUDE_CONFIG_DIR``. A program started from a shell whose
    variable names a hand-made profile (``~/.claude-work``) or a session
    profile would otherwise read, back up and overwrite that profile's login
    as if it were the default one.
    """
    if not os.environ.get("CLAUDE_CONFIG_DIR"):
        return
    configured = get_claude_config_home()
    try:
        is_default = configured.resolve() == get_default_claude_config_home().resolve()
    except OSError:
        is_default = False
    if not is_default:
        raise ConfigError(
            f"CLAUDE_CONFIG_DIR points at {configured}, not the default profile "
            f"{get_default_claude_config_home()}; claude-swap's live store would "
            "follow it. Unset CLAUDE_CONFIG_DIR (env -u CLAUDE_CONFIG_DIR ...) "
            "and retry."
        )


def open_switcher() -> ClaudeAccountSwitcher:
    """Construct a switcher for the default profile.

    Runs claude-swap's one-time data migrations like the CLI does. Raises
    :class:`ConfigError` when ``CLAUDE_CONFIG_DIR`` names any profile but the
    default one.

    The switcher serves a long-running process, so its credential store
    re-probes the macOS Keychain early (``CredentialStore.early_reprobe``):
    another item's ``security`` timeout no longer hides the active login for
    the next minute.
    """
    _refuse_foreign_profile()
    with _quiet():
        switcher = ClaudeAccountSwitcher()
    switcher._store.early_reprobe = True
    return switcher


def accounts_json(
    switcher: ClaudeAccountSwitcher, fetch: Collection[str] | None = frozenset()
) -> dict[str, Any]:
    """The ``cswap list --json`` payload (``schemaVersion`` 1).

    ``fetch`` names the account numbers whose usage may be fetched from the
    network in this call; the default (empty) serves usage from the cache
    only. ``None`` refreshes what is due, as ``cswap list`` does: every
    account whose persisted poll plan is due and whose reading is older than
    the serve TTL is fetched (failure backoff, 429 handling and claims
    against concurrent collectors apply, and the adapted plans are saved);
    the rest is served from the cache. The auto-switch engine never polls a
    disabled account, so this is what keeps one's usage fresh. With no
    accounts managed yet the payload has an empty list.
    """
    wanted = None if fetch is None else set(fetch)
    with _quiet():
        payload = switcher.list_accounts(json_output=True, fetch=wanted)
    if payload is None:  # pragma: no cover - json_output never returns None
        raise ConfigError("claude-swap returned no account list")
    return payload


def project_account(
    switcher: ClaudeAccountSwitcher, path: str | os.PathLike[str]
) -> ProjectAccount:
    """The account ``cswap map`` assigned to ``path`` (longest parent wins)."""
    with _quiet():
        slot, email = switcher.slot_for_directory(Path(path))
    return ProjectAccount(slot=slot, email=email)


def active_account(switcher: ClaudeAccountSwitcher) -> str | None:
    """Slot number of the live default login, or ``None`` when it is unmanaged
    (an access token another computer lent counts as unmanaged, sak.13)."""
    with _quiet():
        return switcher.current_account_number()


def switch_to(switcher: ClaudeAccountSwitcher, identifier: str) -> dict[str, Any]:
    """Switch the default login to ``identifier`` (slot, email or alias).

    Returns the ``cswap switch --json`` payload. An email that matches several
    accounts raises :class:`ConfigError` instead of prompting.
    """
    with _quiet():
        payload = switcher.switch_to(identifier, json_output=True)
    if payload is None:  # pragma: no cover - json_output never returns None
        raise SwitchError(f"claude-swap returned no switch result for {identifier}")
    return payload


def add_current_login(
    switcher: ClaudeAccountSwitcher, *, alias: str | None = None
) -> AddedAccount:
    """Register the default profile's live login (the silent ``cswap add``).

    The login is whatever Claude Code is logged in to in the default profile
    (``claude``, then ``/login``, with ``CLAUDE_CONFIG_DIR`` unset). A login
    that is already managed is refreshed in its slot; a new one takes the
    next free slot. ``alias`` sets the account's alias (unique). Raises
    :class:`ConfigError` without a live login or from a foreign profile and
    :class:`ValidationError` for a bad or taken alias, a credential that
    belongs to another account, or an access token alone that another
    computer lent (sak.13: it cannot be renewed here; log in on this computer
    first). Never prompts.
    """
    _refuse_foreign_profile()
    with _quiet():
        before = set((switcher._get_sequence_data() or {}).get("accounts", {}))
        switcher.add_account(slot=None, assume_yes=True, alias=alias)
        slot = switcher.current_account_number()
        if slot is None:  # pragma: no cover - add_account just stored it
            raise ConfigError("the live login was not registered")
        email = switcher.account_email(slot)
    return AddedAccount(slot=slot, email=email, created=slot not in before)


def add_profile_login(
    switcher: ClaudeAccountSwitcher,
    config_dir: str | os.PathLike[str],
    *,
    alias: str | None = None,
) -> AddedAccount:
    """Register the login of a separate Claude profile; the default profile
    is never read or written.

    For adding an account without touching the default login: log in with
    ``CLAUDE_CONFIG_DIR=<config_dir> claude auth login`` (an empty
    directory), then pass the same string here. A new account takes the next
    free slot with ``alias``; an account already managed gets this login as
    its stored login and keeps its alias; the account the default profile is
    logged in to keeps following the live login (nothing changes, or the
    live login is registered when it is not managed yet). The recorded active
    account does not change. The profile keeps its login until
    :func:`forget_profile_login`.

    Raises :class:`ConfigError` when the profile has no login or its
    credential belongs to another account, :class:`CredentialError` when the
    credential is missing or unreadable, or when the default profile is
    logged in to that account and its login cannot be read cleanly now
    (sak.13: whether to follow it cannot be told, so nothing is stored), and
    :class:`ValidationError` for a bad or taken alias or an API-key login.
    Never prompts.
    """
    _refuse_foreign_profile()
    with _quiet():
        slot, created = switcher.add_account_from_profile(
            os.fspath(config_dir), alias=alias
        )
        email = switcher.account_email(slot)
    return AddedAccount(slot=slot, email=email, created=created)


def forget_profile_login(config_dir: str | os.PathLike[str]) -> None:
    """Remove the login a separate Claude profile keeps (its keychain entry
    on macOS and its ``.credentials.json``), once :func:`add_profile_login`
    stored it or it is abandoned.

    ``config_dir`` is the exact string the login ran with: the keychain
    entry is named after it. Removing a login that is already gone is fine.
    Raises :class:`CredentialError` when it stays.
    """
    with _quiet():
        forget_config_dir_login(os.fspath(config_dir))


def remove_account(switcher: ClaudeAccountSwitcher, identifier: str) -> tuple[str, str]:
    """Remove a managed account (the silent ``cswap remove``) and its mappings.

    ``identifier`` is a slot, alias or email; an email that matches several
    accounts raises :class:`ConfigError` instead of prompting. An account
    with a live ``cswap run`` session is refused (:class:`SessionError`).
    Removing the active account leaves the default profile logged in; it is
    only no longer managed. Returns ``(slot, email)`` of the removed account.
    """
    _refuse_foreign_profile()
    with _quiet():
        slot, email, _org = switcher.resolve_account(identifier)
        switcher.remove_account(slot, assume_yes=True)
    return slot, email


def set_alias(
    switcher: ClaudeAccountSwitcher, identifier: str, alias: str | None
) -> str | None:
    """Set, rename or (``alias=None``) clear an account's alias.

    Returns the stored (normalised) alias, or ``None`` after clearing.
    Raises :class:`ValidationError` for an invalid alias and
    :class:`ConfigError` for one another account uses.
    """
    with _quiet():
        if alias is None:
            switcher.unset_alias(identifier)
            return None
        _num, stored = switcher.set_alias(identifier, alias)
    return stored


def map_project(
    switcher: ClaudeAccountSwitcher,
    path: str | os.PathLike[str],
    identifier: str,
) -> ProjectAccount:
    """Map a directory to an account (the silent ``cswap map``).

    The directory and everything below it then resolves to that account
    (:func:`project_account`); a mapping on the same directory is replaced.
    """
    from claude_swap.mappings import MappingStore

    with _quiet():
        slot, email, org_uuid = switcher.resolve_account(identifier)
        MappingStore(switcher.backup_dir).set(Path(path), email, org_uuid)
    return ProjectAccount(slot=slot, email=email)


def unmap_project(switcher: ClaudeAccountSwitcher, path: str | os.PathLike[str]) -> bool:
    """Remove the mapping of exactly this directory; returns whether one existed."""
    from claude_swap.mappings import MappingStore

    with _quiet():
        return MappingStore(switcher.backup_dir).remove(Path(path))


def project_mappings(switcher: ClaudeAccountSwitcher) -> list[ProjectMapping]:
    """Every directory mapping, sorted by directory."""
    from claude_swap.mappings import MappingStore

    with _quiet():
        stored = MappingStore(switcher.backup_dir).all()
        data = switcher._get_sequence_data() or {}
        mappings: list[ProjectMapping] = []
        for path, record in sorted(stored.items()):
            email = str(record.get("email", ""))
            org = str(record.get("organizationUuid", "") or "")
            slot = switcher._find_account_slot(data, email, org) if data else None
            mappings.append(ProjectMapping(path=path, slot=slot, email=email))
    return mappings


def set_rotation(
    switcher: ClaudeAccountSwitcher, identifier: str, enabled: bool
) -> None:
    """Put an account in (``enabled=True``) or out of automatic rotation.

    The silent equivalent of ``cswap enable`` / ``cswap disable``. A disabled
    account stays managed and remains an explicit switch target.
    """
    with _quiet():
        switcher.set_account_disabled(identifier, not enabled)


def live_session_accounts(switcher: ClaudeAccountSwitcher) -> dict[str, tuple[int, ...]]:
    """Slots with live ``cswap run`` session-mode Claude processes, and their PIDs.

    Only slots with at least one live process appear. The auto-switch engine
    never switches the default login to such an account.
    """
    with _quiet():
        data = switcher._get_sequence_data() or {}
        live: dict[str, tuple[int, ...]] = {}
        for num, record in sorted(data.get("accounts", {}).items()):
            pids = switcher.live_session_pids_for(str(num), record.get("email", ""))
            if pids:
                live[str(num)] = tuple(sorted(pids))
    return live


def create_engine(
    switcher: ClaudeAccountSwitcher,
    on_event: Callable[[AutoSwitchEvent], None],
    *,
    dry_run: bool,
) -> AutoSwitchEngine:
    """An auto-switch engine with the user's saved settings (``cswap config``).

    The engine is single-use: run :meth:`AutoSwitchEngine.run_loop` on a
    dedicated thread and call :meth:`AutoSwitchEngine.stop` to end it.
    ``on_event`` runs on that thread; an exception it raises ends the loop.
    ``dry_run`` evaluates and polls but never switches or writes state.
    """
    with _quiet():
        settings = load_settings(switcher.backup_dir)
        return AutoSwitchEngine(switcher, settings, on_event, dry_run=dry_run)


def version() -> str:
    """The installed claude-swap distribution version (e.g. ``0.27.0b1+sak.1``)."""
    return _dist_version("claude-swap")


def inject_system_trust() -> bool:
    """Route TLS verification through the OS trust store, as the CLI does.

    Returns whether the injection happened; failure leaves the stdlib ``ssl``
    defaults in place.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 - best-effort, mirrors cli._use_native_tls
        _logger.debug("api: truststore injection failed", exc_info=True)
        return False
    return True


def _checked_env(extra_env: Mapping[str, str] | None) -> dict[str, str]:
    """``extra_env`` (sak.11) as a child's environment can hold it; refuses
    names and values it cannot (the values are never repeated)."""
    checked: dict[str, str] = {}
    for name, value in (extra_env or {}).items():
        if not isinstance(name, str) or not name or "=" in name or "\0" in name:
            raise ValidationError(f"extra_env: {name!r} is not a variable name.")
        if not isinstance(value, str) or "\0" in value:
            raise ValidationError(
                f"extra_env: the value of {name} is not text a variable can hold."
            )
        checked[name] = value
    return checked


def prepare_session_profile(
    switcher: ClaudeAccountSwitcher,
    identifier: str,
    *,
    share_history: bool = True,
    extra_env: Mapping[str, str] | None = None,
) -> SessionProfile:
    """Create or refresh the session-mode profile of ``identifier``.

    Adds the guard ``cswap run`` applies and ``SessionManager.setup_session``
    alone lacks: the account that is the active default login is refused
    (:class:`SessionError`), because a second copy of a live credential drifts
    when either copy refreshes. API-key accounts are refused too.
    ``share_history`` shares conversation history with the default profile
    (POSIX only). Settings and MCP servers are always shared, as with
    ``cswap run``. May run ``claude auth status`` and refresh a token.

    ``extra_env`` (sak.11): variables added to the environment of that
    ``claude auth status`` (an embedding program's privacy settings, for
    example). claude-swap's own variables win over them: the profile's
    ``CLAUDE_CONFIG_DIR``, and the login overrides it strips
    (``strip_env``), which are never passed on. A name or value an
    environment cannot hold raises :class:`ValidationError`.
    """
    _refuse_foreign_profile()
    if share_history and switcher.platform == Platform.WINDOWS:
        raise SessionError("share_history is not supported on Windows")
    extra = _checked_env(extra_env)
    with _quiet():
        account_num, email, org_uuid = switcher.resolve_account(identifier)
        # A login another computer lent (sak.13) is no copy of this
        # account's stored login: the profile may hold that one.
        current = switcher._own_live_identity()
        if current is not None and current == (email, org_uuid):
            raise SessionError(
                f"Account-{account_num} ({email}) is the active default login; "
                "a session profile would hold a second copy of its live "
                "credential. Switch the default login to another account first."
            )
        session_dir, slot, slot_email = SessionManager(
            switcher, extra_env=extra
        ).setup_session(identifier, True, share_history)
    return SessionProfile(
        config_dir=str(session_dir),
        slot=str(slot),
        email=slot_email,
        strip_env=tuple(AUTH_OVERRIDE_ENV_VARS),
    )


def running_autoswitchers(
    switcher: ClaudeAccountSwitcher,
) -> list[AutoSwitcherInstance]:
    """Auto-switch engines running against this account store, oldest first.

    Every engine records itself while its loop runs (``cswap auto``, the menu
    bar, an embedding program). Records of processes that are gone are pruned.
    """
    with _quiet():
        return running_instances(switcher.backup_dir)


def _profile_keychain_override(directory: str) -> str | None:
    """The unsuffixed Keychain item for the default profile, else ``None``."""
    try:
        if Path(directory).resolve() == get_default_claude_config_home().resolve():
            return CLAUDE_CODE_KEYCHAIN_SERVICE
    except OSError:
        return None
    return None


def shared_grants(
    switcher: ClaudeAccountSwitcher,
    profile_dirs: Collection[str | os.PathLike[str]],
) -> list[SharedGrant]:
    """Managed accounts whose stored login is the login of one of ``profile_dirs``.

    Each directory is read the way Claude Code reads a ``CLAUDE_CONFIG_DIR``
    profile: on macOS the Keychain item named from the directory string (the
    unsuffixed item for the default profile), elsewhere its
    ``.credentials.json``. Logins compare by credential fingerprint (the
    refresh-token hash), so an access-token rotation still matches. Missing or
    logged-out directories match nothing. Results carry no token material.
    """
    with _quiet():
        logins: list[tuple[str, str]] = []
        for raw in profile_dirs:
            directory = os.path.expanduser(os.fspath(raw))
            creds = read_config_dir_credentials(
                directory, keychain_service=_profile_keychain_override(directory)
            )
            fingerprint = credential_fingerprint(creds or "")
            if fingerprint is not None:
                logins.append((directory, fingerprint))
        if not logins:
            return []
        data = switcher._get_sequence_data() or {}
        grants: list[SharedGrant] = []
        for num, record in sorted(
            data.get("accounts", {}).items(), key=lambda item: int(item[0])
        ):
            email = record.get("email", "")
            stored = credential_fingerprint(
                switcher.read_account_credentials(str(num), email)
            )
            if stored is None:
                continue
            for directory, fingerprint in logins:
                if fingerprint == stored:
                    grants.append(
                        SharedGrant(slot=str(num), email=email, directory=directory)
                    )
    return grants


# -- moving a login in, adding a setup-token (sak.7) --------------------------------

LOGIN_MOVE_TIMEOUT = 120.0
"""How long Claude Code may take to turn a refresh token into a new login."""

SETUP_TOKEN_PREFIX = "sk-ant-oat"
"""How a token ``claude setup-token`` prints starts."""

_MAX_CREDENTIALS_BYTES = 1 << 20
"""A credentials file is a few hundred bytes; anything this big is not one."""

_MOVE_ENV_REMOVED = frozenset(
    {
        *AUTH_OVERRIDE_ENV_VARS,
        # set below: the target profile, and the login to move
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
        "CLAUDE_CODE_OAUTH_SCOPES",
        # would keep the new login in another profile's secure store
        "CLAUDE_SECURESTORAGE_CONFIG_DIR",
        # Claude Code's nested-session markers
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_SSE_PORT",
        "CLAUDE_CODE_SESSION_ID",
    }
)
"""Variables the ``claude auth login`` that moves a login must not inherit."""


@dataclass(frozen=True)
class _Found:
    """A login :func:`_find_login` read: what may be shown, and what only
    the move itself may use."""

    info: LoginSource
    refresh_token: str
    scopes: tuple[str, ...]
    fingerprint: str
    config_dir: str | None
    """For a profile: the exact ``CLAUDE_CONFIG_DIR`` string its login is
    kept under (macOS names the Keychain item after it)."""


def _oauth_entry(credentials: str) -> dict[str, Any] | None:
    try:
        data = json.loads(credentials)
    except (TypeError, ValueError):
        return None
    entry = data.get("claudeAiOauth") if isinstance(data, dict) else None
    return entry if isinstance(entry, dict) else None


def _identity(path: Path) -> tuple[int, int] | None:
    """What ``path`` names, as the file system tells files apart (device and
    inode, after symlinks), or ``None`` when it cannot be looked at."""
    try:
        status = path.stat()
    except (OSError, ValueError):
        return None
    return status.st_dev, status.st_ino


def _anchor(path: Path) -> tuple[Path, tuple[int, int] | None, tuple[str, ...]]:
    """``path``'s nearest existing folder (itself when it exists), that
    folder's identity, and the names below it that do not exist yet."""
    try:
        full = path.resolve()
    except (OSError, RuntimeError):  # a symlink loop
        full = Path(os.path.abspath(path))
    below: list[str] = []
    current = full
    while True:
        identity = _identity(current)
        if identity is not None or current.parent == current:
            return current, identity, tuple(reversed(below))
        below.append(current.name)
        current = current.parent


def _folded(names: Sequence[str]) -> tuple[str, ...]:
    """Names as a case-insensitive disk compares them."""
    return tuple(unicodedata.normalize("NFC", name).casefold() for name in names)


def _inside(path: Path, root: Path) -> bool:
    """Whether ``path`` is ``root`` or lies inside it, as the file system
    sees it: by file identity, so another spelling of the same folder counts
    too (letter case on a case-insensitive disk, macOS's
    ``/System/Volumes/Data`` firmlink, a symlink, ``..``, a trailing slash).
    A path or root that does not exist yet compares through its nearest
    existing folder and the names below it (letter case ignored)."""
    place, place_id, below = _anchor(path)
    _, root_id, root_below = _anchor(root)
    if place_id is None or root_id is None:
        return False
    if not root_below:
        return any(_identity(folder) == root_id for folder in (place, *place.parents))
    return place_id == root_id and _folded(below[: len(root_below)]) == _folded(
        root_below
    )


def _refuse_own_store(switcher: ClaudeAccountSwitcher, path: Path) -> None:
    """The default profile and claude-swap's own store hold logins that are
    managed already; moving one would log out the login in use."""
    if _inside(path, get_default_claude_config_home()):
        raise ConfigError(
            f"{path} is Claude Code's default profile; its login is registered "
            "without moving it (add the current login instead)"
        )
    if _inside(path, Path(switcher.backup_dir)):
        raise ConfigError(
            f"{path} belongs to claude-swap's own store; its logins are managed "
            "already"
        )


def _profile_identity(directory: Path) -> tuple[str, str] | None:
    """``(email, organization_uuid)`` the profile's ``.claude.json`` records,
    as Claude Code resolves it; ``None`` when it records none."""
    legacy = directory / ".config.json"
    path = legacy if legacy.exists() else directory / ".claude.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    account = data.get("oauthAccount") if isinstance(data, dict) else None
    if not isinstance(account, dict) or not account.get("emailAddress"):
        return None
    return str(account["emailAddress"]), str(account.get("organizationUuid") or "")


def _profile_credentials(raw: str) -> tuple[str, str]:
    """The login a profile folder keeps and the exact string it is kept
    under: the folder as given, else without a trailing slash, else its
    absolute path (Claude Code names the macOS Keychain item after the
    ``CLAUDE_CONFIG_DIR`` it ran with)."""
    candidates = list(dict.fromkeys([raw, raw.rstrip("/") or "/", os.path.abspath(raw)]))
    for candidate in candidates:
        creds = read_config_dir_credentials(candidate, strict_keychain=True)
        if creds:
            return creds, candidate
    raise CredentialReadError(f"No Claude login found in the profile {raw}.")


def _read_file_credentials(path: Path) -> str:
    try:
        if path.stat().st_size > _MAX_CREDENTIALS_BYTES:
            raise ValidationError(f"{path} is too big to be a credentials file.")
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        raise CredentialReadError(f"Cannot read {path}: {e}") from e


def _movable(
    credentials: str, where: str, verb: str = "move"
) -> tuple[str, tuple[str, ...], str | None]:
    """The refresh token, scopes and login expiry of a login Claude Code can
    ``verb`` (move, or use as a copy); refuses anything else."""
    if looks_like_api_key(credentials):
        raise ValidationError(f"{where} holds an API key, not a Claude login.")
    entry = _oauth_entry(credentials)
    if entry is None:
        raise ValidationError(
            f"{where} holds no Claude login (no claudeAiOauth entry)."
        )
    refresh = entry.get("refreshToken")
    if not isinstance(refresh, str) or not refresh.strip():
        if is_setup_token(credentials):
            raise ValidationError(
                f"{where} holds a setup-token, not a login: register the token "
                "itself instead."
            )
        raise ValidationError(
            f"{where} holds no refresh token: the login was logged out or emptied."
        )
    expires = entry.get("refreshTokenExpiresAt")
    if (
        isinstance(expires, (int, float))
        and not isinstance(expires, bool)
        and 0 < expires <= time.time() * 1000
    ):
        raise ValidationError(
            f"{where} holds a login that expired on "
            f"{login_expires_at_iso(credentials)}; log in again instead."
        )
    scopes = entry.get("scopes")
    if (
        not isinstance(scopes, list)
        or not scopes
        or not all(isinstance(s, str) and s and not s.isspace() for s in scopes)
    ):
        raise ValidationError(
            f"{where} does not record the login's scopes, which Claude Code needs "
            f"to {verb} it."
        )
    return refresh.strip(), tuple(scopes), login_expires_at_iso(credentials)


def _refuse_managed_grant(
    switcher: ClaudeAccountSwitcher, fingerprint: str, where: str
) -> None:
    """The same login (one refresh-token lineage) must not be moved away from
    where claude-swap uses it."""
    default = str(get_default_claude_config_home())
    live = read_config_dir_credentials(
        default, keychain_service=_profile_keychain_override(default)
    )
    if live and credential_fingerprint(live) == fingerprint:
        raise ConfigError(
            f"{where} holds the login Claude Code's default profile uses now; it "
            "is managed already."
        )
    data = switcher._get_sequence_data() or {}
    for num, record in sorted(
        data.get("accounts", {}).items(), key=lambda item: int(item[0])
    ):
        email = record.get("email", "")
        stored = switcher.read_account_credentials(str(num), email)
        if stored and credential_fingerprint(stored) == fingerprint:
            raise ConfigError(
                f"{where} holds the login of account {num} ({email}), which is "
                "managed already."
            )


def _find_login(
    switcher: ClaudeAccountSwitcher, source: str | os.PathLike[str]
) -> _Found:
    raw = os.path.expanduser(os.fspath(source)).strip()
    if not raw:
        raise ConfigError("Name a credentials file or a Claude profile folder.")
    path = Path(raw)
    if not path.exists():
        raise ConfigError(f"{raw} does not exist.")
    _refuse_own_store(switcher, path)
    identity: tuple[str, str] | None = None
    config_dir: str | None = None
    if path.is_dir():
        kind = "profile"
        credentials, config_dir = _profile_credentials(raw)
        identity = _profile_identity(path)
    elif path.is_file():
        kind = "file"
        credentials = _read_file_credentials(path)
    else:
        raise ConfigError(f"{raw} is neither a file nor a folder.")
    refresh, scopes, expires = _movable(credentials, raw)
    fingerprint = credential_fingerprint(credentials) or ""
    _refuse_managed_grant(switcher, fingerprint, raw)
    slot: str | None = None
    email: str | None = None
    if identity is not None:
        email, org_uuid = identity
        # A login another computer lent (sak.13) is not this account's
        # login here: one moved in is stored for it.
        current = switcher._own_live_identity()
        data = switcher._get_sequence_data() or {}
        slot = switcher._find_account_slot(data, email, org_uuid) if data else None
        if current == identity:
            managed = f" (account {slot})" if slot else ""
            raise ConfigError(
                f"{raw} is logged in to {email}, the account Claude Code's default "
                f"profile uses{managed}; there is nothing to move."
            )
    return _Found(
        info=LoginSource(
            path=raw, kind=kind, email=email, slot=slot, expires_at=expires
        ),
        refresh_token=refresh,
        scopes=scopes,
        fingerprint=fingerprint,
        config_dir=config_dir,
    )


def check_login_source(
    switcher: ClaudeAccountSwitcher, source: str | os.PathLike[str]
) -> LoginSource:
    """What :func:`import_login` would move from ``source``, refusing what it
    would refuse; nothing changes.

    ``source`` is a Claude Code credentials file (``.credentials.json``) or a
    Claude profile folder (a ``CLAUDE_CONFIG_DIR``; on macOS its login is read
    from the Keychain item named after the folder). Raises
    :class:`ConfigError` for a missing source, the default profile,
    claude-swap's own store, or a login that is managed already;
    :class:`CredentialError` when no login can be read there (a locked
    Keychain included); :class:`ValidationError` for an API key, a
    setup-token, a logged-out or expired login, or one without its scopes.
    Never prompts; reads no network.
    """
    _refuse_foreign_profile()
    with _quiet():
        return _find_login(switcher, source).info


def _free_alias(
    switcher: ClaudeAccountSwitcher, alias: str | None, slot: str | None
) -> str | None:
    """``alias`` normalised, when no account but ``slot`` uses it."""
    if alias is None:
        return None
    try:
        normalized = normalize_alias(alias)
    except ValueError as e:
        raise ValidationError(str(e)) from e
    owner = switcher._alias_in_use(normalized, exclude_num=slot)
    if owner is not None:
        raise ValidationError(f"Alias '{normalized}' is already used by account {owner}")
    return normalized


def _empty_profile(
    switcher: ClaudeAccountSwitcher,
    config_dir: str | os.PathLike[str],
    source: str | None,
    verb: str = "move",
) -> str:
    target = os.fspath(config_dir)
    path = Path(target)
    if not path.is_dir():
        raise ConfigError(f"The profile to {verb} the login into, {target}, is missing.")
    _refuse_own_store(switcher, path)
    if source is not None and (_inside(path, Path(source)) or _inside(Path(source), path)):
        raise ConfigError("The login cannot go into the folder it comes from.")
    if read_config_dir_credentials(target) or _profile_identity(path) is not None:
        raise ConfigError(f"The profile {target} holds a login already.")
    return target


def _moved_login(target: str, found: _Found) -> bool:
    """Whether the target profile holds a login of its own now."""
    try:
        creds = read_config_dir_credentials(target, strict_keychain=True)
    except CredentialError:
        return False
    fingerprint = credential_fingerprint(creds or "")
    return fingerprint is not None and fingerprint != found.fingerprint


def _summary(output: str, secret: str) -> str:
    """The last line Claude Code printed, never with the token in it."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    text = lines[-1] if lines else ""
    return text.replace(secret, "<token>")[:300]


def _exchange(
    found: _Found,
    target: str,
    claude: str | os.PathLike[str] | None,
    extra_env: Mapping[str, str],
) -> None:
    """Let Claude Code turn the refresh token into a new login in ``target``
    (``claude auth login`` with ``CLAUDE_CODE_OAUTH_REFRESH_TOKEN``: no
    browser). Once it has, the source's copy no longer works. ``extra_env``
    is added to its environment; the variables set or removed here win."""
    executable = os.fspath(claude) if claude is not None else shutil.which("claude")
    if not executable:
        raise SessionError("'claude' was not found on PATH. Install Claude Code first.")
    env = child_env(_MOVE_ENV_REMOVED, extra_env)
    env["CLAUDE_CONFIG_DIR"] = target
    env["CLAUDE_CODE_OAUTH_REFRESH_TOKEN"] = found.refresh_token
    env["CLAUDE_CODE_OAUTH_SCOPES"] = " ".join(found.scopes)
    failure: str | None = None
    try:
        done = subprocess.run(
            [executable, "auth", "login"],
            env=env,
            cwd=target,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=LOGIN_MOVE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        failure = f"it did not finish within {LOGIN_MOVE_TIMEOUT:.0f} s"
    except OSError as e:
        raise SessionError(f"Could not run {executable}: {e}") from None
    else:
        if done.returncode != 0:
            failure = (
                _summary(done.stdout + "\n" + done.stderr, found.refresh_token)
                or f"exit status {done.returncode}"
            )
    # A login Claude Code stored before it failed or hung is the only working
    # copy now: register it rather than leave it behind.
    if failure is not None and not _moved_login(target, found):
        raise CredentialError(f"Claude Code could not move the login: {failure}")
    if failure is not None:
        _logger.warning("api: claude auth login failed after storing the login: %s", failure)


def _remove_source(found: _Found) -> tuple[bool, str | None]:
    """Remove the old copy, which no longer works: the file, or the login the
    profile keeps (its Keychain item and ``.credentials.json``; the folder
    stays)."""
    try:
        if found.config_dir is not None:
            forget_config_dir_login(found.config_dir)
        else:
            Path(found.info.path).unlink(missing_ok=True)
    except (ClaudeSwitchError, OSError) as e:
        return False, str(e)
    return True, None


def import_login(
    switcher: ClaudeAccountSwitcher,
    source: str | os.PathLike[str],
    config_dir: str | os.PathLike[str],
    *,
    alias: str | None = None,
    claude: str | os.PathLike[str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> MovedLogin:
    """Move a Claude login in from ``source`` (see :func:`check_login_source`)
    without copying it: the one-way move.

    Claude Code itself turns the login's refresh token into a new login in
    ``config_dir``, an empty, temporary profile the caller owns
    (``CLAUDE_CONFIG_DIR=<config_dir> claude auth login`` with
    ``CLAUDE_CODE_OAUTH_REFRESH_TOKEN`` and ``CLAUDE_CODE_OAUTH_SCOPES``, as
    its documentation describes for automated setups; no browser) and records
    the account's email and organisation, which a credentials file lacks. The
    new login is registered as :func:`add_profile_login` registers one (a new
    account takes the next free slot with ``alias``; an account already
    managed gets it as its stored login), and the old copy at ``source`` is
    removed: it no longer works, since each refresh token works once. Never
    log out at the source: that can revoke the login on the server.

    ``config_dir`` keeps the new login until :func:`forget_profile_login`.
    ``claude`` is the Claude Code executable (default: ``claude`` on PATH).
    ``extra_env`` (sak.11): variables added to the environment of that
    ``claude auth login`` (an embedding program's privacy settings, for
    example). claude-swap's own variables win over them:
    ``CLAUDE_CONFIG_DIR``, ``CLAUDE_CODE_OAUTH_REFRESH_TOKEN`` and
    ``CLAUDE_CODE_OAUTH_SCOPES`` are set by the move, and the variables it
    keeps from Claude Code (the login overrides,
    ``CLAUDE_SECURESTORAGE_CONFIG_DIR``, the nested-session markers) are
    dropped from ``extra_env`` too.

    Raises what :func:`check_login_source` raises, before anything moves, and
    :class:`ValidationError` for a bad alias or one another account uses, or
    an ``extra_env`` name or value an environment cannot hold;
    :class:`SessionError` when Claude Code cannot run; :class:`CredentialError`
    when it could not move the login (nothing moved then). After the move,
    a failed registration raises :class:`LoginNotAddedError`: ``config_dir``
    holds the only working copy (also when the default profile is logged in
    to the moved login's account and that login cannot be read cleanly now,
    sak.13). The token is never printed or logged.
    """
    _refuse_foreign_profile()
    extra = _checked_env(extra_env)
    with _quiet():
        found = _find_login(switcher, source)
        alias = _free_alias(switcher, alias, found.info.slot)
        target = _empty_profile(switcher, config_dir, found.info.path)
    # Outside the output lock: Claude Code may take a while, and every other
    # call of this module waits for that lock.
    _exchange(found, target, claude, extra)
    with _quiet():
        try:
            identity = _profile_identity(Path(target))
            default_login = identity is not None and switcher._live_login_followed_for(
                *identity
            )
            slot, created = switcher.add_account_from_profile(target, alias=alias)
            email = switcher.account_email(slot)
        except ClaudeSwitchError as e:
            raise LoginNotAddedError(
                f"The login moved into {target}, but it could not be registered: {e}",
                target,
            ) from e
        removed, error = _remove_source(found)
    return MovedLogin(
        account=AddedAccount(slot=slot, email=email, created=created),
        source=found.info,
        default_login=default_login,
        source_removed=removed,
        source_error=error,
    )


def _setup_token_slot(switcher: ClaudeAccountSwitcher, email: str) -> str | None:
    """The setup-token account registered under ``email``, which a new token
    replaces; refuses an email a login (or an API key) is registered under,
    whose stored login the token would replace or share an email with."""
    data = switcher._get_sequence_data() or {}
    for num, record in sorted(
        data.get("accounts", {}).items(), key=lambda item: int(item[0])
    ):
        if record.get("email") != email:
            continue
        stored = switcher.read_account_credentials(str(num), email)
        if record.get("organizationUuid") or not is_setup_token(stored):
            raise ValidationError(
                f"{email} is the email of account {num}, which is not a "
                "setup-token; give the setup-token another email, or none."
            )
        return str(num)
    return None


def add_setup_token(
    switcher: ClaudeAccountSwitcher,
    token: str,
    *,
    email: str | None = None,
    alias: str | None = None,
    account: str | None = None,
) -> AddedAccount:
    """Register a setup-token (``claude setup-token``) as an account: the
    silent ``cswap add-token`` for setup-tokens only.

    A setup-token lasts a year and never renews itself, so copies of it do
    not invalidate each other; it can only run Claude: its usage cannot be
    read, and Remote Control and claude.ai connectors do not work with it.
    ``email`` labels the account (default ``setup-token-<slot>@token.local``):
    a setup-token already registered under that email gets the new token and
    keeps its alias; otherwise a new account takes the next free slot with
    ``alias``. When the token was stored is recorded (``setupTokenAddedAt``
    in :func:`accounts_json`).

    ``account`` (sak.9) names the setup-token account (slot, alias or email)
    that gets the new token in place of its own instead; it keeps its email
    and alias, and the new token counts from now. ``email`` and ``alias`` do
    not go with it. An account that runs on a login (or an API key) is
    refused, as a setup-token cannot show whose it is, and so is the token
    the account has already.

    When Claude Code's default profile runs on the account that gets a new
    token (either way), the new token goes into the default profile too
    (sak.10); if it cannot, the account keeps its old token and the error is
    raised.

    Raises :class:`ValidationError` for an empty value, an API key, anything
    that is not a setup-token, a bad email, a bad or taken alias, or an
    ``account`` that cannot take it; :class:`AccountNotFoundError` for an
    unknown ``account`` and :class:`CredentialReadError` when its stored
    login cannot be read now; a switch's errors (:class:`SwitchError`, …)
    when the default profile cannot take the new token. Never prompts and
    never prints the token.
    """
    _refuse_foreign_profile()
    value = token.strip()
    if not value:
        raise ValidationError("The setup-token is empty.")
    if looks_like_api_key(value):
        raise ValidationError(
            "This is an API key, not a setup-token; only a setup-token made with "
            "`claude setup-token` can be added this way."
        )
    if not value.startswith(SETUP_TOKEN_PREFIX) or any(c.isspace() for c in value):
        raise ValidationError(
            "This is not a setup-token: `claude setup-token` prints one that "
            f"starts with {SETUP_TOKEN_PREFIX}."
        )
    if account is not None:
        if email is not None or alias is not None:
            raise ConfigError(
                "Name the account that gets the new setup-token, or give an email "
                "and an alias for a new one; not both."
            )
        with _quiet():
            return _replace_setup_token(switcher, account, value)
    with _quiet():
        before = set((switcher._get_sequence_data() or {}).get("accounts", {}))
        label = email.strip() if email else None
        existing = _setup_token_slot(switcher, label) if label else None
        normalized = _free_alias(switcher, alias, existing)
        live = _live_setup_token(switcher, existing, label)
        switcher.add_account_from_token(value, email=label, assume_yes=True)
        if live is not None:
            _activate_new_token(switcher, live)
        data = switcher._get_sequence_data() or {}
        accounts = data.get("accounts", {})
        if label:
            slot = switcher._find_account_slot(data, label, "")
        else:
            new = sorted(set(accounts) - before, key=int)
            slot = new[-1] if new else None
        if slot is None:  # pragma: no cover - add_account_from_token just stored it
            raise ConfigError("The setup-token was not registered.")
        created = slot not in before
        record = accounts[slot]
        if created and normalized:
            record["alias"] = normalized
        record["setupTokenAddedAt"] = get_timestamp()
        switcher._write_json(switcher.sequence_file, data)
        stored_email = str(record.get("email", label or ""))
    return AddedAccount(slot=slot, email=stored_email, created=created)


# -- giving an account a new login in place (sak.9) ------------------------------------


def _stored_login(switcher: ClaudeAccountSwitcher, slot: str, email: str) -> str:
    """The account's stored login (``""`` when it has none); refuses one
    that cannot be read now rather than taking it for missing."""
    stored, unreadable = switcher._read_account_credentials_ex(slot, email)
    if unreadable:
        raise CredentialReadError(
            f"Account {slot}'s stored login is unreadable (locked or busy) — "
            "unlock the keychain and retry."
        )
    return stored


@dataclass(frozen=True)
class _LiveToken:
    """The setup-token account the default profile runs on, and its token."""

    slot: str
    email: str
    stored: str


def _live_setup_token(
    switcher: ClaudeAccountSwitcher, slot: str | None, email: str | None
) -> _LiveToken | None:
    """The setup-token account ``slot`` with what it stores, when the default
    profile runs on it (its new token has to go live too); None otherwise."""
    if slot is None or email is None:
        return None
    if switcher._get_current_account() != (email, ""):
        return None
    return _LiveToken(slot, email, _stored_login(switcher, slot, email))


def _activate_new_token(switcher: ClaudeAccountSwitcher, live: _LiveToken) -> None:
    """Put the account's new setup-token into the default profile, which runs
    on it: a switch away backs the live login up into the account, and would
    put the old token back over the new one. When the default profile cannot
    take it, the account gets back the token it runs on, and the error is
    raised."""
    activated = False
    try:
        switcher.switch_to(live.slot, json_output=True, force=True)
        activated = True
    finally:
        if not activated and live.stored:
            switcher._write_account_credentials(live.slot, live.email, live.stored)


def _replace_setup_token(
    switcher: ClaudeAccountSwitcher, account: str, value: str
) -> AddedAccount:
    """Store ``value`` as the setup-token account's new token, in its slot."""
    slot, email, org_uuid = switcher.resolve_account(account)
    stored = _stored_login(switcher, slot, email)
    if looks_like_api_key(stored):
        raise ValidationError(
            f"Account {slot} ({email}) is an API key, not a setup-token."
        )
    if org_uuid or (stored and not is_setup_token(stored)):
        raise ValidationError(
            f"Account {slot} ({email}) runs on a login, not a setup-token: a "
            "setup-token cannot show whose it is, so it never replaces a login."
        )
    if stored and (_oauth_entry(stored) or {}).get("accessToken") == value:
        raise ValidationError(
            f"Account {slot} ({email}) has this setup-token already; nothing "
            "changed."
        )
    live = _live_setup_token(switcher, slot, email)
    switcher.add_account_from_token(value, email=email, assume_yes=True)
    if live is not None:
        _activate_new_token(switcher, live)
    data = switcher._get_sequence_data() or {}
    if switcher._find_account_slot(data, email, "") != slot:  # pragma: no cover
        raise ConfigError(f"The setup-token was not stored for account {slot}.")
    data["accounts"][slot]["setupTokenAddedAt"] = get_timestamp()
    switcher._write_json(switcher.sequence_file, data)
    return AddedAccount(slot=slot, email=email, created=False)


def _default_login_refusal(slot: str, email: str) -> ConfigError:
    return ConfigError(
        f"Account {slot} ({email}) is the account Claude Code's default profile "
        "is logged in to: its stored login follows that live login, so a copy "
        "cannot replace it. Log in again there, or switch to another account "
        "first."
    )


def _copy_target(
    switcher: ClaudeAccountSwitcher, account: str, alias: str | None
) -> tuple[str, str, str]:
    """``(slot, email, organization)`` of the account a copy is for, refusing
    one that cannot take a copy of its login."""
    if alias is not None:
        raise ConfigError(
            "An account that gets a new login keeps its alias; give no alias "
            "with it."
        )
    slot, email, org_uuid = switcher.resolve_account(account)
    stored = _stored_login(switcher, slot, email)
    if looks_like_api_key(stored):
        raise ValidationError(f"Account {slot} ({email}) is an API key, not a login.")
    if not org_uuid and is_setup_token(stored):
        raise ValidationError(
            f"Account {slot} ({email}) runs on a setup-token: it takes a new "
            "setup-token instead of a login."
        )
    if switcher._own_live_identity() == (email, org_uuid):
        raise _default_login_refusal(slot, email)
    return slot, email, org_uuid


def _refuse_own_login(
    switcher: ClaudeAccountSwitcher,
    target: tuple[str, str, str],
    fingerprint: str,
    where: str,
) -> None:
    """The copy must not be the login the account has already."""
    slot, email, _org = target
    stored = switcher.read_account_credentials(slot, email)
    if stored and credential_fingerprint(stored) == fingerprint:
        raise ConfigError(
            f"{where} is the login account {slot} ({email}) has already; nothing "
            "changed."
        )


def _refuse_other_login(
    switcher: ClaudeAccountSwitcher,
    target: tuple[str, str, str],
    identity: tuple[str, str, str | None],
    where: str,
    renewed: bool,
) -> None:
    """The copy must be the login of the account it is for: the one
    :meth:`add_account_from_profile` would give it to."""
    slot, email, org_uuid = target
    owner, owner_org, owner_org_name = identity
    data = switcher._get_sequence_data() or {}
    if switcher._find_account_slot(data, owner, owner_org) == slot:
        if switcher._own_live_identity() == (email, org_uuid):
            raise _default_login_refusal(slot, email)
        return
    whose = owner
    if owner == email:
        whose = f"{owner} in another organization ({owner_org_name or owner_org})"
    ending = "." if renewed else "; nothing changed."
    raise ValidationError(
        f"{where} is the login of {whose}, not of account {slot} ({email}){ending}"
    )


# -- copying a login in (sak.8) -----------------------------------------------------

_GIVEN = "The text given"
"""How :func:`add_login_copy`'s messages name credentials given as text."""


def _copy_source(
    switcher: ClaudeAccountSwitcher,
    credentials: str | None,
    source: str | os.PathLike[str] | None,
) -> tuple[str, str, str | None]:
    """``(where, credentials, path)``: the login to copy, how messages name
    it, and the file it was read from (``None`` for text)."""
    if (credentials is None) == (source is None):
        raise ConfigError("Give either the credentials or a credentials file.")
    if credentials is not None:
        if len(credentials) > _MAX_CREDENTIALS_BYTES:
            raise ValidationError(f"{_GIVEN} is too big to be a Claude login.")
        return _GIVEN, credentials.strip(), None
    raw = os.path.expanduser(os.fspath(source)).strip()
    if not raw:
        raise ConfigError("Name a credentials file.")
    path = Path(raw)
    if not path.exists():
        raise ConfigError(f"{raw} does not exist.")
    _refuse_own_store(switcher, path)
    if not path.is_file():
        raise ConfigError(f"{raw} is not a file; name a Claude Code credentials file.")
    return raw, _read_file_credentials(path).strip(), raw


def _live_access_token(credentials: str) -> str | None:
    """The access token, unless it is missing or has lapsed."""
    entry = _oauth_entry(credentials) or {}
    token = entry.get("accessToken")
    if not isinstance(token, str) or not token.strip():
        return None
    if oauth.is_oauth_token_expired(entry.get("expiresAt")):
        return None
    return token.strip()


def _keep_copy(target: str, credentials: str) -> None:
    """Store the copy as Claude Code keeps a profile's login outside the
    macOS Keychain: ``.credentials.json``, readable by its owner only."""
    path = Path(target) / ".credentials.json"
    temporary = path.with_name(".credentials.json.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(credentials)
        os.replace(temporary, path)
    except OSError as e:
        raise CredentialError(f"Could not keep the login in {target}: {e}") from e


def _renew(credentials: str, where: str) -> str:
    """The copy with a new access token. Its refresh token is spent then, so
    the copy it came from no longer works."""
    outcome = oauth.try_refresh_oauth_credentials(credentials)
    if outcome.credentials is not None:
        return outcome.credentials
    if outcome.error == "invalid_grant":
        raise CredentialError(
            f"{where} holds a login that no longer works: it was renewed, moved or "
            "logged out elsewhere since this copy was made. Log in again instead."
        )
    raise CredentialError(
        f"{where} holds a login whose access has lapsed, and it could not be "
        f"renewed now ({outcome.error}); check the connection and try again."
    )


def _copy_identity(target: str, credentials: str) -> tuple[str, str, str | None]:
    """Ask Anthropic whose login it is, and record that in the target profile
    as Claude Code does (``oauthAccount``). Returns ``(email, organization,
    the organization's name)``."""
    token = _live_access_token(credentials)
    profile = oauth.fetch_oauth_profile(token) if token else None
    email = profile.get("email") if profile else None
    org_uuid = profile.get("organizationUuid") if profile else None
    if not profile or not email or not org_uuid:
        raise CredentialError(
            "Anthropic did not say whose login this is (offline, or the login "
            "no longer works); try again later."
        )
    account = {
        "accountUuid": profile["uuid"],
        "emailAddress": email,
        "organizationUuid": org_uuid,
        "organizationName": profile.get("organizationName"),
    }
    try:
        (Path(target) / ".claude.json").write_text(
            json.dumps({"oauthAccount": account}), encoding="utf-8"
        )
    except OSError as e:
        raise CredentialError(f"Could not record the account in {target}: {e}") from e
    return email, org_uuid, profile.get("organizationName")


def add_login_copy(
    switcher: ClaudeAccountSwitcher,
    config_dir: str | os.PathLike[str],
    *,
    credentials: str | None = None,
    source: str | os.PathLike[str] | None = None,
    alias: str | None = None,
    account: str | None = None,
) -> CopiedLogin:
    """Register a copy of a Claude login: the credentials JSON Claude Code
    keeps (``{"claudeAiOauth": ...}``), given as text (``credentials``) or as
    a credentials file (``source``, read here and left as it is). Exactly one
    of the two.

    Unlike :func:`import_login`, nothing moves: both copies work until either
    renews the login, and from then on only that one does (each refresh
    token works once). Use the copy only here. The credentials name no
    account, so Anthropic is asked whose login it is (``/api/oauth/profile``,
    with its access token); when that token has lapsed, the login is renewed
    first, and the copy it came from stops working right away.

    The copy is kept in ``config_dir``, an empty, temporary profile the
    caller owns, and registered as :func:`add_profile_login` registers a
    login: a new account takes the next free slot with ``alias``; an account
    already managed gets it as its stored login; the account the default
    profile is logged in to keeps following its live login.
    ``config_dir`` keeps the copy until :func:`forget_profile_login`.

    ``account`` (sak.9) names the managed account (slot, alias or email) the
    copy is for: it must be that account's own login, which then replaces
    its stored login; the account keeps its slot and alias (``alias`` does
    not go with it). Refused before anything happens: a setup-token or
    API-key account, the account the default profile is logged in to (its
    stored login follows the live one) and the login the account has
    already; refused once Anthropic says whose the copy is: another
    account's login.

    Raises :class:`ConfigError` for neither or both of ``credentials`` and
    ``source``, a missing file, a folder, the default profile, claude-swap's
    own store, or a login that is managed already; :class:`ValidationError`
    for a bare token, an API key, a setup-token, a logged-out or expired
    login, one without its scopes, or a bad alias or one another account
    uses; :class:`CredentialError` when the file cannot be read, the login
    cannot be renewed, Anthropic does not say whose it is, or the default
    profile is logged in to its account and that login cannot be read
    cleanly now (sak.13). With
    ``account`` also :class:`AccountNotFoundError`, :class:`ConfigError` for
    the default login's account, the login it has already or an alias, and
    :class:`ValidationError` for an account of another kind or another
    account's login. Nothing is
    registered then, and unless the login was renewed nothing changed. A
    renewed copy that cannot be registered raises
    :class:`LoginNotAddedError`: ``config_dir`` holds the only working copy.
    Never prompts; the tokens are never printed or logged.
    """
    _refuse_foreign_profile()
    with _quiet():
        where, copy, path = _copy_source(switcher, credentials, source)
        if copy.startswith(SETUP_TOKEN_PREFIX):
            raise ValidationError(
                f"{where} is a bare token, not a login's credentials (the JSON "
                "with claudeAiOauth); register a setup-token as one."
            )
        wanted = None if account is None else _copy_target(switcher, account, alias)
        _movable(copy, where, "use")
        fingerprint = credential_fingerprint(copy) or ""
        if wanted is not None:
            _refuse_own_login(switcher, wanted, fingerprint, where)
        _refuse_managed_grant(switcher, fingerprint, where)
        alias = _free_alias(switcher, alias, None)
        target = _empty_profile(switcher, config_dir, path, "copy")
        _keep_copy(target, copy)
    # Outside the output lock: Anthropic may take a while to answer, and
    # every other call of this module waits for that lock.
    renewed = False
    if _live_access_token(copy) is None:
        copy = _renew(copy, where)
        renewed = True
        try:
            _keep_copy(target, copy)  # the only working copy from here on
        except CredentialError as e:
            raise CredentialError(
                f"{e}. The login was renewed, so the copy it came from no longer "
                "works: log in again."
            ) from e
    try:
        identity = _copy_identity(target, copy)
        with _quiet():
            if wanted is not None:
                _refuse_other_login(switcher, wanted, identity, where, renewed)
            default_login = switcher._live_login_followed_for(*identity[:2])
            slot, created = switcher.add_account_from_profile(target, alias=alias)
            email = switcher.account_email(slot)
    except ClaudeSwitchError as e:
        if not renewed:
            raise
        raise LoginNotAddedError(
            f"The login was renewed into {target}, but it could not be "
            f"registered: {e}",
            target,
        ) from e
    return CopiedLogin(
        account=AddedAccount(slot=slot, email=email, created=created),
        default_login=default_login,
        renewed=renewed,
    )


# -- handing a login over (sak.11) --------------------------------------------------


def _handed_over_file(
    switcher: ClaudeAccountSwitcher, destination: str | os.PathLike[str]
) -> Path:
    """The file a handed-over login goes to: new, outside the default
    profile and claude-swap's own store (a login there would be taken for a
    managed or live one)."""
    raw = os.path.expanduser(os.fspath(destination)).strip()
    if not raw:
        raise ConfigError("Name the file the login goes to.")
    path = Path(raw)
    for root, what in (
        (get_default_claude_config_home(), "Claude Code's default profile"),
        (Path(switcher.backup_dir), "claude-swap's own store"),
    ):
        if _inside(path, root) or _inside(path.parent, root):
            raise ConfigError(f"{raw} is inside {what}; name a file elsewhere.")
    if not path.parent.is_dir():
        raise ConfigError(f"{path.parent} is not a folder.")
    return path


HAND_OVER_LOCK_TIMEOUT = 30.0
"""How long :func:`hand_over_login` waits for a renewal of the account's
login to end (a renewal's request is bounded at 10 s)."""


def _write_new_file(path: Path, credentials: str) -> None:
    """Write the login as Claude Code keeps it outside the macOS Keychain:
    a new file, readable by its owner only (an existing file is never
    replaced), and on the disk itself before this returns, since it becomes
    the login's only copy once the account goes."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as e:
        raise ConfigError(f"{path} exists already; name a new file.") from e
    except OSError as e:
        raise CredentialError(f"Could not write the login to {path}: {e}") from e
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(credentials)
            f.flush()
            sync_file(f.fileno())
    except OSError as e:
        with contextlib.suppress(OSError):
            path.unlink()
        raise CredentialError(f"Could not write the login to {path}: {e}") from e
    sync_folder(path.parent)


def _handover_refusal(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str, stored: str
) -> ClaudeSwitchError | None:
    """Why the account's login cannot leave claude-swap, or ``None``."""
    if not stored:
        return CredentialError(
            f"Account {slot} ({email}) has no stored login to hand over."
        )
    if looks_like_api_key(stored):
        return ValidationError(
            f"Account {slot} ({email}) is an API key, not a login; add the key "
            "where it is needed instead."
        )
    if not org_uuid and is_setup_token(stored):
        return ValidationError(
            f"Account {slot} ({email}) runs on a setup-token, not a login; add "
            "the setup-token where it is needed instead."
        )
    if switcher._own_live_identity() == (email, org_uuid):
        return ConfigError(
            f"Account {slot} ({email}) is the account Claude Code's default "
            "profile is logged in to; switch to another account first."
        )
    if switcher.live_session_pids_for(slot, email):
        return SessionError(
            f"A Claude session runs on account {slot} ({email}); end it first."
        )
    try:
        # what import_login takes on the other computer, checked before the
        # account goes: a login it refuses would be stuck in transit
        _movable(stored, f"Account {slot} ({email})'s stored login")
    except ValidationError as e:
        return e
    return None


@contextlib.contextmanager
def _renewals_held(switcher: ClaudeAccountSwitcher, slot: str) -> Iterator[None]:
    """Keep the account's stored login from being renewed meanwhile: every
    renewal holds the slot's consume lock, and a renewal would spend the copy
    being handed over."""
    lock = FileLock(switcher.consume_lock_file(slot), timeout=HAND_OVER_LOCK_TIMEOUT)
    if not lock.acquire():
        raise LockError(
            f"Account {slot}'s login is being renewed right now; try again in a moment."
        )
    try:
        yield
    finally:
        lock.release()


def _newest_login(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str
) -> str:
    """The account's stored login, after taking in a newer one where there
    is one: a renewal whose saving failed (claude-swap set it aside) and one
    Claude Code made in the account's ``cswap run`` profile (as a switch
    takes it in). The caller holds :func:`_renewals_held`."""
    with FileLock(switcher.lock_file):
        current, unreadable = switcher._read_account_credentials_ex(slot, email)
        if current and not unreadable:
            switcher._adopt_stashed_successor(slot, email, current)
    switcher._adopt_session_credential(slot, email, org_uuid)
    if switcher._session_profile_ahead(slot, email, org_uuid):
        raise SessionError(
            f"Account {slot} ({email}) was renewed in its `cswap run` profile, and "
            "that newer login cannot be taken in now (a Claude session may run "
            "there); end it and try again."
        )
    return _stored_login(switcher, slot, email)


def _still_stored(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, stored: str
) -> bool:
    """Whether the account still keeps the login that was handed over (a
    removal that failed took nothing); False when that cannot be read now."""
    try:
        current, unreadable = switcher._read_account_credentials_ex(slot, email)
    except (ClaudeSwitchError, OSError):
        return False
    if unreadable or not current:
        return False
    fingerprint = credential_fingerprint(stored)
    if fingerprint is None:
        return current.strip() == stored.strip()
    return credential_fingerprint(current) == fingerprint


def hand_over_login(
    switcher: ClaudeAccountSwitcher,
    identifier: str,
    destination: str | os.PathLike[str],
) -> HandedOver:
    """Hand an account's login over to another computer: the other half of
    the one-way move (:func:`import_login` takes it in there).

    Writes the account's stored login to ``destination``, a new Claude Code
    credentials file (``{"claudeAiOauth": ...}``, mode 0600, never an
    existing file, flushed to the disk), then removes the account from
    claude-swap (its stored login, config and mappings) as
    :func:`remove_account` does. From then on the file is the account's only
    copy here: deliver it, then delete it. The login handed over is the
    newest one: a renewal claude-swap set aside, or one Claude Code made in
    the account's ``cswap run`` profile, is taken in first, and claude-swap
    does not renew the login while it leaves (a renewal under way is waited
    for, up to :data:`HAND_OVER_LOCK_TIMEOUT` seconds).

    If the removal fails while the account still keeps its login, the file
    is deleted and the error is raised: nothing changed. If it fails after
    the stored login went (or when that cannot be read), the file is kept,
    as the login's only copy, and the result says what failed
    (``removal_error``); claude-swap may still list the account then,
    without a login.

    ``identifier`` is a slot, alias or email. Refused before anything is
    written: the account the default profile is logged in to (switch to
    another account first), an account a live ``cswap run`` session uses,
    an API-key or setup-token account, an account without a stored login,
    a login :func:`import_login` would refuse (no refresh token, expired,
    without its scopes), and a destination that exists, whose folder is
    missing, or that lies in the default profile or claude-swap's own store.

    Raises :class:`AccountNotFoundError` for an unknown account,
    :class:`ConfigError` for the default login's account, an ambiguous email
    or a refused destination, :class:`SessionError` for a live session (or a
    newer login in its profile that cannot be taken in now),
    :class:`ValidationError` for an API key, a setup-token or a login that
    cannot move, :class:`LockError` while a renewal of the account does not
    end, and :class:`CredentialError` when the stored login is missing or
    unreadable or the file cannot be written. Never prompts; the token is
    never printed or logged.
    """
    _refuse_foreign_profile()
    with _quiet():
        slot, email, org_uuid = switcher.resolve_account(identifier)
        stored = _stored_login(switcher, slot, email)
        refusal = _handover_refusal(switcher, slot, email, org_uuid, stored)
        if refusal is not None:
            raise refusal
        path = _handed_over_file(switcher, destination)
    # The renewals are waited for outside the output lock (every other call
    # of this module waits for that lock), which is taken once they are held.
    with _renewals_held(switcher, slot), _quiet():
        stored = _newest_login(switcher, slot, email, org_uuid)
        refusal = _handover_refusal(switcher, slot, email, org_uuid, stored)
        if refusal is not None:
            raise refusal
        record = (switcher._get_sequence_data() or {}).get("accounts", {}).get(slot, {})
        handed = HandedOver(
            slot=slot,
            email=email,
            organization_uuid=org_uuid or None,
            organization=str(record.get("organizationName") or "") or None,
            alias=str(record.get("alias") or "") or None,
            expires_at=login_expires_at_iso(stored),
        )
        _write_new_file(path, stored)
        try:
            switcher.remove_account(slot, assume_yes=True)
        except BaseException as e:
            if _still_stored(switcher, slot, email, stored):
                with contextlib.suppress(OSError):
                    path.unlink()
                raise
            if not isinstance(e, Exception):
                raise  # the file stays: it is the login's only copy
            _logger.warning(
                "api: account %s's login was handed over, but the rest of the "
                "account could not be removed: %s",
                slot,
                e,
            )
            return handed._replace(removal_error=str(e) or type(e).__name__)
    return handed


# -- an access token for another computer (sak.12) ----------------------------------

ACCESS_TOKEN_LOCK_TIMEOUT = 15.0
"""How long :func:`access_token` waits for the account's consume lock and
claude-swap's own lock: a renewal of the account's login holds them for its
request (bounded at 10 s), a switch for its local writes."""

_LOCKED_RENEWAL_TIMEOUT = 6.0
"""The request bound of a renewal made under claude-swap's own lock (and,
for the default profile's login, Claude Code's credential locks too): well
inside the 10 s a switch waits for that lock, as in claude-swap's own locked
renewal of the default profile's login."""

_UNMIRRORED = (
    "CLAUDE_SECURESTORAGE_CONFIG_DIR is set, so Claude Code keeps its login "
    "where claude-swap does not look; unset it and try again."
)

# Why a renewal :func:`access_token` made was set aside in claude-swap's
# unclaimed store instead of stored: the consume gate's own reasons for the
# same cases (``cswap unclaimed`` lists them). The next look at the account
# takes the renewal in while the store still holds the login it renewed; a
# CAS-conflict one is retired on sight, the store having moved on to a login
# of its own.
_ASIDE_PERSIST_FAILED = "consume-gate-persist-failed"
_ASIDE_STORE_UNREADABLE = "consume-gate-store-unreadable"
_ASIDE_SLOT_REMOVED = "consume-gate-slot-removed"
_ASIDE_CAS_CONFLICT = "consume-gate-cas-conflict"

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AccessTokenInfo(NamedTuple):
    """An account's access token, for another computer to use without its
    login (:func:`access_token`, sak.12). ``repr`` never shows the token."""

    slot: str
    email: str
    organization_uuid: str | None
    organization: str | None
    """The organization's name, when claude-swap recorded one."""
    account_uuid: str | None
    """Claude's id of the account (``oauthAccount.accountUuid``)."""
    access_token: str
    expires_at: str
    """When the access token lapses: ISO 8601 UTC with milliseconds
    (``2026-10-16T21:47:12.345Z``)."""
    scopes: tuple[str, ...]
    subscription_type: str | None
    rate_limit_tier: str | None
    renewed: bool
    """This call renewed the login: the old refresh token is spent, and the
    new login is stored wherever the old one was."""

    def __repr__(self) -> str:
        shown = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in self._fields
            if name != "access_token"
        )
        return f"AccessTokenInfo({shown}, access_token=<hidden>)"


def _renew_window(renew_within: float) -> int:
    """``renew_within`` in milliseconds; refuses what is not a duration."""
    if (
        isinstance(renew_within, bool)
        or not isinstance(renew_within, (int, float))
        or not math.isfinite(renew_within)
        or renew_within < 0
    ):
        raise ValidationError(
            f"renew_within is a number of seconds, 0 or more; not {renew_within!r}."
        )
    return int(renew_within * 1000)


def _due(credentials: str, window_ms: int) -> bool:
    """Whether the login's access token is missing or lapses within
    ``window_ms`` from now (or has lapsed)."""
    entry = _oauth_entry(credentials) or {}
    token = entry.get("accessToken")
    expires = entry.get("expiresAt")
    if not isinstance(token, str) or not token.strip():
        return True
    if isinstance(expires, bool) or not isinstance(expires, (int, float)):
        return True
    return time.time() * 1000 + window_ms >= expires


def _login_scopes(entry: Mapping[str, Any]) -> tuple[str, ...] | None:
    """The login's scopes as :func:`keep_access_login` takes them (names
    without spaces), or ``None`` when it has none it could keep."""
    scopes = entry.get("scopes")
    if not isinstance(scopes, list) or not scopes:
        return None
    if not all(
        isinstance(s, str) and s and not any(c.isspace() for c in s) for s in scopes
    ):
        return None
    return tuple(scopes)


def _no_scopes(slot: str, email: str) -> ValidationError:
    return ValidationError(
        f"Account {slot} ({email})'s login does not say what it may do (it has no "
        "scopes), so another computer could not use it; log in again."
    )


def _kind_refusal(
    slot: str, email: str, org_uuid: str, login: str
) -> ClaudeSwitchError | None:
    """Why the account has no access token to hand out although it has
    something stored, or ``None``."""
    if looks_like_api_key(login):
        return ValidationError(
            f"Account {slot} ({email}) is an API key, not a login: it has no "
            "access token to hand out."
        )
    if not org_uuid and is_setup_token(login):
        return ValidationError(
            f"Account {slot} ({email}) runs on a setup-token, not a login: add "
            "the setup-token where it is needed instead."
        )
    entry = _oauth_entry(login)
    if entry is None:
        return CredentialError(
            f"Account {slot} ({email})'s login is not one Claude Code keeps (no "
            "claudeAiOauth entry); log in again."
        )
    if _login_scopes(entry) is None:
        return _no_scopes(slot, email)
    return None


def _live_unreadable(slot: str, email: str) -> CredentialReadError:
    return CredentialReadError(
        f"The login of Claude Code's default profile (account {slot}, {email}) "
        "cannot be read now (the Keychain is locked or busy); unlock it and try "
        "again."
    )


def _renewal_failure(slot: str, email: str, error: str | None) -> ClaudeSwitchError:
    """What a renewal that did not happen raises."""
    who = f"Account {slot} ({email})"
    if error == "invalid_grant":
        return CredentialError(
            f"{who}'s login no longer works: it was renewed, moved or logged out "
            "elsewhere. Log in again."
        )
    if error == "no_refresh_token":
        return CredentialError(
            f"{who}'s login has no refresh token, so it cannot be renewed; log "
            "in again."
        )
    if error == "invalid_client":
        return CredentialError(
            f"Anthropic refused to renew {who}'s login (invalid_client); the login "
            "itself may still work. Try again later."
        )
    return CredentialError(
        f"{who}'s login could not be renewed now (no connection, or Anthropic did "
        "not answer); try again in a few minutes."
    )


def _session_renews_it(slot: str, email: str) -> SessionError:
    return SessionError(
        f"A Claude session runs on account {slot} ({email}) with this login, which "
        "is due for renewal: that session renews it itself the next time it is "
        "used (renewing it here would leave that session with a spent login). "
        "Try again after that, or end the session."
    )


def _session_holds(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str, login: str
) -> bool:
    """Whether a Claude session may be running on the account's ``cswap
    run`` profile with ``login`` itself (the same refresh token), which that
    session renews on its own: renewing it here would spend the session's
    copy. A session record, or a login there, that cannot be read counts as
    one; a session on another login of the account does not."""
    session_dir = switcher._session_dir(slot, email)
    if profile_is_quiescent(session_dir):
        return False
    if session_identity_drifted(session_dir, email, org_uuid):
        return False
    try:
        held = read_config_dir_credentials(str(session_dir), strict_keychain=True)
    except CredentialReadError:
        return True
    return bool(held) and credential_fingerprint(held) == credential_fingerprint(login)


def _set_aside_unreadable(slot: str, email: str) -> CredentialReadError:
    return CredentialReadError(
        f"A renewal of account {slot} ({email})'s login that claude-swap set aside "
        "cannot be read now (a locked Keychain?); unlock it and try again."
    )


def _take_in_set_aside(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, current: str
) -> str | None:
    """A renewal of ``current`` that claude-swap set aside because it could
    not store it (the consume gate's, or :func:`access_token`'s own), now
    the account's stored login; ``None`` when there is none. The caller
    holds claude-swap's own lock."""
    try:
        return switcher._adopt_stashed_successor(slot, email, current)
    except CredentialReadError as e:
        raise _set_aside_unreadable(slot, email) from e


def _set_aside(
    switcher: ClaudeAccountSwitcher, slot: str, consumed: str, renewed: str, reason: str
) -> None:
    """Keep a renewal that was not stored in claude-swap's unclaimed store,
    keyed by the login it renewed, as the consume gate keeps one: the next
    look at the account takes it in. The old refresh token is spent, so the
    new login must not be lost. Never raises."""
    try:
        switcher._store._write_unclaimed_credential(
            renewed,
            {
                "reason": reason,
                "configSlot": slot,
                "consumedFp": credential_fingerprint(consumed),
                "fingerprint": credential_fingerprint(renewed),
            },
        )
    except (ClaudeSwitchError, OSError, ValueError) as e:
        _logger.error(
            "api: account %s's renewed login could not be stored or set aside: %s; "
            "log in again if Claude Code asks for it",
            slot,
            e,
        )
        return
    _logger.warning(
        "api: account %s's renewed login was not stored and is set aside (%s)",
        slot,
        reason,
    )


def _store_renewal(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, consumed: str, renewed: str
) -> None:
    """Store a renewal of the account's stored login as the consume gate
    stores one: only while the store still holds the login it renewed, read
    again now, since adding or removing the account does not wait for
    claude-swap's lock (a login added meanwhile stays; a removed one is not
    brought back). Otherwise, and when the write fails, the renewal is set
    aside. Never raises: the old refresh token is spent, and the renewal is
    handed out all the same."""
    now, unreadable = switcher._read_account_credentials_ex(slot, email)
    if unreadable:
        reason = _ASIDE_STORE_UNREADABLE
    elif not now:
        reason = _ASIDE_SLOT_REMOVED
    elif credential_fingerprint(now) != credential_fingerprint(consumed):
        reason = _ASIDE_CAS_CONFLICT
    else:
        try:
            switcher._write_account_credentials(slot, email, renewed)
        except (ClaudeSwitchError, OSError) as e:
            _logger.warning(
                "api: account %s's renewed login could not be kept as its stored "
                "login: %s",
                slot,
                e,
            )
            reason = _ASIDE_PERSIST_FAILED
        else:
            return
    _set_aside(switcher, slot, consumed, renewed, reason)


def _store_live_renewal(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, consumed: str, renewed: str
) -> None:
    """Store a renewal of the default profile's login wherever the old login
    was: the account's stored login and the default profile (Claude Code
    takes it from there at its next request); set it aside when neither
    takes it. Never raises: the old refresh token is spent, and the new
    login is handed out all the same."""
    stored = True
    try:
        switcher._write_account_credentials(slot, email, renewed)
    except (ClaudeSwitchError, OSError) as e:
        stored = False
        _logger.warning(
            "api: account %s's renewed login could not be kept as its stored "
            "login: %s",
            slot,
            e,
        )
    try:
        with claude_config_lock():
            switcher._write_credentials(renewed)
    except (ClaudeSwitchError, OSError) as e:
        _logger.warning(
            "api: account %s's renewed login could not be written to Claude "
            "Code's default profile: %s",
            slot,
            e,
        )
        if not stored:
            _set_aside(switcher, slot, consumed, renewed, _ASIDE_PERSIST_FAILED)


def _write_live(switcher: ClaudeAccountSwitcher, slot: str, credentials: str) -> None:
    """Put a renewal of the default profile's login that was set aside back
    where Claude Code reads it; the caller holds Claude Code's credential
    locks (:func:`_claude_code_locks`). Best effort: it is the stored login
    now, and claude-swap restores the default profile from there once Claude
    Code's copy stops working."""
    try:
        with claude_config_lock():
            switcher._write_credentials(credentials)
    except (ClaudeSwitchError, OSError) as e:
        _logger.warning(
            "api: account %s's login that was set aside could not be written to "
            "Claude Code's default profile: %s",
            slot,
            e,
        )


def _attribute_live(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str, live: str
) -> None:
    """Let claude-swap confirm a login Claude Code renewed in the default
    profile as the account's own (it asks Anthropic whose the access token
    is, while that still works), so that it may be handed out and renewed
    here. Best effort: claude-swap's own check, which never raises and
    stores the login as the account's when it is."""
    if _due(live, 0):
        return
    stored, unreadable = switcher._read_account_credentials_ex(slot, email)
    if unreadable or (stored and credential_fingerprint(stored) == credential_fingerprint(live)):
        return
    switcher._resync_rotated_backup(slot, email, org_uuid, live)


def _confirm_live(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str
) -> None:
    """When the account is the one Claude Code's default profile is logged
    in to, run :func:`_attribute_live` on its login there. Before the locks:
    asking Anthropic is network, which never runs under claude-swap's lock
    but for a renewal's one request. The check that counts is made under
    the locks (:func:`_require_own_live`)."""
    if not switcher._live_identity_matches(email, org_uuid):
        return
    active = switcher._read_active_credentials()
    if not active.value or active.degraded or active.keychain_unavailable:
        return
    if oauth.is_lent_login(active.value):
        return  # another computer lent it (sak.13): nothing to confirm
    _attribute_live(switcher, slot, email, org_uuid, active.value)


def _renewing_now(slot: str) -> LockError:
    return LockError(
        f"Account {slot}'s login is being renewed or switched right now; try "
        "again in a moment."
    )


@contextlib.contextmanager
def _account_locks(switcher: ClaudeAccountSwitcher, slot: str) -> Iterator[None]:
    """Hold, in claude-swap's order, the account's consume lock (every
    renewal of its login holds it) and claude-swap's own lock (every switch
    holds it, from reading the login it puts in the default profile to
    writing it there): no other renewal and no switch comes in between."""
    consume = FileLock(switcher.consume_lock_file(slot), timeout=ACCESS_TOKEN_LOCK_TIMEOUT)
    if not consume.acquire():
        raise _renewing_now(slot)
    try:
        own = FileLock(switcher.lock_file, timeout=ACCESS_TOKEN_LOCK_TIMEOUT)
        if not own.acquire():
            raise _renewing_now(slot)
        try:
            yield
        finally:
            own.release()
    finally:
        consume.release()


@contextlib.contextmanager
def _claude_code_locks(slot: str) -> Iterator[None]:
    """Hold Claude Code's credential locks (taken after claude-swap's own, as
    claude-swap does), so that Claude Code does not renew the default
    profile's login meanwhile."""
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(claude_credentials_lock())
        except ClaudeCodeLockTimeout as e:
            raise _renewing_now(slot) from e
        except OSError as e:
            raise LockError(f"Claude Code's credential lock cannot be taken: {e}") from e
        yield


def _require_own_live(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, live: str
) -> None:
    """Refuse a login in the default profile that is not the account's own:
    its refresh token is neither the stored login's nor one claude-swap
    confirmed as the account's (:func:`_confirm_live`). Claude Code's store
    can hold another account's login under this account's name (a partial
    sync of that store; a ``/login`` between its two writes), and handing it
    out would put that account's token under this one's name."""
    lineage = credential_fingerprint(live) or ""
    stored, unreadable = switcher._read_account_credentials_ex(slot, email)
    if not unreadable and stored and credential_fingerprint(stored) == lineage:
        return
    verdict = switcher._probe_verdicts.get(switcher._lineage_key(slot, email, lineage))
    if verdict is True:
        return
    who = f"Account {slot} ({email})"
    if verdict is False:
        raise CredentialError(
            f"Claude Code's default profile is set to account {slot} ({email}) but "
            "holds another account's login; log in again there."
        )
    if unreadable:
        raise CredentialReadError(
            f"{who}'s stored login cannot be read now (the Keychain is locked or "
            "busy), so its login in Claude Code's default profile cannot be "
            "checked; unlock it and try again."
        )
    if _due(live, 0):
        raise CredentialError(
            f"{who}'s login in Claude Code's default profile is not the one "
            "claude-swap keeps for it, and its access token has lapsed, so "
            "claude-swap cannot check whose it is. Use Claude Code on this "
            "computer once (it renews the login there), then try again."
        )
    raise CredentialError(
        f"{who}'s login in Claude Code's default profile is not the one "
        "claude-swap keeps for it (Claude Code renewed it there, or someone "
        "logged in) and is not confirmed as this account's yet; try again in a "
        "few minutes."
    )


def _own_live_login(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str
) -> tuple[str, bool]:
    """The login of Claude Code's default profile, which runs on the
    account, once it is known to be the account's own; and whether it is a
    renewal of that login claude-swap set aside, taken in as the account's
    stored login but not in the default profile yet. A read that fell back
    past a failed Keychain read is refused: on macOS Claude Code renews in
    the Keychain only, so the fallback may be an older login whose access
    token no longer works. The caller holds :func:`_account_locks`."""
    active = switcher._read_active_credentials()
    if active.value is None or active.degraded or active.keychain_unavailable:
        raise _live_unreadable(slot, email)
    live = active.value
    if not live:
        raise CredentialError(
            f"Account {slot} ({email}) is the account Claude Code's default "
            "profile is logged in to, but that profile has no login now; log "
            "in again there."
        )
    refusal = _kind_refusal(slot, email, org_uuid, live)
    if refusal is not None:
        raise refusal
    taken = _take_in_set_aside(switcher, slot, email, live)
    if taken is not None:
        return taken, True
    _require_own_live(switcher, slot, email, live)
    return live, False


def _live_token(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str, window_ms: int
) -> tuple[str, bool]:
    """The login of Claude Code's default profile, renewed first when it is
    due, as Claude Code renews it: under Claude Code's credential locks too,
    read again under them (a renewal Claude Code made meanwhile is taken as
    it is). A renewal set aside is put back in the default profile under
    those locks as well. The caller holds :func:`_account_locks`: no switch
    and no other renewal of claude-swap's comes in between."""
    live, set_aside = _own_live_login(switcher, slot, email, org_uuid)
    if not set_aside and not _due(live, window_ms):
        return live, False
    with _claude_code_locks(slot):
        if not set_aside:
            live, set_aside = _own_live_login(switcher, slot, email, org_uuid)
        if set_aside:
            _write_live(switcher, slot, live)
        if not _due(live, window_ms):
            return live, False
        if _session_holds(switcher, slot, email, org_uuid, live):
            raise _session_renews_it(slot, email)
        result = oauth.try_refresh_oauth_credentials(
            live, timeout_s=_LOCKED_RENEWAL_TIMEOUT
        )
        if result.error is not None or not result.credentials:
            raise _renewal_failure(slot, email, result.error)
        renewed = result.credentials
        switcher._probe_verdicts[
            switcher._lineage_key(slot, email, credential_fingerprint(renewed) or "")
        ] = True
        _store_live_renewal(switcher, slot, email, live, renewed)
    return renewed, True


def _stored_token(
    switcher: ClaudeAccountSwitcher, slot: str, email: str, org_uuid: str, window_ms: int
) -> tuple[str, bool]:
    """The account's newest stored login, renewed first when it is due. The
    caller holds :func:`_account_locks`: no other renewal of the login comes
    in between, and no switch puts the login in the default profile while
    it is renewed.

    Takes in what claude-swap set aside from a renewal it could not store,
    and what Claude Code renewed in the account's ``cswap run`` profile once
    no session runs there (as claude-swap itself does). While a session
    runs, a newer login in its profile is served as it is."""
    current, unreadable = switcher._read_account_credentials_ex(slot, email)
    if unreadable:
        raise CredentialReadError(
            f"Account {slot} ({email})'s stored login cannot be read now (the "
            "Keychain is locked or busy); unlock it and try again."
        )
    if current:
        current = _take_in_set_aside(switcher, slot, email, current) or current
    if not current:
        raise CredentialError(f"Account {slot} ({email}) has no stored login.")
    refusal = _kind_refusal(slot, email, org_uuid, current)
    if refusal is not None:
        raise refusal
    ahead = switcher._session_profile_ahead(slot, email, org_uuid)
    if ahead is not None and profile_is_quiescent(switcher._session_dir(slot, email)):
        try:
            # what claude-swap's own take-in does (_adopt_session_credential),
            # under the lock held here
            switcher._store._write_account_credentials(slot, email, ahead)
        except (ClaudeSwitchError, OSError) as e:
            raise CredentialWriteError(
                f"Account {slot} ({email})'s newer login from its `cswap run` "
                f"profile could not be stored: {e}"
            ) from e
        current, ahead = ahead, None
    login = ahead or current
    if not _due(login, window_ms):
        return login, False
    if ahead is not None or _session_holds(switcher, slot, email, org_uuid, login):
        raise _session_renews_it(slot, email)
    result = oauth.try_refresh_oauth_credentials(login, timeout_s=_LOCKED_RENEWAL_TIMEOUT)
    if result.error is not None or not result.credentials:
        raise _renewal_failure(slot, email, result.error)
    _store_renewal(switcher, slot, email, login, result.credentials)
    return result.credentials, True


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _instant(epoch_ms: int) -> str:
    """Epoch milliseconds as ISO 8601 UTC with milliseconds."""
    moment = _EPOCH + timedelta(milliseconds=epoch_ms)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _token_info(
    slot: str,
    email: str,
    org_uuid: str,
    record: Mapping[str, Any],
    login: str,
    renewed: bool,
) -> AccessTokenInfo:
    entry = _oauth_entry(login) or {}
    token = entry.get("accessToken")
    expires = entry.get("expiresAt")
    if (
        not isinstance(token, str)
        or not token.strip()
        or isinstance(expires, bool)
        or not isinstance(expires, (int, float))
        or not math.isfinite(expires)
    ):
        raise CredentialError(
            f"Account {slot} ({email})'s login has no access token with an expiry; "
            "log in again."
        )
    scopes = _login_scopes(entry)
    if scopes is None:
        raise _no_scopes(slot, email)
    return AccessTokenInfo(
        slot=slot,
        email=email,
        organization_uuid=org_uuid or None,
        organization=_text(record.get("organizationName")),
        account_uuid=_text(record.get("uuid")),
        access_token=token.strip(),
        expires_at=_instant(int(expires)),
        scopes=scopes,
        subscription_type=_text(entry.get("subscriptionType")),
        rate_limit_tier=_text(entry.get("rateLimitTier")),
        renewed=renewed,
    )


def access_token(
    switcher: ClaudeAccountSwitcher, identifier: str, *, renew_within: float = 0.0
) -> AccessTokenInfo:
    """The newest access token of an account on this computer, for another
    computer to use without the login (sak.12): its refresh token never
    leaves, so only this computer ever renews the login.

    The newest login is the default profile's live login when the account
    is the one Claude Code's default profile is logged in to, handed out
    only when it is the account's own (the stored login's refresh token, or
    one claude-swap confirmed by asking Anthropic whose it is); otherwise the
    stored login, after taking in a newer one (a renewal claude-swap set
    aside; one Claude Code made in the account's ``cswap run`` profile).

    The login is read, and renewed, under the account's consume lock and
    claude-swap's own lock (the one a switch holds), so no other renewal of
    it and no switch comes in between. When the access token lapses within
    ``renew_within`` seconds (or has lapsed), the login is renewed first,
    the way claude-swap and Claude Code renew it here, so no copy on this
    computer is left with a spent refresh token: the default profile's login
    under Claude Code's own credential locks as well, written back to the
    default profile and the stored login; a stored login written back to
    the store while it still holds the login renewed, as claude-swap's
    consume gate writes one (a login added or removed meanwhile is left as
    it is). A renewal that is not stored is set aside, and the next call
    takes it in. A login renewed meanwhile is taken as it is. A login
    is not renewed while a Claude session runs on it in the account's
    ``cswap run`` profile (that session holds the same refresh token and
    renews it itself the next time it is used); a session on another login
    of the account does not hold this one.

    Raises :class:`AccountNotFoundError` for an unknown account,
    :class:`ConfigError` for an ambiguous email, a foreign
    ``CLAUDE_CONFIG_DIR``, or ``CLAUDE_SECURESTORAGE_CONFIG_DIR`` set (Claude
    Code then keeps its login where claude-swap does not look),
    :class:`ValidationError` for a bad ``renew_within``, an API-key or a
    setup-token account, or a login without scopes (another computer could
    not use it), :class:`CredentialError` when there is no login (``has no
    stored login``; the default profile ``has no login now``), it no longer
    works or cannot be renewed now, or the default profile's login is not
    the account's or cannot be confirmed as the account's yet,
    :class:`CredentialReadError` when it cannot be read now,
    :class:`LockError` while another renewal or a switch does not end within
    :data:`ACCESS_TOKEN_LOCK_TIMEOUT` seconds, and :class:`SessionError`
    while a Claude session holds a login that is due. Never prompts; the
    token is never printed or logged.
    """
    _refuse_foreign_profile()
    window_ms = _renew_window(renew_within)
    if os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR"):
        raise ConfigError(_UNMIRRORED)
    with _quiet():
        slot, email, org_uuid = switcher.resolve_account(identifier)
        record = (switcher._get_sequence_data() or {}).get("accounts", {}).get(slot, {})
    _confirm_live(switcher, slot, email, org_uuid)
    # The rest runs under the account's locks and outside the output lock:
    # it waits for locks and the network, and every other call of this
    # module waits for that lock.
    with _account_locks(switcher, slot):
        # A default login another computer lent (sak.13) is not this
        # account's login here: the stored login is the newest one then.
        live = switcher._live_identity_matches(email, org_uuid)
        if live and not switcher._live_login_is_lent():
            login, renewed = _live_token(switcher, slot, email, org_uuid, window_ms)
        else:
            login, renewed = _stored_token(switcher, slot, email, org_uuid, window_ms)
    return _token_info(slot, email, org_uuid, record, login, renewed)


# -- a profile that runs on an access token only (sak.12) ---------------------------


def _access_profile(config_dir: str | os.PathLike[str]) -> str:
    """The profile folder of a kept access token, as the exact string Claude
    Code runs with (``CLAUDE_CONFIG_DIR``); refuses a place that is not a
    folder of its own, by whatever spelling (:func:`_inside`), before
    anything is written."""
    raw = os.fspath(config_dir)
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError("Name the profile folder.")
    if not os.path.isabs(raw):
        raise ConfigError(f"Name the profile folder by its full path, not {raw}.")
    path = Path(raw)
    for root, what in (
        (get_default_claude_config_home(), "Claude Code's default profile"),
        (get_backup_root(), "claude-swap's own store"),
        (get_legacy_backup_root(), "claude-swap's own store"),
    ):
        if _inside(path, root) or _inside(root, path):
            raise ConfigError(
                f"{raw} is {what}, or inside or above it; name a folder of its own."
            )
    if path.exists() and not path.is_dir():
        raise ConfigError(f"{raw} is not a folder.")
    return raw


_REFRESH_TOKEN_PREFIX = "sk-ant-ort"
"""How Anthropic's refresh tokens start (``sk-ant-ort01-...``)."""


def _checked_token(access_token: object) -> str:
    if (
        not isinstance(access_token, str)
        or not access_token
        or not access_token.isprintable()
        or any(c.isspace() for c in access_token)
    ):
        raise ValidationError("The access token is empty or not one token.")
    if looks_like_api_key(access_token):
        raise ValidationError("This is an API key, not an access token.")
    if access_token.startswith(_REFRESH_TOKEN_PREFIX):
        raise ValidationError(
            "This is a refresh token, not an access token; a refresh token never "
            "goes to another computer."
        )
    return access_token


def _epoch_ms(expires_at: object) -> int:
    moment: datetime | None = None
    if isinstance(expires_at, str):
        try:
            moment = datetime.fromisoformat(expires_at)
        except ValueError:
            moment = None
    if moment is None or moment.tzinfo is None:
        raise ValidationError(
            "expires_at is an ISO 8601 time with its zone "
            f"(2026-10-16T21:47:00Z); not {expires_at!r}."
        )
    return (moment - _EPOCH) // timedelta(milliseconds=1)


def _checked_scopes(scopes: object) -> list[str]:
    if isinstance(scopes, str) or not isinstance(scopes, Sequence):
        raise ValidationError("scopes is a list of the login's scopes.")
    kept = list(scopes)
    if not kept or not all(
        isinstance(s, str) and s and not any(c.isspace() for c in s) for s in kept
    ):
        raise ValidationError(
            "scopes lists the login's scopes (user:inference, ...): Claude Code "
            "reads them to know what the login may do."
        )
    return kept


def _checked_identity(value: object, what: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{what} is text{'' if required else ' or None'}.")
    if required and not value.strip():
        raise ValidationError(f"{what} is empty.")
    return value.strip() or None


@contextlib.contextmanager
def _profile_lock(
    lock_dir: Path, staleness: float, what: str
) -> Iterator[None]:
    """Hold one of Claude Code's locks of a profile (proper-lockfile)."""
    with contextlib.ExitStack() as held:
        try:
            held.enter_context(proper_lockfile(lock_dir, staleness=staleness))
        except ClaudeCodeLockTimeout as e:
            raise LockError(
                f"Claude Code is changing {what} right now; try again in a moment."
            ) from e
        except OSError as e:
            raise CredentialWriteError(
                f"Claude Code's lock of {what} cannot be taken: {e}"
            ) from e
        yield


def _read_keychain(service: str) -> str | None:
    """One Keychain item of this user (``None`` when there is none), with the
    bounded retry of claude-swap's active read (``_ACTIVE_READ_ATTEMPTS``
    tries, ``_ACTIVE_READ_RETRY_DELAY`` apart): a transient lock or timeout
    of ``security`` is ridden out; the last error is raised."""
    account = macos_keychain.keychain_account_name()
    attempt = 1
    while True:
        try:
            return macos_keychain.get_password(service, account)
        except macos_keychain.KEYCHAIN_ERRORS:
            if attempt >= _ACTIVE_READ_ATTEMPTS:
                raise
        attempt += 1
        time.sleep(_ACTIVE_READ_RETRY_DELAY)


def _profile_item(raw: str) -> str | None:
    """The profile's own Keychain item (macOS), which Claude Code reads
    before its ``.credentials.json``; ``None`` when there is none."""
    if Platform.detect() != Platform.MACOS:
        return None
    try:
        return _read_keychain(keychain_service_name(raw))
    except macos_keychain.KEYCHAIN_ERRORS as e:
        raise CredentialReadError(
            f"The Keychain item of the profile {raw} cannot be read now (locked or "
            f"busy); unlock the keychain and retry: {e}"
        ) from e


def _profile_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except (OSError, ValueError) as e:
        raise CredentialReadError(f"Cannot read {path}: {e}") from e


def _refuse_full_login(raw: str, current: str) -> None:
    """A login with a refresh token was not put there by
    :func:`keep_access_login` (someone logged in there): it is never
    replaced or removed, which would lose it."""
    entry = _oauth_entry(current) or {}
    refresh = entry.get("refreshToken")
    if isinstance(refresh, str) and refresh.strip():
        raise ConfigError(
            f"The profile {raw} holds a login of its own (someone logged in there), "
            "so it is left as it is; log out there or use another folder."
        )


def _profile_keys(current: str) -> dict[str, Any]:
    """What Claude Code keeps in the profile's credentials besides the login
    (an MCP server's login, a plugin's secrets)."""
    try:
        data = json.loads(current) if current.strip() else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if key != "claudeAiOauth"}


_TEMPORARY_SUFFIX = ".tmp"


def _temporary_prefix(path: Path) -> str:
    """How :func:`_write_private` names its temporary copy of ``path``."""
    return f".{path.name}."


def _sweep_temporaries(path: Path) -> None:
    """Remove the temporary copies of ``path`` that a write left behind when
    its process died before it could (each holds the token). The caller
    holds the lock that writes of ``path`` are made under."""
    prefix = _temporary_prefix(path)
    try:
        names = os.listdir(path.parent)
    except OSError as e:
        _logger.debug("api: could not look for leftovers in %s: %s", path.parent, e)
        return
    for name in names:
        if (
            name.startswith(prefix)
            and name.endswith(_TEMPORARY_SUFFIX)
            and len(name) > len(prefix) + len(_TEMPORARY_SUFFIX)
        ):
            try:
                os.unlink(path.parent / name)
            except OSError as e:
                _logger.debug("api: could not remove %s: %s", path.parent / name, e)


def _write_private(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step, readable by its owner only,
    on the disk (its folder's entry too) before this returns; line ends are
    written as ``text`` has them, on every system. The temporary copy never
    outlives the call; one a dead process left behind goes with
    :func:`_sweep_temporaries`."""
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=_temporary_prefix(path), suffix=_TEMPORARY_SUFFIX
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            sync_file(f.fileno())
        replace_with_retry(temporary, path)
        temporary = None
        sync_folder(path.parent)
    except OSError as e:
        raise CredentialWriteError(f"Could not write {path}: {e}") from e
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary)


def _remove_profile_item(raw: str) -> None:
    try:
        macos_keychain.delete_password(
            keychain_service_name(raw), macos_keychain.keychain_account_name()
        )
    except macos_keychain.KEYCHAIN_ERRORS as e:
        raise CredentialWriteError(
            f"Could not remove the Keychain item of the profile {raw}, which would "
            f"keep Claude Code from reading the new login: {e}"
        ) from e


def _touch_newer(path: Path) -> None:
    """Give the file a newer modification time: a running Claude Code checks
    it before each request and reads its login again when it moved."""
    try:
        status = path.stat()
        newer = max(time.time_ns(), status.st_mtime_ns + 1_000_000)
        os.utime(path, ns=(status.st_atime_ns, newer))
    except OSError:
        _logger.debug("api: could not touch %s", path, exc_info=True)


def _keep_access(raw: str, entry: Mapping[str, Any]) -> None:
    """Make ``entry`` the profile's login, where Claude Code reads it: its
    ``.credentials.json``, with the profile's Keychain item removed (macOS),
    which would hide the file. Under Claude Code's own lock for the
    profile's stored credentials; what else Claude Code keeps there (an MCP
    server's login, also when it moved it into the Keychain item) stays."""
    path = Path(raw)
    target = path / ".credentials.json"
    with _profile_lock(
        storage_write_lock_dir(path), STORAGE_WRITE_STALENESS_S, f"the login of {raw}"
    ):
        item = _profile_item(raw)
        current = item if item is not None else _profile_file(target)
        _refuse_full_login(raw, current)
        kept = _profile_keys(current)
        kept["claudeAiOauth"] = dict(entry)
        _sweep_temporaries(target)
        _write_private(target, json.dumps(kept))
        if item is not None:
            _remove_profile_item(raw)
            _touch_newer(target)


def _default_theme() -> str:
    """The default profile's theme, so the new profile looks the same."""
    try:
        data = json.loads(get_default_global_config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "dark"
    theme = data.get("theme") if isinstance(data, dict) else None
    return theme if isinstance(theme, str) and theme else "dark"


def _keep_identity(directory: Path, account: Mapping[str, str | None]) -> None:
    """Record whose login the profile holds in its ``.claude.json``
    (``oauthAccount``, as Claude Code does), under Claude Code's lock for
    that file; what Claude Code recorded about the same account stays."""
    config_path = directory / ".claude.json"
    given = {key: value for key, value in account.items() if value is not None}
    with _profile_lock(
        directory / ".claude.json.lock", CONFIG_STALENESS_S, str(config_path)
    ):
        if config_path.is_symlink() or (config_path.exists() and not config_path.is_file()):
            raise ConfigError(f"{config_path} is not a file of the profile's own.")
        existing: dict[str, Any] = {}
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            loaded = {}
        except (OSError, ValueError):
            loaded = {}  # a broken file is replaced, as a session profile's is
        if isinstance(loaded, dict):
            existing = loaded
        current = existing.get("oauthAccount")
        same = (
            isinstance(current, dict)
            and current.get("emailAddress") == given.get("emailAddress")
            and (current.get("organizationUuid") or None)
            == given.get("organizationUuid")
        )
        merged = {**current, **given} if same and isinstance(current, dict) else given
        if (
            existing.get("oauthAccount") == merged
            and existing.get("hasCompletedOnboarding") is True
            and existing.get("theme")
        ):
            return
        existing["oauthAccount"] = merged
        existing["hasCompletedOnboarding"] = True
        if not existing.get("theme"):
            existing["theme"] = _default_theme()
        try:
            atomic_write_json(config_path, existing)
        except OSError as e:
            raise ConfigError(f"Could not write {config_path}: {e}") from e


class _SharingHost:
    """What :class:`SessionManager`'s sharing reads of a switcher, for a
    profile that is no claude-swap account: no account store is opened."""

    def __init__(self) -> None:
        self.platform = Platform.detect()
        self.backup_dir = get_backup_root()
        self._logger = _logger


def keep_access_login(
    config_dir: str | os.PathLike[str],
    *,
    access_token: str,
    expires_at: str,
    scopes: Sequence[str],
    subscription_type: str | None,
    rate_limit_tier: str | None,
    email: str,
    organization_uuid: str | None,
    organization: str | None,
    account_uuid: str | None,
    share_history: bool = True,
) -> None:
    """Make ``config_dir`` a Claude Code profile signed in with an access
    token alone (sak.12): another computer's :func:`access_token`, with no
    refresh token, so Claude Code here uses it and never renews it (a new
    one comes from that computer). The token it had is replaced; a running
    Claude Code takes the new one at its next request.

    ``config_dir`` is the exact string Claude Code runs with
    (``CLAUDE_CONFIG_DIR``), a full path; the folder is made when it is
    missing. The login goes into its ``.credentials.json`` (``refreshToken``
    null, 0600, replaced in one step), under Claude Code's own lock for the
    profile's stored credentials; on macOS the profile's Keychain item is
    removed, since Claude Code reads it first (what Claude Code keeps there
    besides a login, such as an MCP server's login, moves to the file). The
    account goes into its ``.claude.json`` (``oauthAccount``), and the
    profile shares settings, MCP servers and, with ``share_history``,
    conversation history with the default profile, as ``cswap run``
    profiles do. It is no claude-swap account and is never registered.

    ``expires_at`` is ISO 8601 with its zone, as :func:`access_token` gives
    it. Refuses Claude Code's default profile and claude-swap's own store
    (and anything inside or above them, by whatever spelling: letter case,
    macOS's ``/System/Volumes/Data``, a symlink, ``..``) with
    :class:`ConfigError` before anything is written, and a profile that
    holds a login of its own (with a refresh token: someone logged in there)
    the same way; :class:`ValidationError` for a value that is not what it
    should be (an API key or a refresh token, an empty token, a time without
    a zone, no scopes); :class:`SessionError` for ``share_history`` on Windows;
    :class:`CredentialReadError` when the Keychain item cannot be read now;
    :class:`CredentialWriteError` when the login cannot be written, or the
    item cannot be removed; :class:`LockError` while Claude Code holds its
    lock. Never prompts; the token is never printed or logged.
    """
    raw = _access_profile(config_dir)
    entry = {
        "accessToken": _checked_token(access_token),
        "refreshToken": None,
        "expiresAt": _epoch_ms(expires_at),
        "scopes": _checked_scopes(scopes),
        "subscriptionType": _checked_identity(subscription_type, "subscription_type"),
        "rateLimitTier": _checked_identity(rate_limit_tier, "rate_limit_tier"),
    }
    account = {
        "accountUuid": _checked_identity(account_uuid, "account_uuid"),
        "emailAddress": _checked_identity(email, "email", required=True),
        "organizationUuid": _checked_identity(organization_uuid, "organization_uuid"),
        "organizationName": _checked_identity(organization, "organization"),
    }
    if share_history and Platform.detect() == Platform.WINDOWS:
        raise SessionError("share_history is not supported on Windows")
    path = Path(raw)
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as e:
        raise ConfigError(f"Could not make the profile folder {raw}: {e}") from e
    _keep_access(raw, entry)
    _keep_identity(path, account)
    with _quiet():
        sharing = SessionManager(cast(ClaudeAccountSwitcher, _SharingHost()))
        sharing._sync_sharing(path, True, share_history)


def forget_access_login(config_dir: str | os.PathLike[str]) -> None:
    """Remove the access token :func:`keep_access_login` kept in
    ``config_dir`` (sak.12), under Claude Code's own lock for the profile's
    stored credentials: the login leaves its ``.credentials.json`` and, on
    macOS, the profile's Keychain item is removed, as
    :func:`forget_profile_login` does. What Claude Code keeps there besides
    a login (an MCP server's login, also when it moved it into the Keychain
    item) stays in the file, as :func:`keep_access_login` keeps it; the file
    goes when nothing else is left. The folder and the rest of the profile
    stay; a missing folder is fine.

    Refuses the places :func:`keep_access_login` refuses and a profile that
    holds a login of its own (someone logged in there) with
    :class:`ConfigError`; raises :class:`CredentialReadError` when the
    Keychain item cannot be read now, :class:`CredentialError` when the
    login stays, and :class:`LockError` while Claude Code holds its lock.
    """
    raw = _access_profile(config_dir)
    path = Path(raw)
    if not path.is_dir():
        return
    target = path / ".credentials.json"
    with _profile_lock(
        storage_write_lock_dir(path), STORAGE_WRITE_STALENESS_S, f"the login of {raw}"
    ):
        item = _profile_item(raw)
        current = item if item is not None else _profile_file(target)
        _refuse_full_login(raw, current)
        _sweep_temporaries(target)
        kept = _profile_keys(current)
        if not kept:
            forget_config_dir_login(raw)
            return
        _write_private(target, json.dumps(kept))
        if item is not None:
            _remove_profile_item(raw)
            _touch_newer(target)


# -- Claude Code's default login on a device (sak.13) -------------------------------

DEFAULT_LOGIN_LOCK_TIMEOUT = 15.0
"""How long :func:`use_access_login` and :func:`forget_default_access_login`
wait for claude-swap's own lock: a switch holds it for its local writes."""


@dataclass(frozen=True, slots=True)
class DefaultLoginInfo:
    """What Claude Code's default profile is signed in with now
    (:func:`default_login`, sak.13). Never a token."""

    kind: str
    """``"none"``: no login. ``"access"``: an access token alone (no
    refresh token, its expiry recorded, not a setup-token) of an account
    that is none of claude-swap's accounts here: what
    :func:`use_access_login` puts there. ``"own"``: a login of this
    computer's own: a refresh token (someone logged in here, or one of
    claude-swap's accounts is active), a setup-token, an access token
    without its expiry, or an access token alone of one of claude-swap's
    accounts here. ``"api_key"``: a managed API key."""
    email: str | None
    """Its account (``~/.claude.json`` ``oauthAccount``), when known; never
    for ``"none"`` or ``"api_key"``."""
    organization_uuid: str | None
    expires_at: str | None
    """ISO 8601 UTC, when the login records it: for ``"access"`` when its
    access token lapses (the login stops working then), for ``"own"`` when
    the login itself lapses."""
    leftover_access: bool = False
    """For ``"own"`` and ``"api_key"``: one of the two stores still holds an
    access token alone as :func:`use_access_login` writes it, next to the
    computer's own login or API key: a leftover
    :func:`forget_default_access_login` removes (a stale
    ``.credentials.json`` next to an own login in the Keychain item, or a
    Keychain item hiding an own login in the file from Claude Code, also when
    it keeps an MCP server's login besides)."""
    own_kind: str | None = None
    """What a replace (:func:`use_access_login` with ``replace_own``,
    sak.14) does to the computer's own login or key. ``"login"``: a login
    made here (a refresh token, a setup-token, an access token of its own)
    whose account is none of claude-swap's accounts here; a replace removes
    it. ``"account"``: the account ``~/.claude.json`` records is one of
    claude-swap's accounts here; a replace switches away from it as ``cswap
    switch`` does, so its login stays in claude-swap's store. ``"api_key"``:
    a managed API key, also one next to a login; never replaced. ``None``
    for ``"none"`` and ``"access"``. When claude-swap's account list cannot
    be read now, an own login reads as ``"login"`` (a replace refuses until
    it can be read)."""


class OwnLoginError(ConfigError):
    """Claude Code's default profile holds a login of its own (a refresh
    token: someone logged in there, or claude-swap's own account; a
    setup-token; an access token alone of one of claude-swap's accounts
    here) or a managed API key: it is never replaced or removed for an
    access-only login (sak.13), unless the caller asks for a replace
    (``replace_own``, sak.14); a managed API key never is."""


def _parsed(text: str | None) -> object:
    """``text`` as Claude Code parses a store of its credentials; ``None``
    when the store holds nothing it can read (none, empty, not JSON)."""
    if text is None or not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _login_kind(text: str | None) -> str:
    """What one store of Claude Code's credentials holds, as
    :class:`DefaultLoginInfo` names it. An empty store, ``{}``, what Claude
    Code keeps besides a login (an MCP server's login) and an emptied login
    hold none."""
    if text is None:
        return "none"
    if looks_like_api_key(text):
        return "api_key"
    entry = _oauth_entry(text)
    if entry is None:
        return "none"
    refresh = entry.get("refreshToken")
    if isinstance(refresh, str) and refresh.strip():
        return "own"
    access = entry.get("accessToken")
    if not isinstance(access, str) or not access.strip():
        return "none"
    return "access" if oauth.is_access_only_login(text) else "own"


@dataclass(frozen=True)
class _DefaultStores:
    """What Claude Code's default profile holds now, store by store."""

    item: str | None
    """Its macOS Keychain item; ``None`` when there is none (or no Keychain)."""
    file: str | None
    """Its ``.credentials.json``; ``None`` when there is none."""
    api_key: bool
    """A managed API key is set (the ``Claude Code`` Keychain item, or
    ``primaryApiKey``)."""
    config: dict[str, Any] | None
    """``~/.claude.json``; ``None`` when there is none, or it is broken."""
    config_broken: bool
    """``~/.claude.json`` is there but holds no JSON object."""
    managed: bool = False
    """The account ``~/.claude.json`` records is one of claude-swap's
    accounts here (:func:`_claude_swap_slot`; looked up only when a store
    holds an access token alone)."""

    @property
    def effective(self) -> str | None:
        """What Claude Code reads its login from: the Keychain item when it
        holds JSON, else the file (an empty item does not hide the file)."""
        return self.item if _parsed(self.item) is not None else self.file

    @property
    def account(self) -> dict[str, Any]:
        """The account ``~/.claude.json`` records (``oauthAccount``); empty
        when it records none."""
        account = (self.config or {}).get("oauthAccount")
        return cast(dict[str, Any], account) if isinstance(account, dict) else {}

    @property
    def kind(self) -> str:
        """The default profile's login, as :class:`DefaultLoginInfo` names
        it. A login of its own in either store counts, also one Claude Code
        does not read now: a write would replace it there. So does an access
        token alone of one of claude-swap's accounts here: that account's
        login of this computer's own."""
        stores = tuple(
            "own" if kind == "access" and self.managed else kind
            for kind in (_login_kind(self.item), _login_kind(self.file))
        )
        if "own" in stores:
            return "own"
        if self.api_key or "api_key" in stores:
            return "api_key"
        return "access" if _login_kind(self.effective) == "access" else "none"


def _leftover_stores(stores: _DefaultStores) -> tuple[str, ...]:
    """The stores (``"item"``, ``"file"``) that hold an access token alone
    as :func:`use_access_login` writes it (:func:`oauth.is_lent_login`)
    next to a login of the computer's own in the other store, or a managed
    API key: leftovers no login of this computer depends on. An access
    token alone that is itself the reason for ``"own"`` (one of
    claude-swap's accounts here) is none."""
    own_item = _login_kind(stores.item) in ("own", "api_key")
    own_file = _login_kind(stores.file) in ("own", "api_key")
    held = (
        ("item", stores.item, own_file or stores.api_key),
        ("file", stores.file, own_item or stores.api_key),
    )
    return tuple(
        name for name, text, beside in held if beside and oauth.is_lent_login(text)
    )


def _read_if_there(path: Path) -> str | None:
    """The file's text as it is written, its line ends too; ``None`` when
    there is no such file."""
    try:
        with path.open(encoding="utf-8", newline="") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise CredentialReadError(f"Cannot read {path} now: {e}") from e


def _default_item(service: str) -> str | None:
    """One of Claude Code's Keychain items of the default profile; ``None``
    when there is none."""
    try:
        return _read_keychain(service)
    except macos_keychain.KEYCHAIN_ERRORS as e:
        raise CredentialReadError(
            "Claude Code's default login in the Keychain cannot be read now (the "
            f"Keychain is locked or busy); unlock it and try again: {e}"
        ) from e


def _default_credentials_path() -> Path:
    return get_default_claude_config_home() / ".credentials.json"


def _claude_swap_root() -> Path:
    """claude-swap's store, with a legacy store moved to it first (Linux and
    WSL), as opening claude-swap moves it (``ClaudeAccountSwitcher``): a lock
    or a read here never leaves the new store beside an unmoved legacy one,
    which every later opening would refuse."""
    root = get_backup_root()
    try:
        migrate_legacy_backup_dir(root)
    except MigrationError as e:
        raise ConfigError(
            f"claude-swap's store cannot be used now: {e}"
        ) from e
    return root


def _claude_swap_slot(account: Mapping[str, Any]) -> str | None:
    """Which of claude-swap's accounts here ``account`` (an ``oauthAccount``)
    is, matched as :meth:`ClaudeAccountSwitcher.current_account_number`
    matches the live login (email and organization); ``None`` for none.
    claude-swap's account list is only read: one that cannot be read now is
    refused (:class:`ConfigError`), never taken for no accounts."""
    email = account.get("emailAddress")
    if not isinstance(email, str) or not email:
        return None
    path = _claude_swap_root() / "sequence.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise ConfigError(
            f"claude-swap's account list {path} cannot be read now ({e}), so it "
            "cannot be told whether Claude Code's default login is one of its "
            "accounts; try again in a moment."
        ) from e
    data = _parsed(text)
    accounts = data.get("accounts", {}) if isinstance(data, dict) else None
    if not isinstance(accounts, dict) or not all(
        isinstance(record, dict) for record in accounts.values()
    ):
        raise ConfigError(
            f"claude-swap's account list {path} could not be parsed, so it cannot "
            "be told whether Claude Code's default login is one of its accounts. "
            "Repair or move the file, then try again."
        )
    organization = account.get("organizationUuid", "") or ""
    return ClaudeAccountSwitcher._find_account_slot(
        cast(dict[str, Any], data), email, organization
    )


def _read_default_stores() -> _DefaultStores:
    """What Claude Code's default profile holds, read where Claude Code
    opened by hand reads it; what cannot be read now is refused
    (:class:`CredentialReadError`, :class:`ConfigError`), never taken for
    empty. For an access token alone, claude-swap's account list tells
    whether its account is one of claude-swap's here."""
    item: str | None = None
    api_key = False
    if Platform.detect() == Platform.MACOS:
        item = _default_item(CLAUDE_CODE_KEYCHAIN_SERVICE)
        managed = _default_item(CLAUDE_CODE_MANAGED_KEYCHAIN_SERVICE)
        api_key = bool(managed and managed.strip())
    file = _read_if_there(_default_credentials_path())
    text = _read_if_there(get_default_global_config_path())
    config: dict[str, Any] | None = None
    broken = False
    if text is not None:
        loaded = _parsed(text) if text.strip() else {}
        if isinstance(loaded, dict):
            config = cast(dict[str, Any], loaded)
        else:
            broken = True
    key = (config or {}).get("primaryApiKey")
    api_key = api_key or (isinstance(key, str) and bool(key.strip()))
    stores = _DefaultStores(
        item=item, file=file, api_key=api_key, config=config, config_broken=broken
    )
    if stores.kind in ("own", "api_key") or "access" not in (
        _login_kind(item),
        _login_kind(file),
    ):
        return stores
    return replace(stores, managed=_claude_swap_slot(stores.account) is not None)


def _instant_or_none(epoch_ms: object) -> str | None:
    """Epoch milliseconds as :func:`_instant` gives them; ``None`` for what
    is no time."""
    if isinstance(epoch_ms, bool) or not isinstance(epoch_ms, (int, float)):
        return None
    try:
        return _instant(int(epoch_ms))
    except (OverflowError, ValueError):
        return None


def _login_lapse(credentials: str) -> str | None:
    """When the login itself lapses (:func:`login_expires_at_iso`); ``None``
    for a value that is no time."""
    try:
        return login_expires_at_iso(credentials)
    except (OverflowError, OSError, ValueError):
        return None


def _own_kind(stores: _DefaultStores, *, strict: bool = False) -> str | None:
    """:attr:`DefaultLoginInfo.own_kind` of what the default profile holds.
    An own login whose account cannot be told because claude-swap's account
    list cannot be read now reads as ``"login"``, unless ``strict``
    (a replace), which raises the :class:`ConfigError`."""
    stored = (_login_kind(stores.item), _login_kind(stores.file))
    if stores.api_key or "api_key" in stored:
        return "api_key"
    if stores.kind != "own":
        return None
    if stores.managed:
        return "account"
    try:
        slot = _claude_swap_slot(stores.account)
    except ConfigError:
        if strict:
            raise
        return "login"
    return "login" if slot is None else "account"


def _default_info(stores: _DefaultStores) -> DefaultLoginInfo:
    kind = stores.kind
    leftover = bool(_leftover_stores(stores))
    own_kind = _own_kind(stores)
    if kind not in ("access", "own"):
        return DefaultLoginInfo(
            kind=kind,
            email=None,
            organization_uuid=None,
            expires_at=None,
            leftover_access=leftover,
            own_kind=own_kind,
        )
    held = (stores.effective, stores.item, stores.file)
    own = next((t for t in held if _login_kind(t) == "own"), None)
    if own is not None:
        expires = _login_lapse(own)
    else:  # an access token alone: the login lapses with it
        alone = next((t for t in held if _login_kind(t) == "access"), None)
        entry = _oauth_entry(alone or "") or {}
        expires = _instant_or_none(entry.get("expiresAt"))
    account = stores.account
    return DefaultLoginInfo(
        kind=kind,
        email=_text(account.get("emailAddress")),
        organization_uuid=_text(account.get("organizationUuid")),
        expires_at=expires,
        leftover_access=leftover,
        own_kind=own_kind,
    )


def _refuse_other_store() -> None:
    """Read and write only where Claude Code opened by hand keeps its login,
    as :func:`access_token` refuses anything else."""
    _refuse_foreign_profile()
    if os.environ.get("CLAUDE_SECURESTORAGE_CONFIG_DIR"):
        raise ConfigError(_UNMIRRORED)


def default_login() -> DefaultLoginInfo:
    """What Claude Code's default profile is signed in with now (sak.13):
    the login Claude Code opened by hand uses (``claude`` in a terminal, an
    editor's Claude agent). Changes nothing (but for moving a legacy store
    on Linux/WSL, as opening claude-swap does) and carries no token.

    Read where Claude Code reads it: on macOS the Keychain item ``Claude
    Code-credentials`` first, then ``~/.claude/.credentials.json``, and a
    managed API key after both; the account from ``~/.claude.json``
    (``oauthAccount``). An empty store, one that holds ``{}`` or only what
    Claude Code keeps besides a login (an MCP server's login), and an
    emptied login read as ``"none"``. An access token alone in the store
    Claude Code reads (no refresh token: null, missing or ``""``; its
    ``expiresAt`` recorded; not a setup-token's scopes) reads as
    ``"access"``, unless its account is one of claude-swap's accounts here
    (as :meth:`ClaudeAccountSwitcher.current_account_number` matches the
    live login: email and organization), which makes it that account's
    login and ``"own"``. A login with a refresh token, a setup-token, or an
    access token without its ``expiresAt`` reads as ``"own"``, in either
    store (a write would replace it there too); a managed API key as
    ``"api_key"``. ``leftover_access`` says when an access token alone is
    left in one store next to such a login or key
    (:func:`forget_default_access_login` removes it), and ``own_kind``
    (sak.14) what a replace would do to the login or key of the computer's
    own. Each Keychain read rides out a transient failure with
    claude-swap's bounded retry.

    Raises :class:`CredentialReadError` when a store cannot be read now (a
    locked Keychain) and :class:`ConfigError` with a foreign
    ``CLAUDE_CONFIG_DIR`` or ``CLAUDE_SECURESTORAGE_CONFIG_DIR`` set, or
    when claude-swap's account list cannot be read for an access token
    alone. Never prompts.
    """
    _refuse_other_store()
    return _default_info(_read_default_stores())


@contextlib.contextmanager
def _default_login_locks() -> Iterator[None]:
    """Hold, in claude-swap's order, claude-swap's own lock (every switch
    holds it), Claude Code's credential locks of the default profile (its
    renewals hold them), its storage-write lock (its other changes of the
    login, such as an MCP server's login) and the lock of
    ``~/.claude.json``: no switch, renewal or write of Claude Code's comes
    in between."""
    store = _claude_swap_root()
    own = FileLock(store / ".lock", timeout=DEFAULT_LOGIN_LOCK_TIMEOUT)
    try:
        store.mkdir(mode=0o700, parents=True, exist_ok=True)
        taken = own.acquire()
    except OSError as e:
        raise LockError(f"claude-swap's own lock cannot be taken: {e}") from e
    if not taken:
        raise LockError(
            "claude-swap is switching Claude Code's default login right now; try "
            "again in a moment."
        )
    home = get_default_claude_config_home()
    config = get_default_global_config_path()
    # Claude Code's two credential locks in its own order, then its
    # storage-write lock (its renewal takes them so), then ~/.claude.json's.
    locks = (
        (home / ".oauth_refresh.lock", CREDENTIALS_STALENESS_S, "its login"),
        (home.with_name(f"{home.name}.lock"), CREDENTIALS_STALENESS_S, "its login"),
        (storage_write_lock_dir(home), STORAGE_WRITE_STALENESS_S, "its login"),
        (config.with_name(f"{config.name}.lock"), CONFIG_STALENESS_S, str(config)),
    )
    try:
        with contextlib.ExitStack() as held:
            for lock_dir, staleness, what in locks:
                held.enter_context(_profile_lock(lock_dir, staleness, what))
            yield
    finally:
        own.release()


_OWN_API_KEY = (
    "Claude Code's default login on this computer uses an API key of its "
    "own, so it is left as it is; nothing was changed."
)


def _refuse_own_default(stores: _DefaultStores) -> None:
    """A login of this computer's own, or a managed API key, is never
    replaced or removed (:class:`OwnLoginError`)."""
    kind = stores.kind
    if kind == "own":
        raise OwnLoginError(
            "Claude Code's default login on this computer is signed in on its own "
            "(someone logged in here, or one of claude-swap's own accounts is in "
            "use), so it is left as it is; nothing was changed."
        )
    if kind == "api_key":
        raise OwnLoginError(_OWN_API_KEY)


def _refuse_broken_config(stores: _DefaultStores) -> None:
    """``~/.claude.json`` carries all of Claude Code's state: one that cannot
    be read is never written over."""
    if stores.config_broken:
        raise ConfigError(
            f"{get_default_global_config_path()} is not a file Claude Code can read "
            "(it holds no JSON object), so the account cannot be recorded there; "
            "nothing was changed. Repair or move the file, then try again."
        )


def _refuse_claude_swap_account(account: Mapping[str, Any]) -> None:
    """An access token alone of one of claude-swap's accounts here never goes
    into the default profile: :func:`default_login` would read it as that
    account's login of this computer's own, so it could not be replaced or
    removed again; claude-swap's own login of that account is the one to
    use here (``cswap switch``)."""
    slot = _claude_swap_slot(account)
    if slot is not None:
        raise ConfigError(
            f"{account.get('emailAddress')} is account {slot} of claude-swap on this "
            "computer, so Claude Code's default login here uses claude-swap's own "
            f"login of it (`cswap switch {slot}`), not one another computer sent; "
            "nothing was changed."
        )


def _profile_account(directory: Path) -> dict[str, Any] | None:
    """The account a profile's ``.claude.json`` records (``oauthAccount``,
    as Claude Code resolves the file); ``None`` without one."""
    legacy = directory / ".config.json"
    path = legacy if legacy.exists() else directory / ".claude.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    account = data.get("oauthAccount") if isinstance(data, dict) else None
    if not isinstance(account, dict) or not _text(account.get("emailAddress")):
        return None
    return cast(dict[str, Any], account)


def _lent_login(raw: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """The access-only login the profile ``raw`` holds (as Claude Code reads
    it there, under the profile's storage-write lock) with ``refreshToken``
    null, and the account it records; refuses anything else
    (:class:`ConfigError`) before anything is written. What goes into the
    default profile is a login claude-swap tells for lent
    (:func:`oauth.is_lent_login`), so it never stores or backs it up."""
    path = Path(raw)
    if not path.is_dir():
        raise ConfigError(f"The profile {raw} does not exist, so it holds no login.")
    with _profile_lock(
        storage_write_lock_dir(path), STORAGE_WRITE_STALENESS_S, f"the login of {raw}"
    ):
        item = _profile_item(raw)
        text = item if _parsed(item) is not None else _read_if_there(
            path / ".credentials.json"
        )
    entry = dict(_oauth_entry(text or "") or {})
    entry["refreshToken"] = None
    if _login_kind(text) != "access" or not oauth.is_lent_login(
        json.dumps({"claudeAiOauth": entry})
    ):
        raise ConfigError(
            f"The profile {raw} holds no login on an access token alone with its "
            "scopes (it holds none, one with a refresh token, or one without its "
            "scopes), so it cannot be used; nothing was changed."
        )
    account = _profile_account(path)
    if account is None:
        raise ConfigError(
            f"The profile {raw} does not record whose login it holds (no "
            "oauthAccount in its .claude.json); nothing was changed."
        )
    return entry, account


def _held_login(stores: _DefaultStores) -> dict[str, Any] | None:
    """The login Claude Code reads in the default profile, with a refresh
    token that is missing or ``""`` written as null."""
    entry = _oauth_entry(stores.effective or "")
    if entry is None:
        return None
    held = dict(entry)
    refresh = held.get("refreshToken")
    if not isinstance(refresh, str) or not refresh.strip():
        held["refreshToken"] = None
    return held


def _with_login(stores: _DefaultStores, entry: Mapping[str, Any]) -> str:
    """The default profile's credentials with ``entry`` as their login: what
    Claude Code keeps there besides a login (an MCP server's login, a
    plugin's secrets) stays; what belongs to another login (its device
    enrollment) does not."""
    base = _parsed(stores.effective)
    kept: dict[str, Any] = {}
    if isinstance(base, dict):
        kept = {
            key: value
            for key, value in cast(dict[str, Any], base).items()
            if key not in ACCOUNT_CREDENTIAL_KEYS
        }
    kept["claudeAiOauth"] = dict(entry)
    return json.dumps(kept)


class _DefaultProfileHost:
    """What :class:`CredentialStore` reads of a switcher, for writing Claude
    Code's default login as a switch writes it: no account store is
    opened."""

    def __init__(self) -> None:
        self.platform = Platform.detect()
        self.credentials_dir = get_backup_root() / "credentials"
        self._logger = _logger


def _write_default_login(credentials: str) -> None:
    """Write Claude Code's default login where a switch writes it: on macOS
    the Keychain item when it takes it, and an existing
    ``.credentials.json`` too, so running sessions read it again; else the
    file, with any stale Keychain item removed. The caller holds
    :func:`_default_login_locks`."""
    CredentialStore(_DefaultProfileHost())._write_oauth_credentials(credentials)


def _put_back(stores: _DefaultStores) -> None:
    """Best effort: put back what Claude Code's default profile held before
    a write that could not be finished, store by store, so a Keychain that
    takes no write does not keep the file from being put back."""
    if Platform.detect() == Platform.MACOS:
        account = macos_keychain.keychain_account_name()
        try:
            if stores.item is None:
                macos_keychain.delete_password(CLAUDE_CODE_KEYCHAIN_SERVICE, account)
            else:
                macos_keychain.set_password(
                    CLAUDE_CODE_KEYCHAIN_SERVICE, account, stores.item
                )
        except macos_keychain.KEYCHAIN_ERRORS as e:
            _logger.warning(
                "api: Claude Code's Keychain item could not be put back as it was: %s",
                e,
            )
    path = _default_credentials_path()
    try:
        if stores.file is None:
            path.unlink(missing_ok=True)
        else:
            _write_private(path, stores.file)
    except (ClaudeSwitchError, OSError) as e:
        _logger.warning("api: %s could not be put back as it was: %s", path, e)


def _write_default_config(config: Mapping[str, Any]) -> None:
    """Replace ``~/.claude.json`` with ``config`` in one step (through a
    symlink, never over it), owner-only, as a switch writes it. The caller
    holds its lock (:func:`_default_login_locks`)."""
    path = get_default_global_config_path()
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    _sweep_temporaries(target)
    _write_private(target, json.dumps(config, indent=2))


def _record_default_account(stores: _DefaultStores, account: Mapping[str, Any]) -> bool:
    """Record whose login the default profile holds in ``~/.claude.json``:
    ``oauthAccount``, merged as :func:`keep_access_login` merges it (what
    Claude Code recorded about the same account stays), and
    ``hasCompletedOnboarding``; nothing else changes. Whether it wrote."""
    config = dict(stores.config or {})
    given = {key: value for key, value in account.items() if value is not None}
    current = config.get("oauthAccount")
    same = (
        isinstance(current, dict)
        and current.get("emailAddress") == given.get("emailAddress")
        and (current.get("organizationUuid") or None)
        == (given.get("organizationUuid") or None)
    )
    merged = {**current, **given} if same and isinstance(current, dict) else given
    if current == merged and config.get("hasCompletedOnboarding") is True:
        return False
    config["oauthAccount"] = merged
    config["hasCompletedOnboarding"] = True
    _write_default_config(config)
    return True


def use_access_login(
    config_dir: str | os.PathLike[str], *, replace_own: bool = False
) -> bool:
    """Sign Claude Code's default profile in with the access-only login the
    profile ``config_dir`` holds (sak.13): one :func:`keep_access_login`
    keeps (its ``claudeAiOauth`` with ``refreshToken`` null, and its
    ``.claude.json`` ``oauthAccount``), so that Claude Code opened by hand
    on this computer uses that account. Another access-only login there is
    replaced (that is a switch); a running Claude Code takes the new one at
    its next request.

    With ``replace_own`` (sak.14; only with the user's OK) a login of this
    computer's own there gives way too, as :attr:`DefaultLoginInfo.own_kind`
    says. A login made here (``"login"``) is removed from every store
    Claude Code reads: its Keychain item and ``.credentials.json`` lose it
    (with its device enrollment; what Claude Code keeps there besides a
    login stays), and ``~/.claude.json`` records the new account in place
    of its account. While one of claude-swap's accounts is active
    (``"account"``) its live login is kept in claude-swap's store first, as
    ``cswap switch`` keeps it when it leaves the account, so the account
    stays and ``cswap switch`` goes back to it. A managed API key
    (``"api_key"``, also one next to a login) is never replaced. A store
    that still holds the own login once the new one is written is cleared
    of it; when that fails, both stores are put back as they were (best
    effort) and the error is raised. Without ``replace_own``, a login of
    the computer's own is refused as in sak.13.

    The login is written where ``cswap switch`` writes the default login: on
    macOS the Keychain item ``Claude Code-credentials`` when it takes it,
    and an existing ``~/.claude/.credentials.json`` too, so running sessions
    read it again; else that file, with any stale Keychain item removed.
    What Claude Code keeps there besides a login (an MCP server's login, a
    plugin's secrets) stays; an empty item, or one with only those, is
    replaced in place. ``~/.claude.json`` gets the account (``oauthAccount``,
    merged as :func:`keep_access_login` merges it) and
    ``hasCompletedOnboarding``; nothing else there changes. All of it under
    claude-swap's own lock, Claude Code's credential locks of the default
    profile, its storage-write lock and the lock of ``~/.claude.json``, and
    the source profile's storage-write lock while it is read. If the
    account cannot be recorded, the login is put back as it was (best
    effort) and the error is raised.

    Returns False when the default profile held that very login already
    (and its account), True when it wrote. Raises :class:`OwnLoginError`
    before anything is written when the default profile holds a login of
    its own (``"own"`` in :func:`default_login`: a refresh token, a
    setup-token or an access token without its expiry in either store, or
    an access token alone of one of claude-swap's accounts here) and
    ``replace_own`` is not given, or a managed API key, which is never
    touched; :class:`ConfigError` when ``config_dir`` is the default
    profile, claude-swap's store, inside or above them, not a full path, or
    holds no access-only login or no account, when that account is one of
    claude-swap's accounts here (its own login of it is the one to use),
    when ``~/.claude.json`` or claude-swap's account list cannot be read,
    and with a foreign ``CLAUDE_CONFIG_DIR`` or
    ``CLAUDE_SECURESTORAGE_CONFIG_DIR``;
    :class:`CredentialReadError` when a store cannot be read now (a locked
    Keychain); :class:`CredentialWriteError` when the login or the account
    cannot be written, or the own login cannot be kept or removed;
    :class:`LockError` while Claude Code or a switch holds a lock. Never
    prompts; the token is never printed or logged.
    """
    _refuse_other_store()
    raw = _access_profile(config_dir)
    switcher: ClaudeAccountSwitcher | None = None
    provenance: dict[str, Any] = {}
    if replace_own:
        _claude_swap_root()  # a store it cannot move is refused as below
        switcher = open_switcher()
        # Whose the live login is may take a network call, and none runs
        # under the locks: asked before them, as a switch asks.
        with _quiet():
            provenance = switcher._prefetch_live_identity()
    with _default_login_locks():
        entry, account = _lent_login(raw)
        stores = _read_default_stores()
        own_kind = _own_kind(stores, strict=True) if replace_own else None
        if own_kind == "api_key":
            raise OwnLoginError(_OWN_API_KEY)
        if not replace_own:
            _refuse_own_default(stores)
        _refuse_broken_config(stores)
        _refuse_claude_swap_account(account)
        if own_kind == "account" and switcher is not None:
            _keep_account_login(switcher, stores.effective, provenance)
        wrote = _held_login(stores) != entry
        removed = False
        if wrote:
            _write_default_login(_with_login(stores, entry))
        try:
            if own_kind is not None:
                removed = _remove_own_logins()
            recorded = _record_default_account(stores, account)
        except BaseException:
            if wrote or removed:
                _put_back(stores)
            raise
    if own_kind is not None:
        _logger.info(
            "api: Claude Code's default login replaced a login of this computer's "
            "own (%s) with the login of %s",
            own_kind,
            raw,
        )
    elif wrote or recorded:
        _logger.debug("api: Claude Code's default login now uses the login of %s", raw)
    return wrote or removed or recorded


def _keep_account_login(
    switcher: ClaudeAccountSwitcher, live: str | None, provenance: dict[str, Any]
) -> None:
    """Keep the live login of claude-swap's active account in its store
    before a replace (sak.14), as a switch away from the account keeps it
    (:meth:`ClaudeAccountSwitcher._back_up_replaced_login`). Only the login
    Claude Code reads is the account's live one, where its renewals land; a
    copy it does not read is not kept. The caller holds
    :func:`_default_login_locks`, which include every lock a switch holds.
    A store that cannot be written is a :class:`CredentialWriteError`, and
    nothing is replaced then."""
    if live is None or _login_kind(live) != "own":
        return
    try:
        with _quiet():
            switcher._back_up_replaced_login(live, provenance)
    except OSError as e:
        raise CredentialWriteError(
            "The login of claude-swap's active account could not be kept in its "
            f"store ({e}), so Claude Code's default login was left as it is."
        ) from e


def _remove_own_logins() -> bool:
    """Take what is left of a login of this computer's own out of Claude
    Code's default profile once :func:`use_access_login` wrote the new one
    (sak.14): a store the write did not reach (a ``.credentials.json`` it
    could not refresh beside the Keychain item, the Keychain item a write
    that fell back to the file could not remove, or a store beside one that
    held the new login already). What Claude Code keeps there besides a
    login stays; the own login's device enrollment goes with it. Whether one
    went; :class:`CredentialWriteError` (or :class:`CredentialReadError`)
    while one stays."""
    removed = False
    if Platform.detect() == Platform.MACOS:
        item = _default_item(CLAUDE_CODE_KEYCHAIN_SERVICE)
        if _login_kind(item) == "own":
            _forget_in_item(item or "", ACCOUNT_CREDENTIAL_KEYS)
            removed = True
    file = _read_if_there(_default_credentials_path())
    if _login_kind(file) == "own":
        _forget_in_file(file or "", ACCOUNT_CREDENTIAL_KEYS)
        removed = True
    return removed


_LOGIN_ONLY = ("claudeAiOauth",)
"""What :func:`forget_default_access_login` takes out of a store: the login."""


def _without_login(
    text: str, login: Collection[str] = _LOGIN_ONLY
) -> dict[str, Any]:
    """What a store of Claude Code's credentials keeps besides its login
    (the keys ``login`` names)."""
    data = _parsed(text)
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, value in cast(dict[str, Any], data).items()
        if key not in login
    }


def _forget_in_item(text: str, login: Collection[str] = _LOGIN_ONLY) -> None:
    """Remove the login from the default profile's Keychain item; the item
    goes when nothing else is left in it."""
    kept = _without_login(text, login)
    account = macos_keychain.keychain_account_name()
    try:
        if kept:
            macos_keychain.set_password(
                CLAUDE_CODE_KEYCHAIN_SERVICE, account, json.dumps(kept)
            )
        else:
            macos_keychain.delete_password(CLAUDE_CODE_KEYCHAIN_SERVICE, account)
    except macos_keychain.KEYCHAIN_ERRORS as e:
        raise CredentialWriteError(
            f"Could not remove the login from Claude Code's Keychain item: {e}"
        ) from e


def _forget_in_file(text: str, login: Collection[str] = _LOGIN_ONLY) -> None:
    """Remove the login from the default profile's ``.credentials.json``;
    the file goes when nothing else is left in it."""
    path = _default_credentials_path()
    kept = _without_login(text, login)
    if kept:
        _sweep_temporaries(path)
        _write_private(path, json.dumps(kept))
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        raise CredentialWriteError(f"Could not remove {path}: {e}") from e


def _skip_whitespace(text: str, index: int) -> int:
    """The first position at or after ``index`` that is not JSON whitespace."""
    while index < len(text) and text[index] in " \t\n\r":
        index += 1
    return index


def _written_values(text: str) -> dict[str, tuple[Any, str]]:
    """The top-level values of the JSON object ``text`` by key, each with the
    text it is written as there; a key written twice holds its last value, as
    :func:`json.loads` reads it. Raises ValueError when ``text`` is not one
    JSON object."""
    decoder = json.JSONDecoder()
    values: dict[str, tuple[Any, str]] = {}
    index = _skip_whitespace(text, 0)
    if not text.startswith("{", index):
        raise ValueError("not a JSON object")
    index = _skip_whitespace(text, index + 1)
    more = not text.startswith("}", index)
    while more:
        key, index = decoder.raw_decode(text, index)
        index = _skip_whitespace(text, index)
        if not isinstance(key, str) or not text.startswith(":", index):
            raise ValueError("not a JSON object")
        start = _skip_whitespace(text, index + 1)
        value, end = decoder.raw_decode(text, start)
        values[key] = (value, text[start:end])
        index = _skip_whitespace(text, end)
        more = text.startswith(",", index)
        if more:
            index = _skip_whitespace(text, index + 1)
        elif not text.startswith("}", index):
            raise ValueError("not a JSON object")
    if _skip_whitespace(text, index + 1) != len(text):
        raise ValueError("more than one JSON value")
    return values


def _shared_beside_own_login(item: str, file: str) -> str | None:
    """``file`` (the default profile's ``.credentials.json``, with the
    computer's own login) with what the Keychain ``item`` keeps besides a
    login added; ``None`` when that adds nothing. Only what the file lacks is
    added: a key, or an entry of a dict both hold (an MCP server's login in
    ``mcpOAuth``). The file's own keys and entries win, and the item's
    ``claudeAiOauth`` and ``trustedDeviceToken`` (its login's) are not added.
    What the file holds stays as it is written there, byte for byte, its
    login above all. Raises ValueError when ``file`` is not one JSON
    object."""
    written = _written_values(file)
    shared = _parsed(item)
    if not isinstance(shared, dict):
        return None
    merged = {key: value for key, (value, _) in written.items()}
    changed: set[str] = set()
    for key, value in cast(dict[str, Any], shared).items():
        if key in ACCOUNT_CREDENTIAL_KEYS:
            continue
        if key not in merged:
            merged[key] = value
            changed.add(key)
            continue
        here = merged[key]
        if isinstance(here, dict) and isinstance(value, dict):
            missing = {name: entry for name, entry in value.items() if name not in here}
            if missing:
                merged[key] = {**here, **missing}
                changed.add(key)
    if not changed:
        return None
    entries = (
        f"{json.dumps(key)}: "
        + (json.dumps(value) if key in changed else written[key][1])
        for key, value in merged.items()
    )
    return "{" + ", ".join(entries) + "}"


def _forget_item_beside_own_file(item: str, file: str) -> None:
    """Remove the default profile's Keychain item: a leftover access token
    that hides the computer's own login in ``.credentials.json`` from Claude
    Code, which reads the item first while it holds JSON. What the item keeps
    besides a login moves into the file first, in one owner-only write
    (:func:`_shared_beside_own_login`); when that write fails nothing changes.
    When the removal fails the item stays, its token too, and a retry
    finishes it: the file already holds what moved."""
    path = _default_credentials_path()
    try:
        merged = _shared_beside_own_login(item, file)
    except ValueError as e:
        raise CredentialWriteError(f"Could not read {path} as JSON: {e}") from e
    if merged is not None:
        _sweep_temporaries(path)
        _write_private(path, merged)
    try:
        macos_keychain.delete_password(
            CLAUDE_CODE_KEYCHAIN_SERVICE, macos_keychain.keychain_account_name()
        )
    except macos_keychain.KEYCHAIN_ERRORS as e:
        raise CredentialWriteError(
            "Could not remove Claude Code's Keychain item, which hides this "
            f"computer's own login in {path}: {e}"
        ) from e
    _touch_newer(path)


def forget_default_access_login() -> bool:
    """Remove an access-only login from Claude Code's default profile
    (sak.13): from its Keychain item (macOS) and from its
    ``.credentials.json``, keeping what Claude Code keeps there besides a
    login (an MCP server's login); an item or file with nothing else left
    goes. The account (``oauthAccount``) leaves ``~/.claude.json``; nothing
    else there changes. Under the locks :func:`use_access_login` holds.

    Next to a login of the computer's own or a managed API key
    (:attr:`DefaultLoginInfo.leftover_access`), only the leftover access
    token goes, from the store that holds it, and ``~/.claude.json`` is not
    touched. A leftover file next to an own login in the Keychain item only
    loses its token: Claude Code reads the item first. A leftover Keychain
    item next to an own login in ``.credentials.json`` goes whole, as Claude
    Code would still read the item and not that login. What the item keeps
    besides a login (an MCP server's login, a plugin's secrets) moves into the
    file first, in one owner-only write: only what the file lacks, the file's
    own keys win, and its login stays byte for byte. When that write fails
    nothing changes, and the leftover still shows.

    Returns False when there was no such login (nothing changes). Raises
    :class:`OwnLoginError` when the default profile holds a login of its own
    (``"own"`` in :func:`default_login`, an access token alone of one of
    claude-swap's accounts here too) or a managed API key, and no leftover,
    which is never removed; :class:`ConfigError` when ``~/.claude.json``
    cannot be read as JSON or claude-swap's account list cannot be read, and
    with a foreign ``CLAUDE_CONFIG_DIR`` or
    ``CLAUDE_SECURESTORAGE_CONFIG_DIR``; :class:`CredentialReadError` when a
    store cannot be read now; :class:`CredentialWriteError` when the login
    stays; :class:`LockError` while Claude Code or a switch holds a lock.
    Never prompts.
    """
    _refuse_other_store()
    with _default_login_locks():
        stores = _read_default_stores()
        leftovers = _leftover_stores(stores)
        if leftovers:
            if "item" in leftovers and _login_kind(stores.file) == "own":
                _forget_item_beside_own_file(stores.item or "", stores.file or "")
            elif "item" in leftovers:
                _forget_in_item(stores.item or "")
            if "file" in leftovers:
                _forget_in_file(stores.file or "")
            _logger.debug(
                "api: a leftover access token left Claude Code's default profile"
            )
            return True
        _refuse_own_default(stores)
        in_item = _login_kind(stores.item) == "access"
        in_file = _login_kind(stores.file) == "access"
        if not (in_item or in_file):
            return False
        _refuse_broken_config(stores)
        # The account goes first: a removal that fails then leaves a login
        # this reads again, and a retry finishes it.
        if stores.config is not None and "oauthAccount" in stores.config:
            _write_default_config(
                {
                    key: value
                    for key, value in stores.config.items()
                    if key != "oauthAccount"
                }
            )
        if in_item:
            _forget_in_item(stores.item or "")
        if in_file:
            _forget_in_file(stores.file or "")
    _logger.debug("api: the access-only login left Claude Code's default profile")
    return True
