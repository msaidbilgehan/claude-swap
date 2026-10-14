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
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, NamedTuple

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
from claude_swap.credentials import CLAUDE_CODE_KEYCHAIN_SERVICE, looks_like_api_key
from claude_swap.exceptions import (
    AccountNotFoundError,
    ClaudeSwitchError,
    ConfigError,
    CredentialError,
    CredentialReadError,
    LockError,
    SessionError,
    SwitchError,
    ValidationError,
)
from claude_swap.json_output import SCHEMA_VERSION
from claude_swap.models import Platform, get_timestamp, normalize_alias
from claude_swap.oauth import (
    credential_fingerprint,
    is_setup_token,
    login_expires_at_iso,
)
from claude_swap.paths import get_claude_config_home, get_default_claude_config_home
from claude_swap.session import (
    AUTH_OVERRIDE_ENV_VARS,
    SessionManager,
    forget_config_dir_login,
    read_config_dir_credentials,
)
from claude_swap.settings import AutoSwitchSettings, load_settings
from claude_swap.switcher import ClaudeAccountSwitcher

__all__ = [
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


class LoginNotAddedError(CredentialError):
    """Claude Code moved the login into ``config_dir``, but it could not be
    registered: that profile holds the only working copy now. Fix the cause
    and import from ``config_dir``; do not discard it."""

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
    """Slot number of the live default login, or ``None`` when it is unmanaged."""
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
    :class:`ValidationError` for a bad or taken alias or a credential that
    belongs to another account. Never prompts.
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
    credential is missing or unreadable, and :class:`ValidationError` for a
    bad or taken alias or an API-key login. Never prompts.
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


def prepare_session_profile(
    switcher: ClaudeAccountSwitcher, identifier: str, *, share_history: bool = True
) -> SessionProfile:
    """Create or refresh the session-mode profile of ``identifier``.

    Adds the guard ``cswap run`` applies and ``SessionManager.setup_session``
    alone lacks: the account that is the active default login is refused
    (:class:`SessionError`), because a second copy of a live credential drifts
    when either copy refreshes. API-key accounts are refused too.
    ``share_history`` shares conversation history with the default profile
    (POSIX only). Settings and MCP servers are always shared, as with
    ``cswap run``. May run ``claude auth status`` and refresh a token.
    """
    _refuse_foreign_profile()
    if share_history and switcher.platform == Platform.WINDOWS:
        raise SessionError("share_history is not supported on Windows")
    with _quiet():
        account_num, email, org_uuid = switcher.resolve_account(identifier)
        current = switcher._get_current_account()
        if current is not None and current == (email, org_uuid):
            raise SessionError(
                f"Account-{account_num} ({email}) is the active default login; "
                "a session profile would hold a second copy of its live "
                "credential. Switch the default login to another account first."
            )
        session_dir, slot, slot_email = SessionManager(switcher).setup_session(
            identifier, True, share_history
        )
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


def _inside(path: Path, root: Path) -> bool:
    try:
        resolved, base = path.resolve(), root.resolve()
    except OSError:
        return False
    return resolved == base or base in resolved.parents


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


def _movable(credentials: str, where: str) -> tuple[str, tuple[str, ...], str | None]:
    """The refresh token, scopes and login expiry of a login that can move;
    refuses anything else."""
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
            f"The login in {where} expired on {login_expires_at_iso(credentials)}; "
            "log in again instead."
        )
    scopes = entry.get("scopes")
    if (
        not isinstance(scopes, list)
        or not scopes
        or not all(isinstance(s, str) and s and not s.isspace() for s in scopes)
    ):
        raise ValidationError(
            f"{where} does not record the login's scopes, which Claude Code needs "
            "to move it."
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
        current = switcher._get_current_account()
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
    switcher: ClaudeAccountSwitcher, config_dir: str | os.PathLike[str], found: _Found
) -> str:
    target = os.fspath(config_dir)
    path = Path(target)
    if not path.is_dir():
        raise ConfigError(f"The profile to move the login into, {target}, is missing.")
    _refuse_own_store(switcher, path)
    if _inside(path, Path(found.info.path)) or _inside(Path(found.info.path), path):
        raise ConfigError("The login cannot move into the folder it comes from.")
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


def _exchange(found: _Found, target: str, claude: str | os.PathLike[str] | None) -> None:
    """Let Claude Code turn the refresh token into a new login in ``target``
    (``claude auth login`` with ``CLAUDE_CODE_OAUTH_REFRESH_TOKEN``: no
    browser). Once it has, the source's copy no longer works."""
    executable = os.fspath(claude) if claude is not None else shutil.which("claude")
    if not executable:
        raise SessionError("'claude' was not found on PATH. Install Claude Code first.")
    env = {k: v for k, v in os.environ.items() if k not in _MOVE_ENV_REMOVED}
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

    Raises what :func:`check_login_source` raises, before anything moves, and
    :class:`ValidationError` for a bad alias or one another account uses;
    :class:`SessionError` when Claude Code cannot run; :class:`CredentialError`
    when it could not move the login (nothing moved then). After the move,
    a failed registration raises :class:`LoginNotAddedError`: ``config_dir``
    holds the only working copy. The token is never printed or logged.
    """
    _refuse_foreign_profile()
    with _quiet():
        found = _find_login(switcher, source)
        alias = _free_alias(switcher, alias, found.info.slot)
        target = _empty_profile(switcher, config_dir, found)
    # Outside the output lock: Claude Code may take a while, and every other
    # call of this module waits for that lock.
    _exchange(found, target, claude)
    with _quiet():
        try:
            identity = _profile_identity(Path(target))
            default_login = (
                identity is not None and switcher._get_current_account() == identity
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

    Raises :class:`ValidationError` for an empty value, an API key, anything
    that is not a setup-token, a bad email, or a bad or taken alias. Never
    prompts and never prints the token.
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
    with _quiet():
        before = set((switcher._get_sequence_data() or {}).get("accounts", {}))
        label = email.strip() if email else None
        existing = _setup_token_slot(switcher, label) if label else None
        normalized = _free_alias(switcher, alias, existing)
        switcher.add_account_from_token(value, email=label, assume_yes=True)
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
