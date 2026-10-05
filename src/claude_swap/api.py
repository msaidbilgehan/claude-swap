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
import logging
import os
import threading
from collections.abc import Callable, Collection, Iterator
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
from claude_swap.credentials import CLAUDE_CODE_KEYCHAIN_SERVICE
from claude_swap.exceptions import (
    AccountNotFoundError,
    ClaudeSwitchError,
    ConfigError,
    CredentialError,
    LockError,
    SessionError,
    SwitchError,
    ValidationError,
)
from claude_swap.json_output import SCHEMA_VERSION
from claude_swap.models import Platform
from claude_swap.oauth import credential_fingerprint
from claude_swap.paths import get_claude_config_home, get_default_claude_config_home
from claude_swap.session import (
    AUTH_OVERRIDE_ENV_VARS,
    SessionManager,
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
    "create_engine",
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
    """The account :func:`add_current_login` registered or refreshed.

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
    """
    _refuse_foreign_profile()
    with _quiet():
        return ClaudeAccountSwitcher()


def accounts_json(
    switcher: ClaudeAccountSwitcher, fetch: Collection[str] = frozenset()
) -> dict[str, Any]:
    """The ``cswap list --json`` payload (``schemaVersion`` 1).

    ``fetch`` names the account numbers whose usage may be fetched from the
    network in this call; the default (empty) serves usage from the cache
    only. With no accounts managed yet the payload has an empty list.
    """
    with _quiet():
        payload = switcher.list_accounts(json_output=True, fetch=set(fetch))
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
