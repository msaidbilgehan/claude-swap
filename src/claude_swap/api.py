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
)
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
from claude_swap.paths import get_claude_config_home, get_default_claude_config_home
from claude_swap.settings import AutoSwitchSettings, load_settings
from claude_swap.switcher import ClaudeAccountSwitcher

__all__ = [
    "SCHEMA_VERSION",
    "AccountNotFoundError",
    "AllExhaustedEvent",
    "AutoSwitchEngine",
    "AutoSwitchEvent",
    "AutoSwitchSettings",
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
    "project_account",
    "set_rotation",
    "switch_to",
    "version",
]

_logger = logging.getLogger("claude-swap")
_output_lock = threading.RLock()


class ProjectAccount(NamedTuple):
    """The account a directory mapping (``cswap map``) resolves to.

    ``(None, None)``: no mapping covers the directory. ``(None, email)``: a
    mapping exists but its account was removed. ``(slot, email)``: resolved.
    """

    slot: str | None
    email: str | None


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
