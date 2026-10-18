"""Run a command confined to a filesystem allowlist with Landlock.

The cost agent's fallback sandbox for hosts where bubblewrap cannot create a
user namespace (Ubuntu's ``apparmor_restrict_unprivileged_userns``). Landlock
needs no privileges: the process restricts itself, then execs the command,
which inherits the restriction along with everything it starts. Only paths
beneath ``--ro`` (read and execute) and ``--rw`` (also write) stay reachable;
everything else, ``/proc`` included, fails with ``EACCES``.

Run as a script with only the standard library, before any restriction::

    python -I landlock.py --chdir DIR --ro PATH... --rw PATH... -- CMD ARG...
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys

_SYS_CREATE_RULESET, _SYS_ADD_RULE, _SYS_RESTRICT_SELF = 444, 445, 446  # x86_64
_CREATE_RULESET_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_PDEATHSIG, _PR_SET_NO_NEW_PRIVS = 1, 38

_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1, 1 << 1, 1 << 2, 1 << 3
# Remove dir/file and make char/dir/reg/sock/fifo/block/sym.
_MODIFY_TREE = sum(1 << bit for bit in range(4, 13))
_REFER, _TRUNCATE = 1 << 13, 1 << 14

_libc = ctypes.CDLL(None, use_errno=True)


class _RulesetAttr(ctypes.Structure):  # pylint: disable=too-few-public-methods
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):  # pylint: disable=too-few-public-methods
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def abi_version() -> int:
    """The kernel's Landlock ABI version, or 0 when Landlock is unavailable."""
    return max(
        int(_libc.syscall(_SYS_CREATE_RULESET, None, 0, _CREATE_RULESET_VERSION)), 0
    )


def _check(result: int, what: str) -> int:
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"landlock {what}: {os.strerror(errno)}")
    return result


def restrict(read_only: list[str], read_write: list[str]) -> None:
    """Confine this process (and its future children) to the given paths."""
    abi = abi_version()
    if abi < 1:
        raise OSError("Landlock is not available on this kernel")
    read = _EXECUTE | _READ_FILE | _READ_DIR
    write = (
        _WRITE_FILE
        | _MODIFY_TREE
        | (_REFER if abi >= 2 else 0)
        | (_TRUNCATE if abi >= 3 else 0)
    )
    file_rights = _EXECUTE | _READ_FILE | _WRITE_FILE | (_TRUNCATE if abi >= 3 else 0)
    attr = _RulesetAttr(read | write)
    ruleset = _check(
        _libc.syscall(_SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0),
        "create_ruleset",
    )
    for paths, rights in ((read_only, read), (read_write, read | write)):
        for path in paths:
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            # A rule on a file may only carry rights that apply to files.
            allowed = rights if os.path.isdir(path) else rights & file_rights
            rule = _PathBeneathAttr(allowed, parent)
            _check(
                _libc.syscall(
                    _SYS_ADD_RULE, ruleset, _RULE_PATH_BENEATH, ctypes.byref(rule), 0
                ),
                f"add_rule {path}",
            )
            os.close(parent)
    _check(_libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
    _check(_libc.syscall(_SYS_RESTRICT_SELF, ruleset, 0), "restrict_self")
    os.close(ruleset)


def main(argv: list[str]) -> None:
    """Parse ``--chdir/--ro/--rw ... -- CMD``, restrict, and exec ``CMD``."""
    split = argv.index("--")
    command = argv[split + 1 :]
    lists: dict[str, list[str]] = {"--ro": [], "--rw": []}
    chdir = None
    current = None
    options = iter(argv[:split])
    for arg in options:
        if arg == "--chdir":
            chdir = next(options)
        elif arg in lists:
            current = lists[arg]
        else:
            assert current is not None, f"path {arg!r} before --ro/--rw"
            current.append(arg)
    # The bwrap sandbox's --die-with-parent: the agent dies with the evaluation.
    _check(_libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0), "pdeathsig")
    if chdir is not None:
        os.chdir(chdir)
    restrict(lists["--ro"], lists["--rw"])
    os.execvp(command[0], command)


if __name__ == "__main__":
    main(sys.argv[1:])
