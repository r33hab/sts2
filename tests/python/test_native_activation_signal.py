"""The library survives the runtime's activation signal in a spawned worker (macOS).

The NativeAOT runtime stops a thread for the GC by sending it SIGUSR1, and its handler
then chains to the SIGUSR1 action it found when it started. .NET 9 calls that action's
`sa_sigaction` whenever its flags say SA_SIGINFO, and checks for SIG_DFL and SIG_IGN only
on the other branch (dotnet/runtime#132581). macOS resets a caught signal's handler
across execve but keeps its flags, so a worker spawned by a process that had loaded the
library -- every runner worker -- inherits SIGUSR1 as {SIG_DFL, SA_SIGINFO}, and the
first background GC that stopped its main thread mid-call jumped to address 0: SIGSEGV,
exit -11. `native.py` clears the stale flag before it loads the library.

Each test sends the signal the GC's thread suspension sends, to the thread it sends it
to, so none of them waits for a GC to happen.
"""

import ctypes
import multiprocessing
import os
import signal
import subprocess
import sys
import textwrap
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sts2_gym import native  # noqa: E402 -- this process now holds the library

MACOS_ONLY = unittest.skipUnless(
    sys.platform == "darwin",
    "the runtime's activation signal is SIGUSR1 only on macOS, and only macOS keeps "
    "sa_flags across execve",
)
SA_SIGINFO = 0x40  # <sys/signal.h> on macOS


class _Sigaction(ctypes.Structure):
    """macOS's `struct sigaction`: the handler, a 32-bit mask, the flags."""

    _fields_ = [
        ("handler", ctypes.c_void_p),
        ("mask", ctypes.c_uint32),
        ("flags", ctypes.c_int),
    ]


def _sigusr1_flags() -> int:
    """Return SIGUSR1's sa_flags in this process, as the kernel reports them.

    Raises:
        OSError: If sigaction cannot read SIGUSR1.

    """
    action = _Sigaction()
    if ctypes.CDLL(None).sigaction(signal.SIGUSR1, None, ctypes.byref(action)) != 0:
        raise OSError("sigaction(SIGUSR1) could not be read")
    return action.flags


def _take_an_activation_signal() -> None:
    """In a spawned worker: take the GC's activation signal mid-run, then carry on."""
    handle = native.run_create()
    signal.pthread_kill(threading.get_ident(), signal.SIGUSR1)
    native.run_destroy(handle)


def _run_child(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=60,
    )


# The child sets SIGUSR1 to what a launcher can hand it, then loads the library and takes
# the activation signal. `{handler}` is 0 (SIG_DFL) or 1 (SIG_IGN).
_INHERITED = """
    import ctypes, signal, sys, threading
    class Sigaction(ctypes.Structure):
        _fields_ = [("handler", ctypes.c_void_p), ("mask", ctypes.c_uint32),
                    ("flags", ctypes.c_int)]
    libc = ctypes.CDLL(None)
    stale = Sigaction({handler}, 0, 0x40 | 0x2)  # SA_SIGINFO | SA_RESTART, no handler
    assert libc.sigaction(signal.SIGUSR1, ctypes.byref(stale), None) == 0
    from sts2_gym import native
    handle = native.run_create()
    signal.pthread_kill(threading.get_ident(), signal.SIGUSR1)
    native.run_destroy(handle)
    print("survived")
"""

# A handler of the process's own, installed before the library loads, is left in place:
# the runtime chains to it, and it still runs.
_OWN_HANDLER = """
    import signal, sys, threading
    seen = []
    signal.signal(signal.SIGUSR1, lambda signum, frame: seen.append(signum))
    from sts2_gym import native
    handle = native.run_create()
    signal.pthread_kill(threading.get_ident(), signal.SIGUSR1)
    native.run_destroy(handle)
    print("handled" if seen == [signal.SIGUSR1] else f"not handled: {seen}")
"""


@MACOS_ONLY
class ActivationSignalTests(unittest.TestCase):
    def test_a_worker_spawned_by_a_process_holding_the_library_survives_it(self):
        # What makes the worker's inheritance stale: the runtime here installed its
        # handler with SA_SIGINFO, and execve keeps the flag while dropping the handler.
        self.assertTrue(
            _sigusr1_flags() & SA_SIGINFO,
            "the runtime no longer installs SIGUSR1 with SA_SIGINFO; this test would "
            "pass without exercising the inherited flag",
        )
        worker = multiprocessing.get_context("spawn").Process(
            target=_take_an_activation_signal,
        )
        worker.start()
        self.addCleanup(worker.kill)  # a no-op once it has exited
        worker.join(60)
        self.assertEqual(worker.exitcode, 0)

    def test_a_default_or_ignored_sigusr1_inherited_with_sa_siginfo_is_survived(self):
        for name, handler in (("SIG_DFL", 0), ("SIG_IGN", 1)):
            with self.subTest(inherited=name):
                child = _run_child(_INHERITED.format(handler=handler))
                self.assertEqual(
                    (child.returncode, child.stdout.strip()),
                    (0, "survived"),
                    child.stderr,
                )

    def test_a_handler_installed_before_the_library_loads_still_runs(self):
        child = _run_child(_OWN_HANDLER)
        self.assertEqual(
            (child.returncode, child.stdout.strip()),
            (0, "handled"),
            child.stderr,
        )


if __name__ == "__main__":
    unittest.main()
