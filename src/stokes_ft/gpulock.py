"""The machine-wide accelerator lock shared by the PhysicsNeMo projects of this machine.

Same protocol as ``physicsnemo-pino-darcy/src/pino_darcy/gpulock.py``: an exclusive ``flock`` on
``~/.cache/physicsnemo/gpu.lock`` (or ``$PHYSICSNEMO_GPU_LOCK``), held for the whole accelerator
job. The kernel releases it when the holder exits, so a killed job never leaves it held.
"""

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_PATH = Path.home() / ".cache" / "physicsnemo" / "gpu.lock"


def lock_path() -> Path:
    return Path(os.environ.get("PHYSICSNEMO_GPU_LOCK", DEFAULT_PATH))


@contextmanager
def gpu_lock(device_type: str, path: Path | None = None, poll_seconds: float = 30.0, timeout: float | None = None,
             log=print):
    """Hold the lock for the body of the ``with`` block when ``device_type`` is ``mps`` or ``cuda``.

    Waits, reporting the holder, until the lock is free or ``timeout`` seconds have passed.
    """
    if device_type not in ("mps", "cuda"):
        yield None
        return
    path = Path(path or lock_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if timeout is not None and time.monotonic() - start >= timeout:
                    raise TimeoutError(f"accelerator lock {path} still held after {timeout:g} s") from None
                holder = os.pread(fd, 512, 0).decode(errors="replace").strip() or "unknown"
                log(f"[gpu-lock] waiting for {path} (held by {holder})")
                time.sleep(poll_seconds)
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"pid={os.getpid()} cwd={os.getcwd()} since={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n".encode(), 0)
        log(f"[gpu-lock] acquired {path}")
        yield path
    finally:
        os.close(fd)
