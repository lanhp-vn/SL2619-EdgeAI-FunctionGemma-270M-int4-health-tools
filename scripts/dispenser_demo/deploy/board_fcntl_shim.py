"""Minimal fcntl shim for Yocto Python 3.12 on the SL2619 (the board's Python
is built without the fcntl module). pybleno only needs fcntl.ioctl; back it
with libc ioctl(3) via ctypes. Staged on PYTHONPATH ahead of stdlib for the
pybleno process only — see docs/deployment/sl2619-ble-bringup.md."""
import array as _array
import ctypes as _c
import ctypes.util as _cu
import os as _os

_libc = _c.CDLL(_cu.find_library("c") or "libc.so.6", use_errno=True)
_libc.ioctl.restype = _c.c_int


def ioctl(fd, request, arg=0, mutate_flag=True):
    if hasattr(fd, "fileno"):
        fd = fd.fileno()
    fd = int(fd)
    request = int(request)
    if isinstance(arg, int):
        res = _libc.ioctl(_c.c_int(fd), _c.c_ulong(request), _c.c_int(arg))
        if res < 0:
            e = _c.get_errno()
            raise OSError(e, _os.strerror(e))
        return res
    # buffer-style arg (array.array / bytes / bytearray)
    buf = bytearray(arg)
    cbuf = (_c.c_char * len(buf)).from_buffer(buf)
    res = _libc.ioctl(_c.c_int(fd), _c.c_ulong(request), cbuf)
    if res < 0:
        e = _c.get_errno()
        raise OSError(e, _os.strerror(e))
    if mutate_flag and isinstance(arg, _array.array):
        arg[:] = _array.array(arg.typecode, bytes(buf))
        return res
    return bytes(buf)


# pybleno doesn't use these, but provide them so `import fcntl` is complete.
def flock(fd, op):  # noqa: D401
    pass


LOCK_SH, LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 4, 8
