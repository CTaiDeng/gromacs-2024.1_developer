from __future__ import annotations

import os
import re
import sys
from typing import Any

try:
    from colorama import Fore, Style, init as colorama_init
    _HAS_COLOR = True
    # Initialize colorama on Windows; no-op on POSIX. autoreset keeps prints simple.
    colorama_init(autoreset=True)
except Exception:  # pragma: no cover
    _HAS_COLOR = False

    class _Dummy:
        RESET_ALL = ""
        BRIGHT = ""
        RED = ""
        GREEN = ""
        CYAN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""

    Fore = _Dummy()  # type: ignore[assignment]
    Style = _Dummy()  # type: ignore[assignment]


def _want_color() -> bool:
    env = os.getenv("VIDEO_PROD_COLOR")
    if env is not None:
        v = str(env).strip().lower()
        return v not in ("0", "false", "no", "off")
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_TAG_RE = re.compile(r"^(\s*)\[(ERROR|OK|DEBUG|WARN|READ|WAIT|SDK|INFO|DONE|GEMINI)\](.*)", re.DOTALL)
_TAG_COLOR = {
    "ERROR": Fore.RED + Style.BRIGHT,
    "OK": Fore.GREEN + Style.BRIGHT,
    "DEBUG": Fore.CYAN,
    "WARN": Fore.YELLOW + Style.BRIGHT,
    "READ": Fore.BLUE,
    "WAIT": Fore.YELLOW,
    "SDK": Fore.MAGENTA,
    "INFO": Fore.CYAN + Style.BRIGHT,
    "DONE": Fore.GREEN + Style.BRIGHT,
    "GEMINI": Fore.MAGENTA + Style.BRIGHT,
}


def colorize(text: str) -> str:
    s = str(text)
    if not _HAS_COLOR or not _want_color():
        return s
    m = _TAG_RE.match(s)
    if m:
        ws, tag, rest = m.groups()
        color = _TAG_COLOR.get(tag, "")
        return f"{ws}{color}[{tag}]{Style.RESET_ALL}{rest}"
    return s


def cc(text: str) -> str:
    """Return a colorized string if coloring is enabled.

    Useful for sys.stdout.write(...) scenarios (spinners / progress bars).
    """
    return colorize(text)


def cprint(*args: Any, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
    """Drop-in replacement for print() that colorizes known tags.

    Recognized tags at the beginning of the message (case-sensitive):
    [ERROR], [OK], [DEBUG], [WARN], [READ], [WAIT], [SDK]
    """
    s = sep.join(str(a) for a in args)
    s = colorize(s)
    print(s, end=end, file=file or sys.stdout, flush=flush)


__all__ = [
    "cprint",
    "cc",
]

