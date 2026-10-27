"""Safe arithmetic and Russian-locale number formatting (no UI dependencies)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

NBSP = " "  # narrow no-break space as thousands separator
DASH = "—"


def safe_div(num: Any, den: Any) -> Any:
    """Divide without raising: zero or missing denominator gives NaN.

    Works on scalars, numpy arrays and pandas Series (index of the first Series is kept).
    """
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(d == 0, np.nan, n / d)
    out = np.where(np.isfinite(out), out, np.nan)
    for ref in (num, den):
        if isinstance(ref, pd.Series):
            return pd.Series(out, index=ref.index)
    return float(out) if out.ndim == 0 else out


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v)) or v is pd.NA


def _group(text: str) -> str:
    return text.replace(",", NBSP)


def fmt_int(v: Any) -> str:
    return DASH if _is_missing(v) else _group(f"{v:,.0f}")


def fmt_num(v: Any, digits: int = 1) -> str:
    return DASH if _is_missing(v) else _group(f"{v:,.{digits}f}").replace(".", ",")


def fmt_pct(v: Any, digits: int = 1) -> str:
    """Fraction -> percent string (0.123 -> '12,3%')."""
    return DASH if _is_missing(v) else f"{v * 100:.{digits}f}".replace(".", ",") + "%"


def fmt_thousands(v: Any, digits: int = 1) -> str:
    """Rubles -> thousands of rubles."""
    return DASH if _is_missing(v) else fmt_num(v / 1000.0, digits)


_FORMATTERS = {
    "int": fmt_int,
    "k": fmt_thousands,
    "k0": lambda v: fmt_thousands(v, 0),
    "rub0": lambda v: fmt_num(v, 0),
    "rub1": lambda v: fmt_num(v, 1),
    "rub2": lambda v: fmt_num(v, 2),
    "pct": fmt_pct,
    "num2": lambda v: fmt_num(v, 2),
    "days": lambda v: fmt_num(v, 0),
}


def format_value(v: Any, kind: str) -> str:
    """Format a raw metric value according to its display kind."""
    if kind == "text":
        return DASH if _is_missing(v) or v == "" else str(v)
    try:
        return _FORMATTERS[kind](v)
    except KeyError as exc:
        raise ValueError(f"Unknown format kind: {kind!r}") from exc
