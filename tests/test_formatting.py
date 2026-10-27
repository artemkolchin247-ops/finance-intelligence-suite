import math

import numpy as np
import pandas as pd
import pytest

from core.formatting import fmt_pct, fmt_thousands, format_value, safe_div


def test_safe_div_scalar_zero_denominator_is_nan():
    assert math.isnan(safe_div(5, 0))
    assert safe_div(6, 3) == 2


def test_safe_div_series_keeps_index_and_masks_zero():
    num = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    den = pd.Series([2.0, 0.0, np.nan], index=["a", "b", "c"])
    res = safe_div(num, den)
    assert res["a"] == 0.5
    assert res[["b", "c"]].isna().all()


def test_formatters_use_russian_style_and_dash_for_missing():
    assert fmt_pct(0.1234) == "12,3%"
    assert fmt_thousands(1_234_500) == "1 234,5"
    assert format_value(float("nan"), "pct") == "—"
    assert format_value(None, "int") == "—"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        format_value(1.0, "bogus")
