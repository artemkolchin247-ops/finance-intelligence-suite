"""Hypothesis log helpers: effect of a change = N days after its date vs N days before."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core import rnp_calc as R
from core.formatting import safe_div
from core.insights import _num

# order-based metrics only: sales (and fact margin) lag orders by days, which would distort a short before/after
EFFECT_KEYS = ("orders", "orders_rub", "aov_before", "clicks", "cr_click_order", "ad_total", "drr_total_orders",
               "adv_cpc", "ext_views", "ext_clicks", "fc_margin", "fc_margin_before_ads_pct", "fc_margin_pct")
TOTAL_MODEL_NAMES = {"", "-", "—", "итого", "все"}


@dataclass(frozen=True)
class EffectRow:
    key: str
    label: str
    kind: str
    before: float
    after: float
    delta: float  # relative change; percentage points for ratio metrics
    is_pp: bool
    sentiment: str


def _per_day(values: dict, key: str, n_days: int) -> float:
    """Flow metrics are compared per day so windows of different length stay comparable."""
    v = _num(values.get(key))
    m = R.METRIC_BY_KEY[key]
    return v / n_days if m.kind in ("int", "k") else v


def effect(fact: pd.DataFrame, stocks_ms: pd.DataFrame, creators: pd.DataFrame, mp: str, model: str,
           date: pd.Timestamp, days_before: int = 7, days_after: int = 7,
           last_date: pd.Timestamp | None = None) -> list[EffectRow] | None:
    """Metrics of the model in [date-N, date-1] vs [date, date+N-1]. None if either window has no data."""
    models = () if model.strip().lower() in TOTAL_MODEL_NAMES else (model,)
    mp_key = "WB" if mp.lower() == "wb" else "Ozon"
    end_after = date + pd.Timedelta(days=days_after - 1)
    if last_date is not None:
        end_after = min(end_after, last_date)
    if end_after < date:
        return None
    before = R.build_report(fact, stocks_ms, creators, R.Selection(mp_key, date - pd.Timedelta(days=days_before),
                                                                     date - pd.Timedelta(days=1), models))
    after = R.build_report(fact, stocks_ms, creators, R.Selection(mp_key, date, end_after, models))
    if before is None or after is None:
        return None
    n_b, n_a = days_before, (end_after - date).days + 1
    rows = []
    for key in EFFECT_KEYS:
        m = R.METRIC_BY_KEY[key]
        b, a = _per_day(before.total.values, key, n_b), _per_day(after.total.values, key, n_a)
        if pd.isna(a) or pd.isna(b):
            continue
        is_pp = m.kind == "pct"
        delta = (a - b) if is_pp else float(safe_div(a - b, abs(b)))
        if pd.isna(delta):
            continue
        if abs(delta) < 0.005 or m.good is None:
            sentiment = "neutral"
        else:
            sentiment = "good" if (delta > 0) == (m.good == "up") else "bad"
        rows.append(EffectRow(key, m.display_label, m.kind, b, a, delta, is_pp, sentiment))
    return rows
