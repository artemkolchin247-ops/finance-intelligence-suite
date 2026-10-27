"""'What changed' analytics on top of the report: period comparison, margin bridge, anomalies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.formatting import fmt_int, fmt_num, fmt_pct, safe_div
from core.rnp_calc import METRIC_BY_KEY

SHORT = {"orders": "Заказы", "orders_rub": "Заказы, ₽", "clicks": "Клики", "cr_click_order": "Конверсия клик → заказ",
         "aov_before": "Средний чек", "buyout": "Выкуп", "ad_total": "Реклама", "drr_total_sales": "ДРР",
         "margin": "Маржа, ₽", "margin_pct": "Маржа, %", "fc_margin": "Маржа (прогноз), ₽",
         "fc_margin_pct": "Маржа (прогноз), %"}
HEADLINE_KEYS = ("orders", "orders_rub", "clicks", "cr_click_order", "aov_before", "buyout",
                 "ad_total", "drr_total_sales", "margin", "margin_pct", "fc_margin", "fc_margin_pct")


@dataclass(frozen=True)
class Change:
    key: str
    label: str
    current: float
    previous: float
    delta: float  # relative change, or percentage points for ratio metrics
    is_pp: bool
    sentiment: str  # "good" | "bad" | "neutral"


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else float("nan")


def compare(cur: Mapping, prev: Mapping, keys=HEADLINE_KEYS, eps: float = 0.005) -> list[Change]:
    """Change of each metric vs the previous period; `eps` = share below which a change is 'neutral'."""
    out = []
    for k in keys:
        m = METRIC_BY_KEY[k]
        c, p = _num(cur.get(k)), _num(prev.get(k))
        if math.isnan(c) or math.isnan(p):
            continue
        is_pp = m.kind == "pct"
        delta = (c - p) if is_pp else float(safe_div(c - p, abs(p)))
        if math.isnan(delta):
            continue
        if abs(delta) < eps or m.good is None:
            sentiment = "neutral"
        else:
            sentiment = "good" if (delta > 0) == (m.good == "up") else "bad"
        out.append(Change(k, SHORT.get(k, m.display_label), c, p, delta, is_pp, sentiment))
    return out


def fmt_delta(ch: Change) -> str:
    sign = "+" if ch.delta >= 0 else "−"
    num = f"{abs(ch.delta) * 100:.1f}".replace(".", ",")
    return f"{sign}{num} п.п." if ch.is_pp else f"{sign}{num}%"


def orders_drivers(cur: Mapping, prev: Mapping) -> tuple[float, float] | None:
    """Split the log change of orders into traffic (clicks) and conversion (orders / clicks) parts."""
    try:
        c1, c0 = cur["clicks"], prev["clicks"]
        o1, o0 = cur["orders"], prev["orders"]
    except KeyError:
        return None
    if min(c1, c0, o1, o0) <= 0:
        return None
    traffic = c1 / c0 - 1
    conversion = (o1 / c1) / (o0 / c0) - 1
    return traffic, conversion


def margin_bridge(cur: Mapping, prev: Mapping) -> dict[str, float] | None:
    """Exact decomposition of the forecast margin change (order-based, free of the buyout lag).

    M = S * m - ads  =>  dM = (S1 - S0) * m0 [volume] + S1 * (m1 - m0) [unit margin] - (ads1 - ads0) [advertising]
    """
    try:
        s0, s1 = prev["fc_sales"], cur["fc_sales"]
        m0, m1 = prev["fc_margin_before_ads_pct"], cur["fc_margin_before_ads_pct"]
        a0, a1 = prev["ad_total"], cur["ad_total"]
        start, end = prev["fc_margin"], cur["fc_margin"]
    except KeyError:
        return None
    vals = [_num(x) for x in (s0, s1, m0, m1, a0, a1, start, end)]
    if any(math.isnan(v) for v in vals) or s0 <= 0 or s1 <= 0:
        return None
    return {"start": start, "volume": (s1 - s0) * m0, "unit_margin": s1 * (m1 - m0), "ads": -(a1 - a0), "end": end}


def narrative(cur: Mapping, prev: Mapping) -> list[tuple[str, str]]:
    """Plain-language findings as (sentiment, text); at most six, most important first."""
    lines: list[tuple[str, str]] = []
    ch = {c.key: c for c in compare(cur, prev)}
    if "orders" in ch:
        o = ch["orders"]
        text = f"Заказы {fmt_int(o.current)} шт ({fmt_delta(o)} к предыдущему периоду)"
        drv = orders_drivers(cur, prev)
        if drv:
            text += f": трафик {drv[0] * 100:+.1f}%, конверсия {drv[1] * 100:+.1f}%".replace(".", ",")
        lines.append((o.sentiment, text))
    if "aov_before" in ch and abs(ch["aov_before"].delta) >= 0.03:
        a = ch["aov_before"]
        lines.append(("neutral", f"Средний чек {fmt_num(a.current, 0)} ₽ ({fmt_delta(a)}) — "
                                 "изменение цены или структуры заказов"))
    if "drr_total_sales" in ch:
        d = ch["drr_total_sales"]
        lines.append((d.sentiment, f"ДРР общий {fmt_pct(d.current)} ({fmt_delta(d)}); "
                                   f"реклама {fmt_num(cur['ad_total'] / 1000, 0)} тыс. ₽"))
    if "buyout" in ch and abs(ch["buyout"].delta) >= 0.02:
        b = ch["buyout"]
        lines.append((b.sentiment, f"Выкуп {fmt_pct(b.current)} ({fmt_delta(b)})"))
    bridge = margin_bridge(cur, prev)
    if bridge:
        parts = sorted((("объёма продаж", bridge["volume"]), ("маржинальности до рекламы", bridge["unit_margin"]),
                        ("рекламных расходов", bridge["ads"])), key=lambda x: -abs(x[1]))
        total = bridge["end"] - bridge["start"]
        top = parts[0]
        verb = "выросла" if total >= 0 else "снизилась"
        lines.append(("good" if total >= 0 else "bad",
                      f"Прогнозная маржа {verb} на {fmt_num(abs(total) / 1000, 0)} тыс. ₽; главный фактор — "
                      f"{'рост' if top[1] >= 0 else 'снижение'} {top[0]} ({fmt_num(top[1] / 1000, 0)} тыс. ₽)"))
    return lines[:6]


def anomalies(day_frame: pd.DataFrame, keys=("orders", "orders_rub", "drr_total_sales", "margin"),
              window: int = 14, z_limit: float = 2.0) -> list[dict]:
    """Days where a metric deviates more than `z_limit` sigma from its trailing `window`-day mean."""
    found = []
    for k in keys:
        if k not in day_frame:
            continue
        s = day_frame[k].astype(float)
        mean = s.shift(1).rolling(window, min_periods=7).mean()
        std = s.shift(1).rolling(window, min_periods=7).std()
        z = (s - mean) / std.replace(0, np.nan)
        for dt, val in z[z.abs() >= z_limit].items():
            found.append({"date": dt, "key": k, "z": float(val), "value": float(s[dt])})
    return sorted(found, key=lambda r: r["date"])
