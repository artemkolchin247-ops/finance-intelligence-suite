"""ОПиУ: P&L by months / quarters / years; marketplace lines come from the same facts as the РНП."""
from __future__ import annotations

import math
from html import escape

import pandas as pd
import streamlit as st

from core import pnl
from core.formatting import format_value
from ui import finance_charts as fc
from ui.components import Kpi, chart_card, kpi_row, page_header, section_title
from ui.export import to_excel
from ui.state import get_finance
from ui.statement_html import KIND_FMT, Col, TRow, render_statement

page_header("ОПиУ", "Отчёт о прибылях и убытках: что заработали, на что потратили и сколько осталось")
try:
    fin = get_finance()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

fm = fin.fact_monthly
closed = sorted(m for m in fm["month"].unique() if m + pd.offsets.MonthEnd(0) <= fin.as_of)
closed = [pd.Timestamp(m) for m in closed]
labels = {m: f"{m:%m.%Y}" for m in closed}

with st.container(border=True):
    c1, c2, c3 = st.columns([1, 2, 1.4])
    mp = c1.segmented_control("Маркетплейс", ["Все", "WB", "Ozon"], default="Все", key="pl_mp") or "Все"
    kw = {} if "pl_range" in st.session_state else {"value": (labels[closed[-6]], labels[closed[-1]])}
    lo, hi = c2.select_slider("Период (месяцы)", options=list(labels.values()), key="pl_range", **kw)
    gran = c3.segmented_control("Детализация", ["Месяцы", "Кварталы", "Годы"], default="Месяцы", key="pl_gran") or "Месяцы"
opts = list(labels.values())
months = closed[opts.index(lo): opts.index(hi) + 1]
if not months:
    st.warning("Выберите период.")
    st.stop()

result = pnl.build_pnl(fm, fin.journal, fin.articles, months, mp, gran)
cols = result.columns
last, prev = cols[-1], (cols[-2] if len(cols) > 1 else None)
st.caption(f"{mp} · {cols[0].label} – {last.label} · суммы в тыс. ₽ · косвенные расходы, налоги и прочее"
           + (" распределены по доле выручки маркетплейса" if mp != "Все" else " — по дате начисления (Дата P&L) из журнала операций"))

# ---- what changed
if prev:
    lines = pnl.highlights(result, prev.key, last.key)
    items = "".join(f'<div class="it"><span class="dot {s}"></span><span>{escape(t)}</span></div>' for s, t in lines)
    st.markdown(f'<div class="story"><div class="t">Что изменилось · {last.label} к {prev.label}</div>{items}</div>', unsafe_allow_html=True)

# ---- KPI
section_title("Ключевые показатели")


def kpi(label: str, key: str, fmt: str, good_up: bool = True) -> Kpi:
    r = result.row(key)
    v = r.values[last.key]
    series = tuple(0.0 if (isinstance(x, float) and math.isnan(x)) else x for x in (r.values[c.key] for c in cols))
    if prev is None or math.isnan(r.values[prev.key]):
        return Kpi(label, format_value(v, fmt), spark=series)
    d = v - r.values[prev.key]
    pct = d * 100 if fmt == "pct" else (d / abs(r.values[prev.key]) * 100 if r.values[prev.key] else math.nan)
    text = f"{'▲' if d >= 0 else '▼'} {str(round(abs(pct), 1)).replace('.', ',')}{' п.п.' if fmt == 'pct' else '%'}"
    return Kpi(label, format_value(v, fmt), text, (d >= 0) == good_up if abs(d) > 1e-9 else None, series)


kpi_row([kpi("Выручка до СПП, тыс. ₽", "rev_before", "k0"), kpi("Маржа, тыс. ₽", "margin", "k0"),
         kpi("Маржинальность", "margin_pct_before", "pct"), kpi("EBITDA, тыс. ₽", "ebitda", "k0"),
         kpi("EBITDA, %", "ebitda_pct", "pct"), kpi("Чистая прибыль, тыс. ₽", "net_profit", "k0"),
         kpi("ROI", "roi", "pct"), kpi("Оборачиваемость, дн.", "turnover", "days", good_up=False)])

# ---- table
section_title("Отчёт")
cols_ui = [Col(c.key, c.label, "Итог" if gran != "Месяцы" else "Факт") for c in cols]
if prev:
    cols_ui += [Col("d_abs", "Δ, тыс. ₽", f"{prev.label} → {last.label}", "Δ", delta=True),
                Col("d_rel", "Δ, %", "к пред. периоду", "Δ", fmt="pct", delta=True)]
dl = pnl.delta(result, prev.key, last.key) if prev else {}
rows: list[TRow] = []
for r in result.rows:
    if r.key == "avg_stock_units":
        rows.append(TRow("hdr_ref", "Справочно:", {}, 0, "title"))
    kind = "memo" if r.kind == "memo" else ("total" if r.bold else ("group" if r.level == 0 else "item"))
    vals = dict(r.values)
    if prev:
        vals["d_abs"], vals["d_rel"] = dl[r.key]
    rows.append(TRow(r.key, r.label, vals, r.level if r.level < 3 else 2, kind, KIND_FMT.get(r.kind, "k0"), r.parent, r.good))
folded = [r.id for r in rows if r.id.startswith("cat|")] + ["direct"]
st.iframe(render_statement(cols_ui, rows, height=760, folded=folded), height=760)
st.markdown('<div class="note">Выручка, себестоимость, комиссии, логистика, хранение и реклама маркетплейсов берутся из тех же данных, что и РНП; '
            "комиссия после СПП = комиссия до СПП − скидка СПП. Косвенные расходы, налоги, дивиденды — из журнала операций по дате P&amp;L. "
            "Δ % по затратным строкам — изменение их доли в выручке (как в рабочем файле ОПиУ). Клик по строке сворачивает детализацию.</div>",
            unsafe_allow_html=True)

df_x = pd.DataFrame({"Показатель": [r.label for r in result.rows], **{c.label: [r.values[c.key] for r in result.rows] for c in cols}})
st.download_button("Скачать ОПиУ в Excel", to_excel({"ОПиУ": df_x}), file_name="opiu.xlsx", key="pl_dl")

# ---- charts
section_title("Динамика и структура")
a, b = st.columns(2)
with a:
    chart_card(fc.pnl_waterfall({r.key: r.values[last.key] for r in result.rows}, f"От выручки до чистой прибыли · {last.label}, тыс. ₽"), "pl_wf")
with b:
    chart_card(fc.pnl_trend([c.label for c in cols], {"Маржинальность": [result.row("margin_pct_before").values[c.key] for c in cols],
                                                     "EBITDA": [result.row("ebitda_pct").values[c.key] for c in cols],
                                                     "Чистая прибыль": [result.row("net_margin").values[c.key] for c in cols]}), "pl_tr")
chart_card(fc.indirect_structure([c.label for c in cols], {pnl.SHORT_CATEGORY[cat]: [result.row(f"cat|{cat}").values[c.key] for c in cols]
                                                           for cat in pnl.CATEGORIES}), "pl_ind")
