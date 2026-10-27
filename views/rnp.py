"""РНП: daily performance report with filters, 'what changed' summary, KPIs, charts and the interactive table."""
from __future__ import annotations

from datetime import timedelta
from html import escape

import pandas as pd
import streamlit as st

from core import insights as I
from core import rnp_calc as R
from core.data_repo import MARKETPLACES
from core.formatting import format_value
from ui import charts
from ui.components import Kpi, chart_card, kpi_row, page_header, section_title
from ui.state import DEFAULT_END, DEFAULT_START, cached_report, load_dataset, load_hypotheses
from ui.table_html import render_table

CHIP = {"В работе": "work", "Подтверждено": "ok", "Опровергнуто": "no"}


def _kpi_delta(key: str, cur: dict, prev: dict | None) -> tuple[str, bool | None]:
    if not prev:
        return "", None
    found = {c.key: c for c in I.compare(cur, prev, keys=(key,), eps=0.0005)}
    ch = found.get(key)
    if not ch:
        return "", None
    arrow = "▲" if ch.delta >= 0 else "▼"
    text = I.fmt_delta(ch).lstrip("+−")
    good = None if ch.sentiment == "neutral" else ch.sentiment == "good"
    return f"{arrow} {text}", good


def _story(lines: list[tuple[str, str]], prev_label: str) -> None:
    items = "".join(f'<div class="it"><span class="dot {s}"></span><span>{escape(t)}</span></div>' for s, t in lines)
    st.markdown(f'<div class="story"><div class="t">Что изменилось · по сравнению с {escape(prev_label)}</div>{items}</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ page
page_header("РНП", "Ежедневная динамика заказов, продаж, трафика, воронки, рекламы и маржи — и что к этому привело")

try:
    ds = load_dataset()
    hyp_all = load_hypotheses()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

data_min, data_max = ds.fact["date"].min().normalize(), ds.fact["date"].max().normalize()

# jump from the hypotheses page: widget keys are only settable before the widgets are created
goto = st.session_state.pop("_rnp_goto", None)
if goto:
    for k, v in goto.items():
        st.session_state[k] = v

all_models = sorted(ds.dim["model"].unique(), key=str.lower)
with st.container(border=True):
    c1, c2, c3 = st.columns([1, 1.5, 1])
    # defaults live in session state so the "open from hypotheses" jump can override them without widget warnings
    st.session_state.setdefault("rnp_mp", "WB")
    st.session_state.setdefault("rnp_period", (max(DEFAULT_START, data_min).date(), min(DEFAULT_END, data_max).date()))
    mp = c1.segmented_control("Маркетплейс", MARKETPLACES, key="rnp_mp") or "WB"
    period = c2.date_input("Период", min_value=data_min.date(), max_value=data_max.date(),
                           format="DD.MM.YYYY", key="rnp_period")
    status = c3.selectbox("Статус артикулов", R.STATUSES, key="rnp_status",
                          help="«Продается» — действующий ассортимент, «Выводим» — выводимые из ассортимента")
    d1, d2 = st.columns([1, 1.5])
    models = d1.multiselect("Модель", all_models, key="rnp_models", placeholder="ИТОГО — все модели")
    scope = ds.dim[ds.dim["model"].isin(models)] if models else ds.dim
    if status != R.STATUS_ALL:
        scope = scope[scope["status"] == status]
    sku_options = sorted(scope["sku"])
    st.session_state["rnp_skus"] = [s for s in st.session_state.get("rnp_skus", []) if s in sku_options]
    skus = d2.multiselect("SKU (артикул)", sku_options, key="rnp_skus", placeholder="Все артикулы выбранных моделей")

if not (isinstance(period, (tuple, list)) and len(period) == 2):
    st.info("Выберите дату окончания периода.")
    st.stop()
start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])

report = cached_report(mp, str(start.date()), str(end.date()), tuple(models), tuple(skus), status)
if report is None:
    st.warning("За выбранный период и фильтры нет данных. Измените период или сбросьте фильтры.")
    st.stop()

n_days = (end - start).days + 1
prev_start, prev_end = start - timedelta(days=n_days), start - timedelta(days=1)
prev_report = cached_report(mp, str(prev_start.date()), str(prev_end.date()), tuple(models), tuple(skus), status) \
    if prev_start >= data_min else None
cur = report.total.values
prev = prev_report.total.values if prev_report else None
days = report.day_frame()
hyps = R.hypotheses_by_date(hyp_all, mp, models, start, end)

scope_label = ", ".join(models) if 0 < len(models) <= 3 else (f"{len(models)} моделей" if models else "все модели")
st.caption(f"{mp} · {scope_label} · {start:%d.%m.%Y} – {end:%d.%m.%Y} ({n_days} дн.)"
           + (f" · активных моделей: {len(report.active_models)}" if len(report.active_models) > 1 else ""))

# ---- what changed
if prev:
    lines = I.narrative(cur, prev)
    if hyps:
        n_h = sum(len(v) for v in hyps.values())
        lines.append(("neutral", f"В периоде зафиксировано гипотез: {n_h} — они отмечены 💡 в таблице и на графиках"))
    _story(lines, f"{prev_start:%d.%m}–{prev_end:%d.%m}")
else:
    st.info("Для сравнения с предыдущим периодом нужны данные за такой же отрезок перед выбранным.")

# ---- KPI
section_title("Ключевые показатели")
st.caption("Стрелки — изменение к предыдущему периоду такой же длины; линия — динамика по дням.")
spark = lambda k: tuple(days[k].fillna(0).tolist()) if k in days else ()  # noqa: E731
kpi_row([
    Kpi("Заказы, шт", format_value(cur["orders"], "int"), *_kpi_delta("orders", cur, prev), spark("orders"),
        "Количество заказов за период"),
    Kpi("Заказы, тыс. ₽", format_value(cur["orders_rub"], "k"), *_kpi_delta("orders_rub", cur, prev), spark("orders_rub")),
    Kpi("Клик → заказ", format_value(cur["cr_click_order"], "pct"), *_kpi_delta("cr_click_order", cur, prev),
        spark("cr_click_order"), "Конверсия клика по карточке в заказ"),
    Kpi("Реклама, тыс. ₽", format_value(cur["ad_total"], "k"), *_kpi_delta("ad_total", cur, prev), spark("ad_total")),
    Kpi("ДРР от продаж", format_value(cur["drr_total_sales"], "pct"), *_kpi_delta("drr_total_sales", cur, prev),
        spark("drr_total_sales")),
    Kpi("Маржа, тыс. ₽", format_value(cur["margin"], "k"), *_kpi_delta("margin", cur, prev), spark("margin"),
        "Факт: учитывает отставание выкупов от заказов"),
    Kpi("Маржа (прогноз)", format_value(cur["fc_margin"], "k"), *_kpi_delta("fc_margin", cur, prev),
        spark("fc_margin"), "По заказам и выкупу лагового окна — без отставания, тыс. ₽"),
    Kpi("Оборачиваемость, дн.", format_value(cur["turnover"], "days"),
        f"Остатки {format_value(cur['stock'] + cur['stock_ms'], 'int')} шт", None, spark("turnover")),
])

# ---- charts
section_title("Динамика")
st.caption("💡 — дата гипотезы (наведите, чтобы прочитать), кружок — аномальный день (отклонение от тренда за 14 дней).")
c1, c2 = st.columns(2)
with c1:
    chart_card(charts.orders_chart(days, hyps), "ch_orders")
with c2:
    chart_card(charts.traffic_chart(days, hyps), "ch_traffic")
c3, c4 = st.columns(2)
with c3:
    chart_card(charts.ads_chart(days, hyps), "ch_ads")
with c4:
    chart_card(charts.margin_chart(days, hyps), "ch_margin")
c5, c6 = st.columns(2)
with c5:
    chart_card(charts.funnel_chart(cur), "ch_funnel")
with c6:
    bridge = I.margin_bridge(cur, prev) if prev else None
    if bridge:
        chart_card(charts.bridge_chart(bridge), "ch_bridge")
    else:
        st.info("Мост маржи появится, когда есть данные предыдущего периода.")

# ---- table
section_title("Таблица РНП")
t1, t2 = st.columns([2.2, 1])
sections = t1.multiselect("Блоки показателей", R.SECTIONS, default=R.SECTIONS, key="rnp_sections",
                          help="В таблице можно сворачивать блоки кликом по заголовку, а дни недели — кликом по «Итого за неделю»")
delta_label = t2.radio("Стрелки ▲▼ к пред. дню/неделе", ("Ключевые строки", "Все строки", "Выключить"), horizontal=True,
                       key="rnp_delta")
delta_mode = {"Ключевые строки": "key", "Все строки": "all", "Выключить": "off"}[delta_label]
if sections:
    st.iframe(render_table(report, hyps, delta_mode=delta_mode, sections=sections, height=780), height=780)
    st.markdown('<div class="note">Проценты и стоимости считаются от сумм окна. Продажи отстают от заказов на 3–10 дней, поэтому '
                "по дням факт маржи и выкупа «плавает» — для оценки изменений смотрите блок «Прогноз» (по заказам). "
                "Остатки — на конец окна. Креаторы заполняются только в недельных колонках.</div>", unsafe_allow_html=True)
else:
    st.info("Выберите хотя бы один блок показателей.")

# ---- hypotheses in the period
if hyps:
    section_title("Гипотезы периода")
    for day, items in sorted(hyps.items()):
        for h in items:
            chip = CHIP.get(h["status"], "work")
            st.markdown(
                f'<div class="hcard"><div class="h">{escape(h["text"])}<span class="chip {chip}">{escape(h["status"])}</span></div>'
                f'<div class="m">{day:%d.%m.%Y} · {escape(h["model"])} — подробности и эффект на странице «Гипотезы»</div></div>',
                unsafe_allow_html=True)
