"""Управленческий баланс: assets, liabilities and capital by month-end."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from core import balance as B
from core import pnl
from core.formatting import format_value
from ui import finance_charts as fc
from ui.components import Kpi, chart_card, kpi_row, page_header, section_title
from ui.export import to_excel
from ui.state import get_finance
from ui.statement_html import Col, TRow, render_statement

page_header("Баланс", "Управленческий баланс компании: что есть, что должны и чей это капитал")
try:
    fin = get_finance()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

fm = fin.fact_monthly
closed = [pd.Timestamp(m) for m in sorted(fm["month"].unique()) if m + pd.offsets.MonthEnd(0) <= fin.as_of]
opts = {m: f"{m:%m.%Y}" for m in closed}
with st.container(border=True):
    lo, hi = st.select_slider("Период (концы месяцев)", options=list(opts.values()), key="bs_range",
                              value=(opts[closed[-12]], opts[closed[-1]]))
labels = list(opts.values())
months = closed[labels.index(lo): labels.index(hi) + 1]

res = pnl.build_pnl(fm, fin.journal, fin.articles, closed)
npf = {pd.Timestamp(c.start): res.row("net_profit").values[c.key] for c in res.columns}
dv = {pd.Timestamp(c.start): res.row("dividends").values[c.key] for c in res.columns}
brows = B.build_balance(fin.journal, fin.accounts, fm, fin.balance_1c, npf, dv, months, fin.as_of)
by = {r.key: r for r in brows}
k_last, k_prev = f"{months[-1]:%Y-%m}", (f"{months[-2]:%Y-%m}" if len(months) > 1 else None)


def kpi(label: str, key: str, good_up: bool = True) -> Kpi:
    v = by[key].values[k_last]
    series = tuple(by[key].values[f"{m:%Y-%m}"] for m in months)
    if not k_prev:
        return Kpi(label, format_value(v, "k0"), spark=series)
    p = by[key].values[k_prev]
    d = v - p
    pct = d / abs(p) * 100 if p else math.nan
    return Kpi(label, format_value(v, "k0"), f"{'▲' if d >= 0 else '▼'} {abs(pct):.1f}%".replace(".", ","),
               (d >= 0) == good_up if abs(d) > 1 else None, series)


cogs = res.row("cogs")
cogs_day = cogs.values[f"{months[-1]:%Y-%m}"] / months[-1].days_in_month
stock_days = by["stock"].values[k_last] / cogs_day if cogs_day else math.nan
kpi_row([kpi("Активы, тыс. ₽", "assets"), kpi("Деньги, тыс. ₽", "cash"), kpi("Запасы и предоплаты, тыс. ₽", "stock"),
         kpi("Обязательства, тыс. ₽", "liabilities", good_up=False), kpi("Капитал, тыс. ₽", "equity"),
         kpi("Оборотный капитал, тыс. ₽", "wc"),
         Kpi("Дней запаса", format_value(stock_days, "days"), "запасы / себестоимость в день")])

section_title("Баланс")
cols_ui = [Col(f"{m:%Y-%m}", f"{m:%m.%Y}", "на конец месяца") for m in months]
if k_prev:
    cols_ui.append(Col("d", "Δ, тыс. ₽", "к пред. месяцу", "Δ", delta=True))
ui_rows = []
good = {"assets": "up", "cash": "up", "recv": "up", "stock": "up", "liabilities": "down", "payables": "down", "taxes": "down",
        "payroll": "down", "debt": "down", "equity": "up", "retained": "up"}
for r in brows:
    vals = dict(r.values)
    if k_prev:
        vals["d"] = r.values[k_last] - r.values[k_prev] if r.kind != "title" else math.nan
    ui_rows.append(TRow(r.key, r.label, vals, r.level, r.kind, "k0", r.parent, good.get(r.key)))
st.iframe(render_statement(cols_ui, ui_rows, height=720, folded=["cash"], first_col="Статья"), height=720)
st.markdown('<div class="note">Деньги — по журналу операций (все счета, фонды, депозиты, карты). Товар на маркетплейсах и на складе — '
            "остатки из РНП по себестоимости единицы. Расчёты с маркетплейсами, товары в пути, предоплаты, кредиторская задолженность, "
            "налоги, ФОТ и кредиты — данные учётной системы. Нераспределённая прибыль — расчётная (активы − обязательства − уставный капитал); "
            "для контроля ниже показана накопленная чистая прибыль по ОПиУ.</div>", unsafe_allow_html=True)
df_x = pd.DataFrame({"Статья": [r.label for r in brows], **{f"{m:%m.%Y}": [r.values[f"{m:%Y-%m}"] for r in brows] for m in months}})
st.download_button("Скачать баланс в Excel", to_excel({"Баланс": df_x}), file_name="balance.xlsx", key="bs_dl")

section_title("Структура и динамика")
lab = [f"{m:%m.%Y}" for m in months]
g = lambda key: [by[key].values[f"{m:%Y-%m}"] for m in months]  # noqa: E731
a, b = st.columns(2)
with a:
    chart_card(fc.balance_structure(lab, {"Деньги": g("cash"), "Расчёты с МП": g("recv"), "Товар на МП": g("stock_mp"),
                                          "Товар на складе": g("stock_ms"), "В пути": g("transit"), "Предоплаты": g("prepay")}), "bs_str")
with b:
    chart_card(fc.equity_chart(lab, g("equity"), g("debt")), "bs_eq")
