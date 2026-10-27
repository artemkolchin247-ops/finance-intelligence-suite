"""ДДС: cash flow by month (facts + 12-month forecast), balances, scenario levers and drill-down."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from core import cashflow as C
from core.formatting import format_value
from ui import finance_charts as fc
from ui.components import Kpi, chart_card, kpi_row, page_header, section_title
from ui.export import to_excel
from ui.scenario import scenario_select
from ui.state import get_schedule, get_finance, scenario
from ui.statement_html import Col, TRow, render_statement

page_header("ДДС", "Движение денежных средств: факт и прогноз на 12 месяцев вперёд")
try:
    fin = get_finance()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

as_of = fin.as_of
cur = as_of.to_period("M").to_timestamp()
with st.container(border=True):
    c0, c1 = st.columns([2.2, 2.8])
    with c0:
        scenario_select("dds")
    c1.caption("Прогноз строится на странице «Прогноз»: поступления — оборот год назад × затухающий рост, расходы — среднее за 3 последних месяца плюс "
               "корректировки из загруженного плана. Известные платежи из «Платёжного календаря» заменяют прогноз по своим статьям.")

assump = scenario()
schedule = get_schedule()

with st.container(border=True):
    c1, c2 = st.columns([1.4, 1])
    mode = c1.segmented_control("Период", ["Факт 6 мес. + прогноз 12 мес.", "Только прогноз", "Факт и прогноз за 2 года"],
                                default="Факт 6 мес. + прогноз 12 мес.", key="dds_mode") or "Факт 6 мес. + прогноз 12 мес."
    entity = c2.selectbox("Юрлицо", ["Все", "ООО Вуки", "ИП Петрова"], key="dds_entity",
                          help="Фильтр применяется к фактическим операциям по счетам юрлица и прогнозным платежам")

first = {"Факт 6 мес. + прогноз 12 мес.": cur - pd.DateOffset(months=6), "Только прогноз": cur,
         "Факт и прогноз за 2 года": cur - pd.DateOffset(months=12)}[mode]
months = list(pd.date_range(first, cur + pd.DateOffset(months=11), freq="MS"))

journal, accounts = fin.journal, fin.accounts
if entity != "Все":
    acc_ok = accounts[accounts["entity"] == entity]["account"]
    journal = journal[journal["account"].isin(acc_ok)]
    accounts = accounts[accounts["entity"] == entity]
    schedule = schedule[schedule["entity"] == entity]

stmt = C.build_dds(journal, accounts, schedule, as_of, months)
rows = {r.key: r for r in stmt.rows}
close = rows["close"]
fut = [c for c in stmt.columns if c.month is not None and c.month >= cur]
cash_now = float(C.cash_by_account(journal, accounts, as_of)["balance"].sum())
min_col = min(fut, key=lambda c: close.values[c.key])
below = [c for c in fut if close.values[c.key] < assump.buffer]
f_in = sum(rows["dir|Операционная|in"].values[c.key] for c in fut)
f_out = sum(rows["dir|Операционная|out"].values[c.key] for c in fut)
st.caption(f"Данные на {as_of:%d.%m.%Y}: факт до этой даты, далее план. Юрлицо: {entity}. Суммы в тыс. ₽. "
           f"Минимальный остаток для контроля: {assump.buffer / 1000:,.0f} тыс. ₽.".replace(",", " "))

kpi_row([
    Kpi("Деньги на сегодня, тыс. ₽", format_value(cash_now, "k0")),
    Kpi("Поступления 12 мес., тыс. ₽", format_value(f_in, "k0"), spark=tuple(rows["dir|Операционная|in"].values[c.key] for c in fut)),
    Kpi("Платежи 12 мес., тыс. ₽", format_value(f_out, "k0"), spark=tuple(rows["dir|Операционная|out"].values[c.key] for c in fut)),
    Kpi("ЧДП 12 мес., тыс. ₽", format_value(sum(rows["net"].values[c.key] for c in fut), "k0"),
        good=sum(rows["net"].values[c.key] for c in fut) >= 0, spark=tuple(rows["net"].values[c.key] for c in fut)),
    Kpi("Остаток через 12 мес., тыс. ₽", format_value(close.values[fut[-1].key], "k0")),
    Kpi("Минимум остатка", format_value(close.values[min_col.key], "k0"), f"{min_col.label}", good=close.values[min_col.key] >= assump.buffer),
    Kpi("Месяцев ниже минимума", str(len(below)), ", ".join(c.label for c in below[:3]) or "нет рисков", good=not below),
])
if below:
    st.warning(f"Остаток опускается ниже {assump.buffer / 1000:,.0f} тыс. ₽ в месяцах: {', '.join(c.label for c in below)}. "
               "Посмотрите «Платёжный календарь»: там видно, какие платежи можно сдвинуть.".replace(",", " "))

# ---- statement
section_title("Отчёт о движении денежных средств")
cols_ui = [Col(c.key, c.label, c.state, c.state if c.state != "Факт" else "Факт") for c in stmt.columns]
ui_rows: list[TRow] = []
for r in stmt.rows:
    kind = {"title": "title", "total": "total", "group": "group", "item": "item", "balance": "balance", "memo": "memo"}[r.kind]
    ui_rows.append(TRow(r.key, r.label, r.values, r.level, kind, "k0", r.parent,
                        alert_below=assump.buffer if r.key == "close" else None))
folded = [r.id for r in ui_rows if r.kind == "group"]
st.iframe(render_statement(cols_ui, ui_rows, height=720, folded=folded, first_col="Статья"), height=720)
st.markdown('<div class="note">Остаток на конец периода: до даты отчёта — по журналу, дальше — остаток + накопленный прогноз. '
            "Внутренние переводы между счетами не меняют общий остаток и в потоках не показываются. "
            "Клик по строке сворачивает/разворачивает группу. Красным выделены месяцы с остатком ниже минимума.</div>", unsafe_allow_html=True)
export = pd.DataFrame({"Статья": [r.label for r in stmt.rows], **{c.label: [r.values.get(c.key) for r in stmt.rows] for c in stmt.columns}})
st.download_button("Скачать ДДС в Excel", to_excel({"ДДС": export, "График платежей": schedule.drop(columns=["month"], errors="ignore")}),
                   file_name="dds.xlsx", key="dds_dl")

# ---- charts
section_title("Графики")
lab = [c.label for c in stmt.columns if c.key != "total"]
sts = [c.state for c in stmt.columns if c.key != "total"]
x1, x2 = st.columns(2)
with x1:
    chart_card(fc.dds_balance(lab, sts, [close.values[c.key] for c in stmt.columns if c.key != "total"], assump.buffer), "dds_bal")
with x2:
    op = rows.get("net|Операционная")
    fin_r = rows.get("net|Финансовая")
    chart_card(fc.dds_net(lab, [op.values[c.key] if op else 0 for c in stmt.columns if c.key != "total"],
                          [fin_r.values[c.key] if fin_r else 0 for c in stmt.columns if c.key != "total"]), "dds_net")
groups: dict[str, list[float]] = {}
for r in stmt.rows:
    if r.kind == "group" and "|out|" in r.key:
        name = r.label
        vals = [r.values[c.key] for c in stmt.columns if c.key != "total"]
        groups[name] = [a + b for a, b in zip(groups.get(name, [0.0] * len(vals)), vals)]
chart_card(fc.dds_structure(lab, groups), "dds_str")

# ---- drill-down
section_title("Расшифровка статьи")
items = sorted({r.label for r in stmt.rows if r.kind == "item"})
d1, d2 = st.columns([2, 1])
article = d1.selectbox("Статья", items, index=items.index("Закупка товара") if "Закупка товара" in items else 0, key="dds_art")
month = d2.selectbox("Месяц", months, index=months.index(cur), format_func=lambda m: f"{m:%m.%Y}", key="dds_month")
det = C.drilldown(journal, schedule, as_of, article, month)
if det.empty:
    st.info("В выбранном месяце по статье нет операций.")
else:
    shown = det.assign(Сумма=det["Сумма"].round(0), Дата=det["Дата"].dt.strftime("%d.%m.%Y"))
    st.dataframe(shown, hide_index=True, width="stretch", height=min(60 + 35 * len(shown), 420),
                 column_config={"Сумма": st.column_config.NumberColumn("Сумма, ₽", format="%d")})
    st.caption(f"Итого по статье за месяц: {det['Сумма'].sum() / 1000:,.0f} тыс. ₽ (факт {det[det['Тип'] == 'Факт']['Сумма'].sum() / 1000:,.0f}, "
               f"план {det[det['Тип'] == 'План']['Сумма'].sum() / 1000:,.0f})".replace(",", " "))
