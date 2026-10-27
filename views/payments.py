"""Платёжный календарь: 12 weeks of payments and receipts, cash-gap control and deferral advice."""
from __future__ import annotations

import math
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from core import cashflow as C
from core import payment_calendar as P
from core.formatting import format_value
from core.ledger import PLAN_COLUMNS, PLAN_STATUSES, PRIORITIES
from ui import finance_charts as fc
from ui.calendar_html import render_calendar
from ui.components import Kpi, chart_card, kpi_row, page_header, section_title
from ui.export import to_excel
from ui.state import get_ledger_repo, get_schedule, get_finance, load_payment_plan, scenario
from ui.statement_html import Col, TRow, render_statement

page_header("Платёжный календарь", "Что и когда платим и получаем в ближайшие 12 недель, где не хватит денег и что можно сдвинуть")
try:
    fin = get_finance()
    plan_df = load_payment_plan()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

assump = scenario()
as_of = fin.as_of
base_schedule = get_schedule()
sig = (assump, len(base_schedule), round(float(base_schedule["amount"].sum())) if len(base_schedule) else 0)
if st.session_state.get("cal_sig") != sig:  # moves refer to row ids: drop them when the scenario or plan changes
    st.session_state["cal_sig"] = sig
    st.session_state["cal_moves"] = {}
    st.session_state["cal_rejected"] = set()
moves: dict[str, pd.Timestamp] = st.session_state.setdefault("cal_moves", {})
rejected: set = st.session_state.setdefault("cal_rejected", set())
schedule = P.apply_moves(base_schedule, moves)

opening = float(C.cash_by_account(fin.journal, fin.accounts, as_of)["balance"].sum())
start = P.calendar_start(as_of)
end = start + pd.Timedelta(days=P.CAL_WEEKS * 7 - 1)
buffer = assump.buffer

with st.container(border=True):
    f1, f2, f3 = st.columns(3)
    prio = f1.multiselect("Приоритет", list(PRIORITIES), key="cal_prio", placeholder="Все")
    src = f2.multiselect("Источник", sorted(schedule["source"].unique()), key="cal_src", placeholder="Все")
    ent = f3.multiselect("Юрлицо", ["ООО Вуки", "ИП Петрова"], key="cal_ent", placeholder="Все")
view = schedule
if prio:
    view = view[(view["priority"].isin(prio)) | (view["direction"] == "in")]
if src:
    view = view[view["source"].isin(src)]
if ent:
    view = view[view["entity"].isin(ent)]
opening_view = opening if not ent else float(C.cash_by_account(fin.journal, fin.accounts[fin.accounts["entity"].isin(ent)], as_of)["balance"].sum())

weeks = P.weekly_summary(view, opening_view, start, buffer)
daily = P.daily_table(view, opening_view, start)
win = P.window(view, start)
low = daily.loc[daily["close"].idxmin()]
bad_weeks = [w for w in weeks if w.status != "ok"]
gap_weeks = [w for w in weeks if w.status == "gap"]
critical = win[(win["direction"] == "out") & (win["priority"] == "Критичный")]["amount"].sum()
st.caption(f"Горизонт: {start:%d.%m.%Y} – {end:%d.%m.%Y} (12 недель). Деньги на {as_of:%d.%m.%Y}: {opening_view / 1e6:.2f} млн ₽. "
           f"Минимальный остаток: {buffer / 1000:,.0f} тыс. ₽ (задаётся на странице «ДДС»).".replace(",", " "))

kpi_row([
    Kpi("Свободные деньги, тыс. ₽", format_value(opening_view, "k0")),
    Kpi("Поступления 12 нед., тыс. ₽", format_value(daily["inflow"].sum(), "k0"), spark=tuple(w.net for w in weeks)),
    Kpi("Платежи 12 нед., тыс. ₽", format_value(daily["outflow"].sum(), "k0")),
    Kpi("Критичные платежи, тыс. ₽", format_value(critical, "k0"), "налоги, ФОТ, закупки, кредит"),
    Kpi("Минимум остатка, тыс. ₽", format_value(low["close"], "k0"), f"{low['date']:%d.%m}", good=low["close"] >= buffer),
    Kpi("Недель ниже минимума", str(len(bad_weeks)), "кассовый разрыв!" if gap_weeks else ("риск" if bad_weeks else "всё в порядке"), good=not bad_weeks),
])

advice = P.suggest_deferrals(view, opening_view, start, buffer, exclude=rejected)
if gap_weeks:
    st.error(f"Кассовый разрыв: в неделях {', '.join(str(w.week) for w in gap_weeks)} остаток уходит в минус.")
if bad_weeks:
    first = bad_weeks[0]
    st.warning(f"Неделя {first.week} ({first.start:%d.%m}–{first.end:%d.%m}): остаток снижается до {first.min_balance / 1e6:.2f} млн ₽, "
               f"что ниже минимума {buffer / 1e6:.1f} млн ₽.")
    if advice:
        total = sum(a["amount"] for a in advice)
        with st.expander(f"Рекомендации: отложить {len(advice)} платежей на {total / 1e6:.2f} млн ₽ на неделю", expanded=True):
            st.caption("Каждую рекомендацию можно применить или отклонить отдельно: после отклонения предложим другой платёж.")
            for a in advice:
                r1, r2, r3, r4, r5 = st.columns([3, 1.3, 2, 1.1, 1.1])
                r1.markdown(f"**{a['article']}**")
                r2.write(f"{a['amount'] / 1000:,.0f} тыс. ₽".replace(",", "\u202f"))
                r3.write(f"{a['from']:%d.%m.%Y} → {a['to']:%d.%m.%Y}")
                if r4.button("Применить", key=f"adv_ok_{a['id']}"):
                    moves[a["id"]] = a["to"]
                    st.rerun()
                if r5.button("Отклонить", key=f"adv_no_{a['id']}"):
                    rejected.add(a["id"])
                    st.rerun()
            if st.button("Применить все рекомендации", type="primary", key="cal_apply_all"):
                moves.update({a["id"]: a["to"] for a in advice})
                st.rerun()
    elif rejected:
        st.info("Подходящих платежей для переноса больше нет — верните отклонённые рекомендации или сдвиньте платёж вручную ниже.")
if rejected and st.button(f"Вернуть отклонённые рекомендации ({len(rejected)})", key="cal_unreject"):
    st.session_state["cal_rejected"] = set()
    st.rerun()
if moves and st.button(f"Сбросить переносы ({len(moves)})", key="cal_reset"):
    st.session_state["cal_moves"] = {}
    st.rerun()
if moves:
    st.info(f"Применено переносов: {len(moves)} — перенесённые платежи помечены статусом «Отложен».")

# ---- weekly matrix
section_title("Недельный план")
cols_ui = [Col(f"w{w.week}", f"Нед. {w.week}", f"{w.start:%d.%m}–{w.end:%d.%m}", "curr" if w.week == 1 else "fact") for w in weeks]
rows: list[TRow] = []
v = lambda f: {f"w{w.week}": f(w) for w in weeks}  # noqa: E731
rows.append(TRow("open", "Остаток на начало недели", v(lambda w: w.opening), 0, "balance"))
rows.append(TRow("in", "Поступления", v(lambda w: sum(w.inflow.values())), 0, "total"))
for art in sorted({a for w in weeks for a in w.inflow}):
    rows.append(TRow(f"in|{art}", art, v(lambda w, art=art: w.inflow.get(art, 0.0)), 1, "item", parent="in"))
rows.append(TRow("out", "Платежи", v(lambda w: sum(w.outflow.values())), 0, "total"))
for g in P.CAL_GROUPS:
    rows.append(TRow(f"out|{g}", g, v(lambda w, g=g: w.outflow.get(g, 0.0)), 1, "group", parent="out"))
    arts = win[(win["direction"] == "out") & (win["cal_group"] == g)]
    for art in sorted(arts["article"].unique()):
        vals = {f"w{w.week}": float(arts[(arts["article"] == art) & (arts["date"] >= w.start) & (arts["date"] <= w.end)]["amount"].sum()) for w in weeks}
        rows.append(TRow(f"out|{g}|{art}", art, vals, 2, "item", parent=f"out|{g}"))
for pr in P.PRIORITY_ORDER:
    rows.append(TRow(f"pr|{pr}", f"в т.ч. {pr.lower()}", v(lambda w, pr=pr: w.by_priority.get(pr, 0.0)), 1, "memo", parent="out"))
rows.append(TRow("net", "Чистый денежный поток недели", v(lambda w: w.net), 0, "total"))
rows.append(TRow("close", "Остаток на конец недели", v(lambda w: w.closing), 0, "balance", alert_below=buffer))
rows.append(TRow("min", "Минимальный остаток внутри недели", v(lambda w: w.min_balance), 1, "memo", alert_below=buffer))
rows.append(TRow("status", "Статус", v(lambda w: {"ok": "🟢 в норме", "below": "🟡 ниже минимума", "gap": "🔴 разрыв"}[w.status]), 0, "total"))
st.iframe(render_statement(cols_ui, rows, height=640, folded=[r.id for r in rows if r.kind == "group"], first_col="Статья"), height=640)

# ---- calendar and chart
section_title("Календарь по дням")
st.iframe(render_calendar(daily, win, buffer, as_of), height=40 + 76 * P.CAL_WEEKS)
big = win[(win["direction"] == "out")].sort_values("amount", ascending=False).head(8)
chart_card(fc.calendar_balance(daily, buffer, big), "cal_chart")

# ---- registry
section_title("Реестр платежей и поступлений")
reg = win.sort_values(["date", "direction"], ascending=[True, False]).copy()
table = pd.DataFrame({
    "Дата": reg["date"].dt.strftime("%d.%m.%Y"), "Тип": np.where(reg["direction"] == "in", "Поступление", "Платёж"),
    "Статья": reg["article"], "Контрагент": reg["counterparty"], "Юрлицо": reg["entity"], "Сумма, ₽": reg["amount"].round(0),
    "Приоритет": np.where(reg["direction"] == "in", "", reg["priority"]), "Статус": reg["status"], "Источник": reg["source"],
    "Комментарий": reg["comment"], "ID": reg["id"],
})
st.dataframe(table, hide_index=True, width="stretch", height=380,
             column_config={"Сумма, ₽": st.column_config.NumberColumn(format="%d")})
st.download_button("Скачать календарь в Excel", to_excel({"Реестр": table, "Недели": pd.DataFrame(
    [{"Неделя": w.week, "С": w.start, "По": w.end, "Остаток на начало": w.opening, "Поступления": sum(w.inflow.values()),
      "Платежи": sum(w.outflow.values()), "ЧДП": w.net, "Остаток на конец": w.closing, "Минимум": w.min_balance} for w in weeks])}),
    file_name="payment_calendar.xlsx", key="cal_dl")

# ---- move a single payment
outs = reg[(reg["direction"] == "out") & (reg["priority"] != "Критичный")]
if len(outs):
    with st.expander("Перенести платёж вручную"):
        m1, m2, m3 = st.columns([3, 1.2, 1])
        pick = m1.selectbox("Платёж", outs["id"], key="cal_pick",
                            format_func=lambda i: f"{reg.loc[reg['id'] == i, 'date'].iloc[0]:%d.%m} · {reg.loc[reg['id'] == i, 'article'].iloc[0]} · "
                                                  f"{reg.loc[reg['id'] == i, 'amount'].iloc[0] / 1000:,.0f} тыс. ₽".replace(",", " "))
        new_date = m2.date_input("Новая дата", value=(reg.loc[reg["id"] == pick, "date"].iloc[0] + pd.Timedelta(days=7)).date(),
                                 min_value=start.date(), max_value=end.date(), format="DD.MM.YYYY", key="cal_newdate")
        m3.write("")
        if m3.button("Перенести", key="cal_move"):
            moves[pick] = pd.Timestamp(new_date)
            st.rerun()
        st.caption("Критичные платежи (налоги, ФОТ, закупки, кредит) не переносятся. Переносы действуют в рамках сессии.")

# ---- manual payment plan
section_title("План платежей (ручной ввод)")
st.caption("Известные счета и обязательства. Они заменяют прогноз по своим статьям и учитываются в календаре и в ДДС. "
           "Просроченные платежи попадают на первый день календаря.")
articles = sorted(fin.articles["article"])
edited = st.data_editor(
    plan_df, num_rows="dynamic", width="stretch", hide_index=True, key="plan_editor",
    column_config={
        "Дата": st.column_config.DateColumn("Дата", format="DD.MM.YYYY", required=True),
        "Статья операции": st.column_config.SelectboxColumn("Статья", options=articles, required=True),
        "Юрлицо": st.column_config.SelectboxColumn("Юрлицо", options=["ООО Вуки", "ИП Петрова"], required=True),
        "Сумма": st.column_config.NumberColumn("Сумма, ₽", min_value=1, format="%d", required=True),
        "Приоритет": st.column_config.SelectboxColumn("Приоритет", options=list(PRIORITIES), required=True),
        "Статус": st.column_config.SelectboxColumn("Статус", options=list(PLAN_STATUSES), required=True),
    })
if st.button("Сохранить план платежей", key="plan_save"):
    bad = edited[edited[["Дата", "Статья операции", "Сумма", "Приоритет", "Статус"]].isna().any(axis=1)]
    if len(bad):
        st.error(f"В {len(bad)} строках не заполнены дата, статья, сумма, приоритет или статус.")
    else:
        out = edited.copy()
        out["ID"] = [i if str(i).strip() else f"PP-{n + 1:03d}" for n, i in enumerate(out["ID"].fillna(""))]
        try:
            get_ledger_repo().save_payment_plan(out)
        except (OSError, ValueError) as exc:
            st.error(f"Не удалось сохранить план платежей: {exc}")
        else:
            st.success("План платежей сохранён, календарь и ДДС пересчитаны.")
            st.rerun()
