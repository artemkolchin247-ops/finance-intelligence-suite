"""Планирование: expense corrections, receipts factors, initiatives and the effect on the cash flow."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import cashflow
from core import planning as PL
from core.formatting import format_value
from ui import finance_charts as fc
from ui.components import Kpi, chart_card, kpi_row, page_header, section_title
from ui.export import to_excel
from ui.scenario import scenario_select
from ui.state import cached_schedule, finance_key, get_finance, get_ledger_repo, get_plan, load_payment_plan, scenario

page_header("Планирование", "Корректировки расходов, факторы и новые направления: что меняется в прогнозе ДДС и платёжном календаре")
try:
    fin = get_finance()
    repo = get_ledger_repo()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

ss = st.session_state
if "plan_saved" not in ss:
    ss["plan_saved"] = PL.plan_from_sheets(repo.get_plan_sheets())
ss.setdefault("plan_source", ss["plan_saved"])  # what the editors start from; edits stay in the widgets
ss.setdefault("plan_ver", 0)
source: PL.PlanInputs = ss["plan_source"]
ver: int = ss["plan_ver"]

with st.container(border=True):
    c1, c2 = st.columns([1, 3])
    with c1:
        scenario_select("plan")
    c2.markdown("**Как это работает.** Расходы по умолчанию равны среднему за последние 3 закрытых месяца по статье. Любой месяц можно заменить своей "
                "цифрой; новые направления и расширение команды вносятся как инициативы. Поступления: база по маркетплейсам с трендом и сезонностью, "
                "поверх — факторы (цена, заказы, выкуп…), ручная замена и выручка новых направлений с разгоном. Всё сразу попадает в ДДС и календарь.")
assump = scenario()
base_layers = PL.build_layers(fin.journal, fin.as_of, assump, PL.empty_plan())
months = base_layers.months
groups = fin.articles.set_index("article")["dds_group"].to_dict()

# ---- file actions
section_title("Файл плана")
f1, f2, f3, f4 = st.columns([1.2, 1.4, 1.2, 1.2])
live_for_template = get_plan()
f1.download_button("Скачать шаблон плана", to_excel(PL.to_template(PL.build_layers(fin.journal, fin.as_of, assump, live_for_template),
                                                                  live_for_template, groups)), file_name="plan_template.xlsx",
                   key="pl_tpl", width="stretch", help="Excel с тремя листами для заполнения и справочным листом с базой")
upload = f2.file_uploader("Загрузить заполненный план", type=["xlsx"], key=f"pl_upload_{ver}", label_visibility="collapsed")
if upload is not None:
    try:
        parsed, errors = PL.parse_template(pd.read_excel(upload, sheet_name=None))
    except Exception as exc:  # noqa: BLE001 - unreadable file is reported to the user
        parsed, errors = None, [f"Не удалось прочитать файл «{upload.name}»: {exc}"]
    if errors:
        st.error("План не принят:\n\n" + "\n".join(f"- {e}" for e in errors[:15]) + (f"\n- … и ещё {len(errors) - 15}" if len(errors) > 15 else ""))
    elif st.button("Применить загруженный план", type="primary", key="pl_apply"):
        ss["plan_source"], ss["plan_live"], ss["plan_ver"] = parsed, parsed, ver + 1
        st.rerun()
if f3.button("Сохранить как основной план", key="pl_save", width="stretch", help="Записывает план в файл данных; его увидят все пользователи"):
    plan_now = get_plan()
    problems = PL.validate_plan(plan_now)
    if problems:
        st.error("План не сохранён: " + "; ".join(problems[:5]))
    else:
        try:
            repo.save_plan_sheets(PL.plan_to_sheets(plan_now))
        except (OSError, ValueError) as exc:
            st.error(f"Не удалось сохранить план: {exc}")
        else:
            ss["plan_saved"] = plan_now
            st.success("План сохранён в рабочий файл данных.")
if f4.button("Сбросить правки", key="pl_reset", width="stretch", help="Вернуться к сохранённому плану"):
    ss.pop("plan_live", None)
    ss["plan_source"], ss["plan_ver"] = ss["plan_saved"], ver + 1
    st.rerun()

tab_exp, tab_rec, tab_init, tab_res = st.tabs(["Расходы", "Поступления", "Инициативы", "Результат"])
mlabels = [PL.month_label(m) for m in months]
month_cfg = {l: st.column_config.NumberColumn(l, min_value=0, format="%d", help="Пусто — база; число заменяет сумму за месяц, ₽") for l in mlabels}

with tab_exp:
    st.caption("База — среднее за 3 последних месяца (₽ в месяц). Впишите число в нужный месяц, чтобы заменить базу: например, увеличить ФОТ "
               "с октября или убрать разовый платёж. Расходы новых направлений добавляйте на вкладке «Инициативы».")
    exp_df = PL.expense_matrix(base_layers, source, groups)
    exp_edit = st.data_editor(exp_df, hide_index=True, width="stretch", height=520, key=f"pl_exp_{ver}",
                              disabled=["Статья", "Группа ДДС", "База, ₽/мес"],
                              column_config={**month_cfg, "База, ₽/мес": st.column_config.NumberColumn("База, ₽/мес", format="%d")})

with tab_rec:
    st.caption("Поступления по маркетплейсам: база с трендом и сезонностью (шаг 1), поправка факторов в % (шаг 2), ручная замена месяца (шаг 3).")
    rec_edit = st.data_editor(PL.receipts_matrix(base_layers, source), hide_index=True, width="stretch", key=f"pl_rec_{ver}",
                              disabled=["Строка"], column_config=month_cfg)
    st.markdown("**Факторы** — всё, что меняет выручку, но не видно в истории: цена, рост заказов, выкуп, реклама, ассортимент, остатки.")
    fac_src = source.factors.copy()
    for col in ("month_from", "month_to"):
        fac_src[col] = pd.to_datetime(fac_src[col], errors="coerce")
    fac_edit = st.data_editor(
        fac_src, num_rows="dynamic", hide_index=True, width="stretch", key=f"pl_fac_{ver}",
        column_config={"mp": st.column_config.SelectboxColumn("Маркетплейс", options=[*PL.STREAMS, "Все"], required=True),
                       "factor": st.column_config.SelectboxColumn("Фактор", options=list(PL.FACTORS), required=True),
                       "month_from": st.column_config.DateColumn("С месяца", format="MM.YYYY", required=True),
                       "month_to": st.column_config.DateColumn("По месяц (пусто — до конца)", format="MM.YYYY"),
                       "effect_pct": st.column_config.NumberColumn("Эффект, % к выручке", format="%.1f", min_value=-100, max_value=500),
                       "comment": st.column_config.TextColumn("Комментарий", width="large")})

with tab_init:
    st.caption("Инициатива — это новое направление или расширение: выручка с разгоном плюс связанные расходы. «Месяцев разгона» — за сколько месяцев "
               "выручка выходит на зрелый уровень. Закупки и реклама считаются в % от выручки, ФОТ команды и прочее — фиксированными суммами.")
    init_src = PL.initiatives_editor_frame(source.initiatives)
    init_edit = st.data_editor(
        init_src, num_rows="dynamic", hide_index=True, width="stretch", key=f"pl_init_{ver}",
        column_config={"name": st.column_config.TextColumn("Название", required=True), "active": st.column_config.CheckboxColumn("Активна"),
                       "mp": st.column_config.SelectboxColumn("Куда поступает выручка", options=list(PL.STREAMS)),
                       "start": st.column_config.DateColumn("Старт (месяц)", format="MM.YYYY", required=True),
                       "ramp_months": st.column_config.NumberColumn("Месяцев разгона", min_value=1, max_value=36, format="%d"),
                       "revenue": st.column_config.NumberColumn("Выручка в зрелом состоянии, ₽/мес", min_value=0, format="%d"),
                       "purchases_pct": st.column_config.NumberColumn("Закупки, % выручки", min_value=0, max_value=100, format="%.0f"),
                       "ads_pct": st.column_config.NumberColumn("Реклама, % выручки", min_value=0, max_value=100, format="%.0f"),
                       "payroll": st.column_config.NumberColumn("ФОТ команды, ₽/мес", min_value=0, format="%d"),
                       "payroll_article": st.column_config.SelectboxColumn("Статья ФОТ", options=list(("ФОТ управление", "ФОТ Маркетинг", "ФОТ Склад и логистика"))),
                       "other": st.column_config.NumberColumn("Прочие расходы, ₽/мес", min_value=0, format="%d"),
                       "end": st.column_config.DateColumn("Окончание (месяц)", format="MM.YYYY"),
                       "comment": st.column_config.TextColumn("Комментарий", width="large")})

# ---- assemble the live plan from the editors
def _months(series: pd.Series) -> pd.Series:
    return series.map(lambda v: PL.parse_month(v) if not pd.isna(v) else pd.NaT)


factors_live = fac_edit.copy()
factors_live["month_from"], factors_live["month_to"] = _months(factors_live["month_from"]), _months(factors_live["month_to"])
factors_live = factors_live.dropna(how="all").reset_index(drop=True)
init_live = init_edit.copy()
init_live["start"], init_live["end"] = _months(init_live["start"]), _months(init_live["end"])
for col in ("name", "mp", "payroll_article", "comment"):
    init_live[col] = init_live[col].fillna("").astype(str)
init_live = init_live[init_live["name"].str.strip() != ""].reset_index(drop=True)
live = PL.PlanInputs(PL.cells_from_matrices(exp_edit, rec_edit, months), factors_live, init_live)
problems = PL.validate_plan(live)
if problems:
    with tab_init:
        st.error("Исправьте, чтобы план применился: " + "; ".join(problems[:6]))
else:
    ss["plan_live"] = live

with tab_rec:
    st.markdown("**Из чего складываются поступления, тыс. ₽**")
    layers = PL.build_layers(fin.journal, fin.as_of, assump, get_plan())
    rows = []
    for s in PL.STREAMS:
        for label, key in (("база", "base"), ("факторы, %", "factor_pct"), ("замена", "override"), ("инициативы", "initiative"), ("итого", "final")):
            vals = layers.receipts[key][s]
            if key == "factor_pct":
                rows.append({"Строка": f"{s}: {label}", **{PL.month_label(m): float(vals[m]) * 100 for m in months}})
            else:
                rows.append({"Строка": f"{s}: {label}", **{PL.month_label(m): (None if pd.isna(vals[m]) else float(vals[m]) / 1000) for m in months}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={l: st.column_config.NumberColumn(l, format="%.1f") for l in mlabels})

# ---- result
with tab_res:
    plan_now = get_plan()
    key_fin = finance_key()
    sched_plan = cached_schedule(assump, load_payment_plan(), key_fin, plan_now.key(), plan_now, fin)
    base_plan = PL.empty_plan()
    sched_base = cached_schedule(assump, load_payment_plan(), key_fin, base_plan.key(), base_plan, fin)
    opening = float(cashflow.cash_by_account(fin.journal, fin.accounts, fin.as_of)["balance"].sum())

    def monthly(sched: pd.DataFrame) -> pd.DataFrame:
        g = sched.pivot_table(index="month", columns="direction", values="amount", aggfunc="sum").reindex(months).fillna(0.0)
        for col in ("in", "out"):
            if col not in g:
                g[col] = 0.0
        out = pd.DataFrame({"inflow": g["in"], "outflow": g["out"]})
        out["net"] = out["inflow"] - out["outflow"]
        out["close"] = opening + out["net"].cumsum()
        return out

    mb, mp_ = monthly(sched_base), monthly(sched_plan)
    d_close = mp_["close"] - mb["close"]
    kpi_row([
        Kpi("Поступления 12 мес., тыс. ₽", format_value(mp_["inflow"].sum(), "k0"), f"{(mp_['inflow'].sum() - mb['inflow'].sum()) / 1000:+,.0f} к базе".replace(",", " ")),
        Kpi("Платежи 12 мес., тыс. ₽", format_value(mp_["outflow"].sum(), "k0"), f"{(mp_['outflow'].sum() - mb['outflow'].sum()) / 1000:+,.0f} к базе".replace(",", " ")),
        Kpi("ЧДП 12 мес., тыс. ₽", format_value(mp_["net"].sum(), "k0"), f"{(mp_['net'].sum() - mb['net'].sum()) / 1000:+,.0f} к базе".replace(",", " "),
            good=mp_["net"].sum() >= mb["net"].sum()),
        Kpi("Остаток через 12 мес., тыс. ₽", format_value(mp_["close"].iloc[-1], "k0"), f"{d_close.iloc[-1] / 1000:+,.0f} к базе".replace(",", " "),
            good=d_close.iloc[-1] >= 0),
        Kpi("Минимум остатка, тыс. ₽", format_value(mp_["close"].min(), "k0"), PL.month_label(mp_["close"].idxmin()), good=mp_["close"].min() >= assump.buffer),
    ])
    table = pd.DataFrame({
        "Месяц": mlabels, "Поступления, база": mb["inflow"].values / 1000, "Поступления, план": mp_["inflow"].values / 1000,
        "Платежи, база": mb["outflow"].values / 1000, "Платежи, план": mp_["outflow"].values / 1000,
        "Остаток, база": mb["close"].values / 1000, "Остаток, план": mp_["close"].values / 1000, "Δ остатка": d_close.values / 1000})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.0f") for c in table.columns if c != "Месяц"})
    chart_card(fc.plan_vs_base(mlabels, mb["close"].values, mp_["close"].values, assump.buffer), "pl_cmp")
    st.caption("Месяцы заканчиваются остатком на конец месяца. Текущий месяц учитывает только оставшуюся часть. Ручной «План платежей» "
               "в обоих вариантах одинаковый и заменяет прогноз по своим статьям.")
