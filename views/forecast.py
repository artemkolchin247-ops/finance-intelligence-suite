"""Прогноз: simple forecast of receipts (GMV x fading YoY x buy-out x (1 - costs)) and expenses; the plan is edited in Excel only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core import cashflow
from core import forecast as F
from core.formatting import format_value
from ui import finance_charts as fc
from ui.components import Kpi, chart_card, kpi_row, page_header, section_title
from ui.export import to_excel_styled
from ui.scenario import scenario_select
from ui.state import (cached_schedule, finance_key, get_finance, get_ledger_repo, get_plan, load_payment_plan, scenario, scenario_name)
from ui.statement_html import Col, TRow, render_statement

page_header("Прогноз", "Прогноз выручки и расходов для ДДС и платёжного календаря: оборот год назад × затухающий рост, деньги на р/с и расходы по средним")
try:
    fin = get_finance()
    repo = get_ledger_repo()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

ss = st.session_state
plan = get_plan()
override = ss.get("plan_override")
assump = scenario()
view = F.build_forecast(fin.fact_monthly, fin.as_of, assump.scenario_lambda, plan, fin.journal)

# ------------------------------------------------------------------ source, scenario, file
kpi_row([
    Kpi("Источник плана", "Загрузка" if override else "Файл данных", override["name"] if override else "рабочий файл приложения"),
    Kpi("Сценарий", scenario_name(), f"затухание роста λ = {assump.scenario_lambda:.2f}"),
    Kpi("Поступления 12 мес., тыс. ₽", format_value(view.receipts.loc[view.months, "Итого"].sum(), "k0")),
    Kpi("Рост YoY в 1-й месяц", f"WB {view.yoy.loc[view.months[0], 'WB'] * 100:+.0f}% · Ozon {view.yoy.loc[view.months[0], 'Ozon'] * 100:+.0f}%".replace(".", ","),
        "затухает каждый месяц"),
    Kpi("Данные на дату", f"{fin.as_of:%d.%m.%Y}", "факт до этой даты"),
])
with st.container(border=True):
    st.markdown("**Все вводные меняются только в Excel-файле.** Скачайте готовый прогноз (Excel уже заполнен расчётом выбранного сценария), измените жёлтые ячейки "
                "по месяцам, маркетплейсам и статьям и загрузите обратно — прогноз, ДДС и платёжный календарь пересчитаются. В программе можно только выбрать сценарий.")
    scenario_select("fc")
    f1, f2, f3, f4 = st.columns([1.3, 1.6, 1.3, 1.1])
    articles = sorted(fin.articles["article"])
    template = to_excel_styled(F.to_template(view, plan, articles))
    f1.download_button("Скачать прогноз (Excel)", template, file_name="forecast.xlsx", key="fc_tpl", width="stretch")
    upload = f2.file_uploader("Загрузить заполненный план", type=["xlsx"], key="fc_upload", label_visibility="collapsed")
    if upload is not None:
        try:
            parsed, errors = F.parse_template(pd.read_excel(upload, sheet_name=None), set(articles), fin.fact_monthly, fin.as_of)
        except Exception as exc:  # noqa: BLE001 - an unreadable file is reported to the user
            parsed, errors = None, [f"Не удалось прочитать файл «{upload.name}»: {exc}"]
        if errors:
            st.error("План не принят:\n\n" + "\n".join(f"- {e}" for e in errors[:15]) + (f"\n- … и ещё {len(errors) - 15}" if len(errors) > 15 else ""))
        elif st.button("Применить загруженный план", type="primary", key="fc_apply"):
            ss["plan_override"] = {"plan": parsed, "name": upload.name}
            st.rerun()
    if override and f3.button("Сохранить как основной", key="fc_save", width="stretch", help="Записывает план в файл данных; его увидят все пользователи"):
        try:
            repo.save_plan_sheets(F.plan_to_sheets(override["plan"]))
        except (OSError, ValueError) as exc:
            st.error(f"Не удалось сохранить план: {exc}")
        else:
            ss["plan_saved"] = override["plan"]
            ss.pop("plan_override", None)
            st.success("План сохранён в рабочий файл данных.")
            st.rerun()
    if override and f4.button("Вернуться", key="fc_back", width="stretch", help="К плану из файла данных"):
        ss.pop("plan_override", None)
        st.rerun()

# ------------------------------------------------------------------ the table (as the "forecast" sheet of the reference workbook)
section_title("Прогноз по маркетплейсам")
months = view.hist_months + view.months
cols = [Col(f"{m:%Y-%m}", F.month_label(m), "Факт" if m in view.hist_months else "План", "Факт" if m in view.hist_months else "План") for m in months]
key = lambda m: f"{m:%Y-%m}"  # noqa: E731


def block(title: str, source: pd.DataFrame, names: list[str], fmt: str, total: bool = False, prefix: str = "") -> list[TRow]:
    rows = [TRow(f"t|{prefix}{title}", title, {c.key: np.nan for c in cols}, 0, "title")]
    for n in names:
        rows.append(TRow(f"{prefix}{title}|{n}", n, {key(m): source.loc[m, n] for m in months}, 1, "item", fmt, f"t|{prefix}{title}"))
    if total:
        rows.append(TRow(f"{prefix}{title}|total", "Итого", {key(m): source.loc[m, names].sum() for m in months}, 0, "total", fmt))
    return rows


rows = block("Оборот (GMV): заказы, тыс. ₽", view.gmv, [*F.STREAMS, "Доп. выручка"], "k0", total=True, prefix="a")
rows += block("Рост год к году, %", view.yoy, list(F.STREAMS), "pct", prefix="b")
rows += block("Выкуп, % от заказов", view.buyout, list(F.STREAMS), "pct", prefix="c")
rows += block("Расходы маркетплейса, % от продаж", view.mp_cost, list(F.STREAMS), "pct", prefix="d")
rows += block("Продажи (выкуплено), тыс. ₽", view.sales, [*F.STREAMS, "Доп. выручка"], "k0", total=True, prefix="e")
rows += block("Выплата маркетплейса начислена, тыс. ₽", view.payout, [*F.STREAMS, "Доп. выручка"], "k0", total=True, prefix="g")
rows += block("Деньги на р/с в месяце, % от GMV", view.cash_ratio, list(F.STREAMS), "pct", prefix="h")
rows += block("Поступления на р/с, тыс. ₽", view.receipts, [*F.STREAMS, "Доп. выручка"], "k0", total=True, prefix="f")
st.iframe(render_statement(cols, rows, height=640, first_col="Показатель"), height=640)
st.caption("Факт — по месячным данным маркетплейсов (те же, что в ОПиУ); прогноз — сценарий выше. Поступления — оценка денег на р/с по формуле "
           "заказы → выкупленные продажи (до 21 дня) → выплата маркетплейса → приход на р/с с задержкой в несколько недель, поэтому деньги месяца "
           "не равны выплате, начисленной в нём.")

scen = pd.DataFrame({"Сценарий": list(F.SCENARIOS), "λ": [F.SCENARIOS[n] for n in F.SCENARIOS]})
scen["Поступления 12 мес., тыс. ₽"] = [F.build_forecast(fin.fact_monthly, fin.as_of, lam, plan, fin.journal).receipts.loc[view.months, "Итого"].sum() / 1000
                                       for lam in scen["λ"]]
st.dataframe(scen, hide_index=True, width="stretch", column_config={"Поступления 12 мес., тыс. ₽": st.column_config.NumberColumn(format="%.0f")})

# ------------------------------------------------------------------ loaded inputs (read only)
section_title("Загруженные вводные")
a1, a2 = st.columns(2)
mp_table = pd.DataFrame([{
    "Маркетплейс": mp, "Рост YoY авто, %": (view.auto[mp]["yoy"] - 1) * 100, "Рост YoY ваш, %": plan.mp_value(mp, "yoy_pct"),
    "Выкуп авто, %": view.auto[mp]["buyout"] * 100, "Выкуп ваш, %": plan.mp_value(mp, "buyout_pct"),
    "Расходы МП авто, %": view.auto[mp]["mp_cost"] * 100, "Расходы МП ваш, %": plan.mp_value(mp, "mp_cost_pct")} for mp in F.STREAMS])
a1.markdown("**Допущения по маркетплейсам** (ваше значение, если задано, заменяет авто)")
a1.dataframe(mp_table, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.1f") for c in mp_table.columns[1:]})
params = pd.DataFrame({"Параметр": list(F.PARAM_KEYS.values()),
                       "Значение": [("Да" if plan.params[k] else "Нет") if (k == "scale_variable" and k in plan.params) else
                                    (f"{plan.params[k]:,.0f}".replace(",", " ") if k in plan.params else "по умолчанию") for k in F.PARAM_KEYS]})
lag_rows = []
for mp in F.STREAMS:
    row = {"Маркетплейс": mp, "Источник": "ваши данные" if plan.lag_profile(mp) else "оценка по журналу"}
    row.update({f"{k} нед.": view.lag_used[mp][k] * 100 for k in range(F.LAG_WEEKS)})
    row["Средняя задержка, нед."] = sum(k * p for k, p in enumerate(view.lag_used[mp]))
    lag_rows.append(row)
a2.markdown("**Задержка выплат:** доля выплаты за продажи недели, пришедшая на р/с через N недель, %")
a2.dataframe(pd.DataFrame(lag_rows), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.0f") for c in [f"{k} нед." for k in range(F.LAG_WEEKS)]}
             | {"Средняя задержка, нед.": st.column_config.NumberColumn(format="%.1f")})
a2.markdown("**Общие допущения**")
a2.dataframe(params, hide_index=True, width="stretch")
b1, b2 = st.columns(2)
extra = plan.extra.assign(start=plan.extra["start"].map(lambda v: F.month_label(v) if not pd.isna(v) else ""),
                          end=plan.extra["end"].map(lambda v: F.month_label(v) if not pd.isna(v) else ""))
b1.markdown("**Доп. выручка (новые направления)**")
if len(extra):
    b1.dataframe(extra.rename(columns={"name": "Название", "mp": "Маркетплейс", "start": "С месяца", "end": "По месяц", "gmv": "GMV в месяц, ₽",
                                       "comment": "Комментарий"}), hide_index=True, width="stretch")
else:
    b1.info("Не заполнено")
adj = plan.adj.assign(start=plan.adj["start"].map(lambda v: F.month_label(v) if not pd.isna(v) else ""),
                      end=plan.adj["end"].map(lambda v: F.month_label(v) if not pd.isna(v) else ""))
b2.markdown("**Корректировки расходов**")
if len(adj):
    b2.dataframe(adj.rename(columns={"article": "Статья", "start": "С месяца", "end": "По месяц", "amount": "Добавить, ₽/мес", "comment": "Комментарий"}),
                 hide_index=True, width="stretch")
else:
    b2.info("Не заполнено — расходы равны среднему за 3 последних месяца")

# ------------------------------------------------------------------ effect on cash
section_title("Что это даёт в деньгах")
key_fin = finance_key()
empty = F.empty_plan()
sched_plan = cached_schedule(assump, load_payment_plan(), key_fin, plan.key(), plan, fin)
base_assump = F.resolve_assumptions(empty, cashflow.default_assumptions(fin.journal, fin.as_of), assump.scenario_lambda)
sched_base = cached_schedule(base_assump, load_payment_plan(), key_fin, empty.key(), empty, fin)
opening = float(cashflow.cash_by_account(fin.journal, fin.accounts, fin.as_of)["balance"].sum())
mlabels = [F.month_label(m) for m in view.months]


def monthly(sched: pd.DataFrame) -> pd.DataFrame:
    g = sched.pivot_table(index="month", columns="direction", values="amount", aggfunc="sum").reindex(view.months).fillna(0.0)
    for col in ("in", "out"):
        if col not in g:
            g[col] = 0.0
    out = pd.DataFrame({"inflow": g["in"], "outflow": g["out"]})
    out["net"] = out["inflow"] - out["outflow"]
    out["close"] = opening + out["net"].cumsum()
    return out


mb, mpl = monthly(sched_base), monthly(sched_plan)
d = lambda a, b: f"{(a - b) / 1000:+,.0f} к авто-прогнозу".replace(",", " ")  # noqa: E731
kpi_row([
    Kpi("Поступления 12 мес., тыс. ₽", format_value(mpl["inflow"].sum(), "k0"), d(mpl["inflow"].sum(), mb["inflow"].sum())),
    Kpi("Платежи 12 мес., тыс. ₽", format_value(mpl["outflow"].sum(), "k0"), d(mpl["outflow"].sum(), mb["outflow"].sum())),
    Kpi("Остаток через 12 мес., тыс. ₽", format_value(mpl["close"].iloc[-1], "k0"), d(mpl["close"].iloc[-1], mb["close"].iloc[-1]),
        good=mpl["close"].iloc[-1] >= mb["close"].iloc[-1]),
    Kpi("Минимум остатка, тыс. ₽", format_value(mpl["close"].min(), "k0"), F.month_label(mpl["close"].idxmin()), good=mpl["close"].min() >= assump.buffer),
])
chart_card(fc.plan_vs_base(mlabels, mb["close"].values, mpl["close"].values, assump.buffer), "fc_cmp")
st.caption("«Авто-прогноз» — то же, но без вашего файла плана (допущения по умолчанию). Закупки, реклама и креаторы по умолчанию равны среднему за 3 месяца "
           "и не растут с выручкой; включить связь с поступлениями можно в шаблоне («Общие допущения»).")

with st.expander("Как считается прогноз"):
    st.markdown(
        "- **Оборот (GMV)** месяца = оборот заказов того же месяца год назад × рост год к году. Сезонность поэтому учтена автоматически.\n"
        "- **Рост год к году** = среднее логарифмического роста за последние 4 закрытых месяца (веса 1-2-3-4, ограничение ×0,8…×3,0) и дальше затухает: "
        "через год сохраняется доля λ от текущего роста (пессимистичный 0,45 · базовый 0,70 · оптимистичный 0,90).\n"
        "- **Заказы → продажи.** Заказ выкупается через 1–21 день (в среднем около 9), поэтому продажи месяца = выкуп % × заказы последних трёх недель, а не только этого месяца. "
        "Выкуп и расходы маркетплейса — по последним 3 месяцам, можно задать свои.\n"
        "- **Продажи → выплата → р/с.** Выплата = продажи × (1 − расходы МП). Отчёт за неделю формируется в понедельник, вывод доступен ещё через неделю, "
        "деньги приходят ещё через 0–2 недели: долю выплаты недели, приходящую через 0…5 недель, мы оцениваем по фактическим поступлениям журнала "
        "(с поправкой к стандартному профилю) или берём из вашего файла. Платежи ложатся на пн/вт (WB) и среду (Ozon); ближайшие недели определяются уже сделанными продажами.\n"
        "- **Расходы** по каждой статье = среднее за 3 последних закрытых месяца; НДС следует за поступлениями, взносы — за ФОТ; "
        "корректировки из шаблона добавляются к среднему. УСН, кредит и проценты считаются по графику.\n"
        "- **Доп. выручка** (новое направление) проходит ту же цепочку выкуп/расходы и добавляется к поступлениям выбранного маркетплейса.")
