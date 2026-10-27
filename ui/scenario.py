"""Forecast scenario: levers kept in a plain session dict (widget values vanish when their page is not shown)."""
from __future__ import annotations

import streamlit as st

from core import cashflow
from core.planning import SCENARIOS


def _store() -> dict:
    return st.session_state.setdefault("scn", {})


def scenario_for(fin) -> cashflow.Assumptions:
    """Assumptions of the current session; anything the user has not touched comes from the history."""
    base = cashflow.default_assumptions(fin.journal, fin.as_of)
    s = _store()
    receipts_scale, init_weight = SCENARIOS[s.get("scenario", "Базовый")]
    return cashflow.Assumptions(
        growth_monthly=s.get("growth", round(base.growth_monthly * 100, 1)) / 100, seasonality=s.get("season", True),
        purchase_scale=s.get("purchase", 100) / 100, payroll_index=s.get("payroll", 0.0) / 100,
        dividends_month=s.get("div", base.dividends_month / 1000) * 1000, buffer=s.get("buffer", 2000) * 1000.0,
        receipts_scale=receipts_scale, initiative_weight=init_weight, scale_variable=s.get("scale_var", False))


def scenario_select(prefix: str) -> str:
    s = _store()
    names = list(SCENARIOS)
    choice = st.selectbox("Сценарий", names, index=names.index(s.get("scenario", "Базовый")), key=f"{prefix}_scenario",
                          help="Оптимистичный: поступления +10%. Пессимистичный: поступления −10% и только половина выручки инициатив")
    s["scenario"] = choice
    return choice


def scenario_controls(prefix: str, fin) -> None:
    """Full set of levers (ДДС page)."""
    base = cashflow.default_assumptions(fin.journal, fin.as_of)
    s = _store()
    a1, a2, a3 = st.columns(3)
    with a1:
        scenario_select(prefix)
        s["growth"] = st.number_input("Рост поступлений от МП, % в месяц", -10.0, 15.0, value=float(s.get("growth", round(base.growth_monthly * 100, 1))),
                                      step=0.5, key=f"{prefix}_growth", help="По умолчанию — тренд последних 6 закрытых месяцев (не выше 2% в месяц)")
        s["season"] = st.checkbox("Учитывать сезонность", value=s.get("season", True), key=f"{prefix}_season")
    with a2:
        s["purchase"] = st.number_input("Закупки и связанные расходы, % от среднего за 3 месяца", 50, 200, value=int(s.get("purchase", 100)),
                                        step=5, key=f"{prefix}_purchase")
        s["scale_var"] = st.checkbox("Менять их вместе с поступлениями", value=s.get("scale_var", False), key=f"{prefix}_scale_var",
                                     help="Закупки, реклама и креаторы растут и падают пропорционально поступлениям; по умолчанию — просто среднее за 3 месяца")
        s["payroll"] = st.number_input("Индексация ФОТ, %", 0.0, 30.0, value=float(s.get("payroll", 0.0)), step=1.0, key=f"{prefix}_payroll")
    with a3:
        s["div"] = st.number_input("Дивиденды, тыс. ₽ в месяц", 0, 20000, value=int(s.get("div", base.dividends_month / 1000)), step=100,
                                   key=f"{prefix}_div")
        s["buffer"] = st.number_input("Минимальный остаток, тыс. ₽", 0, 50000, value=int(s.get("buffer", 2000)), step=250, key=f"{prefix}_buffer",
                                      help="Остаток ниже этого уровня подсвечивается как риск кассового разрыва")
