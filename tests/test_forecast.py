import math

import numpy as np
import pandas as pd
import pytest

from core import cashflow as C
from core import forecast as F
from core.ledger import ExcelLedgerRepository, prepare_journal


@pytest.fixture(scope="module")
def env():
    repo = ExcelLedgerRepository()
    as_of = repo.get_as_of()
    j = prepare_journal(repo.get_journal(), repo.get_articles(), as_of)
    return {"repo": repo, "j": j, "as_of": as_of, "fm": repo.get_fact_monthly(), "b1c": repo.get_balance_1c(), "pp": repo.get_payment_plan(),
            "arts": set(repo.get_articles()["article"]), "a": C.default_assumptions(j, as_of)}


def _fm(growth: float, months: int = 30, start="2024-03-01") -> pd.DataFrame:
    """Synthetic monthly facts: WB grows by `growth` per month, Ozon is flat; simple fixed ratios."""
    rows = []
    for i, m in enumerate(pd.date_range(start, periods=months, freq="MS")):
        for mp, g in (("WB", growth), ("Ozon", 0.0)):
            orders = 10e6 * (1 + g) ** i
            sales = orders * 0.7
            rows.append({"month": m, "mp": mp, "orders_rub": orders, "sales_rub": sales, "sales_spp_rub": sales * 0.8,
                         "comis_spp": sales * 0.4, "logist": sales * 0.1, "storage": sales * 0.02, "accept": 0.0, "other": 0.0, "reclama": sales * 0.05,
                         "days": m.days_in_month})
    return pd.DataFrame(rows)


AS_OF = pd.Timestamp("2026-09-06")
FM = _fm(0.03)


def test_yoy_uses_weights_and_clip():
    auto = F.auto_assumptions(FM, AS_OF)
    assert auto["WB"]["yoy"] == pytest.approx(1.03 ** 12, rel=1e-6)  # constant monthly growth -> constant YoY
    assert auto["Ozon"]["yoy"] == pytest.approx(1.0)
    steep = F.auto_assumptions(_fm(0.2), AS_OF)
    assert steep["WB"]["yoy"] == F.YOY_MAX  # clipped
    # weights 1-2-3-4: the latest month matters most
    fm = FM.copy()
    last = pd.Timestamp("2026-08-01")
    base = F.auto_assumptions(fm, AS_OF)["WB"]["yoy"]
    fm.loc[(fm["month"] == last) & (fm["mp"] == "WB"), "orders_rub"] *= 1.5
    first = pd.Timestamp("2026-05-01")
    fm2 = FM.copy()
    fm2.loc[(fm2["month"] == first) & (fm2["mp"] == "WB"), "orders_rub"] *= 1.5
    assert F.auto_assumptions(fm, AS_OF)["WB"]["yoy"] - base > F.auto_assumptions(fm2, AS_OF)["WB"]["yoy"] - base > 0


def test_forecast_is_last_year_times_fading_growth():
    v = F.build_forecast(FM, AS_OF, 0.70)
    yoy = F.auto_assumptions(FM, AS_OF)["WB"]["yoy"]
    gmv = FM[FM["mp"] == "WB"].set_index("month")["orders_rub"]
    for k, m in enumerate(v.months, start=1):
        expected = gmv[m - pd.DateOffset(months=12)] * math.exp(math.log(yoy) * 0.70 ** (k / 12))
        assert v.gmv.loc[m, "WB"] == pytest.approx(expected)
    assert v.months[0] == pd.Timestamp("2026-09-01") and len(v.months) == 12 and len(v.hist_months) == 12
    # growth fades month after month
    g = [v.yoy.loc[m, "WB"] for m in v.months]
    assert all(a > b for a, b in zip(g, g[1:]))


def test_scenarios_are_ordered_and_lambda_one_keeps_growth():
    total = {n: F.build_forecast(FM, AS_OF, lam).receipts.loc[F.build_forecast(FM, AS_OF, lam).months, "Итого"].sum() for n, lam in F.SCENARIOS.items()}
    assert total["Пессимистичный"] < total["Базовый"] < total["Оптимистичный"]
    full = F.build_forecast(FM, AS_OF, 1.0)
    assert full.yoy.loc[full.months[-1], "WB"] == pytest.approx(F.auto_assumptions(FM, AS_OF)["WB"]["yoy"] - 1)


def test_payout_chain_and_cash_delay():
    v = F.build_forecast(FM, AS_OF, 0.7)
    a = F.auto_assumptions(FM, AS_OF)["Ozon"]
    assert a["buyout"] == pytest.approx(0.7)
    for m in v.months[2:]:  # flat marketplace: sales and payout follow the orders once the buy-out lag has passed
        assert v.sales.loc[m, "Ozon"] == pytest.approx(v.gmv.loc[m, "Ozon"] * a["buyout"], rel=0.03)
        assert v.payout.loc[m, "Ozon"] == pytest.approx(v.sales.loc[m, "Ozon"] * (1 - a["mp_cost"]), rel=0.03)
    # the tail of the last weeks is still on its way at the end of the horizon, so the cash is below the accrued payout
    assert v.receipts.loc[v.months, "WB"].sum() < v.payout.loc[v.months, "WB"].sum()
    # payments are dated after the report date, on the marketplace's weekdays, and add up to the monthly cash
    assert (v.payments["date"] > AS_OF).all()
    assert set(v.payments[v.payments["stream"] == "Ozon"]["date"].dt.weekday) == {2}
    assert v.payments[v.payments["stream"] == "WB"]["date"].dt.weekday.isin([0, 1]).all()
    assert (v.payments["base"] + v.payments["extra"]).sum() == pytest.approx(v.receipts_by_stream.loc[v.months].to_numpy().sum())


def test_delay_profile_shifts_cash_and_conserves_the_total():
    fast = F.PlanInputs(lag=pd.DataFrame([{"mp": "WB", "w0": 100.0}, {"mp": "Ozon", "w0": 100.0}]))
    slow = F.PlanInputs(lag=pd.DataFrame([{"mp": "WB", "w4": 100.0}, {"mp": "Ozon", "w4": 100.0}]))
    vf, vs = F.build_forecast(FM, AS_OF, 0.7, fast), F.build_forecast(FM, AS_OF, 0.7, slow)
    assert vf.lag_used["WB"][0] == 1.0 and vs.lag_used["WB"][4] == 1.0
    assert vf.receipts.loc[vf.months[:4], "Итого"].sum() > vs.receipts.loc[vs.months[:4], "Итого"].sum()
    # the delay only moves money in time: the same payouts arrive, minus the weeks that fall beyond the horizon
    assert vs.receipts.loc[vs.months, "Итого"].sum() == pytest.approx(vf.receipts.loc[vf.months, "Итого"].sum(), rel=0.12)


def test_lag_profile_is_estimated_from_the_journal(env):
    est = F.estimate_lag(env["fm"], env["j"], env["as_of"])
    for mp, true in (("WB", (0.0, 0.04, 0.38, 0.36, 0.17, 0.05)), ("Ozon", (0.0, 0.12, 0.46, 0.30, 0.12, 0.0))):
        assert sum(est[mp]) == pytest.approx(1.0)
        assert sum(k * p for k, p in enumerate(est[mp])) == pytest.approx(sum(k * p for k, p in enumerate(true)), abs=0.25)  # mean delay, weeks
    short = F.estimate_lag(env["fm"], env["j"][env["j"]["dds_date"] > pd.Timestamp("2026-06-01")], env["as_of"])
    assert short == F.DEFAULT_LAG  # too little history -> standard profile


def test_user_values_replace_auto_and_extra_revenue_adds():
    plan = F.PlanInputs(pd.DataFrame([{"mp": "WB", "yoy_pct": 20.0, "buyout_pct": 80.0, "mp_cost_pct": 50.0}]),
                        extra=pd.DataFrame([{"name": "Сайт", "mp": "Ozon", "start": pd.Timestamp("2026-11-01"), "end": pd.Timestamp("2026-12-01"),
                                             "gmv": 1_000_000.0, "comment": ""}]))
    v = F.build_forecast(FM, AS_OF, 0.7, plan)
    assert v.used["WB"]["yoy"] == pytest.approx(1.2) and v.used["WB"]["buyout"] == 0.8 and v.used["WB"]["mp_cost"] == 0.5
    assert v.used["Ozon"] == F.build_forecast(FM, AS_OF, 0.7).used["Ozon"]  # untouched marketplace stays automatic
    base = F.build_forecast(FM, AS_OF, 0.7, F.PlanInputs(plan.mp))
    nov, jan = pd.Timestamp("2026-11-01"), pd.Timestamp("2027-01-01")
    assert v.gmv.loc[nov, "Доп. выручка"] == 1_000_000.0 and v.gmv.loc[jan, "Доп. выручка"] == 0.0
    oz = F.auto_assumptions(FM, AS_OF)["Ozon"]
    extra = v.extra_receipts["Ozon"]
    assert extra[:pd.Timestamp("2026-10-01")].sum() == 0  # nothing before the first month
    assert extra.sum() == pytest.approx(2_000_000.0 * oz["buyout"] * (1 - oz["mp_cost"]), rel=1e-6)  # both months arrive within the horizon, delayed
    assert extra[pd.Timestamp("2026-12-01")] > extra[pd.Timestamp("2026-11-01")] > 0
    assert (v.receipts["Итого"] - base.receipts["Итого"]).loc[v.months].sum() == pytest.approx(extra.sum())


def test_expenses_are_three_month_averages_plus_corrections(env):
    a = env["a"]
    base = F.build_layers(env["j"], env["fm"], env["as_of"], a, None)
    j = env["j"]
    last3 = [pd.Timestamp(f"2026-0{m}-01") for m in (6, 7, 8)]
    h = j[(j["article"] == "Аренда помещения") & (j["dds_date"].dt.to_period("M").dt.to_timestamp().isin(last3))]
    assert base.expenses["base"].loc["Аренда помещения", base.months[3]] == pytest.approx(h["expense"].sum() / 3)
    assert base.expenses["final"].equals(base.expenses["base"].clip(lower=0))
    adj = pd.DataFrame([{"article": "ФОТ Маркетинг", "start": pd.Timestamp("2026-11-01"), "end": pd.Timestamp("2027-01-01"), "amount": 300_000.0, "comment": ""},
                        {"article": "Аренда помещения", "start": pd.Timestamp("2027-02-01"), "end": pd.NaT, "amount": -10_000_000.0, "comment": ""}])
    lay = F.build_layers(j, env["fm"], env["as_of"], a, F.PlanInputs(adj=adj))
    nov, jan, feb = pd.Timestamp("2026-11-01"), pd.Timestamp("2027-01-01"), pd.Timestamp("2027-02-01")
    assert lay.expenses["final"].loc["ФОТ Маркетинг", nov] == pytest.approx(base.expenses["final"].loc["ФОТ Маркетинг", nov] + 300_000)
    assert lay.expenses["final"].loc["ФОТ Маркетинг", feb] == pytest.approx(base.expenses["final"].loc["ФОТ Маркетинг", feb])  # window ended
    assert lay.expenses["final"].loc["Аренда помещения", feb] == 0.0  # a large negative correction never makes an expense negative
    assert lay.expenses["final"].loc[C.PAYROLL_TAX, nov] > base.expenses["final"].loc[C.PAYROLL_TAX, nov]  # contributions follow payroll


def test_purchases_follow_receipts_only_when_asked(env):
    plan = F.PlanInputs(extra=pd.DataFrame([{"name": "N", "mp": "WB", "start": pd.Timestamp("2026-11-01"), "end": pd.NaT, "gmv": 20e6, "comment": ""}]))
    flat = F.build_layers(env["j"], env["fm"], env["as_of"], env["a"], plan).expenses["final"].loc["Закупка товара"]
    tied = F.build_layers(env["j"], env["fm"], env["as_of"], C.Assumptions(**{**env["a"].__dict__, "scale_variable": True}), plan).expenses["final"].loc["Закупка товара"]
    jan = pd.Timestamp("2027-01-01")  # the extra revenue is paid with a delay: fully visible in cash from December
    assert tied[jan] > flat[jan] * 1.5 and tied[pd.Timestamp("2026-09-01")] <= flat[pd.Timestamp("2026-09-01")] * 1.3


def test_schedule_matches_layers_and_plan_moves_cash(env):
    a = env["a"]
    plan = F.PlanInputs(extra=pd.DataFrame([{"name": "N", "mp": "Сайт/другое", "start": pd.Timestamp("2027-01-01"), "end": pd.NaT, "gmv": 2e6, "comment": ""}]),
                        adj=pd.DataFrame([{"article": "ФОТ Маркетинг", "start": pd.Timestamp("2027-01-01"), "end": pd.NaT, "amount": 250_000.0, "comment": ""}]))
    base = C.build_schedule(env["j"], env["b1c"], env["pp"], env["as_of"], a, fact_monthly=env["fm"])
    sched = C.build_schedule(env["j"], env["b1c"], env["pp"], env["as_of"], a, plan=plan, fact_monthly=env["fm"])
    lay = F.build_layers(env["j"], env["fm"], env["as_of"], a, plan)
    m = pd.Timestamp("2027-03-01")
    got = sched[(sched["article"] == "Продажи через сайт") & (sched["month"] == m)]["amount"].sum()
    assert got == pytest.approx(lay.receipts.loc[m, "Сайт/другое"]) and got > 0
    wb = sched[(sched["article"] == "Продажи Wildberries") & (sched["month"] == m)]["amount"].sum()
    assert wb == pytest.approx(lay.receipts.loc[m, "WB"])
    assert sched[sched["direction"] == "in"]["amount"].sum() > base[base["direction"] == "in"]["amount"].sum()
    fot = lambda s: s[(s["article"] == "ФОТ Маркетинг") & (s["month"] == m)]["amount"].sum()  # noqa: E731
    assert fot(sched) - fot(base) == pytest.approx(250_000.0)
    assert set(sched[sched["article"] == "Продажи через сайт"]["source"]) == {"План"}


def test_resolve_assumptions_prefers_the_plan(env):
    base = env["a"]
    plan = F.PlanInputs(params={"purchases_pct": 120.0, "dividends": 0.0, "buffer": 3e6, "payroll_index_pct": 10.0, "scale_variable": True})
    r = F.resolve_assumptions(plan, base, 0.9)
    assert (r.scenario_lambda, r.purchase_scale, r.dividends_month, r.buffer, r.payroll_index, r.scale_variable) == (0.9, 1.2, 0.0, 3e6, 0.1, True)
    d = F.resolve_assumptions(F.empty_plan(), base, 0.7)
    assert d.dividends_month == base.dividends_month and d.purchase_scale == 1.0 and not d.scale_variable


def test_template_roundtrip_and_validation(env):
    view = F.build_forecast(env["fm"], env["as_of"], 0.7)
    plan = F.PlanInputs(pd.DataFrame([{"mp": "Ozon", "yoy_pct": 55.0, "buyout_pct": np.nan, "mp_cost_pct": 40.0}]),
                        {"purchases_pct": 110.0, "scale_variable": True, "dividends": 700_000.0},
                        pd.DataFrame([{"name": "Сайт", "mp": "Сайт/другое", "start": pd.Timestamp("2027-02-01"), "end": pd.NaT, "gmv": 1.5e6, "comment": "новое"}]),
                        pd.DataFrame([{"article": "ФОТ Маркетинг", "start": pd.Timestamp("2026-12-01"), "end": pd.NaT, "amount": 200_000.0, "comment": "найм"}]))
    sheets = F.to_template(view, plan, sorted(env["arts"]))
    assert list(sheets) == ["Прогноз по месяцам", "Допущения МП", "Задержка выплат", "Общие допущения", "Доп. выручка", "Корректировки расходов", "Как заполнять"]
    back, errors = F.parse_template({k: v[0] for k, v in sheets.items()}, env["arts"], env["fm"], env["as_of"])
    assert errors == []
    assert back.mp_value("Ozon", "yoy_pct") == 55.0 and back.mp_value("Ozon", "buyout_pct") is None and back.mp_value("WB", "yoy_pct") is None
    assert back.params["purchases_pct"] == 110.0 and back.params["scale_variable"] is True and back.params["dividends"] == 700_000.0
    assert back.extra.iloc[0]["gmv"] == 1.5e6 and back.extra.iloc[0]["start"] == pd.Timestamp("2027-02-01")
    assert back.adj.iloc[0]["article"] == "ФОТ Маркетинг" and back.adj.iloc[0]["amount"] == 200_000.0
    assert back.key() == F.plan_from_sheets(F.plan_to_sheets(back)).key()
    # the untouched template (example rows without amounts) is an empty plan
    blank, errs = F.parse_template({k: v[0] for k, v in F.to_template(view, F.empty_plan(), []).items()}, env["arts"])
    assert errs == [] and blank.extra.empty and blank.adj.empty and blank.params == {}


def test_import_errors_are_readable(env):
    view = F.build_forecast(env["fm"], env["as_of"], 0.7)
    sheets = {k: v[0].copy() for k, v in F.to_template(view, F.empty_plan(), []).items()}
    sheets["Допущения МП"].loc[0, "Выкуп, %"] = 140
    assert any("от 0 до 100" in e for e in F.parse_template(sheets, env["arts"])[1])
    sheets["Корректировки расходов"] = pd.DataFrame([{"Статья": "Нет такой", "С месяца": "кривая дата", "По месяц": None, "Добавить, ₽/мес": 5000, "Комментарий": ""}])
    _, errors = F.parse_template(sheets, env["arts"])
    assert any("не похоже на месяц" in e for e in errors)
    _, errors = F.parse_template({"Допущения МП": sheets["Допущения МП"]}, env["arts"])
    assert any("нет листа" in e for e in errors)
    sheets2 = {k: v[0].copy() for k, v in F.to_template(view, F.empty_plan(), []).items()}
    sheets2["Корректировки расходов"] = pd.DataFrame([{"Статья": "Нет такой", "С месяца": "10.2026", "По месяц": None, "Добавить, ₽/мес": 5000, "Комментарий": ""}])
    assert any("отсутствует в справочнике" in e for e in F.parse_template(sheets2, env["arts"])[1])


def test_plan_sheets_roundtrip_through_repository(tmp_path):
    path = tmp_path / "l.xlsx"
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"a": [1]}).to_excel(xw, sheet_name="journal", index=False)
    repo = ExcelLedgerRepository(path)
    assert F.plan_from_sheets(repo.get_plan_sheets()).key() == F.empty_plan().key()
    plan = F.PlanInputs(pd.DataFrame([{"mp": "WB", "yoy_pct": 30.0, "buyout_pct": np.nan, "mp_cost_pct": np.nan}]), {"buffer": 1e6},
                        pd.DataFrame([{"name": "N", "mp": "WB", "start": pd.Timestamp("2027-01-01"), "end": pd.NaT, "gmv": 1e6, "comment": ""}]),
                        pd.DataFrame([{"article": "РКО", "start": pd.Timestamp("2027-01-01"), "end": pd.NaT, "amount": 1000.0, "comment": ""}]))
    repo.save_plan_sheets(F.plan_to_sheets(plan))
    again = F.plan_from_sheets(ExcelLedgerRepository(path).get_plan_sheets())
    assert again.key() == plan.key() and "journal" in pd.ExcelFile(path).sheet_names


def test_edited_month_cells_become_overrides_and_roundtrip(env):
    view = F.build_forecast(env["fm"], env["as_of"], 0.9)
    sheets = {k: v[0].copy() for k, v in F.to_template(view, F.empty_plan(), []).items()}
    month = sheets["Прогноз по месяцам"]
    assert (month["Сценарий"] == "Оптимистичный").all() and len(month) == 24
    untouched, errors = F.parse_template(sheets, env["arts"], env["fm"], env["as_of"])
    assert errors == [] and untouched.monthly.empty  # downloading and re-uploading without edits changes nothing
    i = month.index[(month["Маркетплейс"] == "WB") & (month["Месяц"] == "11.2026")][0]
    month.loc[i, "Оборот (GMV), ₽"] = month.loc[i, "Оборот (GMV), ₽"] * 2
    month.loc[i, "Выкуп, %"] = 90.0
    plan, errors = F.parse_template(sheets, env["arts"], env["fm"], env["as_of"])
    nov = pd.Timestamp("2026-11-01")
    assert errors == [] and len(plan.monthly) == 1 and pd.isna(plan.monthly.iloc[0]["mp_cost_pct"])
    edited, base = F.build_forecast(env["fm"], env["as_of"], 0.9, plan), F.build_forecast(env["fm"], env["as_of"], 0.9)
    assert edited.gmv.loc[nov, "WB"] == pytest.approx(base.gmv.loc[nov, "WB"] * 2, rel=1e-5)
    assert edited.buyout.loc[nov, "WB"] == 0.9 and edited.mp_cost.loc[nov, "WB"] == base.mp_cost.loc[nov, "WB"]
    assert edited.sales.loc[nov, "WB"] > base.sales.loc[nov, "WB"] * 1.5  # the edit reaches sales, payout and later cash
    assert edited.receipts.loc[pd.Timestamp("2027-01-01"), "WB"] > base.receipts.loc[pd.Timestamp("2027-01-01"), "WB"]
    oct_ = pd.Timestamp("2026-10-01")
    assert edited.sales.loc[oct_, "WB"] == pytest.approx(base.sales.loc[oct_, "WB"])  # earlier months stay on the model
    assert edited.receipts.loc[oct_, "WB"] == pytest.approx(base.receipts.loc[oct_, "WB"])
    # the edit survives storage and a second download
    again = F.plan_from_sheets(F.plan_to_sheets(plan))
    assert again.key() == plan.key()
    second = F.to_template(edited, plan, [])["Прогноз по месяцам"][0]
    assert second.loc[i, "Выкуп, %"] == 90.0
    month.loc[i, "Выкуп, %"] = 150.0
    assert any("от 0 до 100" in e for e in F.parse_template(sheets, env["arts"], env["fm"], env["as_of"])[1])


def test_lag_sheet_roundtrip_and_validation(env):
    view = F.build_forecast(env["fm"], env["as_of"], 0.7, None, env["j"])
    sheets = {k: v[0].copy() for k, v in F.to_template(view, F.empty_plan(), []).items()}
    lag = sheets["Задержка выплат"]
    assert abs(sum(lag.loc[0, f"Авто, через {k} нед., %"] for k in range(F.LAG_WEEKS)) - 100) < 1  # shows the estimated profile
    plan, errors = F.parse_template(sheets, env["arts"], env["fm"], env["as_of"])
    assert errors == [] and plan.lag.empty  # untouched -> estimated profile stays in use
    lag.loc[0, "Через 1 нед., %"], lag.loc[0, "Через 3 нед., %"] = 50.0, 50.0
    plan, errors = F.parse_template(sheets, env["arts"], env["fm"], env["as_of"])
    assert errors == [] and plan.lag_profile("WB") == (0.0, 0.5, 0.0, 0.5, 0.0, 0.0) and plan.lag_profile("Ozon") is None
    assert F.plan_from_sheets(F.plan_to_sheets(plan)).lag_profile("WB") == plan.lag_profile("WB")
    lag.loc[0, "Через 3 нед., %"] = 5.0
    assert any("ожидается около 100" in e for e in F.parse_template(sheets, env["arts"], env["fm"], env["as_of"])[1])
    lag.loc[0, "Через 3 нед., %"] = -5.0
    assert any("от 0 до 100" in e for e in F.parse_template(sheets, env["arts"], env["fm"], env["as_of"])[1])
