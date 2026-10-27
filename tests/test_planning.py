import numpy as np
import pandas as pd
import pytest

from core import cashflow as C
from core import planning as PL
from core.ledger import ExcelLedgerRepository, prepare_journal


@pytest.fixture(scope="module")
def env():
    repo = ExcelLedgerRepository()
    as_of = repo.get_as_of()
    j = prepare_journal(repo.get_journal(), repo.get_articles(), as_of)
    return {"repo": repo, "j": j, "as_of": as_of, "b1c": repo.get_balance_1c(), "pp": repo.get_payment_plan(),
            "a": C.default_assumptions(j, as_of), "arts": repo.get_articles()}


def _plan(**kw):
    p = PL.empty_plan()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _init(**over):
    row = {"name": "Новое направление", "active": True, "mp": "Сайт/другое", "start": pd.Timestamp("2026-11-01"), "ramp_months": 4,
           "revenue": 1_000_000.0, "purchases_pct": 50.0, "ads_pct": 10.0, "payroll": 200_000.0, "payroll_article": "ФОТ Маркетинг",
           "other": 30_000.0, "end": pd.NaT, "comment": ""}
    row.update(over)
    return pd.DataFrame([row])


def test_expense_base_is_three_month_average(env):
    layers = PL.build_layers(env["j"], env["as_of"], env["a"], None)
    j = env["j"]
    last3 = [pd.Timestamp(f"2026-0{m}-01") for m in (6, 7, 8)]
    h = j[(j["article"] == "Аренда помещения") & (j["dds_date"].dt.to_period("M").dt.to_timestamp().isin(last3))]
    assert layers.expenses["base"].loc["Аренда помещения", layers.months[3]] == pytest.approx(h["expense"].sum() / 3)
    assert layers.expenses["final"].equals(layers.expenses["base"])  # nothing replaced, no initiatives


def test_replacement_wins_over_base_and_reaches_the_schedule(env):
    month = pd.Timestamp("2026-12-01")
    plan = _plan(cells=pd.DataFrame([{"sheet": PL.SHEET_EXP, "row": "Аренда помещения", "month": month, "value": 900_000.0}]))
    layers = PL.build_layers(env["j"], env["as_of"], env["a"], plan)
    assert layers.expenses["final"].loc["Аренда помещения", month] == 900_000.0
    assert layers.expenses["override"].loc["Аренда помещения"].notna().sum() == 1
    sched = C.build_schedule(env["j"], env["b1c"], env["pp"], env["as_of"], env["a"], plan=plan)
    got = sched[(sched["article"] == "Аренда помещения") & (sched["month"] == month)]["amount"].sum()
    assert got == pytest.approx(900_000.0)
    other = sched[(sched["article"] == "Аренда помещения") & (sched["month"] == pd.Timestamp("2027-01-01"))]["amount"].sum()
    assert other == pytest.approx(layers.expenses["base"].loc["Аренда помещения", pd.Timestamp("2027-01-01")])


def test_initiative_adds_revenue_with_ramp_and_its_own_costs(env):
    plan = _plan(initiatives=_init())
    layers = PL.build_layers(env["j"], env["as_of"], env["a"], plan)
    rev = layers.receipts["initiative"]["Сайт/другое"]
    assert rev[pd.Timestamp("2026-10-01")] == 0 and rev[pd.Timestamp("2026-11-01")] == pytest.approx(250_000)  # 1 of 4 ramp months
    assert rev[pd.Timestamp("2027-02-01")] == pytest.approx(1_000_000) and rev[pd.Timestamp("2027-05-01")] == pytest.approx(1_000_000)
    m = pd.Timestamp("2027-03-01")
    inc = layers.expenses["initiative"]
    assert inc.loc["Закупка товара", m] == pytest.approx(500_000) and inc.loc["Оплата внешних рекламных каналов", m] == pytest.approx(100_000)
    assert inc.loc["ФОТ Маркетинг", m] == pytest.approx(200_000) and inc.loc["Прочие операционные расходы", m] == pytest.approx(30_000)
    base = PL.build_layers(env["j"], env["as_of"], env["a"], None)
    assert layers.receipts["final"].loc[m, "WB"] == pytest.approx(base.receipts["final"].loc[m, "WB"])  # other streams untouched
    assert layers.expenses["final"].loc[C.PAYROLL_TAX, m] > base.expenses["final"].loc[C.PAYROLL_TAX, m]  # new team -> contributions


def test_inactive_and_ended_initiatives(env):
    off = PL.build_layers(env["j"], env["as_of"], env["a"], _plan(initiatives=_init(active=False)))
    assert off.receipts["initiative"].to_numpy().sum() == 0
    ended = PL.build_layers(env["j"], env["as_of"], env["a"], _plan(initiatives=_init(end=pd.Timestamp("2026-12-01"))))
    assert ended.receipts["initiative"]["Сайт/другое"][pd.Timestamp("2027-01-01")] == 0


def test_factor_changes_only_its_marketplace_and_months(env):
    f = pd.DataFrame([{"mp": "WB", "factor": "Цена", "month_from": pd.Timestamp("2026-11-01"), "month_to": pd.Timestamp("2026-12-01"),
                       "effect_pct": 10.0, "comment": ""}])
    base = PL.build_layers(env["j"], env["as_of"], env["a"], None).receipts["final"]
    lay = PL.build_layers(env["j"], env["as_of"], env["a"], _plan(factors=f)).receipts["final"]
    nov, jan = pd.Timestamp("2026-11-01"), pd.Timestamp("2027-01-01")
    assert lay.loc[nov, "WB"] == pytest.approx(base.loc[nov, "WB"] * 1.10)
    assert lay.loc[nov, "Ozon"] == pytest.approx(base.loc[nov, "Ozon"]) and lay.loc[jan, "WB"] == pytest.approx(base.loc[jan, "WB"])


def test_receipts_replacement_and_scenarios(env):
    nov = pd.Timestamp("2026-11-01")
    plan = _plan(cells=pd.DataFrame([{"sheet": PL.SHEET_REC, "row": "WB", "month": nov, "value": 20_000_000.0}]))
    assert PL.build_layers(env["j"], env["as_of"], env["a"], plan).receipts["final"].loc[nov, "WB"] == 20_000_000.0
    opt = C.Assumptions(**{**env["a"].__dict__, "receipts_scale": 1.1})
    base = PL.build_layers(env["j"], env["as_of"], env["a"], None).receipts["final"].loc[nov, "WB"]
    assert PL.build_layers(env["j"], env["as_of"], opt, None).receipts["final"].loc[nov, "WB"] == pytest.approx(base * 1.1)
    pess = C.Assumptions(**{**env["a"].__dict__, "initiative_weight": 0.5})
    full = PL.build_layers(env["j"], env["as_of"], env["a"], _plan(initiatives=_init())).receipts["initiative"].to_numpy().sum()
    half = PL.build_layers(env["j"], env["as_of"], pess, _plan(initiatives=_init())).receipts["initiative"].to_numpy().sum()
    assert half == pytest.approx(full / 2)


def test_scale_variable_ties_purchases_to_receipts(env):
    nov = pd.Timestamp("2026-11-01")
    plan = _plan(factors=pd.DataFrame([{"mp": "Все", "factor": "Рост заказов", "month_from": nov, "month_to": pd.NaT, "effect_pct": 30.0, "comment": ""}]))
    flat = PL.build_layers(env["j"], env["as_of"], env["a"], plan).expenses["final"].loc["Закупка товара", nov]
    tied = PL.build_layers(env["j"], env["as_of"], C.Assumptions(**{**env["a"].__dict__, "scale_variable": True}), plan).expenses["final"].loc["Закупка товара", nov]
    assert tied > flat * 1.2


def test_schedule_follows_layers_and_initiative_moves_cash(env):
    plan = _plan(initiatives=_init())
    a = env["a"]
    base = C.build_schedule(env["j"], env["b1c"], env["pp"], env["as_of"], a)
    sched = C.build_schedule(env["j"], env["b1c"], env["pp"], env["as_of"], a, plan=plan)
    m = pd.Timestamp("2027-03-01")
    layers = PL.build_layers(env["j"], env["as_of"], a, plan)
    got = sched[(sched["article"] == "Продажи через сайт") & (sched["month"] == m)]["amount"].sum()
    assert got == pytest.approx(layers.receipts["final"].loc[m, "Сайт/другое"])
    assert sched[sched["direction"] == "in"]["amount"].sum() > base[base["direction"] == "in"]["amount"].sum()
    assert set(sched[sched["article"] == "Продажи через сайт"]["source"]) == {"Планирование"}


def test_template_roundtrip_and_validation(env):
    f = pd.DataFrame([{"mp": "Ozon", "factor": "Реклама", "month_from": pd.Timestamp("2026-10-01"), "month_to": pd.Timestamp("2026-12-01"),
                       "effect_pct": 7.5, "comment": "бюджет"}])
    plan = _plan(cells=pd.DataFrame([{"sheet": PL.SHEET_EXP, "row": "Аренда помещения", "month": pd.Timestamp("2026-12-01"), "value": 1.0e6}]),
                 factors=f, initiatives=PL.initiatives_editor_frame(_init()))
    layers = PL.build_layers(env["j"], env["as_of"], env["a"], plan)
    groups = env["arts"].set_index("article")["dds_group"].to_dict()
    sheets = PL.to_template(layers, plan, groups)
    assert {"Расходы", "Поступления", "Факторы", "Инициативы", "База (справка)"} <= set(sheets)
    back, errors = PL.parse_template(sheets)
    assert errors == [] and back.override(PL.SHEET_EXP, "Аренда помещения", pd.Timestamp("2026-12-01")) == 1.0e6
    assert back.factors.iloc[0]["effect_pct"] == 7.5 and back.initiatives.iloc[0]["name"] == "Новое направление"
    assert back.key() == back.key() and back.key() != PL.empty_plan().key()
    bad = {**sheets, "Расходы": sheets["Расходы"].rename(columns={PL.month_label(layers.months[2]): "кривой заголовок"})}
    assert any("не похож на месяц" in e for e in PL.parse_template(bad)[1])
    neg = sheets["Поступления"].copy()
    neg[PL.month_label(layers.months[1])] = [-5.0, None, None]
    assert any("неотрицательное" in e for e in PL.parse_template({**sheets, "Поступления": neg})[1])
    assert any("не указан месяц старта" in e for e in PL.validate_plan(_plan(initiatives=_init(start=pd.NaT))))
    assert any("от 0 до 100" in e for e in PL.validate_plan(_plan(initiatives=_init(purchases_pct=150.0))))


def test_plan_sheets_roundtrip_through_repository(tmp_path):
    path = tmp_path / "l.xlsx"
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"a": [1]}).to_excel(xw, sheet_name="journal", index=False)
    repo = ExcelLedgerRepository(path)
    assert PL.plan_from_sheets(repo.get_plan_sheets()).key() == PL.empty_plan().key()  # nothing saved yet
    plan = _plan(initiatives=PL.initiatives_editor_frame(_init()),
                 cells=pd.DataFrame([{"sheet": PL.SHEET_REC, "row": "WB", "month": pd.Timestamp("2026-11-01"), "value": 5.0}]))
    repo.save_plan_sheets(PL.plan_to_sheets(plan))
    again = PL.plan_from_sheets(ExcelLedgerRepository(path).get_plan_sheets())
    assert again.override(PL.SHEET_REC, "WB", pd.Timestamp("2026-11-01")) == 5.0 and again.initiatives.iloc[0]["revenue"] == 1_000_000.0
    assert bool(again.initiatives.iloc[0]["active"]) is True
