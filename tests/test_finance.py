import math

import numpy as np
import pandas as pd
import pytest

from core import balance as B
from core import cashflow as C
from core import payment_calendar as P
from core import pnl
from core.ledger import (ACT_INTERNAL, ExcelLedgerRepository, PLAN_COLUMNS, activity_of, pl_amount, prepare_journal)


@pytest.fixture(scope="module")
def repo():
    return ExcelLedgerRepository()


@pytest.fixture(scope="module")
def ctx(repo):
    as_of = repo.get_as_of()
    arts = repo.get_articles()
    j = prepare_journal(repo.get_journal(), arts, as_of)
    return {"j": j, "arts": arts, "as_of": as_of, "acc": repo.get_accounts(), "fm": repo.get_fact_monthly(),
            "b1c": repo.get_balance_1c(), "plan": repo.get_payment_plan()}


@pytest.fixture(scope="module")
def schedule(ctx):
    a = C.default_assumptions(ctx["j"], ctx["as_of"])
    return C.build_schedule(ctx["j"], ctx["b1c"], ctx["plan"], ctx["as_of"], a)


# ------------------------------------------------------------------ journal and classification
def _j(**over):
    base = {"dds_date": pd.Timestamp("2026-01-10"), "pl_date": pd.Timestamp("2026-01-10"), "income": np.nan, "expense": 100.0,
            "article": "РКО", "mp": "", "cabinet": "", "sku": "", "size": "", "qty": np.nan, "account": "A", "comment": ""}
    base.update(over)
    return pd.DataFrame([base])


ARTS = pd.DataFrame({"article": ["РКО", "Возврат оплаты", "Прочие доходы", "Перевод между счетами"],
                     "pl_kind": ["Косвенные расходы (до EBITDA)", "Косвенные расходы (до EBITDA)", "Доходы после EBITDA",
                                 "Не учитывается в P&L или учитывается автоматически"],
                     "pl_category": ["Административные расходы", "Прочие расходы", "", ""],
                     "dds_group": ["Банковские расходы", "Прочее", "Прочее", "Перевод между счетами"]})


def test_prepare_journal_rejects_bad_rows():
    asof = pd.Timestamp("2026-09-06")
    with pytest.raises(ValueError, match="no cash or P&L date"):
        prepare_journal(_j(pl_date=pd.NaT), ARTS, asof)
    with pytest.raises(ValueError, match="neither income nor expense"):
        prepare_journal(_j(expense=np.nan), ARTS, asof)
    with pytest.raises(ValueError, match="missing in the reference"):
        prepare_journal(_j(article="Неизвестная статья"), ARTS, asof)


def test_classification_and_signs():
    j = prepare_journal(pd.concat([_j(), _j(article="Возврат оплаты", income=40.0, expense=np.nan),
                                   _j(article="Прочие доходы", income=70.0, expense=np.nan),
                                   _j(article="Перевод между счетами", dds_date=pd.Timestamp("2026-10-01"))], ignore_index=True),
                        ARTS, pd.Timestamp("2026-09-06"))
    assert list(pl_amount(j)) == [100.0, -40.0, 70.0, 100.0]  # a refund reduces the cost; income after EBITDA is positive
    assert list(j["cash"]) == [-100.0, 40.0, 70.0, -100.0]
    assert j["activity"].tolist() == ["Операционная"] * 3 + [ACT_INTERNAL]
    assert j["planned"].tolist() == [False, False, False, True]
    assert activity_of("Кредиты") == "Финансовая" and activity_of("Вывод дивидендов") == "Финансовая"


def test_reference_follows_the_article_table(ctx):
    arts = ctx["arts"].set_index("article")
    assert arts.loc["РКО", "dds_group"] == "Банковские расходы" and arts.loc["РКО", "pl_category"] == "Административные расходы"
    assert arts.loc["Закупка товара", "pl_kind"].startswith("Не учитывается")
    assert arts.loc["Налог на прибыль", "pl_kind"] == "Налоги"
    assert arts.loc["Перечисление на депозит", "dds_group"] == "Перевод между счетами"
    assert arts.loc["ИИ сервис", "dds_group"] == "Программы"
    assert set(ctx["j"]["article"]) <= set(arts.index)


def test_transfers_net_to_zero_and_balances_tie(ctx):
    j, acc = ctx["j"], ctx["acc"]
    internal = j[(j["activity"] == ACT_INTERNAL) & ~j["planned"]]
    assert abs(internal["cash"].sum()) < 1.0
    total = C.cash_by_account(j, acc, ctx["as_of"])["balance"].sum()
    assert total == pytest.approx(acc["opening"].sum() + j[~j["planned"]]["cash"].sum())
    assert (C.cash_by_account(j, acc, ctx["as_of"])["balance"] >= -1.0).all()  # no account is overdrawn


# ------------------------------------------------------------------ P&L
INP = {"sales_rub": 1000.0, "sales_spp_rub": 800.0, "orders_rub": 1500.0, "orders_spp_rub": 1200.0, "orders_count": 10,
       "sales_qty": 8, "sebes_rub": 200.0, "comp_sebes": 2.0, "comis_spp": 390.0, "logist": 110.0, "storage": 20.0,
       "accept": 5.0, "other": 8.0, "reclama": 60.0, "b_rub": 10.0, "ads_vk_c_rub": 4.0, "ads_vk_s_rub": 6.0,
       "ads_s_c_rub": 3.0, "ads_ya_c_rub": 7.0, "margin_rub": 0.0, "days": 30.0, "stock_units": 100.0, "stock_cost": 20000.0}


def test_pnl_lines_follow_the_workbook_formulas(ctx):
    inp = dict(INP)
    inp["j|Косвенные расходы (до EBITDA)|Административные расходы|РКО"] = 30.0
    inp["j|Косвенные расходы (до EBITDA)|Прочие расходы|Возврат оплаты"] = -5.0
    inp["j|Налоги||Налог на прибыль"] = 12.0
    inp["j|Вывод дивидендов||Выплата дивидендов"] = 20.0
    v = pnl.derive_rows(inp, ctx["arts"])
    assert v["comm_after"] == pytest.approx(390 - (1000 - 800))  # SPP is funded out of the commission
    assert v["direct"] == pytest.approx(190 + 110 + 20 + 5 + 8)
    assert v["promo_ext"] == pytest.approx(30.0) and v["promo_total"] == pytest.approx(90.0)
    assert v["nds"] == pytest.approx(0.05 * 800)
    assert v["margin"] == pytest.approx(800 - 200 - 2 - v["direct"] - 90 - 40)
    assert v["indirect"] == pytest.approx(25.0)  # the refund reduces indirect costs
    assert v["ebitda"] == pytest.approx(v["margin"] - 25)
    assert v["net_profit"] == pytest.approx(v["ebitda"] - 12)
    assert v["retained"] == pytest.approx(v["net_profit"] - 20)
    assert v["roi"] == pytest.approx(v["margin"] / 202) and v["spp_size"] == pytest.approx(0.2)


def test_pnl_zero_revenue_gives_nan_not_errors(ctx):
    inp = {k: 0.0 for k in INP}
    v = pnl.derive_rows(inp, ctx["arts"])
    for k in ("margin_pct_before", "aov_orders", "gmroi", "roi", "spp_size"):
        assert math.isnan(v[k]), k


def test_pnl_margin_equals_sum_of_daily_margin_and_taxes_come_from_journal(ctx):
    months = list(pd.date_range("2026-03-01", "2026-08-01", freq="MS"))
    res = pnl.build_pnl(ctx["fm"], ctx["j"], ctx["arts"], months)
    assert sum(res.row("margin").values.values()) == pytest.approx(ctx["fm"][ctx["fm"]["month"].isin(months)]["margin_rub"].sum(), rel=1e-6)
    j = ctx["j"]
    tax = j[(j["article"] == "Налог на прибыль") & j["pl_date"].between("2026-03-01", "2026-08-31")]["expense"].sum()
    assert sum(res.row("taxes").values.values()) == pytest.approx(tax)
    q = pnl.build_pnl(ctx["fm"], j, ctx["arts"], months, granularity="Кварталы")
    assert [c.label for c in q.columns] == ["1 кв. 2026", "2 кв. 2026", "3 кв. 2026"]
    assert sum(q.row("net_profit").values.values()) == pytest.approx(sum(res.row("net_profit").values.values()))


def test_pnl_marketplace_filter_allocates_shared_items_by_revenue(ctx):
    months = [pd.Timestamp("2026-07-01")]
    allmp = pnl.build_pnl(ctx["fm"], ctx["j"], ctx["arts"], months)
    wb = pnl.build_pnl(ctx["fm"], ctx["j"], ctx["arts"], months, mp="WB")
    oz = pnl.build_pnl(ctx["fm"], ctx["j"], ctx["arts"], months, mp="Ozon")
    key = allmp.columns[0].key
    assert wb.row("rev_before").values[key] + oz.row("rev_before").values[key] == pytest.approx(allmp.row("rev_before").values[key])
    assert wb.row("indirect").values[key] + oz.row("indirect").values[key] == pytest.approx(allmp.row("indirect").values[key])
    assert wb.row("net_profit").values[key] + oz.row("net_profit").values[key] == pytest.approx(allmp.row("net_profit").values[key])


def test_delta_uses_share_of_revenue_for_cost_lines(ctx):
    res = pnl.build_pnl(ctx["fm"], ctx["j"], ctx["arts"], [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01")])
    a, b = "2026-06", "2026-07"
    d = pnl.delta(res, a, b)
    cogs, rev = res.row("cogs").values, res.row("rev_before").values
    expected = ((cogs[b] / rev[b]) - (cogs[a] / rev[a])) / (cogs[a] / rev[a])
    assert d["cogs"][1] == pytest.approx(expected)
    assert d["rev_before"][1] == pytest.approx((rev[b] - rev[a]) / rev[a])
    assert d["cogs"][0] == pytest.approx(cogs[b] - cogs[a])


def test_monthly_fact_equals_daily_aggregation():
    daily = pd.read_excel("data/finance_db.xlsx", sheet_name="fact_daily", usecols=["date", "mp", "sales_rub", "margin_rub", "sebes_rub"])
    fm = ExcelLedgerRepository().get_fact_monthly()
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").dt.to_timestamp()
    agg = daily[daily["month"].between("2026-03-01", "2026-08-01")].groupby(["month", "mp"])[["sales_rub", "sebes_rub"]].sum()
    got = fm.set_index(["month", "mp"]).loc[agg.index, ["sales_rub", "sebes_rub"]]
    np.testing.assert_allclose(got.to_numpy(), agg.to_numpy(), rtol=1e-4)


# ------------------------------------------------------------------ cash flow
def test_actual_dds_matches_journal_without_transfers(ctx):
    j, as_of = ctx["j"], ctx["as_of"]
    months = list(pd.date_range("2025-09-01", "2026-08-01", freq="MS"))
    st = C.build_dds(j, ctx["acc"], pd.DataFrame(), as_of, months)
    net = next(r for r in st.rows if r.key == "net")
    sub = j[(j["dds_date"] >= months[0]) & (j["dds_date"] <= "2026-08-31") & (j["activity"] != ACT_INTERNAL)]
    assert net.values["total"] == pytest.approx(sub["cash"].sum())
    close, open_ = next(r for r in st.rows if r.key == "close"), next(r for r in st.rows if r.key == "open")
    assert close.values["total"] == pytest.approx(open_.values["total"] + net.values["total"] + 0.0, abs=1.0)  # transfers net to zero
    for prev, nxt in zip(months, months[1:]):  # continuity of balances
        assert close.values[f"{prev:%Y-%m}"] == pytest.approx(open_.values[f"{nxt:%Y-%m}"])


def test_schedule_is_deterministic_and_within_horizon(ctx, schedule):
    a = C.default_assumptions(ctx["j"], ctx["as_of"])
    again = C.build_schedule(ctx["j"], ctx["b1c"], ctx["plan"], ctx["as_of"], a)
    pd.testing.assert_frame_equal(schedule, again)
    assert schedule["date"].min() > ctx["as_of"] and schedule["date"].max() <= pd.Timestamp("2027-08-31")
    assert (schedule["date"].dt.weekday < 5).all()  # weekends move to the next business day
    assert (schedule["amount"] > 0).all() and schedule["activity"].ne(ACT_INTERNAL).all()


def test_manual_plan_replaces_forecast_and_overdue_goes_first(ctx, schedule):
    first = ctx["as_of"] + pd.Timedelta(days=1)
    plan = ctx["plan"]
    overdue = plan[plan["Статус"] == "Просрочен"].iloc[0]
    rows = schedule[schedule["comment"].str.startswith(overdue["ID"])]
    assert len(rows) == 1 and rows["date"].iloc[0] >= first and rows["source"].iloc[0] == "План платежей"
    sep = schedule[(schedule["article"] == "Закупка товара") & (schedule["month"] == pd.Timestamp("2026-09-01"))]
    planned = plan[(plan["Статья операции"] == "Закупка товара") & plan["Дата"].between("2026-09-01", "2026-09-30")]["Сумма"].sum()
    assert sep["amount"].sum() >= planned - 1  # forecast only tops the plan up, never double counts it
    assert sep["amount"].sum() < planned * 3


def test_assumptions_move_the_forecast(ctx):
    base = C.default_assumptions(ctx["j"], ctx["as_of"])
    def receipts(a):
        s = C.build_schedule(ctx["j"], ctx["b1c"], ctx["plan"], ctx["as_of"], a)
        return s[s["direction"] == "in"]["amount"].sum()
    assert receipts(C.Assumptions(**{**base.__dict__, "growth_monthly": 0.0})) < receipts(C.Assumptions(**{**base.__dict__, "growth_monthly": 0.04}))
    s0 = C.build_schedule(ctx["j"], ctx["b1c"], ctx["plan"], ctx["as_of"], C.Assumptions(**{**base.__dict__, "dividends_month": 0.0}))
    s1 = C.build_schedule(ctx["j"], ctx["b1c"], ctx["plan"], ctx["as_of"], C.Assumptions(**{**base.__dict__, "dividends_month": 1e6}))
    assert s1[s1["article"] == "Выплата дивидендов"]["amount"].sum() - s0[s0["article"] == "Выплата дивидендов"]["amount"].sum() == pytest.approx(12e6)


def test_statement_plan_months_equal_schedule_totals(ctx, schedule):
    months = list(pd.date_range("2026-09-01", "2027-08-01", freq="MS"))
    st = C.build_dds(ctx["j"], ctx["acc"], schedule, ctx["as_of"], months)
    inflow = next(r for r in st.rows if r.key == "dir|Операционная|in")
    month = pd.Timestamp("2027-02-01")
    assert inflow.values["2027-02"] == pytest.approx(schedule[(schedule["month"] == month) & (schedule["activity"] == "Операционная") & (schedule["direction"] == "in")]["amount"].sum())
    close = next(r for r in st.rows if r.key == "close")
    signed = np.where(schedule["direction"] == "in", schedule["amount"], -schedule["amount"]).sum()
    start = C.cash_by_account(ctx["j"], ctx["acc"], ctx["as_of"])["balance"].sum()
    assert close.values["2027-08"] == pytest.approx(start + signed)
    assert [c.state for c in st.columns[:2]] == ["Факт+План", "План"]


def test_drilldown_lists_operations_of_article_month(ctx, schedule):
    d = C.drilldown(ctx["j"], schedule, ctx["as_of"], "Продажи Wildberries", pd.Timestamp("2026-07-01"))
    assert set(d["Тип"]) == {"Факт"} and d["Сумма"].sum() == pytest.approx(ctx["j"][(ctx["j"]["article"] == "Продажи Wildberries") & ctx["j"]["dds_date"].between("2026-07-01", "2026-07-31")]["income"].sum())
    assert set(C.drilldown(ctx["j"], schedule, ctx["as_of"], "Продажи Wildberries", pd.Timestamp("2026-10-01"))["Тип"]) == {"План"}


# ------------------------------------------------------------------ payment calendar
@pytest.fixture(scope="module")
def cal(ctx, schedule):
    opening = float(C.cash_by_account(ctx["j"], ctx["acc"], ctx["as_of"])["balance"].sum())
    return opening, P.calendar_start(ctx["as_of"])


def test_calendar_start_is_monday(ctx, cal):
    assert cal[1].weekday() == 0 and cal[1] > ctx["as_of"]


def test_daily_weekly_and_statement_agree(ctx, schedule, cal):
    opening, start = cal
    daily = P.daily_table(schedule, opening, start)
    weeks = P.weekly_summary(schedule, opening, start, 2_000_000)
    assert len(daily) == 84 and len(weeks) == 12
    assert daily["close"].iloc[0] == pytest.approx(opening + daily["net"].iloc[0])
    for w in weeks:
        d = daily[daily["week"] == w.week - 1]
        assert w.closing == pytest.approx(d["close"].iloc[-1]) and w.min_balance == pytest.approx(d["close"].min())
        assert sum(w.inflow.values()) - sum(w.outflow.values()) == pytest.approx(w.net)
    assert weeks[0].opening == pytest.approx(opening)
    assert weeks[1].opening == pytest.approx(weeks[0].closing)
    in_window = schedule[schedule["date"] <= start + pd.Timedelta(days=83)]
    assert daily["net"].sum() == pytest.approx(np.where(in_window["direction"] == "in", in_window["amount"], -in_window["amount"]).sum())


def test_cash_gap_flags_and_deferral_advice(ctx, schedule, cal):
    opening, start = cal
    status = [w.status for w in P.weekly_summary(schedule, opening, start, 2_000_000)]
    assert "below" in status
    assert all(w.status != "gap" for w in P.weekly_summary(schedule, opening, start, 0.0))
    advice = P.suggest_deferrals(schedule, opening, start, 2_000_000)
    moved = P.apply_moves(schedule, {a["id"]: a["to"] for a in advice})
    assert advice and moved["status"].eq("Отложен").sum() == len({a["id"] for a in advice})
    assert P.daily_table(moved, opening, start)["close"].min() > P.daily_table(schedule, opening, start)["close"].min()
    assert all(a["to"] >= a["from"] + pd.Timedelta(days=7) for a in advice)
    assert len({a["id"] for a in advice}) == len(advice)  # every payment is proposed at most once
    # priority "Можно отложить" only
    ids = {a["id"] for a in advice}
    assert set(schedule[schedule["id"].isin(ids)]["priority"]) == {"Можно отложить"}


def test_unrealistic_buffer_terminates_without_candidates(ctx, schedule, cal):
    opening, start = cal
    assert len(P.suggest_deferrals(schedule, opening, start, 1e12, max_moves=5)) <= 5


# ------------------------------------------------------------------ balance
def test_balance_balances_and_cash_matches_dds(ctx):
    months = list(pd.date_range("2024-01-01", "2026-08-01", freq="MS"))
    res = pnl.build_pnl(ctx["fm"], ctx["j"], ctx["arts"], months)
    npf = {pd.Timestamp(c.start): res.row("net_profit").values[c.key] for c in res.columns}
    dv = {pd.Timestamp(c.start): res.row("dividends").values[c.key] for c in res.columns}
    rows = {r.key: r for r in B.build_balance(ctx["j"], ctx["acc"], ctx["fm"], ctx["b1c"], npf, dv, months, ctx["as_of"])}
    for m in months:
        k = f"{m:%Y-%m}"
        assert rows["check"].values[k] == pytest.approx(0.0, abs=1e-6)
        assert rows["assets"].values[k] == pytest.approx(rows["cash"].values[k] + rows["recv"].values[k] + rows["stock"].values[k])
    jan = "2026-01"
    expected = C.cash_by_account(ctx["j"], ctx["acc"], pd.Timestamp("2026-01-31"))["balance"].sum()
    assert rows["cash"].values[jan] == pytest.approx(expected)
    assert rows["np_cum"].values["2026-08"] == pytest.approx(sum(npf.values()))


# ------------------------------------------------------------------ repository
def test_payment_plan_roundtrip(tmp_path):
    path = tmp_path / "l.xlsx"
    plan = pd.DataFrame({c: [""] for c in PLAN_COLUMNS})
    plan["Дата"], plan["Сумма"] = pd.Timestamp("2026-09-10"), 1000.0
    with pd.ExcelWriter(path) as xw:
        plan.to_excel(xw, sheet_name="payment_plan", index=False)
        pd.DataFrame({"a": [1]}).to_excel(xw, sheet_name="journal", index=False)
    repo = ExcelLedgerRepository(path)
    new = repo.get_payment_plan()
    new.loc[0, "Статус"] = "Согласован"
    new.loc[len(new)] = ["PP-9", pd.Timestamp("2026-09-12"), "Закупка товара", "Поставщик", "ООО Вуки", 5000.0, "Критичный", "Запланирован", ""]
    repo.save_payment_plan(new)
    again = ExcelLedgerRepository(path).get_payment_plan()
    assert len(again) == 2 and again.loc[0, "Статус"] == "Согласован" and again.loc[1, "Сумма"] == 5000.0
    assert "journal" in pd.ExcelFile(path).sheet_names
    with pytest.raises(ValueError):
        repo.save_payment_plan(new.drop(columns=["Сумма"]))


# ------------------------------------------------------------------ per-recommendation control
def test_rejected_recommendation_is_replaced_by_another(ctx, schedule, cal):
    opening, start = cal
    first = P.suggest_deferrals(schedule, opening, start, 2_000_000)
    again = P.suggest_deferrals(schedule, opening, start, 2_000_000, exclude={first[0]["id"]})
    assert first[0]["id"] not in {a["id"] for a in again} and again
    one = P.apply_moves(schedule, {first[0]["id"]: first[0]["to"]})  # applying a single recommendation changes only that payment
    assert (one["date"] != schedule["date"]).sum() == 1
    assert P.daily_table(one, opening, start)["close"].min() >= P.daily_table(schedule, opening, start)["close"].min()


def test_each_recommendation_helps_on_its_own_and_rejection_never_repeats_a_payment(ctx, schedule, cal):
    opening, start = cal
    base_min = P.daily_table(schedule, opening, start)["close"].min()
    advice = P.suggest_deferrals(schedule, opening, start, 2_000_000)
    for a in advice[:3]:  # applied alone, a recommendation does not worsen the balance and moves only its payment
        one = P.apply_moves(schedule, {a["id"]: a["to"]})
        assert P.daily_table(one, opening, start)["close"].min() >= base_min - 1.0
    rejected = set()
    for _ in range(6):  # reject the first proposal repeatedly: ids stay unique and never come back
        nxt = P.suggest_deferrals(schedule, opening, start, 2_000_000, exclude=rejected)
        assert len({a["id"] for a in nxt}) == len(nxt) and not rejected & {a["id"] for a in nxt}
        if not nxt:
            break
        rejected.add(nxt[0]["id"])
