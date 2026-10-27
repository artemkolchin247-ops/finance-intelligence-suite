import math

import numpy as np
import pandas as pd
import pytest

from core import rnp_calc as R
from core.rnp_calc import NO_DATA, Selection, StockCtx, UnitCosts, calc_metrics
from tests.conftest import make_fact, make_ms

ZERO_T = {f: 0.0 for f in R.SUM_FIELDS} | {"ext_rub": 0.0, "b_has_stats": True}
PC = UnitCosts(sebes=500, comis_pct=0.2, logist=120, storage=25, other=15)
CTX = StockCtx(stock_mp=300, ms_alloc=100, n_days=1, share_valid=True)


def _t(**over) -> dict:
    t = dict(ZERO_T)
    t.update(over)
    t["ext_rub"] = sum(t[R.CHANNEL_BY_KEY[p].rub] for p in R.EXT_PARTS)
    return t


BASE = dict(orders_count=100, orders_rub=150_000, orders_spp_rub=120_000, sales_qty=70, sales_rub=105_000,
            margin_rub=20_000, sebes_rub=35_000, clicks_total=3000, cart_total=300, reclama=6000, adv_orders=25,
            adv_views=50_000, adv_clicks=1500, b_rub=1000, b_views=10_000, b_clicks=200, b_carts=30, b_orders=6,
            ads_ya_c_rub=500, ads_ya_c_views=5000, ads_ya_c_clicks=100, ads_ya_c_orders=4)


def test_orders_sales_funnel():
    d = calc_metrics(_t(**BASE), 0.7, 0.7, PC, CTX)
    assert d["aov_before"] == pytest.approx(1500) and d["aov_after"] == pytest.approx(1200)
    assert d["spp_orders"] == pytest.approx(0.2)
    assert d["asp"] == pytest.approx(1500) and d["buyout"] == 0.7
    assert d["cr_click_cart"] == pytest.approx(0.1) and d["cr_cart_order"] == pytest.approx(1 / 3)
    assert d["cr_click_order"] == pytest.approx(100 / 3000)


def test_total_advertising_is_internal_plus_external_without_creators():
    d = calc_metrics(_t(**BASE), 0.7, 0.7, PC, CTX)
    assert d["ext_rub"] == 1500  # blogger + yandex
    assert d["ad_total"] == 6000 + 1500
    assert d["drr_total_sales"] == pytest.approx(7500 / 105_000)
    assert d["drr_total_orders"] == pytest.approx(7500 / 150_000)


def test_internal_channel_metrics_and_organic():
    d = calc_metrics(_t(**BASE), 0.7, 0.7, PC, CTX)
    assert d["organic"] == 75
    assert d["adv_ctr"] == pytest.approx(1500 / 50_000) and d["adv_cpc"] == pytest.approx(4)
    assert d["adv_cpo"] == pytest.approx(240) and d["adv_cpm"] == pytest.approx(120)
    assert R.calc_metrics(_t(**{**BASE, "adv_orders": 130}), 0.7, 0.7, PC, CTX)["organic"] == 0


def test_external_orders_exclude_blogger_without_stats():
    t = _t(**BASE)
    with_stats = calc_metrics(t, 0.7, 0.7, PC, CTX)
    assert with_stats["ext_orders"] == 6 + 4 and with_stats["blog_orders"] == 6
    t["b_has_stats"] = False
    no_stats = calc_metrics(t, 0.7, 0.7, PC, CTX)
    assert no_stats["blog_orders"] == NO_DATA and no_stats["blog_profit"] == NO_DATA
    assert no_stats["ext_orders"] == 4 and no_stats["ext_rub"] == 1500  # spend still counts


def test_margin_is_net_of_ads_and_margin_before_ads_adds_them_back():
    d = calc_metrics(_t(**BASE), 0.7, 0.7, PC, CTX)
    assert d["margin_pct"] == pytest.approx(20_000 / 105_000)
    assert d["margin_before_ads_pct"] == pytest.approx((20_000 + 7_500) / 105_000)


def test_forecast_formulas():
    t = _t(**BASE)
    d = calc_metrics(t, 0.7, 0.6, PC, CTX)
    check, spp = 1500.0, 0.2
    pm = (check - 500 - 0.2 * check - 120 - 25 - 15 - 0.05 * check * (1 - spp)) / check
    assert d["fc_sales"] == pytest.approx(150_000 * 0.6)
    assert d["fc_margin_before_ads_pct"] == pytest.approx(pm)
    assert d["fc_margin_pct"] == pytest.approx(pm - 7_500 / 90_000)
    assert d["fc_margin"] == pytest.approx((pm - 7_500 / 90_000) * 90_000)
    # channel profit = orders * avg check * buyout * margin before ads - spend
    assert d["adv_profit"] == pytest.approx(25 * 1500 * 0.6 * pm - 6000)
    assert d["adv_romi"] == pytest.approx(d["adv_profit"] / 6000)
    assert d["ext_profit"] == pytest.approx(10 * 1500 * 0.6 * pm - 1500)


def test_stock_turnover_and_gmroi():
    d = calc_metrics(_t(**BASE), 0.7, 0.7, PC, StockCtx(300, 100, 10, True))
    assert d["stock"] == 300 and d["stock_ms"] == 100
    assert d["turnover"] == pytest.approx(400 / (70 / 10))
    capital = 400 * (35_000 / 70)
    assert d["gmroi"] == pytest.approx(20_000 / capital * 36.5)
    assert math.isnan(calc_metrics(_t(**BASE), 0.7, 0.7, PC, StockCtx(300, 100, 10, False))["gmroi"])


def test_zero_denominators_give_nan_without_errors():
    d = calc_metrics(_t(), 0.0, 0.0, UnitCosts(), StockCtx())
    for k in ("aov_before", "asp", "cr_click_cart", "drr_total_sales", "adv_ctr", "adv_cpo",
              "margin_pct", "turnover", "gmroi", "adv_romi", "spp_orders"):
        assert math.isnan(d[k]), k
    assert d["organic"] == 0


# ------------------------------------------------------------------ report structure
def _report(fact, ms, creators, **kw):
    sel = Selection(kw.pop("mp", "WB"), pd.Timestamp(kw.pop("start", "2026-04-20")),
                    pd.Timestamp(kw.pop("end", "2026-04-29")), **kw)
    return R.build_report(fact, ms, creators, sel)


def test_columns_total_weeks_days(fact, stocks_ms, creators):
    rep = _report(fact, stocks_ms, creators)  # 10 days -> weeks of 7 and 3
    kinds = [c.kind for c in rep.columns]
    assert kinds[0] == "total" and kinds.count("week") == 2 and kinds.count("day") == 10
    assert kinds[1:9] == ["week"] + ["day"] * 7
    assert rep.columns[1].name2.startswith("с 20.04")


def test_additive_fields_sum_across_days_and_weeks(fact, stocks_ms, creators):
    rep = _report(fact, stocks_ms, creators)
    for key in ("orders", "orders_rub", "sales", "ad_total", "margin", "clicks"):
        total = rep.total.values[key]
        assert sum(c.values[key] for c in rep.column("day")) == pytest.approx(total)
        assert sum(c.values[key] for c in rep.column("week")) == pytest.approx(total)


def test_ratios_are_recomputed_from_sums_not_averaged(fact, stocks_ms, creators):
    rep = _report(fact, stocks_ms, creators)
    assert rep.total.values["cr_click_order"] == pytest.approx(
        rep.total.values["orders"] / rep.total.values["clicks"])
    assert rep.total.values["drr_total_sales"] == pytest.approx(
        rep.total.values["ad_total"] / rep.total.values["sales_rub"])


def test_forecast_uses_lagged_buyout_window(fact, stocks_ms, creators):
    rep = _report(fact, stocks_ms, creators, start="2026-04-20", end="2026-04-20", models=("A",))
    day = pd.Timestamp("2026-04-20")
    lo, hi = day - pd.Timedelta(days=37), day - pd.Timedelta(days=8)
    w = fact[(fact["model"] == "A") & fact["date"].between(lo, hi)]
    buyout = w["sales_qty"].sum() / w["orders_count"].sum()
    orders_rub = fact[(fact["model"] == "A") & (fact["date"] == day)]["orders_rub"].sum()
    assert rep.total.values["fc_sales"] == pytest.approx(orders_rub * buyout)


def test_forecast_undefined_when_lag_window_has_no_orders(fact, stocks_ms, creators):
    rep = _report(fact, stocks_ms, creators, start="2026-03-05", end="2026-03-07")  # < 37 days of history
    assert math.isnan(rep.total.values["fc_sales"]) and math.isnan(rep.total.values["adv_profit"])


def test_multi_model_forecast_is_sum_of_single_models(stocks_ms, creators):
    fact = make_fact(models=("A", "B", "C"), seed=5)
    ms = make_ms(fact)
    both = _report(fact, ms, creators, models=("A", "B"))
    a = _report(fact, ms, creators, models=("A",))
    b = _report(fact, ms, creators, models=("B",))
    for k in ("fc_sales", "fc_margin", "adv_profit", "ext_profit"):
        assert both.total.values[k] == pytest.approx(a.total.values[k] + b.total.values[k]), k
    assert both.active_models == ["A", "B"]


def test_multi_model_override_differs_from_blended_when_economics_differ(creators):
    fact = make_fact(models=("A", "B"), seed=2)
    fact.loc[fact["model"] == "B", "sebes_rub"] *= 1.8  # model B has much higher unit cost
    ms = make_ms(fact)
    both = _report(fact, ms, creators, models=("A", "B")).total.values["fc_margin"]
    a = _report(fact, ms, creators, models=("A",)).total.values["fc_margin"]
    b = _report(fact, ms, creators, models=("B",)).total.values["fc_margin"]
    assert both == pytest.approx(a + b)


def test_stock_snapshot_and_own_warehouse_allocation(fact, stocks_ms, creators):
    # both marketplaces: WB share of cost of sales over the last 30 days decides the allocation
    f = pd.concat([fact, fact.assign(mp="Ozon", sebes_rub=fact["sebes_rub"] * 3)], ignore_index=True)
    rep = R.build_report(f, stocks_ms, creators, Selection("WB", pd.Timestamp("2026-04-20"), pd.Timestamp("2026-04-29")))
    share = 1 / 4  # WB cost = 1x, Ozon = 3x
    ms_total = stocks_ms[stocks_ms["date"] == "2026-04-29"]["qty"].sum()
    assert rep.total.values["stock"] == 4 * 300
    assert rep.total.values["stock_ms"] == round(ms_total * share)


def test_status_filter_and_sku_filter(fact, stocks_ms, creators):
    fact = fact.copy()
    fact.loc[fact["sku"] == "a/2", "status"] = "Выводим"
    all_orders = _report(fact, stocks_ms, creators).total.values["orders"]
    out = _report(fact, stocks_ms, creators, status="Выводим")
    assert out.total.values["orders"] == fact[(fact["sku"] == "a/2") & fact["date"].between("2026-04-20", "2026-04-29")]["orders_count"].sum()
    assert out.total.values["ext_rub"] == 0  # external ads belong to the selling assortment
    one = _report(fact, stocks_ms, creators, skus=("b/1",))
    assert 0 < one.total.values["orders"] < all_orders
    assert R.build_report(fact, stocks_ms, creators, Selection("Ozon", pd.Timestamp("2026-04-20"), pd.Timestamp("2026-04-29"))) is None


def test_creators_only_in_total_and_week_columns(fact, stocks_ms, creators):
    rep = _report(fact, stocks_ms, creators, start="2026-04-27", end="2026-05-03", models=("A",))
    v = rep.total.values
    assert v["cr_spend"] == 1000 and v["cr_cpc"] == pytest.approx(2.5) and v["cr_ctr"] == pytest.approx(0.02)
    assert v["cr_cr_order"] == pytest.approx(0.25) and v["cr_cro"] == pytest.approx(0.05)
    assert "cr_spend" not in rep.column("day")[0].values
    assert "cr_spend" not in _report(fact, stocks_ms, creators, start="2026-04-27", end="2026-05-03",
                                     models=("B",)).total.values  # creators are filtered by model


def test_prepare_fact_spreads_external_ads_by_order_share():
    fact = make_fact(models=("A",), days=3)
    for c in R.EXT_FIELDS:
        fact = fact.drop(columns=c)
    fact = fact.drop(columns=["model", "status", "b_has_stats"])
    dim = pd.DataFrame({"sku": ["a/1", "a/2"], "model": ["A", "A"], "status": ["Продается"] * 2})
    ext = pd.DataFrame({"date": fact["date"].unique(), "mp": "WB", "model": "A", "b_has_stats": 1})
    for c in R.EXT_FIELDS:
        ext[c] = 100.0
    out = R.prepare_fact(fact, ext, dim)
    assert out["b_rub"].sum() == pytest.approx(100.0 * 3)
    day = out[out["date"] == out["date"].min()]
    shares = day["orders_count"] / day["orders_count"].sum()
    assert (day["b_rub"] / 100.0).tolist() == pytest.approx(shares.tolist())
    zero = fact.copy()
    zero["orders_count"] = 0
    assert R.prepare_fact(zero, ext, dim)["b_rub"].tolist() == pytest.approx([50.0] * 6)  # equal split


def test_hypotheses_by_date_filters_marketplace_model_and_period():
    hyp = pd.DataFrame({
        "Дата": pd.to_datetime(["2026-05-07", "2026-05-07", "2026-05-12", "2026-07-01"]),
        "Маркетплейс": ["WB", "Ozon", "wb", "WB"], "Модель": ["Moon", "Moon", "Vuki", "Moon"],
        "Гипотеза": ["h1", "h2", "h3", "h4"], "Статус": ["В работе"] * 4,
    })
    all_models = R.hypotheses_by_date(hyp, "WB", [], "2026-05-01", "2026-05-31")
    assert sum(len(v) for v in all_models.values()) == 2
    moon = R.hypotheses_by_date(hyp, "WB", ["Moon"], "2026-05-01", "2026-05-31")
    assert list(moon) == [pd.Timestamp("2026-05-07")] and moon[pd.Timestamp("2026-05-07")][0]["text"] == "h1"
