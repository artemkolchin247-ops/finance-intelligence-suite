import math

import numpy as np
import pandas as pd
import pytest

from core import insights as I
from core import rnp_calc as R
from core.hypotheses import effect
from core.data_repo import HYPOTHESIS_COLUMNS, ExcelRepository
from tests.conftest import make_fact, make_ms


def _tot(**over):
    t = dict(orders=100, orders_rub=150_000, clicks=3000, cr_click_order=100 / 3000, aov_before=1500, buyout=0.7,
             ad_total=7500, drr_total_sales=0.07, margin=20_000, margin_pct=0.19, sales_rub=105_000,
             fc_sales=100_000, fc_margin=25_000, fc_margin_pct=0.25, fc_margin_before_ads_pct=0.325)
    t.update(over)
    return t


def test_compare_assigns_sentiment_by_metric_direction():
    cur, prev = _tot(orders=120, ad_total=9000), _tot()
    ch = {c.key: c for c in I.compare(cur, prev)}
    assert ch["orders"].sentiment == "good" and ch["orders"].delta == pytest.approx(0.2)
    assert ch["ad_total"].sentiment == "bad"  # more spend is unfavourable
    assert ch["aov_before"].sentiment == "neutral" and ch["margin"].delta == 0


def test_compare_ratio_metrics_use_percentage_points():
    c = {x.key: x for x in I.compare(_tot(drr_total_sales=0.05), _tot())}["drr_total_sales"]
    assert c.is_pp and c.delta == pytest.approx(-0.02) and c.sentiment == "good"
    assert I.fmt_delta(c) == "−2,0 п.п."


def test_orders_drivers_split_traffic_and_conversion():
    traffic, conversion = I.orders_drivers(_tot(orders=132, clicks=3300), _tot())
    assert traffic == pytest.approx(0.1) and conversion == pytest.approx(0.2)
    assert (1 + traffic) * (1 + conversion) - 1 == pytest.approx(0.32)  # = orders change
    assert I.orders_drivers(_tot(clicks=0), _tot()) is None


def test_margin_bridge_components_sum_to_change():
    cur = _tot(fc_sales=130_000, fc_margin_before_ads_pct=0.30, ad_total=10_000)
    cur["fc_margin"] = 130_000 * 0.30 - 10_000
    prev = _tot()
    prev["fc_margin"] = 100_000 * 0.325 - 7_500
    b = I.margin_bridge(cur, prev)
    assert b["volume"] + b["unit_margin"] + b["ads"] == pytest.approx(b["end"] - b["start"])
    assert I.margin_bridge(_tot(fc_sales=float("nan")), prev) is None


def test_narrative_mentions_orders_and_margin_driver():
    cur = _tot(orders=130, clicks=3000, fc_sales=150_000, ad_total=7500)
    cur["fc_margin"] = 150_000 * 0.325 - 7_500
    prev = _tot(); prev["fc_margin"] = 100_000 * 0.325 - 7_500
    lines = I.narrative(cur, prev)
    assert lines[0][0] == "good" and "Заказы 130" in lines[0][1]
    assert any("Прогнозная маржа выросла" in text and "объёма продаж" in text for _, text in lines)


def test_anomalies_flag_spike_only():
    s = pd.Series([100.0] * 14 + [500.0] + [100.0] * 3, index=pd.date_range("2026-05-01", periods=18))
    s.iloc[:14] += np.tile([0.0, 3.0], 7)  # tiny noise so the std is not zero
    found = I.anomalies(pd.DataFrame({"orders": s}), keys=("orders",))
    assert [f["date"] for f in found if f["z"] > 0][0] == pd.Timestamp("2026-05-15")
    assert I.anomalies(pd.DataFrame({"orders": [1.0] * 5}), keys=("orders",)) == []


def test_effect_compares_windows_around_event_date():
    fact = make_fact(models=("A",), days=80, seed=3)
    fact["orders_count"], fact["orders_rub"], fact["orders_spp_rub"] = 10, 15_000.0, 12_000.0  # flat baseline
    event = pd.Timestamp("2026-04-20")
    fact.loc[fact["date"] >= event, ["orders_count", "orders_rub", "orders_spp_rub"]] *= 2
    rows = {r.key: r for r in effect(fact, make_ms(fact), pd.DataFrame(columns=["mp", "model", "week_start"]), "wb", "A", event)}
    assert rows["orders"].delta == pytest.approx(1.0, rel=0.05) and rows["orders"].sentiment == "good"
    assert rows["aov_before"].delta == pytest.approx(0.0, abs=1e-9)
    assert effect(fact, make_ms(fact), pd.DataFrame(columns=["mp", "model", "week_start"]), "wb", "A",
                  pd.Timestamp("2027-01-01")) is None


def test_hypothesis_save_roundtrip(tmp_path):
    path = tmp_path / "db.xlsx"
    hyp = pd.DataFrame({c: [""] for c in HYPOTHESIS_COLUMNS})
    hyp["Дата"] = pd.Timestamp("2026-05-01")
    hyp["Сроки выполнения"] = pd.Timestamp("2026-05-05")
    with pd.ExcelWriter(path) as xw:
        hyp.to_excel(xw, sheet_name="hypotheses", index=False)
        pd.DataFrame({"sku": ["x"]}).to_excel(xw, sheet_name="dim_sku", index=False)
    repo = ExcelRepository(path)
    new = repo.get_hypotheses()
    new.loc[0, "Гипотеза"] = "Поднять цену"
    new.loc[len(new)] = [pd.Timestamp("2026-05-02"), "WB", "Moon", "Тест", "", "", "Артем",
                         pd.Timestamp("2026-05-09"), "В работе", "", ""]
    repo.save_hypotheses(new)
    again = ExcelRepository(path).get_hypotheses()
    assert len(again) == 2 and again.loc[0, "Гипотеза"] == "Поднять цену" and again.loc[1, "Модель"] == "Moon"
    assert "dim_sku" in pd.ExcelFile(path).sheet_names  # other sheets untouched
    with pytest.raises(ValueError):
        repo.save_hypotheses(new.drop(columns=["Статус"]))


def test_end_to_end_on_workbook_matches_hypothesis_story():
    repo = ExcelRepository()
    fact = R.prepare_fact(repo.get_fact_daily(), repo.get_fact_ext_daily(), repo.get_dim_sku())
    ms, cr = repo.get_stocks_ms(), repo.get_creators_weekly()
    sel = R.Selection("WB", pd.Timestamp("2026-04-27"), pd.Timestamp("2026-05-24"))
    rep = R.build_report(fact, ms, cr, sel)
    assert rep.total.values["orders"] == pytest.approx(
        fact[(fact["mp"] == "WB") & fact["date"].between(sel.start, sel.end)]["orders_count"].sum())
    moon = effect(fact, ms, cr, "WB", "Moon", pd.Timestamp("2026-05-07"))
    assert {r.key: r for r in moon}["orders"].delta > 1.5  # price cut triples orders
    hyp = repo.get_hypotheses()
    assert len(hyp) == 11 and set(hyp["Статус"]) == {"В работе", "Подтверждено", "Опровергнуто"}
    models = set(repo.get_dim_sku()["model"])
    assert set(hyp["Модель"]) <= models
