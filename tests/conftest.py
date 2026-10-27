import numpy as np
import pandas as pd
import pytest

from core import rnp_calc as R


def make_fact(models=("A", "B"), days=70, seed=1, mps=("WB",)) -> pd.DataFrame:
    """Small deterministic fact table with every raw field the report reads (already prepared: model/status/ext)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-03-01", periods=days)
    rows = []
    for model in models:
        for mp in mps:
            for sku in (f"{model.lower()}/1", f"{model.lower()}/2"):
                for d in dates:
                    orders = int(rng.integers(5, 20))
                    price = 1500.0
                    sales = int(orders * 0.7)
                    rec = {
                        "date": d, "mp": mp, "sku": sku, "model": model, "status": "Продается",
                        "orders_count": orders, "orders_rub": orders * price, "orders_spp_rub": orders * price * 0.8,
                        "sales_qty": sales, "sales_rub": sales * price, "sebes_rub": sales * 500.0,
                        "comis_spp": sales * price * 0.2, "logist": sales * 120.0, "storage": sales * 25.0,
                        "other": sales * 15.0, "clicks_total": orders * 30, "cart_total": orders * 3,
                        "adv_views": 1000, "adv_clicks": 40, "adv_orders": 3, "reclama": 400.0,
                        "stock_mp": 300, "b_has_stats": 1,
                    }
                    for f in R.EXT_FIELDS:
                        rec[f] = 0.0
                    rec.update(b_rub=100.0, b_views=2000.0, b_clicks=50.0, b_carts=8.0, b_orders=2.0,
                               ads_ya_c_rub=60.0, ads_ya_c_views=900.0, ads_ya_c_clicks=30.0, ads_ya_c_orders=1.0)
                    rec["margin_rub"] = (rec["sales_rub"] - rec["sebes_rub"] - rec["comis_spp"] - rec["logist"]
                                         - rec["storage"] - rec["other"] - 0.05 * rec["sales_rub"] * 0.8
                                         - rec["reclama"] - 160.0)
                    rows.append(rec)
    return pd.DataFrame(rows)


def make_ms(fact: pd.DataFrame) -> pd.DataFrame:
    out = fact[["date", "sku"]].drop_duplicates().copy()
    out["qty"] = 100.0
    return out


@pytest.fixture
def fact() -> pd.DataFrame:
    return make_fact()


@pytest.fixture
def stocks_ms(fact) -> pd.DataFrame:
    return make_ms(fact)


@pytest.fixture
def creators() -> pd.DataFrame:
    return pd.DataFrame({
        "week_start": pd.to_datetime(["2026-04-27"]), "week_end": pd.to_datetime(["2026-05-03"]),
        "mp": ["WB"], "model": ["A"], "cr_spend": [1000.0], "cr_impr": [20000.0], "cr_clicks": [400.0],
        "cr_carts": [80.0], "cr_orders": [20.0], "cr_orders_rub": [30000.0], "cr_posts": [4],
    })
