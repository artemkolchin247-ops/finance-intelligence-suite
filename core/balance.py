"""Management balance sheet (marketplace business, no equipment).

Cash comes from the operations journal, stock on the marketplaces and on the own warehouse from the same
monthly facts as the РНП, the remaining positions from the accounting system ("1C"). Equity is the balancing
figure (retained earnings, calculated); the P&L-based accumulated profit is shown next to it for control.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from core.cashflow import cash_by_account

ASSET_1C = ("Дебиторская задолженность маркетплейсов", "Товары в пути и на таможне", "Предоплаты поставщикам")
LIABILITY_1C = ("Кредиторская задолженность поставщикам", "Задолженность по налогам",
                "Задолженность по оплате труда и взносам", "Кредиты и займы")
CAPITAL_1C = "Уставный капитал"


@dataclass
class BalanceRow:
    key: str
    label: str
    level: int
    kind: str  # title | group | item | total | memo | check
    values: dict = field(default_factory=dict)
    parent: str | None = None


def stock_values(fm: pd.DataFrame, month: pd.Timestamp) -> tuple[float, float]:
    """(stock on marketplaces, stock on the own warehouse) at cost; month-average levels stand for month-end."""
    f = fm[fm["month"] == month]
    if f.empty:
        return 0.0, 0.0
    unit = (f["sebes_rub"] / f["sales_qty"].replace(0, np.nan)).fillna(0.0)
    return float(f["stock_cost_avg"].sum()), float((f["stock_ms_units_avg"] * unit).sum())


def build_balance(journal: pd.DataFrame, accounts: pd.DataFrame, fact_monthly: pd.DataFrame, balance_1c: pd.DataFrame,
                  net_profit: Mapping[pd.Timestamp, float], dividends: Mapping[pd.Timestamp, float],
                  months: Sequence[pd.Timestamp], as_of: pd.Timestamp) -> list[BalanceRow]:
    """Balance sheet rows with a value per month key 'YYYY-MM' (month end)."""
    one = balance_1c.pivot_table(index="month", columns="item", values="amount", aggfunc="sum")
    groups = sorted(accounts["account_group"].unique())
    keys = [f"{m:%Y-%m}" for m in months]
    v: dict[str, dict[str, float]] = {k: {} for k in keys}
    cum_np = cum_div = 0.0
    all_months = sorted(set(months) | set(net_profit))
    base_start = min(all_months)
    for m in months:
        k = f"{m:%Y-%m}"
        end = min(m + pd.offsets.MonthEnd(0), as_of)
        bal = cash_by_account(journal, accounts, end)
        grp = bal.groupby("account_group")["balance"].sum()
        for g in groups:
            v[k][f"cash|{g}"] = float(grp.get(g, 0.0))
        v[k]["cash"] = float(grp.sum())
        mp_stock, ms_stock = stock_values(fact_monthly, m)
        v[k]["stock_mp"], v[k]["stock_ms"] = mp_stock, ms_stock
        for item in ASSET_1C + LIABILITY_1C + (CAPITAL_1C,):
            v[k][item] = float(one[item].get(m, 0.0)) if item in one else 0.0
        v[k]["stock"] = mp_stock + ms_stock + v[k]["Товары в пути и на таможне"] + v[k]["Предоплаты поставщикам"]
        v[k]["assets"] = v[k]["cash"] + v[k]["Дебиторская задолженность маркетплейсов"] + v[k]["stock"]
        v[k]["liabilities"] = sum(v[k][i] for i in LIABILITY_1C)
        v[k]["retained"] = v[k]["assets"] - v[k]["liabilities"] - v[k][CAPITAL_1C]
        v[k]["equity"] = v[k][CAPITAL_1C] + v[k]["retained"]
        v[k]["passives"] = v[k]["liabilities"] + v[k]["equity"]
        cum_np = sum(val for mm, val in net_profit.items() if mm <= m)
        cum_div = sum(val for mm, val in dividends.items() if mm <= m)
        v[k]["np_cum"], v[k]["div_cum"] = cum_np, cum_div
        v[k]["working_capital"] = (v[k]["cash"] + v[k]["Дебиторская задолженность маркетплейсов"] + v[k]["stock"]
                                   - sum(v[k][i] for i in LIABILITY_1C[:3]))

    def vals(key: str) -> dict:
        return {k: v[k][key] for k in keys}

    R = BalanceRow
    rows = [R("t_assets", "АКТИВЫ", 0, "title", {k: math.nan for k in keys}),
            R("cash", "Денежные средства", 1, "group", vals("cash"), "t_assets")]
    rows += [R(f"cash|{g}", g, 2, "item", vals(f"cash|{g}"), "cash") for g in groups]
    rows += [R("recv", "Расчёты с маркетплейсами (к получению)", 1, "item", vals("Дебиторская задолженность маркетплейсов"), "t_assets"),
             R("stock", "Запасы и предоплаты", 1, "group", vals("stock"), "t_assets"),
             R("stock_mp", "Товар на маркетплейсах", 2, "item", vals("stock_mp"), "stock"),
             R("stock_ms", "Товар на складе («Мой склад»)", 2, "item", vals("stock_ms"), "stock"),
             R("transit", "Товары в пути и на таможне", 2, "item", vals("Товары в пути и на таможне"), "stock"),
             R("prepay", "Предоплаты поставщикам", 2, "item", vals("Предоплаты поставщикам"), "stock"),
             R("assets", "ИТОГО АКТИВЫ", 0, "total", vals("assets")),
             R("t_liab", "ОБЯЗАТЕЛЬСТВА", 0, "title", {k: math.nan for k in keys}),
             R("payables", "Кредиторская задолженность поставщикам", 1, "item", vals("Кредиторская задолженность поставщикам"), "t_liab"),
             R("taxes", "Задолженность по налогам", 1, "item", vals("Задолженность по налогам"), "t_liab"),
             R("payroll", "Задолженность по оплате труда и взносам", 1, "item", vals("Задолженность по оплате труда и взносам"), "t_liab"),
             R("debt", "Кредиты и займы", 1, "item", vals("Кредиты и займы"), "t_liab"),
             R("liabilities", "ИТОГО ОБЯЗАТЕЛЬСТВА", 0, "total", vals("liabilities")),
             R("t_eq", "КАПИТАЛ", 0, "title", {k: math.nan for k in keys}),
             R("charter", "Уставный капитал", 1, "item", vals(CAPITAL_1C), "t_eq"),
             R("retained", "Нераспределённая прибыль (расчётная)", 1, "item", vals("retained"), "t_eq"),
             R("equity", "ИТОГО КАПИТАЛ", 0, "total", vals("equity")),
             R("passives", "ИТОГО ПАССИВЫ", 0, "total", vals("passives")),
             R("check", "Контроль: активы − пассивы", 1, "check", {k: v[k]["assets"] - v[k]["passives"] for k in keys}),
             R("np_cum", "Справочно: накопленная чистая прибыль по ОПиУ", 1, "memo", vals("np_cum")),
             R("div_cum", "Справочно: выплачено дивидендов (накопленно)", 1, "memo", vals("div_cum")),
             R("wc", "Справочно: оборотный капитал", 1, "memo", vals("working_capital"))]
    return rows
