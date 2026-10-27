"""ОПиУ (P&L): marketplace lines come from the monthly facts (same source as the РНП), indirect costs, taxes and
other items come from the operations journal by accrual date (Дата P&L).

Pipeline: additive inputs per month -> summed per period column -> derive_rows() builds every line, so months,
quarters and years use identical formulas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from core.formatting import safe_div
from core.ledger import PL_AFTER_EXP, PL_AFTER_INC, PL_DIVIDENDS, PL_INDIRECT, PL_TAX, pl_amount

NDS_RATE = 0.05  # VAT on revenue after SPP
CATEGORIES = ("Общепроизводственные расходы", "Административные расходы", "Коммерческие расходы", "Прочие расходы")
SHORT_CATEGORY = {"Общепроизводственные расходы": "Общепроизводственные расходы",
                  "Административные расходы": "Административные расходы",
                  "Коммерческие расходы": "Коммерческие расходы", "Прочие расходы": "Прочие"}
FACT_SUM = ("orders_count", "orders_rub", "orders_spp_rub", "sales_qty", "sales_rub", "sales_spp_rub", "sebes_rub",
            "comp_sebes", "comis_spp", "logist", "storage", "accept", "other", "reclama", "b_rub", "ads_vk_c_rub",
            "ads_vk_s_rub", "ads_s_c_rub", "ads_ya_c_rub", "margin_rub")


@dataclass
class Row:
    key: str
    label: str
    kind: str  # amount | qty | pct | unit | days | memo
    level: int = 0  # 0 section/total, 1 line, 2 sub-line
    bold: bool = False
    parent: str | None = None
    good: str | None = None  # "up" / "down": favourable direction of change
    delta_base: str | None = None  # "before"/"after": delta % is the change of the share in that revenue (as in the P&L file)
    values: dict = field(default_factory=dict)


@dataclass
class Column:
    key: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    months: list[pd.Timestamp]


@dataclass
class PnlResult:
    columns: list[Column]
    rows: list[Row]

    def row(self, key: str) -> Row:
        return next(r for r in self.rows if r.key == key)


def make_columns(months: Sequence[pd.Timestamp], granularity: str) -> list[Column]:
    """Period columns: one per month, quarter or year (only the months given belong to a column)."""
    cols: dict[str, Column] = {}
    for m in months:
        if granularity == "Кварталы":
            key, label = f"{m.year}Q{(m.month - 1) // 3 + 1}", f"{(m.month - 1) // 3 + 1} кв. {m.year}"
        elif granularity == "Годы":
            key, label = str(m.year), str(m.year)
        else:
            key, label = f"{m:%Y-%m}", f"{m:%m.%Y}"
        col = cols.setdefault(key, Column(key, label, m, m + pd.offsets.MonthEnd(0), []))
        col.months.append(m)
        col.end = m + pd.offsets.MonthEnd(0)
    return list(cols.values())


def _fact_inputs(fm: pd.DataFrame, months: list[pd.Timestamp], mp: str) -> dict[str, float]:
    f = fm[fm["month"].isin(months)]
    if mp != "Все":
        f = f[f["mp"] == mp]
    out = {c: float(f[c].sum()) for c in FACT_SUM}
    days = float(sum(m.days_in_month for m in months))
    out["days"] = days
    # average stock: stock levels are not additive over months, take the mean over months and sum the marketplaces
    n = max(len(months), 1)
    unit_cost = (f["sebes_rub"] / f["sales_qty"].replace(0, np.nan)).fillna(0.0)
    out["stock_units"] = float((f["stock_units_avg"] + f["stock_ms_units_avg"]).sum()) / n
    out["stock_cost"] = float((f["stock_cost_avg"] + f["stock_ms_units_avg"] * unit_cost).sum()) / n
    return out


def _journal_inputs(j: pd.DataFrame, months: list[pd.Timestamp], share: float) -> dict[str, float]:
    sel = j[j["pl_month"].isin(months)]
    out: dict[str, float] = {}
    amt = pl_amount(sel) * share
    for (kind, cat, art), v in amt.groupby([sel["pl_kind"], sel["pl_category"], sel["article"]]).sum().items():
        out[f"j|{kind}|{cat}|{art}"] = float(v)
    return out


def derive_rows(inp: dict[str, float], articles: pd.DataFrame, mp_label: str = "Все") -> dict[str, float]:
    """Every P&L line of one column from its additive inputs."""
    v: dict[str, float] = {}
    rb, ra = inp["sales_rub"], inp["sales_spp_rub"]
    v["orders_rub_before"], v["orders_rub_after"], v["orders_qty"] = inp["orders_rub"], inp["orders_spp_rub"], inp["orders_count"]
    v["rev_before"], v["rev_after"], v["sales_qty"] = rb, ra, inp["sales_qty"]
    v["cogs"], v["cogs_comp"] = inp["sebes_rub"], inp["comp_sebes"]
    v["commission_before"] = inp["comis_spp"]
    v["comm_after"] = inp["comis_spp"] - (rb - ra)  # the marketplace funds SPP out of its commission
    v["logistics"], v["storage"], v["accept"], v["other_direct"] = inp["logist"], inp["storage"], inp["accept"], inp["other"]
    v["direct"] = v["comm_after"] + v["logistics"] + v["storage"] + v["accept"] + v["other_direct"]
    v["promo_int"] = inp["reclama"]
    v["promo_blog"] = inp["b_rub"]
    v["promo_vk"] = inp["ads_vk_c_rub"] + inp["ads_vk_s_rub"]
    v["promo_other"] = inp["ads_s_c_rub"] + inp["ads_ya_c_rub"]
    v["promo_ext"] = v["promo_blog"] + v["promo_vk"] + v["promo_other"]
    v["promo_total"] = v["promo_int"] + v["promo_ext"]
    v["nds"] = NDS_RATE * ra
    v["margin"] = ra - v["cogs"] - v["cogs_comp"] - v["direct"] - v["promo_total"] - v["nds"]
    v["margin_pct_before"] = safe_div(v["margin"], rb)
    v["margin_pct_after"] = safe_div(v["margin"], ra)

    ind_total = 0.0
    for cat in CATEGORIES:
        arts = articles[(articles["pl_kind"] == PL_INDIRECT) & (articles["pl_category"] == cat)]["article"]
        s = 0.0
        for a in arts:
            val = inp.get(f"j|{PL_INDIRECT}|{cat}|{a}", 0.0)
            v[f"art|{a}"] = val
            s += val
        v[f"cat|{cat}"] = s
        ind_total += s
    v["indirect"] = ind_total
    v["ebitda"] = v["margin"] - ind_total
    v["ebitda_pct"] = safe_div(v["ebitda"], rb)

    def block(kind: str, key: str) -> None:
        total = 0.0
        for a in articles[articles["pl_kind"] == kind]["article"]:
            val = inp.get(f"j|{kind}|{articles.loc[articles['article'] == a, 'pl_category'].iloc[0]}|{a}", 0.0)
            v[f"art|{a}"] = val
            total += val
        v[key] = total

    block(PL_TAX, "taxes")
    block(PL_AFTER_INC, "income_after")
    block(PL_AFTER_EXP, "expense_after")
    block(PL_DIVIDENDS, "dividends")
    v["net_profit"] = v["ebitda"] - v["taxes"] + v["income_after"] - v["expense_after"]
    v["net_margin"] = safe_div(v["net_profit"], rb)
    v["retained"] = v["net_profit"] - v["dividends"]

    # reference block
    v["avg_stock_units"], v["avg_stock_cost"] = inp["stock_units"], inp["stock_cost"]
    days = inp["days"]
    cost_total = v["cogs"] + v["cogs_comp"]
    v["gmroi"] = safe_div(v["margin"] * 365.0 / days, v["avg_stock_cost"]) if days else math.nan
    v["turnover"] = safe_div(v["avg_stock_units"], inp["sales_qty"] / days) if days else math.nan
    v["aov_orders"] = safe_div(inp["orders_rub"], inp["orders_count"])
    v["aov_sales"] = safe_div(ra, inp["sales_qty"])
    v["cost_unit"] = safe_div(cost_total, inp["sales_qty"])
    v["logistics_unit"] = safe_div(v["logistics"], inp["sales_qty"])
    v["storage_unit"] = safe_div(v["storage"], inp["sales_qty"])
    v["spp_size"] = safe_div(rb - ra, rb)
    v["roi"] = safe_div(v["margin"], cost_total)
    v["annual_yield"] = v["roi"] * safe_div(365.0, v["turnover"])
    return v


def row_layout(articles: pd.DataFrame) -> list[Row]:
    """Report layout; the same list is filled with values by build_pnl."""
    R = Row
    rows = [
        R("orders_rub_before", "Заказы, руб (до СПП)", "amount", 1),
        R("orders_rub_after", "Заказы, руб (после СПП)", "amount", 1),
        R("orders_qty", "Заказы, шт", "qty", 1),
        R("rev_before", "Выручка до СПП", "amount", 0, True, good="up"),
        R("rev_after", "Выручка после СПП", "amount", 0, True, good="up"),
        R("sales_qty", "Продажи, шт", "qty", 1),
        R("cogs", "Себестоимость продаж", "amount", 1, good="down", delta_base="before"),
        R("cogs_comp", "Себестоимость компенсаций", "amount", 1, good="down", delta_base="before"),
        R("direct", "Прямые расходы", "amount", 0, True, good="down", delta_base="before"),
        R("commission_before", "Комиссия (до СПП) — справочно", "memo", 2, parent="direct"),
        R("comm_after", "Комиссия (после СПП)", "amount", 2, parent="direct", good="down", delta_base="after"),
        R("logistics", "Логистика", "amount", 2, parent="direct", good="down", delta_base="before"),
        R("storage", "Хранение", "amount", 2, parent="direct", good="down", delta_base="before"),
        R("accept", "Платная приёмка", "amount", 2, parent="direct", good="down", delta_base="before"),
        R("other_direct", "Прочие расходы", "amount", 2, parent="direct", good="down", delta_base="before"),
        R("promo_int", "Продвижение внутр.", "amount", 1, good="down"),
        R("promo_ext", "Продвижение внеш.", "amount", 1, good="down"),
        R("promo_blog", "Продвижение (блогеры)", "amount", 2, parent="promo_ext", good="down"),
        R("promo_vk", "Продвижение ВК", "amount", 2, parent="promo_ext", good="down"),
        R("promo_other", "Посевы и Яндекс (подрядчики)", "amount", 2, parent="promo_ext", good="down"),
        R("promo_total", "Продвижение всего", "amount", 0, True, good="down"),
        R("nds", "НДС", "amount", 1, good="down", delta_base="after"),
        R("margin", "Маржинальная прибыль, руб", "amount", 0, True, good="up"),
        R("margin_pct_before", "Маржинальность до СПП, %", "pct", 1, good="up"),
        R("margin_pct_after", "Маржинальность после СПП, %", "pct", 1, good="up"),
        R("indirect", "Косвенные расходы (до EBITDA)", "amount", 0, True, good="down"),
    ]
    for cat in CATEGORIES:
        rows.append(R(f"cat|{cat}", SHORT_CATEGORY[cat], "amount", 1, parent="indirect", good="down"))
        for a in articles[(articles["pl_kind"] == PL_INDIRECT) & (articles["pl_category"] == cat)]["article"]:
            rows.append(R(f"art|{a}", a, "amount", 2, parent=f"cat|{cat}", good="down"))
    rows += [
        R("ebitda", "Операционная прибыль (EBITDA)", "amount", 0, True, good="up"),
        R("ebitda_pct", "EBITDA, % от выручки до СПП", "pct", 1, good="up"),
        R("taxes", "Налоги", "amount", 0, False, good="down", delta_base="after"),
        R("income_after", "Доходы после EBITDA", "amount", 0, False, good="up"),
        R("expense_after", "Расходы после EBITDA", "amount", 0, False, good="down"),
        R("net_profit", "Чистая прибыль", "amount", 0, True, good="up"),
        R("net_margin", "Чистая рентабельность, % от выручки до СПП", "pct", 1, good="up"),
        R("dividends", "Вывод дивидендов", "amount", 1, good=None),
        R("retained", "Нераспределённая прибыль периода", "amount", 0, True, good="up"),
        R("avg_stock_units", "Средний остаток, шт", "qty", 1),
        R("avg_stock_cost", "Средняя себестоимость склада, руб", "amount", 1),
        R("gmroi", "GMROI, % (годовой)", "pct", 1, good="up"),
        R("turnover", "Оборачиваемость, дни", "days", 1, good="down"),
        R("aov_orders", "Средний чек заказов, руб (до СПП)", "unit", 1),
        R("aov_sales", "Средний чек продаж, руб (после СПП)", "unit", 1),
        R("cost_unit", "Себестоимость на ед., руб.", "unit", 1, good="down"),
        R("logistics_unit", "Логистика на ед., руб.", "unit", 1, good="down"),
        R("storage_unit", "Хранение на ед., руб.", "unit", 1, good="down"),
        R("spp_size", "Размер СПП, %", "pct", 1),
        R("roi", "ROI, %", "pct", 1, good="up"),
        R("annual_yield", "Годовая доходность, %", "pct", 1, good="up"),
    ]
    # income/expense after EBITDA and taxes show their articles as sub-lines
    for kind, parent in ((PL_TAX, "taxes"), (PL_AFTER_INC, "income_after"), (PL_AFTER_EXP, "expense_after"), (PL_DIVIDENDS, "dividends")):
        pos = next(i for i, r in enumerate(rows) if r.key == parent) + 1
        for a in articles[articles["pl_kind"] == kind]["article"][::-1]:
            rows.insert(pos, R(f"art|{a}", a, "amount", 2, parent=parent, good="up" if kind == PL_AFTER_INC else "down"))
    return rows


def prepare_pl_journal(journal: pd.DataFrame) -> pd.DataFrame:
    j = journal.copy()
    j["pl_month"] = j["pl_date"].dt.to_period("M").dt.to_timestamp()
    return j


def build_pnl(fact_monthly: pd.DataFrame, journal: pd.DataFrame, articles: pd.DataFrame, months: Sequence[pd.Timestamp],
              mp: str = "Все", granularity: str = "Месяцы") -> PnlResult:
    """P&L columns for the given months. With a single marketplace selected, items that have no marketplace
    (indirect costs, taxes, ...) are allocated in proportion to its share of revenue."""
    j = prepare_pl_journal(journal) if "pl_month" not in journal.columns else journal
    columns = make_columns(list(months), granularity)
    rows = row_layout(articles)
    for col in columns:
        inp = _fact_inputs(fact_monthly, col.months, mp)
        total_rev = float(fact_monthly[fact_monthly["month"].isin(col.months)]["sales_rub"].sum())
        share = 1.0 if mp == "Все" else float(safe_div(inp["sales_rub"], total_rev) if total_rev else 0.0)
        inp.update(_journal_inputs(j, col.months, share))
        vals = derive_rows(inp, articles)
        for r in rows:
            r.values[col.key] = vals.get(r.key, math.nan)
    return PnlResult(columns, rows)


def delta(result: PnlResult, a: str, b: str) -> dict[str, tuple[float, float]]:
    """(absolute, relative) change from column `a` to column `b` for every row.

    Following the P&L workbook, for cost lines with a revenue base the relative change is the change of the
    line's share in revenue (before or after SPP); for all other lines it is the plain change.
    """
    rev_b = {"before": result.row("rev_before").values, "after": result.row("rev_after").values}
    out = {}
    for r in result.rows:
        va, vb = r.values.get(a, math.nan), r.values.get(b, math.nan)
        d = vb - va
        if r.kind == "pct":
            rel = math.nan
        elif r.delta_base:
            base = rev_b[r.delta_base]
            ratio_a, ratio_b = safe_div(va, base[a]), safe_div(vb, base[b])
            rel = float(safe_div(ratio_b - ratio_a, ratio_a)) if not math.isnan(ratio_a) else math.nan
        else:
            rel = float(safe_div(d, abs(va))) if not math.isnan(va) else math.nan
        out[r.key] = (d, rel)
    return out


def highlights(result: PnlResult, a: str, b: str) -> list[tuple[str, str]]:
    """Plain-language findings for the change of the P&L from column `a` to column `b`: (sentiment, text)."""
    from core.formatting import fmt_num, fmt_pct

    def val(key, col):
        return result.row(key).values[col]

    def signed(x: float, suffix: str) -> str:
        return f"{'+' if x >= 0 else '−'}{abs(x):.1f}".replace(".", ",") + suffix

    out: list[tuple[str, str]] = []
    rev_a, rev_b = val("rev_before", a), val("rev_before", b)
    if rev_a and not math.isnan(rev_a):
        g = (rev_b / rev_a - 1) * 100
        out.append(("good" if g >= 0 else "bad", f"Выручка до СПП {fmt_num(rev_b / 1000, 0)} тыс. ₽ ({signed(g, '%')} к предыдущему периоду)"))
    ma, mb = val("margin_pct_before", a), val("margin_pct_before", b)
    if not (math.isnan(ma) or math.isnan(mb)):
        d = (mb - ma) * 100
        out.append(("good" if d >= 0 else "bad", f"Маржинальность {fmt_pct(mb)} ({signed(d, ' п.п.')}), маржа {fmt_num(val('margin', b) / 1000, 0)} тыс. ₽"))
    shares = []
    for key in ("cogs", "comm_after", "logistics", "storage", "promo_int", "promo_ext", "nds", "indirect"):
        sa, sb = val(key, a) / rev_a if rev_a else math.nan, val(key, b) / rev_b if rev_b else math.nan
        if not (math.isnan(sa) or math.isnan(sb)) and abs(sb - sa) >= 0.001:
            shares.append((abs(sb - sa), sb - sa, result.row(key).label))
    for _, d, label in sorted(shares, reverse=True)[:2]:
        out.append(("bad" if d > 0 else "good", f"Доля статьи «{label}» в выручке {'выросла' if d > 0 else 'снизилась'} на "
                    f"{str(round(abs(d) * 100, 1)).replace('.', ',')} п.п."))
    ea, eb = val("ebitda", a), val("ebitda", b)
    out.append(("good" if eb >= ea else "bad", f"EBITDA {fmt_num(eb / 1000, 0)} тыс. ₽ ({signed((eb - ea) / 1000, ' тыс. ₽')}), чистая прибыль {fmt_num(val('net_profit', b) / 1000, 0)} тыс. ₽"))
    return out
