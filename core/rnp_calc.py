"""РНП (daily performance report): pure pandas/numpy, no UI imports.

The logic is a port of the production reporting script:
  * raw additive fields are summed per day / week / period (prefix sums);
  * every ratio is recomputed from sums, never averaged;
  * external advertising = bloggers + VK (contractor) + VK seeding + seeding (contractor) + Yandex (contractor);
    total advertising excludes creators (a separate weekly block);
  * `margin_rub` is stored net of advertising;
  * forecasts use the buyout rate of the lagged window [d-37, d-8] and the unit economics of the
    selected period, and are computed per model and summed when several models are selected.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from core.formatting import safe_div

NAN = float("nan")
NO_DATA = "Нет данных"
BUYOUT_LAG_FROM, BUYOUT_LAG_TO = 37, 8  # forecast buyout window: [d-37, d-8]
NDS_RATE = 0.05

# ----------------------------------------------------------------------------- raw fields
EXT_FIELDS = (
    "b_rub", "b_views", "b_clicks", "b_carts", "b_orders",
    "ads_vk_c_rub", "ads_vk_c_views", "ads_vk_c_clicks", "ads_vk_c_orders",
    "ads_vk_s_rub", "ads_vk_s_views", "ads_vk_s_clicks", "ads_vk_s_orders",
    "ads_s_c_rub", "ads_s_c_views", "ads_s_c_clicks", "ads_s_c_orders",
    "ads_ya_c_rub", "ads_ya_c_views", "ads_ya_c_clicks", "ads_ya_c_orders",
)
SUM_FIELDS = (
    "orders_rub", "orders_spp_rub", "orders_count", "sales_rub", "sales_qty", "margin_rub", "sebes_rub",
    "comis_spp", "logist", "storage", "other", "clicks_total", "cart_total",
    "adv_views", "adv_clicks", "adv_orders", "reclama",
) + EXT_FIELDS
STATUS_ALL = "ИТОГО"
STATUSES = (STATUS_ALL, "Продается", "Выводим")


@dataclass(frozen=True)
class Channel:
    key: str
    section: str
    rub: str  # field names in the summed totals
    views: str
    clicks: str
    orders: str
    carts: str | None
    labels: tuple[str, ...]  # rub, drr_s, drr_o, views, clicks, [carts], orders, ctr, cpc, cpo, [cpm], profit, romi


def _labels(name: str, impr: str = "Показы", rub: str | None = None) -> tuple[str, ...]:
    return (rub or f"Реклама {name}, ₽", f"ДРР {name} от продаж, %", f"ДРР {name} от заказов, %",
            f"{impr} {name}, шт", f"Клики {name}, шт", f"Заказы {name}, шт", f"CTR {name}, %",
            f"CPC {name}, ₽", f"CPO {name}, ₽", f"Прибыль {name} (прогноз), ₽", f"ROMI {name} (прогноз), %")


CHANNELS: tuple[Channel, ...] = (
    Channel("adv", "Внутренняя реклама", "reclama", "adv_views", "adv_clicks", "adv_orders", None, _labels("внутр.")),
    Channel("ext", "Внешняя реклама", "ext_rub", "ext_views", "ext_clicks", "ext_orders", None,
            _labels("внеш.", "Просмотры")),
    Channel("blog", "Блогеры", "b_rub", "b_views", "b_clicks", "b_orders", "b_carts", (
        "Реклама блогеры, ₽", "ДРР блогеры от продаж, %", "ДРР блогеры от заказов, %", "Просмотры Блогеры, шт",
        "Клики Блогеры, шт", "Корзины Блогеры, шт", "Заказы Блогеры, шт", "CTR Блогеры, %", "CPC Блогеры, ₽",
        "CPO Блогеры, ₽", "Прибыль от Блогеров (прогноз), ₽", "ROMI Блогеров (прогноз), %")),
    Channel("vk_c", "ВК (подрядчик)", "ads_vk_c_rub", "ads_vk_c_views", "ads_vk_c_clicks", "ads_vk_c_orders", None,
            _labels("ВК (подрядчик)")),
    Channel("vk_s", "Посевы ВК", "ads_vk_s_rub", "ads_vk_s_views", "ads_vk_s_clicks", "ads_vk_s_orders", None,
            _labels("Посевы ВК", rub="Посевы ВК, ₽")),
    Channel("s_c", "Посевы (подрядчик)", "ads_s_c_rub", "ads_s_c_views", "ads_s_c_clicks", "ads_s_c_orders", None,
            _labels("Посевы (подрядчик)", rub="Посевы (подрядчик), ₽")),
    Channel("ya_c", "Яндекс (подрядчик)", "ads_ya_c_rub", "ads_ya_c_views", "ads_ya_c_clicks", "ads_ya_c_orders",
            None, _labels("Яндекс (подрядчик)")),
)
EXT_PARTS = ("blog", "vk_c", "vk_s", "s_c", "ya_c")
CHANNEL_BY_KEY = {c.key: c for c in CHANNELS}

CREATOR_FIELDS = ("cr_spend", "cr_impr", "cr_clicks", "cr_carts", "cr_orders", "cr_orders_rub", "cr_posts")


# ----------------------------------------------------------------------------- metric catalogue
@dataclass(frozen=True)
class Metric:
    key: str
    label: str  # as in the report spec; kind "k" is shown in thousands of rubles
    kind: str  # format kind, see core.formatting.format_value
    section: str
    good: str | None = None  # "up" / "down": which direction of change is favourable
    grad: str | None = None  # "up" / "down": row colour scale (more is better / less is better)
    neg_red: bool = False  # red background when negative
    key_row: bool = False  # highlighted headline row
    weekly_only: bool = False  # filled only in total / weekly columns
    forecast: bool = False
    help: str = ""

    @property
    def display_label(self) -> str:
        return self.label.replace(", ₽", ", тыс. ₽") if self.kind == "k" else self.label


_GOOD_UP = {"orders", "orders_rub", "sales", "sales_rub", "buyout", "clicks", "carts", "cr_click_cart",
            "cr_cart_order", "cr_click_order", "margin", "margin_pct", "margin_before_ads_pct", "gmroi",
            "fc_sales", "fc_margin", "fc_margin_pct", "fc_margin_before_ads_pct", "fc_gmroi", "organic"}
_HELP = {
    "orders": "Количество заказов за день/неделю/период (всего по выбранным моделям).",
    "orders_rub": "Сумма заказов по цене до СПП.",
    "spp_orders": "СПП = (заказы до СПП − заказы после СПП) / заказы до СПП.",
    "buyout": "Выкуп = продажи шт / заказы шт за то же окно (продажи отстают от заказов, поэтому по дням значение шумное).",
    "sales_rub": "Выкупленные продажи по цене до СПП.",
    "cr_click_cart": "Корзины / клики по карточке.",
    "cr_cart_order": "Заказы / корзины.",
    "cr_click_order": "Заказы / клики по карточке.",
    "ad_total": "Внутренняя + внешняя реклама (блогеры, ВК, посевы, Яндекс). Креаторы учитываются отдельным блоком.",
    "drr_total_sales": "ДРР = реклама ИТОГО / продажи до СПП.",
    "drr_total_orders": "ДРР от заказов = реклама ИТОГО / заказы до СПП.",
    "organic": "Заказы − заказы с внутренней рекламы (не меньше 0).",
    "margin": "Маржинальная прибыль из учёта: после себестоимости, комиссии, логистики, хранения, НДС и всей рекламы.",
    "margin_pct": "Маржа / продажи до СПП.",
    "margin_before_ads_pct": "(Маржа + реклама ИТОГО) / продажи до СПП.",
    "stock": "Остатки на маркетплейсе на конец окна.",
    "stock_ms": "Остатки «Мой склад», аллоцированные на маркетплейс по доле себестоимости продаж за 30 дней.",
    "turnover": "(Остатки МП + Мой склад) / средние продажи в день за окно.",
    "gmroi": "Маржа / (остатки × факт. себестоимость единицы) × 365 / дней в окне.",
    "fc_sales": "Заказы до СПП × выкуп лагового окна [день−37, день−8].",
    "fc_margin_before_ads_pct": "Юнит-экономика периода: (чек − себестоимость − комиссия − логистика − хранение − прочее − НДС) / чек.",
    "fc_margin_pct": "Маржа до рекламы (прогноз) − реклама ИТОГО / продажи (прогноз).",
    "fc_margin": "Маржа % (прогноз) × продажи (прогноз).",
    "fc_gmroi": "Маржа (прогноз) / (остатки × себестоимость единицы по периоду) × 365 / дней в окне.",
}


def _build_metrics() -> list[Metric]:
    m: list[Metric] = []

    def add(key, label, kind, section, **kw):
        kw.setdefault("good", "up" if key in _GOOD_UP else None)
        kw.setdefault("help", _HELP.get(key, ""))
        m.append(Metric(key, label, kind, section, **kw))

    s = "Заказы"
    add("orders", "Заказы (шт.)", "int", s, grad="up", key_row=True)
    add("orders_rub", "Заказы до СПП, ₽", "k", s, key_row=True)
    add("aov_before", "Ср. чек заказа до СПП, ₽", "rub0", s)
    add("aov_after", "Ср. чек заказа после СПП, ₽", "rub0", s)
    add("spp_orders", "СПП заказы, %", "pct", s)
    s = "Продажи"
    add("sales", "Продажи (шт.)", "int", s, grad="up", key_row=True)
    add("buyout", "Выкупы, %", "pct", s)
    add("sales_rub", "Продажи до СПП, ₽", "k", s, key_row=True)
    add("asp", "Ср. чек продажи до СПП, ₽", "rub0", s)
    s = "Воронка"
    add("clicks", "Клики общие всего, шт", "int", s, grad="up")
    add("carts", "Корзина, шт", "int", s)
    add("cr_click_cart", "CR карточка → корзина, %", "pct", s)
    add("cr_cart_order", "CR корзина → заказ, %", "pct", s)
    add("cr_click_order", "CR общий клик → заказ, %", "pct", s)
    s = "Реклама итого"
    add("ad_total", "Реклама ИТОГО, ₽", "k", s, good="down", grad="down", key_row=True)
    add("drr_total_sales", "ДРР общий от продаж, %", "pct", s, good="down", key_row=True)
    add("drr_total_orders", "ДРР общий от заказов, %", "pct", s, good="down")

    for ch in CHANNELS:
        k, s = ch.key, ch.section
        it = iter(ch.labels)
        add(f"{k}_rub", next(it), "k", s, good="down", grad="down")
        add(f"{k}_drr_s", next(it), "pct", s, good="down")
        add(f"{k}_drr_o", next(it), "pct", s, good="down")
        if k == "adv":
            add("organic", "Заказы органика, шт", "int", s)
        add(f"{k}_views", next(it), "int", s, grad="up" if k == "adv" else None)
        add(f"{k}_clicks", next(it), "int", s)
        if ch.carts:
            add(f"{k}_carts", next(it), "int", s)
        add(f"{k}_orders", next(it), "int", s)
        add(f"{k}_ctr", next(it), "pct", s, good="up")
        add(f"{k}_cpc", next(it), "rub1", s, good="down")
        add(f"{k}_cpo", next(it), "rub0", s, good="down")
        if k == "adv":
            add("adv_cpm", "CPM внутр., ₽", "rub0", s, good="down")
        add(f"{k}_profit", next(it), "k", s, good="up", grad="up", neg_red=True)
        add(f"{k}_romi", next(it), "pct", s, good="up")

        if k == "blog":  # creators follow the bloggers block (weekly columns only)
            s = "Креаторы"
            for key, label, kind, good in (
                ("cr_spend", "Креаторы, ₽", "k", "down"), ("cr_impr", "ИТОГО (показы)", "int", None),
                ("cr_cpv", "CPV", "rub2", "down"), ("cr_cpc", "CPC", "rub1", "down"), ("cr_ctr", "CTR", "pct", "up"),
                ("cr_clicks", "Переходы", "int", None), ("cr_carts", "Корзины", "int", None),
                ("cr_cr_cart", "CR корзина", "pct", "up"), ("cr_orders", "Заказы", "int", "up"),
                ("cr_cr_order", "CR заказ", "pct", "up"), ("cr_cro", "CRO переход - заказ", "pct", "up"),
                ("cr_orders_rub", "Сумма заказов, ₽", "k", "up"), ("cr_posts", "Количество постов", "int", None),
            ):
                add(key, label, kind, s, good=good, weekly_only=True)

    s = "Маржа"
    add("margin", "Маржинальная прибыль, руб.", "k", s, grad="up", neg_red=True, key_row=True)
    add("margin_pct", "Маржа, %", "pct", s, key_row=True)
    add("margin_before_ads_pct", "Маржа до рекламы, %", "pct", s)
    s = "Склад и оборачиваемость"
    add("stock", "Остатки, шт.", "int", s)
    add("stock_ms", "Остатки Мой склад, шт. (аллоц.)", "int", s)
    add("turnover", "Оборачиваемость продаж (дни)", "days", s, good="down")
    add("gmroi", "Годовой Net GMROI, %", "pct", s)
    s = "Прогноз"
    add("fc_sales", "Продажи (прогноз), ₽", "k", s, forecast=True)
    add("fc_margin", "Маржинальная прибыль (прогноз), ₽", "k", s, grad="up", neg_red=True, forecast=True)
    add("fc_margin_pct", "Маржа, % (прогноз)", "pct", s, forecast=True)
    add("fc_margin_before_ads_pct", "Маржа до рекламы, % (прогноз)", "pct", s, forecast=True)
    add("fc_gmroi", "Годовой Net GMROI, % (прогноз)", "pct", s, forecast=True)
    return m


METRICS: list[Metric] = _build_metrics()
METRIC_BY_KEY = {x.key: x for x in METRICS}
SECTIONS: list[str] = list(dict.fromkeys(x.section for x in METRICS))
KEY_SECTIONS = ("Заказы", "Продажи", "Воронка", "Реклама итого", "Маржа", "Прогноз")


# ----------------------------------------------------------------------------- prepared data
def prepare_fact(fact: pd.DataFrame, ext: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """Attach model/status and spread model-level external ads over SKU rows.

    External advertising is stored per model; each SKU row receives the share of its orders in the
    model-day (equal split when the model had no orders), so any SKU selection stays additive.
    """
    df = fact.merge(dim[["sku", "model", "status"]], on="sku", how="left", validate="m:1")
    missing = df["model"].isna()
    if missing.any():
        raise ValueError(f"SKU without a model in dim_sku, e.g. {df.loc[missing, 'sku'].head(3).tolist()}")
    keys = ["date", "mp", "model"]
    orders_sum = df.groupby(keys)["orders_count"].transform("sum")
    n_rows = df.groupby(keys)["sku"].transform("count")
    share = np.where(orders_sum > 0, df["orders_count"] / orders_sum.where(orders_sum > 0), 1.0 / n_rows)
    ext_cols = [c for c in EXT_FIELDS if c in ext.columns]
    joined = df[keys].merge(ext[keys + ext_cols + ["b_has_stats"]], on=keys, how="left")
    for c in ext_cols:
        df[c] = joined[c].fillna(0.0).values * share
    df["b_has_stats"] = joined["b_has_stats"].fillna(0).values
    for c in EXT_FIELDS:
        if c not in df:
            df[c] = 0.0
    return df


@dataclass(frozen=True)
class Selection:
    mp: str
    start: pd.Timestamp
    end: pd.Timestamp
    models: tuple[str, ...] = ()  # empty = all models (ИТОГО)
    skus: tuple[str, ...] = ()
    status: str = STATUS_ALL

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1


@dataclass
class Column:
    kind: str  # "total" | "week" | "day"
    name1: str
    name2: str
    start: pd.Timestamp
    end: pd.Timestamp
    values: dict = field(default_factory=dict)


@dataclass
class Report:
    selection: Selection
    columns: list[Column]
    active_models: list[str]

    def column(self, kind: str) -> list[Column]:
        return [c for c in self.columns if c.kind == kind]

    @property
    def total(self) -> Column:
        return self.columns[0]

    def day_frame(self) -> pd.DataFrame:
        """Numeric metrics (rows) x day columns -> DataFrame indexed by date (one row per day)."""
        days = self.column("day")
        rows = {c.start: {k: (v if isinstance(v, (int, float, np.floating, np.integer)) else NAN)
                          for k, v in c.values.items()} for c in days}
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ----------------------------------------------------------------------------- windows
class _Prefix:
    """Prefix sums over a continuous date range; missing days count as zero."""

    def __init__(self, daily: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
        full = daily.reindex(dates, fill_value=0.0)
        self.dates = dates
        self.sums = {f: np.concatenate([[0.0], np.cumsum(full[f].to_numpy(dtype=float))]) for f in SUM_FIELDS}
        self.sums["b_has_stats"] = np.concatenate([[0.0], np.cumsum(full["b_has_stats"].to_numpy(dtype=float))])

    def window(self, i0: int, i1: int) -> dict:
        t = {f: float(self.sums[f][i1 + 1] - self.sums[f][i0]) for f in SUM_FIELDS}
        t["b_has_stats"] = (self.sums["b_has_stats"][i1 + 1] - self.sums["b_has_stats"][i0]) > 0
        t["ext_rub"] = sum(t[CHANNEL_BY_KEY[p].rub] for p in EXT_PARTS)
        return t

    def buyout(self, i0: int, i1: int) -> float:
        qty = self.sums["sales_qty"][i1 + 1] - self.sums["sales_qty"][i0]
        orders = self.sums["orders_count"][i1 + 1] - self.sums["orders_count"][i0]
        return float(safe_div(qty, orders))


@dataclass(frozen=True)
class UnitCosts:
    sebes: float = 0.0
    comis_pct: float = 0.0
    logist: float = 0.0
    storage: float = 0.0
    other: float = 0.0


def unit_costs(rows: pd.DataFrame) -> UnitCosts:
    """Per-unit economics of the selected period: sums over sold units / sales revenue."""
    qty, rub = rows["sales_qty"].sum(), rows["sales_rub"].sum()
    if qty <= 0:
        return UnitCosts()
    return UnitCosts(
        sebes=rows["sebes_rub"].sum() / qty,
        comis_pct=float(safe_div(rows["comis_spp"].sum(), rub)) if rub > 0 else 0.0,
        logist=rows["logist"].sum() / qty, storage=rows["storage"].sum() / qty, other=rows["other"].sum() / qty,
    )


def _last_known(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Snapshot lookup: the value of the latest known date <= each date (0 before the first one)."""
    if series.empty:
        return pd.Series(0.0, index=dates)
    return series.sort_index().reindex(series.index.union(dates)).ffill().reindex(dates).fillna(0.0)


@dataclass
class StockCtx:
    stock_mp: float = 0.0
    ms_alloc: float = 0.0
    n_days: int = 1
    share_valid: bool = False


# ----------------------------------------------------------------------------- formulas
def _pred_margin_before_ads(t: Mapping, pc: UnitCosts) -> float:
    """Unit-economics margin before ads, share of the average check (forecast base)."""
    co, rev, rev_spp = t["orders_count"], t["orders_rub"], t["orders_spp_rub"]
    avg_check = float(safe_div(rev, co))
    if not avg_check > 0:
        return 0.0
    spp = float(safe_div(rev - rev_spp, rev))
    spp = 0.0 if np.isnan(spp) else spp
    unit = (avg_check - pc.sebes - pc.comis_pct * avg_check - pc.logist - pc.storage - pc.other
            - NDS_RATE * avg_check * (1 - spp))
    return unit / avg_check


def _channel_totals(t: Mapping, ch: Channel) -> tuple[float, float, float, float, float]:
    if ch.key == "ext":
        parts = [CHANNEL_BY_KEY[p] for p in EXT_PARTS]
        rub = sum(t[p.rub] for p in parts)
        views = sum(t[p.views] for p in parts)
        clicks = sum(t[p.clicks] for p in parts)
        orders = sum(t[p.orders] for p in parts if p.key != "blog") + (0.0 if _blog_no_stats(t) else t["b_orders"])
        return rub, views, clicks, orders, 0.0
    return t[ch.rub], t[ch.views], t[ch.clicks], t[ch.orders], t[ch.carts] if ch.carts else 0.0


def _blog_no_stats(t: Mapping) -> bool:
    return t["b_rub"] > 0 and not t["b_has_stats"]


def calc_metrics(t: Mapping, buyout_disp: float, buyout_fc: float, pc: UnitCosts, ctx: StockCtx) -> dict:
    """All report metrics for one window of summed raw fields `t` (day, week or whole period)."""
    d: dict = {}
    co, rev, rev_spp = t["orders_count"], t["orders_rub"], t["orders_spp_rub"]
    qty, sal, mar = t["sales_qty"], t["sales_rub"], t["margin_rub"]
    clicks, carts = t["clicks_total"], t["cart_total"]

    ext_rub = sum(t[CHANNEL_BY_KEY[p].rub] for p in EXT_PARTS)
    all_ads = t["reclama"] + ext_rub
    pred_sal = rev * buyout_fc
    pm = _pred_margin_before_ads(t, pc)
    pred_mar_pct = pm - float(safe_div(all_ads, pred_sal)) if pred_sal > 0 else pm
    pred_mar = pred_mar_pct * pred_sal
    avg_rev_per_order = float(safe_div(rev, co))

    d.update(
        orders=co, orders_rub=rev, aov_before=safe_div(rev, co), aov_after=safe_div(rev_spp, co),
        spp_orders=safe_div(rev - rev_spp, rev) if rev > 0 else NAN,
        sales=qty, buyout=buyout_disp, sales_rub=sal, asp=safe_div(sal, qty),
        clicks=clicks, carts=carts, cr_click_cart=safe_div(carts, clicks), cr_cart_order=safe_div(co, carts),
        cr_click_order=safe_div(co, clicks),
        ad_total=all_ads, drr_total_sales=safe_div(all_ads, sal), drr_total_orders=safe_div(all_ads, rev),
    )

    no_stats = _blog_no_stats(t)
    for ch in CHANNELS:
        rub, views, clk, orders, ch_carts = _channel_totals(t, ch)
        k = ch.key
        hidden = ch.key == "blog" and no_stats
        profit = (0.0 if np.isnan(avg_rev_per_order) else orders * avg_rev_per_order * buyout_fc * pm) - rub
        d[f"{k}_rub"] = rub
        d[f"{k}_drr_s"] = safe_div(rub, sal)
        d[f"{k}_drr_o"] = safe_div(rub, rev)
        if k == "adv":
            d["organic"] = max(0.0, co - t["adv_orders"])
        d[f"{k}_views"] = NO_DATA if hidden else views
        d[f"{k}_clicks"] = NO_DATA if hidden else clk
        if ch.carts:
            d[f"{k}_carts"] = NO_DATA if hidden else ch_carts
        d[f"{k}_orders"] = NO_DATA if hidden else orders
        d[f"{k}_ctr"] = NO_DATA if hidden else safe_div(clk, views)
        d[f"{k}_cpc"] = NO_DATA if hidden else safe_div(rub, clk)
        d[f"{k}_cpo"] = NO_DATA if hidden else safe_div(rub, orders)
        if k == "adv":
            d["adv_cpm"] = safe_div(rub, views) * 1000 if views > 0 else NAN
        d[f"{k}_profit"] = NO_DATA if hidden else profit
        d[f"{k}_romi"] = NO_DATA if hidden else safe_div(profit, rub)

    d["margin"] = mar
    d["margin_pct"] = safe_div(mar, sal)
    d["margin_before_ads_pct"] = safe_div(mar + all_ads, sal)

    stock_units = ctx.stock_mp + ctx.ms_alloc
    d["stock"], d["stock_ms"] = ctx.stock_mp, ctx.ms_alloc
    d["turnover"] = safe_div(stock_units, qty / ctx.n_days) if qty > 0 else NAN
    sebes_fact = float(safe_div(t["sebes_rub"], qty)) if qty > 0 else 0.0
    capital_fact = stock_units * (0.0 if np.isnan(sebes_fact) else sebes_fact)
    d["gmroi"] = (mar / capital_fact) * (365 / ctx.n_days) if (ctx.share_valid and capital_fact > 0 and qty > 0) else NAN
    capital_fc = stock_units * pc.sebes
    d["fc_sales"] = pred_sal
    d["fc_margin"] = pred_mar
    d["fc_margin_pct"] = pred_mar_pct
    d["fc_margin_before_ads_pct"] = pm
    d["fc_gmroi"] = (pred_mar / capital_fc) * (365 / ctx.n_days) if (ctx.share_valid and capital_fc > 0) else NAN
    return d


def calc_model_forecast(t: Mapping, buyout_fc: float, pc: UnitCosts, ctx: StockCtx) -> dict:
    """Forecast-only metrics of one model (summed over models for multi-model reports)."""
    rev, co = t["orders_rub"], t["orders_count"]
    ext_rub = sum(t[CHANNEL_BY_KEY[p].rub] for p in EXT_PARTS)
    all_ads = t["reclama"] + ext_rub
    pred_sal = rev * buyout_fc
    pm = _pred_margin_before_ads(t, pc)
    pred_mar_pct = pm - (all_ads / pred_sal if pred_sal > 0 else 0.0)
    pred_mar = pred_mar_pct * pred_sal
    avg = float(safe_div(rev, co))
    avg = 0.0 if np.isnan(avg) else avg

    def profit(rub, orders):
        return orders * avg * buyout_fc * pm - rub

    ch_profit = {}
    for ch in CHANNELS:
        rub, _, _, orders, _ = _channel_totals(t, ch)
        hidden = ch.key == "blog" and _blog_no_stats(t)
        ch_profit[ch.key] = None if hidden else profit(rub, orders)
    stock_units = ctx.stock_mp + ctx.ms_alloc
    sebes_fact = float(safe_div(t["sebes_rub"], t["sales_qty"])) if t["sales_qty"] > 0 else 0.0
    return {
        "pred_sal": pred_sal, "pred_mar": pred_mar, "pm_rub": pm * pred_sal, "profit": ch_profit,
        "mar": t["margin_rub"], "cap_fact": stock_units * (0.0 if np.isnan(sebes_fact) else sebes_fact),
        "cap_fc": stock_units * pc.sebes,
    }


# ----------------------------------------------------------------------------- report builder
def _activity_mask(rows: pd.DataFrame) -> pd.Series:
    cols = ["orders_count", "sales_qty", "reclama", "b_rub", "ads_vk_c_rub", "ads_vk_s_rub", "ads_s_c_rub", "ads_ya_c_rub"]
    return (rows[cols] > 0).any(axis=1)


def _creator_values(cr: pd.DataFrame) -> dict:
    s = {f: float(cr[f].sum()) if len(cr) else 0.0 for f in CREATOR_FIELDS}
    return {
        "cr_spend": s["cr_spend"], "cr_impr": s["cr_impr"], "cr_cpv": safe_div(s["cr_spend"], s["cr_impr"]),
        "cr_cpc": safe_div(s["cr_spend"], s["cr_clicks"]), "cr_ctr": safe_div(s["cr_clicks"], s["cr_impr"]),
        "cr_clicks": s["cr_clicks"], "cr_carts": s["cr_carts"], "cr_cr_cart": safe_div(s["cr_carts"], s["cr_clicks"]),
        "cr_orders": s["cr_orders"], "cr_cr_order": safe_div(s["cr_orders"], s["cr_carts"]),
        "cr_cro": safe_div(s["cr_orders"], s["cr_clicks"]), "cr_orders_rub": s["cr_orders_rub"],
        "cr_posts": s["cr_posts"],
    } if len(cr) else {}


def build_report(fact: pd.DataFrame, stocks_ms: pd.DataFrame, creators: pd.DataFrame, sel: Selection) -> Report | None:
    """Build the column set (period total, weekly totals, days) for a selection. None if no data."""
    lo = sel.start - pd.Timedelta(days=BUYOUT_LAG_FROM)
    all_dates = pd.date_range(lo, sel.end)
    in_scope = (fact["date"] >= lo) & (fact["date"] <= sel.end)
    attr = pd.Series(True, index=fact.index)
    if sel.models:
        attr &= fact["model"].isin(sel.models)
    if sel.skus:
        attr &= fact["sku"].isin(sel.skus)
    if sel.status != STATUS_ALL:
        attr &= fact["status"] == sel.status
    rows_all_mp = fact[attr & (fact["date"] >= sel.end - pd.Timedelta(days=30)) & (fact["date"] < sel.end)]
    rows = fact[attr & in_scope & (fact["mp"] == sel.mp)].copy()
    if sel.status == "Выводим":  # external ads belong to the selling assortment only
        rows[list(EXT_FIELDS)] = 0.0
    period_rows = rows[(rows["date"] >= sel.start)]
    if period_rows.empty:
        return None

    agg = {f: "sum" for f in SUM_FIELDS}
    agg["b_has_stats"] = "max"
    daily = rows.groupby("date").agg(agg)
    prefix = _Prefix(daily, all_dates)
    pc_all = unit_costs(period_rows)

    # per-model structures (needed for additive forecasts when >= 2 models are active)
    by_model_prefix: dict[str, _Prefix] = {}
    pc_model: dict[str, UnitCosts] = {}
    active: list[str] = []
    for model, grp in rows.groupby("model"):
        in_period = grp[grp["date"] >= sel.start]
        if in_period.empty or not _activity_mask(in_period).any():
            continue
        by_model_prefix[model] = _Prefix(grp.groupby("date").agg(agg), all_dates)
        pc_model[model] = unit_costs(in_period)
        active.append(model)
    multi = len(active) >= 2

    # stocks (marketplace snapshots and own warehouse), lookup of the last known value <= date
    stock_mp = _last_known(rows.groupby("date")["stock_mp"].sum(), all_dates)
    stock_mp_m = {m: _last_known(g.groupby("date")["stock_mp"].sum(), all_dates) for m, g in rows.groupby("model")}
    sku_scope = fact.loc[attr, ["sku", "model"]].drop_duplicates("sku")
    ms_rows = stocks_ms[stocks_ms["sku"].isin(sku_scope["sku"])].merge(sku_scope, on="sku")
    ms_all = _last_known(ms_rows.groupby("date")["qty"].sum(), all_dates)
    ms_m = {m: _last_known(g.groupby("date")["qty"].sum(), all_dates) for m, g in ms_rows.groupby("model")}

    # share of own-warehouse stock allocated to this marketplace: cost of sales over the last 30 days
    l30 = rows_all_mp.groupby(["model", "mp"])["sebes_rub"].sum()
    mp_total = float(l30.xs(sel.mp, level="mp").sum()) if sel.mp in l30.index.get_level_values("mp") else 0.0
    both_total = float(l30.sum())
    share_valid = both_total > 0
    share_all = mp_total / both_total if share_valid else 0.0
    share_m = {}
    for model in l30.index.get_level_values("model").unique():
        s_mp = float(l30.get((model, sel.mp), 0.0))
        s_all = float(l30.xs(model, level="model").sum())
        share_m[model] = s_mp / s_all if s_all > 0 else 0.0

    def ctx_for(snap: pd.Timestamp, n_days: int, model: str | None = None) -> StockCtx:
        if model is None:
            mp_q, ms_q, share = stock_mp.get(snap, 0.0), ms_all.get(snap, 0.0), share_all
        else:
            mp_q = stock_mp_m.get(model, pd.Series(dtype=float)).get(snap, 0.0)
            ms_q = ms_m.get(model, pd.Series(dtype=float)).get(snap, 0.0)
            share = share_m.get(model, 0.0)
        return StockCtx(round(mp_q), round(ms_q * share) if share_valid else 0, n_days, share_valid)

    index_of = {d: i for i, d in enumerate(all_dates)}

    def build_values(first: pd.Timestamp, last: pd.Timestamp) -> dict:
        i0, i1 = index_of[first], index_of[last]
        n_days = i1 - i0 + 1
        t = prefix.window(i0, i1)
        disp = float(safe_div(t["sales_qty"], t["orders_count"]))
        fc_from, fc_to = i0 - BUYOUT_LAG_FROM, i0 - BUYOUT_LAG_TO
        fc = prefix.buyout(fc_from, fc_to) if fc_from >= 0 else NAN
        fc_safe = 0.0 if np.isnan(fc) else fc
        d = calc_metrics(t, disp, fc_safe, pc_all, ctx_for(last, n_days))
        if np.isnan(fc):  # lag window has no orders: forecasts are undefined
            for k in ("fc_sales", "fc_margin", "fc_margin_pct", "fc_margin_before_ads_pct", "fc_gmroi"):
                d[k] = NAN
            for ch in CHANNELS:
                if d[f"{ch.key}_profit"] != NO_DATA:
                    d[f"{ch.key}_profit"] = d[f"{ch.key}_romi"] = NAN
            return d
        if multi:
            _override_forecast(d, t, i0, last, n_days, fc_from, fc_to)
        return d

    def _override_forecast(d: dict, t: Mapping, i0: int, last: pd.Timestamp, n_days: int, fc_from: int, fc_to: int):
        pred_sal = pred_mar = pm_rub = mar_sum = cap_fact = cap_fc = 0.0
        profit = {ch.key: 0.0 for ch in CHANNELS}
        blog_ok = True
        for model in active:
            pf = by_model_prefix[model]
            tm = pf.window(i0, i0 + n_days - 1)
            r = calc_model_forecast(tm, pf.buyout(fc_from, fc_to) if fc_from >= 0 else 0.0,
                                    pc_model[model], ctx_for(last, n_days, model))
            pred_sal += r["pred_sal"]
            pred_mar += r["pred_mar"]
            pm_rub += r["pm_rub"]
            mar_sum += r["mar"]
            cap_fact += r["cap_fact"]
            cap_fc += r["cap_fc"]
            for k, v in r["profit"].items():
                if v is None:
                    blog_ok = blog_ok and k != "blog"
                else:
                    profit[k] += v
        for ch in CHANNELS:
            if ch.key == "blog" and not blog_ok:
                d["blog_profit"] = d["blog_romi"] = NO_DATA
                continue
            if d[f"{ch.key}_profit"] == NO_DATA:
                continue
            rub = d[f"{ch.key}_rub"]
            d[f"{ch.key}_profit"] = profit[ch.key]
            d[f"{ch.key}_romi"] = safe_div(profit[ch.key], rub)
        d["fc_sales"], d["fc_margin"] = pred_sal, pred_mar
        d["fc_margin_pct"] = safe_div(pred_mar, pred_sal) if pred_sal > 0 else NAN
        d["fc_margin_before_ads_pct"] = safe_div(pm_rub, pred_sal) if pred_sal > 0 else NAN
        share_ok = share_valid
        d["gmroi"] = (mar_sum / cap_fact) * (365 / n_days) if (share_ok and cap_fact > 0) else NAN
        d["fc_gmroi"] = (pred_mar / cap_fc) * (365 / n_days) if (share_ok and cap_fc > 0) else NAN

    dates = list(pd.date_range(sel.start, sel.end))
    fmt = lambda d: f"{d:%d.%m}"  # noqa: E731
    wd = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
    cr_scope = creators[creators["mp"] == sel.mp]
    if sel.models:
        cr_scope = cr_scope[cr_scope["model"].isin(sel.models)]

    def creators_for(first: pd.Timestamp, last: pd.Timestamp) -> dict:
        return _creator_values(cr_scope[(cr_scope["week_start"] >= first) & (cr_scope["week_start"] <= last)])

    columns = [Column("total", "Итого", f"с {fmt(dates[0])} по {fmt(dates[-1])}", dates[0], dates[-1],
                      {**build_values(dates[0], dates[-1]), **creators_for(dates[0], dates[-1])})]
    for i in range(0, len(dates), 7):
        chunk = dates[i:i + 7]
        columns.append(Column("week", "Итого за неделю", f"с {fmt(chunk[0])} по {fmt(chunk[-1])}", chunk[0], chunk[-1],
                              {**build_values(chunk[0], chunk[-1]), **creators_for(chunk[0], chunk[-1])}))
        for dt in chunk:
            columns.append(Column("day", wd[dt.weekday()], fmt(dt), dt, dt, build_values(dt, dt)))
    return Report(sel, columns, active)


def month_end_factor(end: pd.Timestamp, n_days: int) -> float:
    """Run-rate multiplier to a full month (used only by tests / simple projections)."""
    return calendar.monthrange(end.year, end.month)[1] / n_days if n_days > 0 else NAN


def hypotheses_by_date(hyp: pd.DataFrame, mp: str, models: Sequence[str], start, end) -> dict[pd.Timestamp, list[dict]]:
    """Hypotheses falling on report days: {date: [{model, text, status}]}; all models when none selected."""
    if hyp.empty:
        return {}
    m = (hyp["Маркетплейс"].str.lower() == mp.lower()) & hyp["Дата"].between(pd.Timestamp(start), pd.Timestamp(end))
    if models:
        low = {x.lower() for x in models}
        m &= hyp["Модель"].str.lower().isin(low)
    out: dict[pd.Timestamp, list[dict]] = {}
    for _, r in hyp[m].sort_values("Дата").iterrows():
        text = str(r["Гипотеза"]).strip()
        if text:
            out.setdefault(r["Дата"], []).append({"model": r["Модель"], "text": text, "status": r["Статус"]})
    return out
