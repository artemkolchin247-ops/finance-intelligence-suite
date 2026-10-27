"""Simple forecast of receipts and expenses (the logic of the category-analytics workbook, reduced to our data).

Receipts, per marketplace:
    GMV(month)      = GMV(same month last year) x exp(ln YoY x lambda^(h/12))
    cash receipts   = GMV x buy-out share x (1 - marketplace costs share)
YoY comes from the last 4 closed months (weights 1-2-3-4, clipped to 0.8...3.0), lambda is the scenario
(pessimistic 0.45 / base 0.70 / optimistic 0.90): growth fades towards lambda of itself a year later.
Expenses: average of the last 3 closed months per article, plus the user's corrections. Every input of the
plan lives in an Excel file (see to_template / parse_template); the application only displays it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import cashflow as C

STREAMS = ("WB", "Ozon")
EXTRA_STREAMS = ("WB", "Ozon", "Сайт/другое")
STREAM_ARTICLE = {"WB": "Продажи Wildberries", "Ozon": "Продажи OZON", "Сайт/другое": "Продажи через сайт"}
SCENARIOS = {"Пессимистичный": 0.45, "Базовый": 0.70, "Оптимистичный": 0.90}
YOY_WEIGHTS = (1, 2, 3, 4)
YOY_MIN, YOY_MAX = 0.8, 3.0
SPECIAL_ARTICLES = {C.INCOME_TAX, C.CREDIT_PRINCIPAL, C.CREDIT_INTEREST, C.CREDIT_IN, C.DEPOSIT_INCOME}
EXTRA_ARTICLES = (C.PAYROLL_TAX, C.VAT, "Выплата дивидендов")
BASE_ARTICLES = ("Закупка товара", "Оплата внешних рекламных каналов", "Прочие операционные расходы")

MP_COLUMNS = ["mp", "yoy_pct", "buyout_pct", "mp_cost_pct"]
PARAM_KEYS = {"purchases_pct": "Закупки и связанные расходы, % от среднего за 3 месяца",
              "scale_variable": "Менять закупки, рекламу и креаторов вместе с поступлениями (Да/Нет)",
              "payroll_index_pct": "Индексация ФОТ, %", "dividends": "Дивиденды, ₽ в месяц", "buffer": "Минимальный остаток, ₽"}
EXTRA_COLUMNS = ["name", "mp", "start", "end", "gmv", "comment"]
ADJ_COLUMNS = ["article", "start", "end", "amount", "comment"]


@dataclass
class PlanInputs:
    mp: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=MP_COLUMNS))
    params: dict = field(default_factory=dict)  # PARAM_KEYS -> value; missing = default
    extra: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=EXTRA_COLUMNS))
    adj: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ADJ_COLUMNS))

    def key(self) -> tuple:
        """Content fingerprint (NaN/NaT-safe) used as a cache key for schedules built from this plan."""
        def rows(df: pd.DataFrame) -> tuple:
            def norm(v):
                if pd.isna(v):
                    return None
                return float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else str(v)
            return tuple(tuple(norm(v) for v in r) for r in df.itertuples(index=False))
        return rows(self.mp), tuple(sorted((k, float(v)) for k, v in self.params.items())), rows(self.extra), rows(self.adj)

    def mp_value(self, mp: str, column: str) -> float | None:
        if self.mp.empty:
            return None
        hit = self.mp[self.mp["mp"] == mp]
        if hit.empty or pd.isna(hit[column].iloc[0]):
            return None
        return float(hit[column].iloc[0])


def empty_plan() -> PlanInputs:
    return PlanInputs()


# --------------------------------------------------------------------------- receipts
def closed_months(fm: pd.DataFrame, as_of: pd.Timestamp) -> list[pd.Timestamp]:
    months = sorted(pd.Timestamp(m) for m in fm["month"].unique())
    return [m for m in months if m + pd.offsets.MonthEnd(0) <= as_of]


def _pivot(fm: pd.DataFrame, column: str) -> pd.DataFrame:
    return fm.pivot_table(index="month", columns="mp", values=column, aggfunc="sum").reindex(columns=list(STREAMS)).fillna(0.0)


def payout_series(fm: pd.DataFrame) -> pd.DataFrame:
    """What the marketplaces pay out per month: revenue after SPP minus everything they withhold."""
    f = fm.copy()
    comm_after = f["comis_spp"] - (f["sales_rub"] - f["sales_spp_rub"])
    f["payout"] = f["sales_spp_rub"] - comm_after - f["logist"] - f["storage"] - f["accept"] - f["other"] - f["reclama"]
    return _pivot(f, "payout")


def auto_assumptions(fm: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, dict[str, float]]:
    """Per marketplace: YoY multiplier, buy-out share and marketplace cost share derived from the history."""
    closed = closed_months(fm, as_of)
    gmv, sales, payout = _pivot(fm, "orders_rub"), _pivot(fm, "sales_rub"), payout_series(fm)
    out = {}
    for mp in STREAMS:
        lns, ws = [], []
        for w, m in zip(YOY_WEIGHTS, closed[-4:]):
            prev = m - pd.DateOffset(months=12)
            if prev in gmv.index and gmv.loc[prev, mp] > 0 and gmv.loc[m, mp] > 0:
                lns.append(math.log(gmv.loc[m, mp] / gmv.loc[prev, mp]))
                ws.append(w)
        yoy = math.exp(float(np.average(lns, weights=ws))) if lns else 1.0
        last3 = closed[-3:]
        s, g, p = sales.loc[last3, mp].sum(), gmv.loc[last3, mp].sum(), payout.loc[last3, mp].sum()
        out[mp] = {"yoy": float(np.clip(yoy, YOY_MIN, YOY_MAX)), "buyout": float(s / g) if g > 0 else 0.0,
                   "mp_cost": float(1 - p / s) if s > 0 else 0.0}
    return out


@dataclass
class ForecastView:
    hist_months: list[pd.Timestamp]
    months: list[pd.Timestamp]
    scenario_lambda: float
    auto: dict[str, dict[str, float]]
    used: dict[str, dict[str, float]]  # values in the calculation (user's or automatic)
    gmv: pd.DataFrame  # hist + forecast months x (WB, Ozon, Доп. выручка)
    yoy: pd.DataFrame  # growth, fraction (actual for history, applied for forecast)
    buyout: pd.DataFrame
    mp_cost: pd.DataFrame
    cash_ratio: pd.DataFrame
    receipts: pd.DataFrame  # months x (WB, Ozon, Доп. выручка, Итого)
    receipts_by_stream: pd.DataFrame  # forecast months x EXTRA_STREAMS (cash article streams)


def build_forecast(fm: pd.DataFrame, as_of: pd.Timestamp, scenario_lambda: float, plan: PlanInputs | None = None) -> ForecastView:
    plan = plan or empty_plan()
    closed = closed_months(fm, as_of)
    last = closed[-1]
    hist_months = closed[-12:]
    months = [last + pd.DateOffset(months=k) for k in range(1, 13)]
    gmv_h, sales_h, payout_h = _pivot(fm, "orders_rub"), _pivot(fm, "sales_rub"), payout_series(fm)
    auto = auto_assumptions(fm, as_of)
    used: dict[str, dict[str, float]] = {}
    for mp in STREAMS:
        yoy_pct = plan.mp_value(mp, "yoy_pct")
        buy_pct, cost_pct = plan.mp_value(mp, "buyout_pct"), plan.mp_value(mp, "mp_cost_pct")
        used[mp] = {"yoy": float(np.clip(1 + yoy_pct / 100, 0.2, 10.0)) if yoy_pct is not None else auto[mp]["yoy"],
                    "buyout": buy_pct / 100 if buy_pct is not None else auto[mp]["buyout"],
                    "mp_cost": cost_pct / 100 if cost_pct is not None else auto[mp]["mp_cost"]}

    all_months = hist_months + months
    gmv = pd.DataFrame(0.0, index=all_months, columns=[*STREAMS, "Доп. выручка"])
    yoy = pd.DataFrame(np.nan, index=all_months, columns=list(STREAMS))
    buyout = pd.DataFrame(np.nan, index=all_months, columns=list(STREAMS))
    cost = pd.DataFrame(np.nan, index=all_months, columns=list(STREAMS))
    for m in hist_months:
        for mp in STREAMS:
            gmv.loc[m, mp] = gmv_h.loc[m, mp]
            prev = m - pd.DateOffset(months=12)
            if prev in gmv_h.index and gmv_h.loc[prev, mp] > 0:
                yoy.loc[m, mp] = gmv_h.loc[m, mp] / gmv_h.loc[prev, mp] - 1
            if gmv_h.loc[m, mp] > 0:
                buyout.loc[m, mp] = sales_h.loc[m, mp] / gmv_h.loc[m, mp]
            if sales_h.loc[m, mp] > 0:
                cost.loc[m, mp] = 1 - payout_h.loc[m, mp] / sales_h.loc[m, mp]
    extra_gmv = pd.DataFrame(0.0, index=months, columns=list(EXTRA_STREAMS))
    for k, m in enumerate(months, start=1):
        for mp in STREAMS:
            prev = m - pd.DateOffset(months=12)
            base = float(gmv_h.loc[prev, mp]) if prev in gmv_h.index else float(gmv_h.loc[last, mp])
            factor = math.exp(math.log(used[mp]["yoy"]) * scenario_lambda ** (k / 12))
            gmv.loc[m, mp] = base * factor
            yoy.loc[m, mp] = factor - 1
            buyout.loc[m, mp], cost.loc[m, mp] = used[mp]["buyout"], used[mp]["mp_cost"]
    for r in plan.extra.itertuples(index=False):
        if pd.isna(r.gmv) or float(r.gmv) <= 0 or pd.isna(r.start):
            continue
        stream = r.mp if r.mp in EXTRA_STREAMS else "Сайт/другое"
        for m in months:
            if m >= r.start and (pd.isna(r.end) or m <= r.end):
                extra_gmv.loc[m, stream] += float(r.gmv)
    gmv.loc[months, "Доп. выручка"] = extra_gmv.sum(axis=1)

    ratio = {mp: used[mp]["buyout"] * (1 - used[mp]["mp_cost"]) for mp in STREAMS}
    ratio["Сайт/другое"] = float(np.mean(list(ratio.values())))  # no own history: the average of the marketplaces
    cash_ratio = pd.DataFrame(np.nan, index=all_months, columns=list(STREAMS))
    for m in hist_months:
        for mp in STREAMS:
            if gmv_h.loc[m, mp] > 0:
                cash_ratio.loc[m, mp] = payout_h.loc[m, mp] / gmv_h.loc[m, mp]
    for m in months:
        for mp in STREAMS:
            cash_ratio.loc[m, mp] = ratio[mp]
    receipts = pd.DataFrame(0.0, index=all_months, columns=[*STREAMS, "Доп. выручка", "Итого"])
    for m in hist_months:
        for mp in STREAMS:
            receipts.loc[m, mp] = payout_h.loc[m, mp]
    by_stream = pd.DataFrame(0.0, index=months, columns=list(EXTRA_STREAMS))
    for m in months:
        for mp in STREAMS:
            by_stream.loc[m, mp] = float(gmv.loc[m, mp]) * ratio[mp]
            receipts.loc[m, mp] = by_stream.loc[m, mp]
        for stream in EXTRA_STREAMS:
            amount = float(extra_gmv.loc[m, stream]) * ratio[stream if stream in ratio else "Сайт/другое"]
            by_stream.loc[m, stream] += amount
            receipts.loc[m, "Доп. выручка"] += amount
    receipts["Итого"] = receipts[[*STREAMS, "Доп. выручка"]].sum(axis=1)
    return ForecastView(hist_months, months, scenario_lambda, auto, used, gmv, yoy, buyout, cost, cash_ratio, receipts, by_stream)


# --------------------------------------------------------------------------- expenses
@dataclass
class Layers:
    months: list[pd.Timestamp]
    receipts: pd.DataFrame  # months x EXTRA_STREAMS: monthly cash receipts
    extra_receipts: pd.DataFrame  # the part coming from additional revenue
    expenses: dict[str, pd.DataFrame]  # base, adjustment, final: articles x months
    direction: dict[str, str]


def build_layers(journal: pd.DataFrame, fm: pd.DataFrame, as_of: pd.Timestamp, assumptions: C.Assumptions,
                 plan: PlanInputs | None) -> Layers:
    plan = plan or empty_plan()
    view = build_forecast(fm, as_of, assumptions.scenario_lambda, plan)
    months = view.months
    hist = journal[(journal["dds_date"] <= as_of) & (journal["activity"] != C.ACT_INTERNAL)]
    recent = C.last_closed_months(as_of, 6)
    recent3 = recent[-3:]

    def net(article: str, ms) -> float:
        h = hist[(hist["article"] == article) & hist["dds_date"].dt.to_period("M").dt.to_timestamp().isin(ms)]
        return float(h["expense"].sum() - h["income"].sum())

    receipts_total = view.receipts["Итого"].reindex(months)
    payout_hist = view.receipts["Итого"].reindex(view.hist_months)
    recent_payout = max(float(payout_hist.reindex(recent).sum()), 1.0)
    mean3 = max(float(payout_hist.reindex(recent3).mean()), 1.0)
    special = SPECIAL_ARTICLES | set(C.RECEIPT_ARTICLES) | set(STREAM_ARTICLE.values())
    adj_articles = set(plan.adj["article"]) if len(plan.adj) else set()
    articles = sorted((set(hist["article"]) - special) | set(EXTRA_ARTICLES) | set(C.PAYROLL) | set(BASE_ARTICLES) | adj_articles)
    n3 = {a: net(a, recent3) / 3 for a in articles}
    vat_ratio = net(C.VAT, recent) / recent_payout
    pay_hist = sum(net(a, recent) for a in C.PAYROLL)
    pt_ratio = net(C.PAYROLL_TAX, recent) / max(pay_hist, 1.0)
    base = pd.DataFrame(0.0, index=articles, columns=months)
    for m in months:
        for a in articles:
            if a == C.VAT:
                v = vat_ratio * float(receipts_total[m])
            elif a in C.PAYROLL:
                v = n3[a] * (1 + assumptions.payroll_index)
            elif a in C.VARIABLE:
                v = n3[a] * assumptions.purchase_scale * (float(receipts_total[m]) / mean3 if assumptions.scale_variable else 1.0)
            elif a == "Выплата дивидендов":
                v = assumptions.dividends_month
            elif a == C.PAYROLL_TAX:
                v = 0.0
            else:
                v = abs(n3[a])
            base.loc[a, m] = max(v, 0.0)
    adj = pd.DataFrame(0.0, index=articles, columns=months)
    for r in plan.adj.itertuples(index=False):
        if pd.isna(r.start) or pd.isna(r.amount):
            continue
        for m in months:
            if m >= r.start and (pd.isna(r.end) or m <= r.end):
                adj.loc[r.article, m] += float(r.amount)
    final = (base + adj).clip(lower=0.0)
    for m in months:  # contributions follow the final payroll
        payroll_m = float(final.loc[list(C.PAYROLL), m].sum())
        base.loc[C.PAYROLL_TAX, m] = pt_ratio * payroll_m
        final.loc[C.PAYROLL_TAX, m] = max(base.loc[C.PAYROLL_TAX, m] + adj.loc[C.PAYROLL_TAX, m], 0.0)
    direction = {a: ("in" if n3[a] < 0 else "out") for a in articles}
    extra_receipts = pd.DataFrame(0.0, index=months, columns=list(EXTRA_STREAMS))
    if len(plan.extra):
        no_extra = build_forecast(fm, as_of, assumptions.scenario_lambda, PlanInputs(plan.mp, plan.params))
        extra_receipts = view.receipts_by_stream - no_extra.receipts_by_stream
    return Layers(months, view.receipts_by_stream, extra_receipts, {"base": base, "adjustment": adj, "final": final}, direction)


# --------------------------------------------------------------------------- assumptions from the plan
def resolve_assumptions(plan: PlanInputs, base: C.Assumptions, scenario_lambda: float) -> C.Assumptions:
    """Global levers: the plan file's values, otherwise defaults derived from the history."""
    p = plan.params
    return C.Assumptions(
        scenario_lambda=scenario_lambda, purchase_scale=float(p.get("purchases_pct", 100.0)) / 100.0,
        payroll_index=float(p.get("payroll_index_pct", 0.0)) / 100.0,
        dividends_month=float(p["dividends"]) if p.get("dividends") is not None else base.dividends_month,
        buffer=float(p["buffer"]) if p.get("buffer") is not None else base.buffer,
        scale_variable=bool(p.get("scale_variable", False)))


# --------------------------------------------------------------------------- storage and Excel
def month_label(m: pd.Timestamp) -> str:
    return f"{m:%m.%Y}"


def parse_month(value) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
        t = pd.Timestamp(value)
        return pd.Timestamp(year=t.year, month=t.month, day=1)
    for fmt in ("%m.%Y", "%Y-%m", "%d.%m.%Y", "%Y-%m-%d", "%m/%Y"):
        try:
            t = pd.to_datetime(str(value).strip(), format=fmt)
            return pd.Timestamp(year=t.year, month=t.month, day=1)
        except ValueError:
            continue
    return None


def plan_to_sheets(plan: PlanInputs) -> dict[str, pd.DataFrame]:
    """Stored form of the plan (the sheets plan_* of the data workbook)."""
    mp = plan.mp.rename(columns={"mp": "Маркетплейс", "yoy_pct": "Рост YoY, %", "buyout_pct": "Выкуп, %", "mp_cost_pct": "Расходы МП, %"})
    params = pd.DataFrame({"Параметр": [PARAM_KEYS[k] for k in plan.params if k in PARAM_KEYS],
                           "Значение": [plan.params[k] for k in plan.params if k in PARAM_KEYS]})
    extra = plan.extra.rename(columns={"name": "Название", "mp": "Маркетплейс", "start": "С месяца", "end": "По месяц",
                                       "gmv": "GMV в месяц, ₽", "comment": "Комментарий"})
    adj = plan.adj.rename(columns={"article": "Статья", "start": "С месяца", "end": "По месяц", "amount": "Добавить, ₽/мес",
                                   "comment": "Комментарий"})
    return {"plan_mp": mp, "plan_params": params, "plan_extra": extra, "plan_adj": adj}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def plan_from_sheets(sheets: dict[str, pd.DataFrame]) -> PlanInputs:
    mp = sheets.get("plan_mp", pd.DataFrame()).rename(columns={"Маркетплейс": "mp", "Рост YoY, %": "yoy_pct", "Выкуп, %": "buyout_pct",
                                                               "Расходы МП, %": "mp_cost_pct"})
    mp = pd.DataFrame({c: mp[c] if c in mp else pd.Series(dtype=object) for c in MP_COLUMNS})
    for c in ("yoy_pct", "buyout_pct", "mp_cost_pct"):
        mp[c] = _num(mp[c])
    mp = mp.dropna(subset=["mp"]).reset_index(drop=True)
    pr = sheets.get("plan_params", pd.DataFrame())
    params: dict = {}
    inv = {v: k for k, v in PARAM_KEYS.items()}
    if {"Параметр", "Значение"} <= set(pr.columns):
        for name, value in zip(pr["Параметр"], pr["Значение"]):
            key = inv.get(str(name))
            if not key or pd.isna(value):
                continue
            if key == "scale_variable":
                params[key] = str(value).strip().lower() in ("да", "yes", "true", "1")
            else:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if not pd.isna(number):
                    params[key] = float(number)
    ex = sheets.get("plan_extra", pd.DataFrame()).rename(columns={"Название": "name", "Маркетплейс": "mp", "С месяца": "start", "По месяц": "end",
                                                                 "GMV в месяц, ₽": "gmv", "Комментарий": "comment"})
    ex = pd.DataFrame({c: ex[c] if c in ex else pd.Series(dtype=object) for c in EXTRA_COLUMNS})
    ex["start"], ex["end"] = ex["start"].map(parse_month), ex["end"].map(parse_month)
    ex["gmv"] = _num(ex["gmv"])
    ex = ex[ex["gmv"].fillna(0) > 0].reset_index(drop=True)
    ad = sheets.get("plan_adj", pd.DataFrame()).rename(columns={"Статья": "article", "С месяца": "start", "По месяц": "end",
                                                               "Добавить, ₽/мес": "amount", "Комментарий": "comment"})
    ad = pd.DataFrame({c: ad[c] if c in ad else pd.Series(dtype=object) for c in ADJ_COLUMNS})
    ad["start"], ad["end"] = ad["start"].map(parse_month), ad["end"].map(parse_month)
    ad["amount"] = _num(ad["amount"])
    ad = ad[ad["amount"].fillna(0) != 0].reset_index(drop=True)
    for df in (ex, ad):
        for c in ("comment", "name", "mp", "article"):
            if c in df:
                df[c] = df[c].astype("string").fillna("").astype(str)
    return PlanInputs(mp, params, ex, ad)


def validate_plan(plan: PlanInputs, known_articles: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    for r in plan.mp.itertuples(index=False):
        if r.mp not in STREAMS:
            errors.append(f"Маркетплейс «{r.mp}» не распознан (WB или Ozon)")
        if not pd.isna(r.yoy_pct) and not -80 <= r.yoy_pct <= 900:
            errors.append(f"{r.mp}: рост YoY {r.yoy_pct:g}% вне разумных границ (−80…900)")
        for col, title in (("buyout_pct", "выкуп"), ("mp_cost_pct", "расходы МП")):
            v = getattr(r, col)
            if not pd.isna(v) and not 0 <= v <= 100:
                errors.append(f"{r.mp}: {title} должен быть от 0 до 100%")
    p = plan.params
    for key in ("purchases_pct", "payroll_index_pct", "dividends", "buffer"):
        if key in p and p[key] < 0:
            errors.append(f"«{PARAM_KEYS[key]}» не может быть отрицательным")
    for i, r in plan.extra.iterrows():
        if pd.isna(r["start"]):
            errors.append(f"Доп. выручка, строка {i + 1}: не указан месяц начала в формате 09.2026")
        if not pd.isna(r["end"]) and not pd.isna(r["start"]) and r["end"] < r["start"]:
            errors.append(f"Доп. выручка, строка {i + 1}: «По месяц» раньше «С месяца»")
    for i, r in plan.adj.iterrows():
        if pd.isna(r["start"]):
            errors.append(f"Корректировки, строка {i + 1}: не указан месяц начала в формате 09.2026")
        if known_articles is not None and r["article"] not in known_articles:
            errors.append(f"Корректировки, строка {i + 1}: статья «{r['article']}» отсутствует в справочнике")
        if not pd.isna(r["end"]) and not pd.isna(r["start"]) and r["end"] < r["start"]:
            errors.append(f"Корректировки, строка {i + 1}: «По месяц» раньше «С месяца»")
    return errors


HOW_TO = [
    "Как заполнять план. Заполняйте только жёлтые ячейки, остальное — справочные значения.",
    "Допущения МП: «Рост YoY, %» — на сколько процентов оборот заказов этого маркетплейса выше, чем в тот же месяц год назад (первый месяц прогноза). "
    "Пусто — берётся авто-значение. Рост со временем затухает: сценарий (пессимистичный / базовый / оптимистичный) выбирается в программе.",
    "«Выкуп, %» и «Расходы МП, %» — доля выкупа от заказов и доля удержаний маркетплейса от продаж. Деньги на р/с = GMV × выкуп × (1 − расходы МП).",
    "Общие допущения: закупки и связанные расходы — % от среднего за 3 последних месяца; дивиденды и минимальный остаток — в рублях. Пусто — по умолчанию.",
    "Доп. выручка: новое направление или канал. Укажите маркетплейс, период (месяцы в виде 09.2026) и GMV в месяц, ₽. Разгон задайте несколькими строками.",
    "Корректировки расходов: статья из справочника, период и сумма в рублях в месяц — положительная добавляет к среднему за 3 месяца, отрицательная убирает.",
]


def to_template(view: ForecastView, plan: PlanInputs, plan_articles: list[str]) -> dict[str, tuple[pd.DataFrame, set[str]]]:
    """Excel template: sheet -> (table, names of the yellow input columns)."""
    mp_rows = []
    for mp in STREAMS:
        a = view.auto[mp]
        mp_rows.append({"Маркетплейс": mp, "Рост YoY, авто, %": round((a["yoy"] - 1) * 100, 1), "Рост YoY, %": plan.mp_value(mp, "yoy_pct"),
                        "Выкуп, авто, %": round(a["buyout"] * 100, 1), "Выкуп, %": plan.mp_value(mp, "buyout_pct"),
                        "Расходы МП, авто, %": round(a["mp_cost"] * 100, 1), "Расходы МП, %": plan.mp_value(mp, "mp_cost_pct")})
    defaults = {"purchases_pct": 100.0, "scale_variable": "Нет", "payroll_index_pct": 0.0, "dividends": None, "buffer": 2_000_000.0}
    params = pd.DataFrame({"Параметр": list(PARAM_KEYS.values()),
                           "По умолчанию": [defaults[k] if defaults[k] is not None else "по истории" for k in PARAM_KEYS],
                           "Ваше значение": [("Да" if plan.params[k] else "Нет") if k == "scale_variable" and k in plan.params
                                             else plan.params.get(k) for k in PARAM_KEYS]})
    sheets = plan_to_sheets(plan)
    extra = sheets["plan_extra"] if len(sheets["plan_extra"]) else pd.DataFrame(
        [{"Название": "Пример: новое направление (удалите или замените)", "Маркетплейс": "Ozon", "С месяца": "01.2027", "По месяц": None,
          "GMV в месяц, ₽": None, "Комментарий": "строка без GMV не учитывается"}])
    adj = sheets["plan_adj"] if len(sheets["plan_adj"]) else pd.DataFrame(
        [{"Статья": "ФОТ Маркетинг", "С месяца": "01.2027", "По месяц": None, "Добавить, ₽/мес": None,
          "Комментарий": "пример: расширение команды; строка без суммы не учитывается"}])
    for df in (extra, adj):
        for c in ("С месяца", "По месяц"):
            df[c] = df[c].map(lambda v: month_label(v) if isinstance(v, pd.Timestamp) and not pd.isna(v) else v)
    return {
        "Допущения МП": (pd.DataFrame(mp_rows), {"Рост YoY, %", "Выкуп, %", "Расходы МП, %"}),
        "Общие допущения": (params, {"Ваше значение"}),
        "Доп. выручка": (extra, set(extra.columns)),
        "Корректировки расходов": (adj, set(adj.columns)),
        "Как заполнять": (pd.DataFrame({"Инструкция": HOW_TO}), set()),
    }


def parse_template(sheets: dict[str, pd.DataFrame], known_articles: set[str]) -> tuple[PlanInputs, list[str]]:
    """Plan from an uploaded template; the second value lists human-readable errors that block the import."""
    errors: list[str] = []
    for name in ("Допущения МП", "Общие допущения", "Доп. выручка", "Корректировки расходов"):
        if name not in sheets:
            errors.append(f"В файле нет листа «{name}»")
    if errors:
        return empty_plan(), errors
    mp_df = sheets["Допущения МП"]
    need = {"Маркетплейс", "Рост YoY, %", "Выкуп, %", "Расходы МП, %"}
    if not need <= set(mp_df.columns):
        return empty_plan(), [f"На листе «Допущения МП» нет столбцов: {', '.join(sorted(need - set(mp_df.columns)))}"]
    mp = mp_df[["Маркетплейс", "Рост YoY, %", "Выкуп, %", "Расходы МП, %"]]
    pr = sheets["Общие допущения"]
    if not {"Параметр", "Ваше значение"} <= set(pr.columns):
        return empty_plan(), ["На листе «Общие допущения» нужны столбцы «Параметр» и «Ваше значение»"]
    for name, value in zip(pr["Параметр"], pr["Ваше значение"]):
        key = {v: k for k, v in PARAM_KEYS.items()}.get(str(name))
        if key and not pd.isna(value) and key != "scale_variable" and pd.isna(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]):
            errors.append(f"«{name}»: нужно число, получено «{value}»")
    plan = plan_from_sheets({"plan_mp": mp, "plan_params": pr[["Параметр", "Ваше значение"]].rename(columns={"Ваше значение": "Значение"}),
                             "plan_extra": sheets["Доп. выручка"], "plan_adj": sheets["Корректировки расходов"]})
    for label, df, cols in (("Доп. выручка", sheets["Доп. выручка"], ("С месяца", "По месяц")),
                            ("Корректировки расходов", sheets["Корректировки расходов"], ("С месяца", "По месяц"))):
        for c in cols:
            if c in df:
                for idx, v in df[c].items():
                    if not pd.isna(v) and parse_month(v) is None:
                        errors.append(f"Лист «{label}», строка {idx + 2}: «{v}» не похоже на месяц (нужен вид 09.2026)")
    if errors:
        return plan, errors
    return plan, validate_plan(plan, known_articles)
