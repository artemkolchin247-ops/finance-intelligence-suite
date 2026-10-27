"""Planning layers of the cash-flow forecast.

Expenses: the base of every article is the average of the last 3 closed months; any month can be replaced by the
user, and initiatives (new direction, team expansion) add their own costs. Receipts: base by marketplace from the
recent level with trend and seasonality, multiplied by the effect of factors (price, orders, ...), optionally replaced
by hand, plus initiative revenue with a ramp-up. The layers are exposed so the UI can show where every figure comes from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core import cashflow as C

STREAMS = ("WB", "Ozon", "Сайт/другое")
STREAM_ARTICLE = {"WB": "Продажи Wildberries", "Ozon": "Продажи OZON", "Сайт/другое": "Продажи через сайт"}
FACTORS = ("Цена", "Рост заказов", "Выкуп", "Реклама", "Ассортимент", "Остатки", "Прочее")
SCENARIOS: dict[str, tuple[float, float]] = {  # (receipts multiplier, share of initiative revenue taken into account)
    "Базовый": (1.0, 1.0), "Оптимистичный": (1.10, 1.0), "Пессимистичный": (0.90, 0.5),
}
SHEET_EXP, SHEET_REC = "Расходы", "Поступления"
SPECIAL_ARTICLES = {C.INCOME_TAX, C.CREDIT_PRINCIPAL, C.CREDIT_INTEREST, C.CREDIT_IN, C.DEPOSIT_INCOME}  # schedule-driven
EXTRA_ARTICLES = (C.PAYROLL_TAX, C.VAT, "Выплата дивидендов")

CELL_COLUMNS = ["sheet", "row", "month", "value"]
FACTOR_COLUMNS = ["mp", "factor", "month_from", "month_to", "effect_pct", "comment"]
INIT_COLUMNS = ["name", "active", "mp", "start", "ramp_months", "revenue", "purchases_pct", "ads_pct", "payroll",
                "payroll_article", "other", "end", "comment"]
INIT_RU = {"name": "Название", "active": "Активна", "mp": "Маркетплейс", "start": "Старт (месяц)", "ramp_months": "Месяцев разгона",
           "revenue": "Выручка в зрелом состоянии, ₽/мес", "purchases_pct": "Закупки, % выручки", "ads_pct": "Реклама, % выручки",
           "payroll": "ФОТ команды, ₽/мес", "payroll_article": "Статья ФОТ", "other": "Прочие расходы, ₽/мес",
           "end": "Окончание (месяц)", "comment": "Комментарий"}
FACTOR_RU = {"mp": "Маркетплейс", "factor": "Фактор", "month_from": "С месяца", "month_to": "По месяц", "effect_pct": "Эффект, % к выручке",
             "comment": "Комментарий"}


@dataclass
class PlanInputs:
    cells: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CELL_COLUMNS))
    factors: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=FACTOR_COLUMNS))
    initiatives: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=INIT_COLUMNS))

    def key(self) -> tuple[int, int, int]:
        """Content fingerprint for caching."""
        def h(df: pd.DataFrame) -> int:
            return int(pd.util.hash_pandas_object(df.astype(str), index=False).sum() % (2 ** 31)) if len(df) else 0
        return h(self.cells), h(self.factors), h(self.initiatives)

    def override(self, sheet: str, row: str, month: pd.Timestamp) -> float | None:
        c = self.cells
        hit = c[(c["sheet"] == sheet) & (c["row"] == row) & (c["month"] == month)] if len(c) else c
        return float(hit["value"].iloc[-1]) if len(hit) else None


def empty_plan() -> PlanInputs:
    return PlanInputs()


# --------------------------------------------------------------------------- layers
@dataclass
class Layers:
    months: list[pd.Timestamp]
    receipts: dict[str, pd.DataFrame]  # base, factor_pct, override, initiative, final: months x streams
    expenses: dict[str, pd.DataFrame]  # base, override, initiative, final: articles x months
    direction: dict[str, str]  # article -> "out" / "in" (net sign of its history)


def _ramp(month: pd.Timestamp, start: pd.Timestamp, ramp: float, end: pd.Timestamp | None) -> float:
    if pd.isna(start) or month < start or (end is not None and not pd.isna(end) and month > end):
        return 0.0
    k = (month.year - start.year) * 12 + month.month - start.month
    return min(1.0, (k + 1) / max(float(ramp or 1), 1.0))


def _active(init: pd.DataFrame) -> pd.DataFrame:
    if init.empty:
        return init
    return init[init["active"].astype(str).str.lower().isin(["true", "1", "да", "yes", "1.0"])]


def build_layers(journal: pd.DataFrame, as_of: pd.Timestamp, assumptions: C.Assumptions, plan: PlanInputs | None,
                 horizon_months: int = 12) -> Layers:
    plan = plan or empty_plan()
    months = [as_of.to_period("M").to_timestamp() + pd.DateOffset(months=i) for i in range(horizon_months)]
    hist = journal[(journal["dds_date"] <= as_of) & (journal["activity"] != C.ACT_INTERNAL)]
    recent = C.last_closed_months(as_of, 6)
    recent3 = recent[-3:]

    def net(article: str, ms) -> float:
        h = hist[(hist["article"] == article) & hist["dds_date"].dt.to_period("M").dt.to_timestamp().isin(ms)]
        return float(h["expense"].sum() - h["income"].sum())

    # ---- receipts: base level x trend x seasonality, factors, replacement, initiatives
    rec_hist = C.receipts_series(journal, as_of)
    total_hist = rec_hist.sum(axis=1)
    season = C.seasonal_index(total_hist) if assumptions.seasonality else pd.Series(1.0, index=range(1, 13))
    base3 = total_hist.reindex(recent3)
    deseason = float(np.nan_to_num(np.mean([base3[m] / season[m.month] for m in recent3]), nan=0.0))
    recent_total = max(float(total_hist.reindex(recent).sum()), 1.0)
    share = {s: (float(rec_hist[STREAM_ARTICLE[s]].reindex(recent).sum()) / recent_total if STREAM_ARTICLE[s] in rec_hist else 0.0)
             for s in STREAMS}
    last_closed = recent[-1]
    base = pd.DataFrame(0.0, index=months, columns=STREAMS)
    for m in months:
        h = (m.year - last_closed.year) * 12 + m.month - last_closed.month
        total = deseason * (1 + assumptions.growth_monthly) ** h * season[m.month] * assumptions.receipts_scale
        for s in STREAMS:
            base.loc[m, s] = total * share[s]
    factor_pct = pd.DataFrame(0.0, index=months, columns=STREAMS)
    for f in plan.factors.itertuples(index=False):
        if pd.isna(f.month_from) or pd.isna(f.effect_pct):
            continue
        to = f.month_to if not pd.isna(f.month_to) else months[-1]
        targets = [s for s in STREAMS if f.mp in (s, "Все")] if f.mp in STREAMS or f.mp == "Все" else []
        for m in months:
            if f.month_from <= m <= to:
                for s in targets:
                    factor_pct.loc[m, s] += float(f.effect_pct) / 100.0
    pre = base * (1 + factor_pct)
    override = pd.DataFrame(np.nan, index=months, columns=STREAMS)
    for m in months:
        for s in STREAMS:
            v = plan.override(SHEET_REC, s, m)
            if v is not None:
                override.loc[m, s] = v
    rec_init = pd.DataFrame(0.0, index=months, columns=STREAMS)
    init = _active(plan.initiatives)
    for r in init.itertuples(index=False):
        stream = r.mp if r.mp in STREAMS else "Сайт/другое"
        for m in months:
            rec_init.loc[m, stream] += float(r.revenue or 0) * _ramp(m, r.start, r.ramp_months, r.end) * assumptions.initiative_weight
    rec_final = override.fillna(pre) + rec_init
    receipts = {"base": base, "factor_pct": factor_pct, "override": override, "initiative": rec_init, "final": rec_final}

    # ---- expenses: average of the last 3 closed months, replacement, initiatives
    special = SPECIAL_ARTICLES | set(C.RECEIPT_ARTICLES) | set(STREAM_ARTICLE.values())
    articles = sorted((set(hist["article"]) - special) | set(EXTRA_ARTICLES) | set(C.PAYROLL) | set(plan.cells[plan.cells["sheet"] == SHEET_EXP]["row"])
                      | {"Закупка товара", "Оплата внешних рекламных каналов", "Прочие операционные расходы"})
    direction: dict[str, str] = {}
    exp_base = pd.DataFrame(0.0, index=articles, columns=months)
    rec_total = rec_final.sum(axis=1)
    mean3 = max(float(total_hist.reindex(recent3).mean()), 1.0)
    vat_ratio = net(C.VAT, recent) / recent_total
    pay_hist = sum(net(a, recent) for a in C.PAYROLL)
    pt_ratio = net(C.PAYROLL_TAX, recent) / max(pay_hist, 1.0)
    n3 = {a: net(a, recent3) / 3 for a in articles}
    for a in articles:
        direction[a] = "in" if n3[a] < 0 else "out"
    exp_override = pd.DataFrame(np.nan, index=articles, columns=months)
    exp_init = pd.DataFrame(0.0, index=articles, columns=months)
    for m in months:
        for a in articles:
            if a == C.VAT:
                v = vat_ratio * float(rec_total[m])
            elif a in C.PAYROLL:
                v = n3[a] * (1 + assumptions.payroll_index)
            elif a in C.VARIABLE:
                v = n3[a] * assumptions.purchase_scale * (float(rec_total[m]) / mean3 if assumptions.scale_variable else 1.0)
            elif a == "Выплата дивидендов":
                v = assumptions.dividends_month
            elif a == C.PAYROLL_TAX:
                v = 0.0  # filled below from the final payroll
            else:
                v = abs(n3[a])
            exp_base.loc[a, m] = max(v, 0.0)
            ov = plan.override(SHEET_EXP, a, m)
            if ov is not None:
                exp_override.loc[a, m] = ov
        for r in init.itertuples(index=False):
            k = _ramp(m, r.start, r.ramp_months, r.end)
            if k == 0.0:
                continue
            rev = float(r.revenue or 0) * k * assumptions.initiative_weight
            exp_init.loc["Закупка товара", m] += rev * float(r.purchases_pct or 0) / 100.0
            exp_init.loc["Оплата внешних рекламных каналов", m] += rev * float(r.ads_pct or 0) / 100.0
            pa = r.payroll_article if r.payroll_article in C.PAYROLL else "ФОТ управление"
            exp_init.loc[pa, m] += float(r.payroll or 0)
            exp_init.loc["Прочие операционные расходы", m] += float(r.other or 0)
    final = exp_override.fillna(exp_base) + exp_init
    for m in months:  # contributions follow the final payroll, so a new team raises them
        payroll_m = float(final.loc[list(C.PAYROLL), m].sum())
        exp_base.loc[C.PAYROLL_TAX, m] = pt_ratio * payroll_m
        if np.isnan(exp_override.loc[C.PAYROLL_TAX, m]):
            final.loc[C.PAYROLL_TAX, m] = exp_base.loc[C.PAYROLL_TAX, m] + exp_init.loc[C.PAYROLL_TAX, m]
    expenses = {"base": exp_base, "override": exp_override, "initiative": exp_init, "final": final}
    return Layers(months, receipts, expenses, direction)


# --------------------------------------------------------------------------- Excel template and import
def month_label(m: pd.Timestamp) -> str:
    return f"{m:%m.%Y}"


def parse_month(value) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)) or hasattr(value, "year"):
        t = pd.Timestamp(value)
        return pd.Timestamp(year=t.year, month=t.month, day=1)
    text = str(value).strip()
    for fmt in ("%m.%Y", "%Y-%m", "%d.%m.%Y", "%Y-%m-%d", "%m/%Y"):
        try:
            t = pd.to_datetime(text, format=fmt)
            return pd.Timestamp(year=t.year, month=t.month, day=1)
        except ValueError:
            continue
    return None


def expense_matrix(layers: Layers, plan: PlanInputs, groups: dict[str, str]) -> pd.DataFrame:
    """Editable matrix: article x months of replacement values (blank = use the base), with the base per month for reference."""
    df = pd.DataFrame({"Статья": layers.expenses["base"].index})
    df["Группа ДДС"] = df["Статья"].map(groups).fillna("Прочее")
    df["База, ₽/мес"] = [float(layers.expenses["base"].loc[a, layers.months[1]] if len(layers.months) > 1 else 0.0)
                         for a in df["Статья"]]
    for m in layers.months:
        df[month_label(m)] = pd.Series([plan.override(SHEET_EXP, a, m) for a in df["Статья"]], dtype="float64").to_numpy()
    return df[(df["База, ₽/мес"] > 0) | df[[month_label(m) for m in layers.months]].notna().any(axis=1)
              | df["Статья"].isin(["Закупка товара", "Оплата внешних рекламных каналов", "Прочие операционные расходы"])].reset_index(drop=True)


def receipts_matrix(layers: Layers, plan: PlanInputs) -> pd.DataFrame:
    df = pd.DataFrame({"Строка": list(STREAMS)})
    for m in layers.months:
        df[month_label(m)] = pd.Series([plan.override(SHEET_REC, s, m) for s in STREAMS], dtype="float64").to_numpy()
    return df


def cells_from_matrices(expenses: pd.DataFrame, receipts: pd.DataFrame, months: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for sheet, df, key in ((SHEET_EXP, expenses, "Статья"), (SHEET_REC, receipts, "Строка")):
        for _, r in df.iterrows():
            for m in months:
                v = r.get(month_label(m))
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    rows.append({"sheet": sheet, "row": r[key], "month": m, "value": float(v)})
    return pd.DataFrame(rows, columns=CELL_COLUMNS)


def initiatives_editor_frame(init: pd.DataFrame) -> pd.DataFrame:
    df = init.copy()
    for c in INIT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[INIT_COLUMNS]
    df["active"] = df["active"].astype(str).str.lower().isin(["true", "1", "да", "yes", "1.0"])
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def validate_plan(plan: PlanInputs) -> list[str]:
    errors = []
    init = plan.initiatives
    for i, r in init.iterrows():
        label = f"Инициатива «{r['name']}»" if str(r["name"]).strip() else f"Инициатива в строке {i + 1}"
        if not str(r["name"]).strip():
            errors.append(f"{label}: не указано название")
        if pd.isna(r["start"]):
            errors.append(f"{label}: не указан месяц старта")
        for col, title in (("revenue", "выручка"), ("payroll", "ФОТ"), ("other", "прочие расходы")):
            if not pd.isna(r[col]) and float(r[col]) < 0:
                errors.append(f"{label}: {title} не может быть отрицательной")
        for col in ("purchases_pct", "ads_pct"):
            if not pd.isna(r[col]) and not 0 <= float(r[col]) <= 100:
                errors.append(f"{label}: проценты должны быть от 0 до 100")
        if not pd.isna(r["start"]) and not pd.isna(r["end"]) and r["end"] < r["start"]:
            errors.append(f"{label}: окончание раньше старта")
    for i, r in plan.factors.iterrows():
        if pd.isna(r["effect_pct"]) or pd.isna(r["month_from"]):
            errors.append(f"Фактор в строке {i + 1}: укажите месяц начала и эффект")
        elif r["mp"] not in (*STREAMS, "Все"):
            errors.append(f"Фактор в строке {i + 1}: неизвестный маркетплейс «{r['mp']}»")
    return errors


def plan_to_sheets(plan: PlanInputs) -> dict[str, pd.DataFrame]:
    """Stored form of a plan (Russian headers)."""
    f = plan.factors.rename(columns=FACTOR_RU)
    i = plan.initiatives.rename(columns=INIT_RU)
    c = plan.cells.rename(columns={"sheet": "Лист", "row": "Строка", "month": "Месяц", "value": "Значение"})
    return {"plan_cells": c, "plan_factors": f, "plan_initiatives": i}


def plan_from_sheets(sheets: dict[str, pd.DataFrame]) -> PlanInputs:
    c = sheets.get("plan_cells", pd.DataFrame()).rename(columns={"Лист": "sheet", "Строка": "row", "Месяц": "month", "Значение": "value"})
    f = sheets.get("plan_factors", pd.DataFrame()).rename(columns={v: k for k, v in FACTOR_RU.items()})
    i = sheets.get("plan_initiatives", pd.DataFrame()).rename(columns={v: k for k, v in INIT_RU.items()})
    cells = pd.DataFrame({col: c[col] if col in c else pd.Series(dtype=object) for col in CELL_COLUMNS})
    cells["month"] = pd.to_datetime(cells["month"], errors="coerce")
    cells["value"] = pd.to_numeric(cells["value"], errors="coerce")
    cells = cells.dropna(subset=["month", "value"]).reset_index(drop=True)
    factors = pd.DataFrame({col: f[col] if col in f else pd.Series(dtype=object) for col in FACTOR_COLUMNS})
    for col in ("month_from", "month_to"):
        factors[col] = pd.to_datetime(factors[col], errors="coerce")
    factors["effect_pct"] = pd.to_numeric(factors["effect_pct"], errors="coerce")
    factors["comment"] = factors["comment"].fillna("").astype(str)
    init = initiatives_editor_frame(pd.DataFrame({col: i[col] if col in i else pd.Series(dtype=object) for col in INIT_COLUMNS}))
    for col in ("ramp_months", "revenue", "purchases_pct", "ads_pct", "payroll", "other"):
        init[col] = pd.to_numeric(init[col], errors="coerce")
    for col in ("name", "mp", "payroll_article", "comment"):
        init[col] = init[col].astype("string").fillna("").astype(str)
    return PlanInputs(cells, factors, init)


def to_template(layers: Layers, plan: PlanInputs, groups: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Excel template of the plan: matrices with replacements, factors, initiatives and a read-only reference of the base."""
    base_exp = layers.expenses["base"].copy()
    base_exp.columns = [month_label(m) for m in layers.months]
    base_exp.insert(0, "Раздел", "Расходы")
    base_rec = layers.receipts["base"].T.copy()
    base_rec.columns = [month_label(m) for m in layers.months]
    base_rec.insert(0, "Раздел", "Поступления")
    ref = pd.concat([base_exp.reset_index(names="Строка"), base_rec.reset_index(names="Строка")], ignore_index=True)
    sheets = plan_to_sheets(plan)
    init = sheets["plan_initiatives"]
    if init.empty:
        init = pd.DataFrame(columns=list(INIT_RU.values()))
    return {SHEET_EXP: expense_matrix(layers, plan, groups), SHEET_REC: receipts_matrix(layers, plan),
            "Факторы": sheets["plan_factors"] if len(sheets["plan_factors"]) else pd.DataFrame(columns=list(FACTOR_RU.values())),
            "Инициативы": init, "База (справка)": ref}


def parse_template(sheets: dict[str, pd.DataFrame]) -> tuple[PlanInputs, list[str]]:
    """Plan from an uploaded template. Returns (plan, errors); errors are human-readable and block the import."""
    errors: list[str] = []
    cells = []
    for sheet, key in ((SHEET_EXP, "Статья"), (SHEET_REC, "Строка")):
        df = sheets.get(sheet)
        if df is None:
            errors.append(f"В файле нет листа «{sheet}»")
            continue
        if key not in df.columns:
            errors.append(f"На листе «{sheet}» нет столбца «{key}»")
            continue
        month_cols = {c: parse_month(c) for c in df.columns if c not in (key, "Группа ДДС", "База, ₽/мес")}
        for c, m in month_cols.items():
            if m is None:
                errors.append(f"Лист «{sheet}»: заголовок «{c}» не похож на месяц (нужен вид 09.2026)")
        for idx, r in df.iterrows():
            for c, m in month_cols.items():
                if m is None or pd.isna(r[c]):
                    continue
                v = pd.to_numeric(pd.Series([r[c]]), errors="coerce").iloc[0]
                if pd.isna(v) or v < 0:
                    errors.append(f"Лист «{sheet}», строка {idx + 2}, {c}: нужно неотрицательное число")
                else:
                    cells.append({"sheet": sheet, "row": str(r[key]).strip(), "month": m, "value": float(v)})
    stored = plan_from_sheets({"plan_factors": sheets.get("Факторы", pd.DataFrame()), "plan_initiatives": sheets.get("Инициативы", pd.DataFrame())})
    for col in ("month_from", "month_to"):
        stored.factors[col] = stored.factors[col].map(lambda v: parse_month(v) if not pd.isna(v) else pd.NaT)
    plan = PlanInputs(pd.DataFrame(cells, columns=CELL_COLUMNS), stored.factors.dropna(how="all").reset_index(drop=True),
                      stored.initiatives[stored.initiatives["name"].astype(str).str.strip() != ""].reset_index(drop=True))
    errors += validate_plan(plan)
    return plan, errors
