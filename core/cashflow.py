"""Cash flow (ДДС): actuals from the journal by cash date, and a forecast engine that builds one dated payment
schedule. The monthly cash-flow statement and the 12-week payment calendar are both views of that schedule,
so their numbers agree by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd

from core.ledger import ACT_FINANCING, ACT_INTERNAL, ACT_OPERATING, activity_of

RECEIPT_ARTICLES = {"Продажи Wildberries": "WB", "Продажи OZON": "Ozon"}
MARKETPLACE_GROUP = {"WB": "Wildberries - взаиморасчеты", "Ozon": "OZON - взаиморасчеты"}
VARIABLE = ("Закупка товара", "Логистика из Китая", "Таможенные платежи", "Маркировка", "Размещение у блогеров",
            "Оплата внешних рекламных каналов", "Услуги контент-креаторов", "Логистика до WB", "Логистика до Озон",
            "Логистика прочая", "Закупка расходных материалов")
PAYROLL = ("ФОТ управление", "ФОТ Маркетинг", "ФОТ Склад и логистика")
PAYROLL_TAX = "Налог на фонд оплаты труда"
VAT, INCOME_TAX = "Налог на добавленную стоимость", "Налог на прибыль"
CREDIT_PRINCIPAL, CREDIT_INTEREST, CREDIT_IN = ("Оплаты по кредитам и займам", "Погашение процентов по кредитам и займам",
                                                "Получение кредитов и займов")
DIVIDENDS = ("Выплата дивидендов", "Вывод прибыли")
DEPOSIT_INCOME = "Прочие поступл. от фин. операций"

CRITICAL = {*PAYROLL, PAYROLL_TAX, VAT, INCOME_TAX, CREDIT_PRINCIPAL, CREDIT_INTEREST, "Закупка товара",
            "Таможенные платежи", "Аренда помещения"}
DEFERRABLE = {*DIVIDENDS, "Услуги контент-креаторов", "Размещение у блогеров", "Оплата внешних рекламных каналов",
              "IT разработка", "Проведение фотосъемок", "Корпоративы и дни рождения", "Обучение и развитие сотрудников",
              "HR, найм персонала", "Юридические услуги", "Подарки для блогеров", "Бартерные интеграции", "Самовыкупы",
              "Фин услуги", "Маркетинговые услуги", "Подрядчики", "Закупка образцов", "Закупка реквизита"}
COUNTERPARTY = {
    "Продажи Wildberries": "Wildberries", "Продажи OZON": "Ozon", "Закупка товара": "Поставщики товара",
    "Логистика из Китая": "Карго-компании", "Таможенные платежи": "Таможенный брокер", VAT: "ФНС (ЕНП)",
    INCOME_TAX: "ФНС (ЕНП)", PAYROLL_TAX: "ФНС (ЕНП)", "Аренда помещения": "Арендодатель",
    CREDIT_PRINCIPAL: "Банк-кредитор", CREDIT_INTEREST: "Банк-кредитор", "Выплата дивидендов": "Учредители",
    "Вывод прибыли": "Учредители", **{a: "Сотрудники" for a in PAYROLL},
}
MAIN_ACCOUNT = {"ООО Вуки": "ООО Вуки Точка ₽", "ИП Петрова": "ИП Петрова Точка ₽"}


def priority_of(article: str) -> str:
    if article in CRITICAL:
        return "Критичный"
    return "Можно отложить" if article in DEFERRABLE else "Обязательный"


def business_day(d: pd.Timestamp) -> pd.Timestamp:
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


@dataclass(frozen=True)
class Assumptions:
    """Global levers of the forecast. The scenario sets how fast growth fades; everything else comes from the plan file."""
    scenario_lambda: float = 0.70  # share of the year-on-year growth that remains a year later
    purchase_scale: float = 1.0  # purchases and related costs vs. their 3-month average
    payroll_index: float = 0.0  # indexation of payroll
    dividends_month: float = 0.0  # monthly dividends
    buffer: float = 2_000_000.0  # minimum free balance
    scale_variable: bool = False  # scale purchases, advertising and creators together with receipts


# --------------------------------------------------------------------------- actuals
def actual_flows(journal: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Cash flows of the journal (internal transfers excluded) by month / activity / direction / group / article."""
    j = journal[(journal["dds_date"] <= as_of) & (journal["activity"] != ACT_INTERNAL)].copy()
    j["month"] = j["dds_date"].dt.to_period("M").dt.to_timestamp()
    rows = []
    for direction, col in (("in", "income"), ("out", "expense")):
        part = j[j[col] > 0].groupby(["month", "activity", "dds_group", "article"])[col].sum().rename("amount").reset_index()
        part["direction"] = direction
        rows.append(part)
    out = pd.concat(rows, ignore_index=True).rename(columns={"dds_group": "group"})
    out["kind"] = "Факт"
    return out


def cash_by_account(journal: pd.DataFrame, accounts: pd.DataFrame, upto: pd.Timestamp) -> pd.DataFrame:
    """Balance of every account at the end of `upto` (opening balance + all movements incl. transfers)."""
    mov = journal[journal["dds_date"] <= upto].groupby("account")["cash"].sum()
    out = accounts.copy()
    out["balance"] = out["opening"] + out["account"].map(mov).fillna(0.0)
    return out


def month_end_cash(journal: pd.DataFrame, accounts: pd.DataFrame, months: list[pd.Timestamp], as_of: pd.Timestamp) -> pd.DataFrame:
    """Closing cash by account group for each month end (months after `as_of` are not available here)."""
    rows = []
    for m in months:
        end = min(m + pd.offsets.MonthEnd(0), as_of)
        bal = cash_by_account(journal, accounts, end)
        row = bal.groupby("account_group")["balance"].sum().to_dict()
        row["month"], row["total"] = m, bal["balance"].sum()
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


# --------------------------------------------------------------------------- assumptions
def receipts_series(journal: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    j = journal[(journal["dds_date"] <= as_of) & journal["article"].isin(RECEIPT_ARTICLES)].copy()
    j["month"] = j["dds_date"].dt.to_period("M").dt.to_timestamp()
    return j.pivot_table(index="month", columns="article", values="income", aggfunc="sum").fillna(0.0)


def last_closed_months(as_of: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    last = (as_of.to_period("M") - 1).to_timestamp() if as_of != as_of + pd.offsets.MonthEnd(0) else as_of.to_period("M").to_timestamp()
    return [last - pd.DateOffset(months=i) for i in range(n - 1, -1, -1)]


def default_assumptions(journal: pd.DataFrame, as_of: pd.Timestamp) -> Assumptions:
    """History-based defaults: dividends = the recent monthly average; the rest is neutral."""
    closed = last_closed_months(as_of, 6)
    div = journal[(journal["article"].isin(DIVIDENDS)) & journal["dds_date"].between(closed[0], as_of)]["expense"].sum() / 6
    return Assumptions(dividends_month=round(float(div), -3))


# --------------------------------------------------------------------------- forecast engine
def _day_profile(ops: pd.DataFrame, col: str) -> list[tuple[int, float]]:
    """Typical payment days of the month: up to three days carrying the most money, with their weights."""
    if ops.empty or ops[col].sum() <= 0:
        return [(15, 1.0)]
    w = ops.groupby(ops["dds_date"].dt.day)[col].sum()
    days = (w.rolling(3, center=True, min_periods=1).sum().sort_values(ascending=False).head(3)).index
    top = w.reindex(sorted(set(days)), fill_value=0.0)
    top = top[top > 0.05 * top.sum()]
    return [(int(d), float(v / top.sum())) for d, v in top.items()] or [(15, 1.0)]


def build_schedule(journal: pd.DataFrame, balance_1c: pd.DataFrame, payment_plan: pd.DataFrame, as_of: pd.Timestamp,
                   assumptions: Assumptions, horizon_months: int = 12, plan=None, fact_monthly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Dated future operations: forecast rules + manual payment plan + planned journal rows.

    Receipts and expenses come from core.forecast: receipts = last year's GMV x fading YoY x buy-out x (1 - marketplace
    costs) plus additional revenue; expenses = 3-month averages plus the plan's corrections. Columns: date, article, group, activity, direction, amount (positive),
    entity, account, priority, status, source, counterparty, comment.
    """
    from core import forecast  # imported here: forecast builds on this module

    if fact_monthly is None:
        raise ValueError("fact_monthly is required: receipts are forecast from the monthly marketplace facts")
    layers = forecast.build_layers(journal, fact_monthly, as_of, assumptions, plan)
    arts = journal.drop_duplicates("article").set_index("article")["dds_group"].to_dict()
    first = (as_of + pd.Timedelta(days=1))
    months = [as_of.to_period("M").to_timestamp() + pd.DateOffset(months=i) for i in range(horizon_months)]
    end_date = months[-1] + pd.offsets.MonthEnd(0)
    hist = journal[(journal["dds_date"] <= as_of) & (journal["activity"] != ACT_INTERNAL)]
    recent = last_closed_months(as_of, 6)
    recent3 = recent[-3:]

    def in_months(df, ms):
        return df[df["dds_date"].dt.to_period("M").dt.to_timestamp().isin(ms)]

    rows: list[dict] = []

    def add(date, article, amount, entity, source, status="Прогноз", counterparty=None, comment="", direction=None, priority=None):
        if amount is None or amount <= 0 or date < first or date > end_date:
            return
        group = arts.get(article, "Прочее")
        rows.append({
            "date": business_day(date), "article": article, "group": group, "activity": activity_of(group),
            "direction": direction or ("in" if article in RECEIPT_ARTICLES or article in (CREDIT_IN, DEPOSIT_INCOME) else "out"),
            "amount": float(amount), "entity": entity, "account": MAIN_ACCOUNT.get(entity, ""), "priority": priority or priority_of(article),
            "status": status, "source": source, "counterparty": counterparty or COUNTERPARTY.get(article, "—"), "comment": comment,
        })

    def entity_split(article: str) -> list[tuple[str, float]]:
        h = in_months(hist[hist["article"] == article], recent)
        tot = (h["income"] + h["expense"]).sum()
        if tot <= 0:
            return [("ООО Вуки", 1.0)]
        ip = (h[h["cabinet"] == "ИП Петрова"][["income", "expense"]].sum().sum()) / tot
        return [("ООО Вуки", 1 - ip), ("ИП Петрова", ip)] if ip > 0.1 else [("ООО Вуки", 1.0)]

    # ---- already known: planned journal rows and the manual payment plan (they replace the rule-based forecast)
    known: dict[tuple[str, pd.Timestamp], float] = {}

    def known_add(article, date, amount):
        key = (article, date.to_period("M").to_timestamp())
        known[key] = known.get(key, 0.0) + amount

    for r in journal[journal["dds_date"] > as_of].itertuples():
        amt = r.expense if r.expense > 0 else r.income
        add(r.dds_date, r.article, amt, r.cabinet or "ООО Вуки", "Журнал (плановый)", "Запланирован", comment=r.comment,
            direction="out" if r.expense > 0 else "in")
        known_add(r.article, r.dds_date, amt)
    manual = payment_plan[~payment_plan["Статус"].isin(["Оплачен"])].dropna(subset=["Дата", "Сумма"])
    for r in manual.itertuples(index=False):
        d = first if (r.Статус == "Просрочен" or r.Дата < first) else r.Дата
        add(d, r[2], r.Сумма, r.Юрлицо or "ООО Вуки", "План платежей", r.Статус, r.Контрагент, f"{r.ID}: {r.Комментарий}".strip(": "),
            priority=r.Приоритет or None)
        known_add(r[2], d, r.Сумма)

    # ---- receipts by stream (marketplaces and additional revenue): weekly payouts of the forecast monthly amount
    payout_hist = forecast.payout_series(fact_monthly).sum(axis=1)
    rec_recent = max(float(payout_hist.reindex(recent).sum()), 1.0)
    rec_final, rec_extra = layers.receipts, layers.extra_receipts
    receipts_fc = {m: float(rec_final.loc[m].sum()) for m in months}
    weekdays = {"WB": (0, 1), "Ozon": (2,)}
    for stream, art in forecast.STREAM_ARTICLE.items():
        for m in months:
            amount_m = float(rec_final.loc[m, stream])
            planned = bool(rec_extra.loc[m, stream] > 0)
            if m == months[0]:  # the current month: what is still to come
                got = journal[(journal["article"] == art) & journal["dds_date"].between(m, as_of)]["income"].sum()
                amount_m = max(amount_m - got, 0.0)
            rng_days = pd.date_range(max(m, first), m + pd.offsets.MonthEnd(0))
            days = [d for d in rng_days if d.weekday() in weekdays[stream]] if stream in weekdays else [d for d in rng_days if d.day in (10, 25)]
            if not days or amount_m <= 0:
                continue
            for entity, share in entity_split(art):
                for d in days:
                    add(d, art, amount_m / len(days) * share, entity, "План" if planned else "Прогноз",
                        comment="Ожидаемая выплата маркетплейса" if stream in weekdays else "Ожидаемое поступление", direction="in")

    # ---- expenses: 3-month averages plus the plan's corrections
    def spread(article: str, month: pd.Timestamp, amount: float, source="Прогноз", note=""):
        h = in_months(hist[hist["article"] == article], recent)
        col = "expense" if h["expense"].sum() >= h["income"].sum() else "income"
        remaining = amount - known.get((article, month), 0.0)
        if remaining <= 0:
            return
        direction = "out" if col == "expense" else "in"
        profile = [(month + pd.Timedelta(days=min(day, month.days_in_month) - 1), wt) for day, wt in _day_profile(h, col)]
        if month == months[0]:  # the current month: days already passed are dropped, their weight goes to the rest
            profile = [(d, wt) for d, wt in profile if d > as_of] or [(first + pd.Timedelta(days=7), 1.0)]
            total_w = sum(wt for _, wt in profile)
            profile = [(d, wt / total_w) for d, wt in profile]
        for d, wt in profile:
            for entity, share in entity_split(article):
                add(d, article, remaining * wt * share, entity, source, comment=note, direction=direction)

    exp = layers.expenses
    for m in months:
        part_of_month = max((m + pd.offsets.MonthEnd(0) - as_of).days / m.days_in_month, 0.0) if m == months[0] else 1.0
        for art in exp["final"].index:
            amount = float(exp["final"].loc[art, m])
            if amount <= 0:
                continue
            spread(art, m, amount * part_of_month, "План" if exp["adjustment"].loc[art, m] != 0 else "Прогноз")

    # income tax: paid on the 28th of the month after each quarter; accrual ratio from recent P&L of the journal
    tax_hist = journal[(journal["article"] == INCOME_TAX) & journal["pl_date"].between(recent[0], recent[-1] + pd.offsets.MonthEnd(0))]
    tax_ratio = float((tax_hist["expense"].sum()) / rec_recent)
    for pay_month in [m for m in months if m.month in (1, 4, 7, 10)]:
        q_start = pay_month - pd.DateOffset(months=3)
        q_months = [q_start + pd.DateOffset(months=i) for i in range(3)]
        q_receipts = sum(float(payout_hist.get(qm, 0.0)) if qm in payout_hist.index and qm <= recent[-1] else receipts_fc.get(qm, 0.0)
                         for qm in q_months)
        amount = tax_ratio * q_receipts - known.get((INCOME_TAX, pay_month), 0.0)
        for entity, share in (("ООО Вуки", 0.88), ("ИП Петрова", 0.12)):
            add(pay_month + pd.Timedelta(days=27), INCOME_TAX, amount * share, entity, "Прогноз",
                comment=f"УСН за {((q_start.month - 1) // 3) + 1} квартал {q_start.year}")

    # credit: constant principal until the debt is repaid, interest on the remaining balance
    debt = float(balance_1c[balance_1c["item"] == "Кредиты и займы"].sort_values("month")["amount"].iloc[-1]) if len(balance_1c) else 0.0
    paid = hist[hist["article"] == CREDIT_PRINCIPAL].groupby(hist["dds_date"].dt.to_period("M"))["expense"].sum()
    principal = float(paid.tail(3).mean()) if len(paid) else 0.0
    interest_last = hist[hist["article"] == CREDIT_INTEREST].sort_values("dds_date")
    rate = (float(interest_last["expense"].iloc[-1]) / max(debt, 1.0)) if len(interest_last) else 0.0
    for m in months:
        if debt <= 0 or principal <= 0:
            break
        d = m + pd.Timedelta(days=9)
        pay = min(principal, debt)
        if d >= first and not known.get((CREDIT_PRINCIPAL, m)):
            add(d, CREDIT_PRINCIPAL, pay, "ООО Вуки", "Прогноз", comment="Основной долг по графику")
            add(d, CREDIT_INTEREST, debt * rate, "ООО Вуки", "Прогноз", comment="Проценты по графику")
        debt -= pay

    schedule = pd.DataFrame(rows)
    if schedule.empty:
        return pd.DataFrame(columns=["date", "article", "group", "activity", "direction", "amount", "entity", "account",
                                     "priority", "status", "source", "counterparty", "comment", "month"])
    schedule["month"] = schedule["date"].dt.to_period("M").dt.to_timestamp()
    schedule = schedule.sort_values(["date", "direction", "article"], kind="stable").reset_index(drop=True)
    schedule["id"] = [f"S-{i + 1:04d}" for i in range(len(schedule))]
    return schedule


# --------------------------------------------------------------------------- cash-flow statement
@dataclass
class StmtColumn:
    key: str
    label: str
    state: str  # "Факт" | "Факт+План" | "План" | "Итого"
    month: pd.Timestamp | None = None


@dataclass
class StmtRow:
    key: str
    label: str
    level: int  # 0 section / total, 1 group, 2 article
    kind: str  # title | group | item | total | balance | memo
    values: dict | None = None
    parent: str | None = None


@dataclass
class Statement:
    columns: list[StmtColumn]
    rows: list[StmtRow]


def forecast_flows(schedule: pd.DataFrame) -> pd.DataFrame:
    if schedule.empty:
        return pd.DataFrame(columns=["month", "activity", "group", "article", "amount", "direction", "kind"])
    s = schedule[schedule["activity"] != ACT_INTERNAL]
    out = s.groupby(["month", "activity", "group", "article", "direction"])["amount"].sum().reset_index()
    out["kind"] = "План"
    return out


def build_dds(journal: pd.DataFrame, accounts: pd.DataFrame, schedule: pd.DataFrame, as_of: pd.Timestamp,
              months: list[pd.Timestamp]) -> Statement:
    """Monthly cash-flow statement: opening balance, flows by activity / direction / group / article, closing balance."""
    act = actual_flows(journal, as_of)
    fc = forecast_flows(schedule)
    flows = pd.concat([act.rename(columns={"amount": "amount"}), fc], ignore_index=True)
    flows = flows[flows["month"].isin(months)]
    cur_month = as_of.to_period("M").to_timestamp()
    cols = [StmtColumn(f"{m:%Y-%m}", f"{m:%m.%Y}", "Факт" if m < cur_month else ("Факт+План" if m == cur_month else "План"), m)
            for m in months]
    cols.append(StmtColumn("total", "Итого", "Итого"))

    def vals(frame: pd.DataFrame) -> dict:
        v = frame.groupby("month")["amount"].sum()
        d = {f"{m:%Y-%m}": float(v.get(m, 0.0)) for m in months}
        d["total"] = sum(d.values())
        return d

    rows: list[StmtRow] = []
    # closing balance per month: the journal up to the as-of date, then the as-of balance plus the cumulated schedule
    cash_asof = float(cash_by_account(journal, accounts, as_of)["balance"].sum())
    signed = (schedule.assign(s=np.where(schedule["direction"] == "in", schedule["amount"], -schedule["amount"]))
              if not schedule.empty else pd.DataFrame({"date": [], "s": []}))

    def balance_at(day: pd.Timestamp) -> float:
        if day <= as_of:
            return float(cash_by_account(journal, accounts, day)["balance"].sum())
        return cash_asof + float(signed[signed["date"] <= day]["s"].sum())

    closing = {m: balance_at(m + pd.offsets.MonthEnd(0)) for m in months}
    opening = {m: balance_at(m - pd.Timedelta(days=1)) for m in months}
    rows.append(StmtRow("open", "Остаток на начало периода", 0, "balance",
                        {**{f"{m:%Y-%m}": opening[m] for m in months}, "total": opening[months[0]]}))

    def section(activity: str, title: str):
        sub = flows[flows["activity"] == activity]
        if sub.empty:
            return None
        rows.append(StmtRow(f"sec|{activity}", title, 0, "title", {c.key: math.nan for c in cols}))
        for direction, dlabel in (("in", "Поступления"), ("out", "Платежи")):
            d = sub[sub["direction"] == direction]
            if d.empty:
                continue
            key = f"dir|{activity}|{direction}"
            rows.append(StmtRow(key, dlabel, 0, "total", vals(d), parent=f"sec|{activity}"))
            for group, g in d.groupby("group"):
                gkey = f"{key}|{group}"
                rows.append(StmtRow(gkey, group, 1, "group", vals(g), parent=key))
                for art, a in g.groupby("article"):
                    rows.append(StmtRow(f"{gkey}|{art}", art, 2, "item", vals(a), parent=gkey))
        net = sub.assign(amount=np.where(sub["direction"] == "in", sub["amount"], -sub["amount"]))
        rows.append(StmtRow(f"net|{activity}", f"Чистый денежный поток — {title.lower()}", 0, "total", vals(net)))
        return net

    nets = [n for n in (section(ACT_OPERATING, "Операционная деятельность"), section(ACT_FINANCING, "Финансовая деятельность")) if n is not None]
    all_net = pd.concat(nets, ignore_index=True) if nets else pd.DataFrame({"month": [], "amount": []})
    rows.append(StmtRow("net", "Чистый денежный поток (ЧДП)", 0, "total", vals(all_net)))
    rows.append(StmtRow("close", "Остаток на конец периода", 0, "balance",
                        {**{f"{m:%Y-%m}": closing[m] for m in months}, "total": closing[months[-1]]}))
    # memo: closing balance by account group (actual months only)
    groups = sorted(accounts["account_group"].unique())
    for g in groups:
        v = {}
        for m in months:
            end = m + pd.offsets.MonthEnd(0)
            v[f"{m:%Y-%m}"] = (float(cash_by_account(journal, accounts, end).query("account_group == @g")["balance"].sum())
                               if end <= as_of else math.nan)
        v["total"] = math.nan
        rows.append(StmtRow(f"memo|{g}", f"в т.ч. {g.lower()}", 1, "memo", v, parent="close"))
    return Statement(cols, rows)


def drilldown(journal: pd.DataFrame, schedule: pd.DataFrame, as_of: pd.Timestamp, article: str, month: pd.Timestamp) -> pd.DataFrame:
    """Operations behind one article in one month: journal rows (actual) and schedule rows (forecast)."""
    m_end = month + pd.offsets.MonthEnd(0)
    a = journal[(journal["article"] == article) & journal["dds_date"].between(month, min(m_end, as_of))]
    fact = pd.DataFrame({"Дата": a["dds_date"], "Сумма": a["cash"], "Счёт / юрлицо": a["account"], "Комментарий": a["comment"],
                         "Тип": "Факт"})
    if schedule.empty:
        return fact.sort_values("Дата")
    s = schedule[(schedule["article"] == article) & (schedule["month"] == month)]
    plan = pd.DataFrame({"Дата": s["date"], "Сумма": np.where(s["direction"] == "in", s["amount"], -s["amount"]),
                         "Счёт / юрлицо": s["entity"], "Комментарий": s["comment"].where(s["comment"] != "", s["counterparty"]),
                         "Тип": "План"})
    return pd.concat([fact, plan], ignore_index=True).sort_values("Дата")


def cash_gap_months(stmt: Statement, buffer: float) -> list[str]:
    close = next(r for r in stmt.rows if r.key == "close")
    return [c.label for c in stmt.columns if c.key != "total" and close.values[c.key] < buffer]
