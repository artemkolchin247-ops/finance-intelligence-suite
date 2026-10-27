"""Builds data/finance_ledger.xlsx: operations journal, accounts, monthly P&L facts, 1C balance items, payment plan.

The journal is derived from the same marketplace figures as the РНП (data/finance_db.xlsx): marketplace payouts,
purchases, advertising and taxes follow the monthly sales; fixed costs follow a company-size index. The journal
carries both dates of an operation: the cash date (Дата ДДС) and the accrual date (Дата P&L).
Run after data/generate_data.py:  python data/generate_ledger.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ledger_reference import ACCOUNTS, ARTICLES, ENTITY_IP, ENTITY_OOO  # noqa: E402

SEED = 11
START = pd.Timestamp("2024-01-01")
AS_OF = pd.Timestamp("2026-09-06")
DB_PATH = Path(__file__).with_name("finance_db.xlsx")
OUT_PATH = Path(__file__).with_name("finance_ledger.xlsx")

MAIN, OOO2, OOO3 = "ООО Вуки Точка ₽", "ООО Вуки Совкомбанк", "ООО Вуки ВТБ"
IP_T, IP_S, IP_V, IP_O = "ИП Петрова Точка ₽", "ИП Петрова Сбербанк ₽", "ИП Петрова ВТБ", "ИП Петрова Ozon банк"
F_TAX, F_NDS, F_FOT, F_DEV, F_TAX_IP = ("Фонд Налоги ООО", "Фонд НДС ООО", "Фонд ФОТ ООО", "Фонд НЗ и развитие ООО",
                                        "Фонд Налоги ИП")
D_DEV, D_TAX, D_IP = "ООО Депозит НЗ и развитие", "ООО Депозит Налоги", "ИП Депозит Налоги"
JOINT, CARD_USD = "Совместный счёт учредителей", "$ Карта компании"
ENTITY_OF = {a[0]: a[2] for a in ACCOUNTS}
DEPOSITS = {D_DEV, D_TAX, D_IP}

MONTHLY_FIELDS = (
    "orders_count", "orders_rub", "orders_spp_rub", "sales_qty", "sales_rub", "sales_spp_rub", "sebes_rub",
    "comp_sebes", "comis_spp", "logist", "storage", "accept", "other", "reclama", "b_rub", "ads_vk_c_rub",
    "ads_vk_s_rub", "ads_s_c_rub", "ads_ya_c_rub", "margin_rub",
)
SEASON = np.array([0.97, 0.94, 1.00, 1.02, 1.05, 0.98, 0.93, 0.96, 1.05, 1.12, 1.17, 1.08])


# --------------------------------------------------------------------------- monthly facts
def build_fact_monthly(rng: np.random.Generator) -> pd.DataFrame:
    """Month x marketplace P&L facts: exact aggregation of the daily data, simulated history before it."""
    sheets = pd.read_excel(DB_PATH, sheet_name=["fact_daily", "fact_ext_daily", "stocks_ms"])
    fact, ext, ms = sheets["fact_daily"], sheets["fact_ext_daily"], sheets["stocks_ms"]
    for df in (fact, ext, ms):
        df["date"] = pd.to_datetime(df["date"])
    fact["month"] = fact["date"].dt.to_period("M").dt.to_timestamp()
    ext["month"] = ext["date"].dt.to_period("M").dt.to_timestamp()
    ms["month"] = ms["date"].dt.to_period("M").dt.to_timestamp()

    base_cols = [c for c in MONTHLY_FIELDS if c in fact.columns]
    daily = fact.groupby(["month", "mp"])[base_cols].sum()
    ext_m = ext.groupby(["month", "mp"])[["b_rub", "ads_vk_c_rub", "ads_vk_s_rub", "ads_s_c_rub", "ads_ya_c_rub"]].sum()
    daily = daily.join(ext_m)
    days = fact.groupby(["month", "mp"])["date"].nunique().rename("days")
    stock_units = fact.groupby(["month", "mp", "date"])["stock_mp"].sum().groupby(["month", "mp"]).mean().rename("stock_units_avg")
    daily = daily.join(days).join(stock_units).reset_index()
    ms_month = ms.groupby(["month", "date"])["qty"].sum().groupby("month").mean()
    daily["ms_total"] = daily["month"].map(ms_month)
    cost = daily["sebes_rub"] / daily["sales_qty"].replace(0, np.nan)
    daily["stock_cost_avg"] = daily["stock_units_avg"] * cost
    share = daily["sebes_rub"] / daily.groupby("month")["sebes_rub"].transform("sum")
    daily["stock_ms_units_avg"] = daily["ms_total"] * share
    daily = daily.drop(columns="ms_total")

    # ratios to revenue from the full months of daily data
    full = daily[daily["days"] >= 25]
    ratios, growth_ref = {}, {}
    for mp, g in full.groupby("mp"):
        ratios[mp] = {c: float(g[c].sum() / g["sales_rub"].sum()) for c in MONTHLY_FIELDS + ("stock_units_avg", "stock_cost_avg",
                                                                                           "stock_ms_units_avg")}
        first = g.sort_values("month").iloc[0]
        growth_ref[mp] = (first["month"], float(first["sales_rub"]))

    hist_rows = []
    first_daily = daily["month"].min()
    for mp, (ref_month, ref_sales) in growth_ref.items():
        g = 1.055 if mp == "WB" else 1.075
        start = START if mp == "WB" else pd.Timestamp("2024-07-01")
        for month in pd.date_range(start, first_daily - pd.offsets.MonthBegin(1), freq="MS"):
            k = (ref_month.year - month.year) * 12 + (ref_month.month - month.month)
            sales = ref_sales / (g ** k) * SEASON[month.month - 1] / SEASON[ref_month.month - 1] * rng.normal(1, 0.03)
            row = {"month": month, "mp": mp, "days": month.days_in_month}
            for c, r in ratios[mp].items():
                row[c] = sales * r * rng.normal(1, 0.02)
            row["sales_rub"] = sales
            hist_rows.append(row)
    hist = pd.DataFrame(hist_rows)
    # margin by the same identity as in the daily data
    comis_after = hist["comis_spp"] - (hist["sales_rub"] - hist["sales_spp_rub"])
    ext_total = hist[["b_rub", "ads_vk_c_rub", "ads_vk_s_rub", "ads_s_c_rub", "ads_ya_c_rub"]].sum(axis=1)
    hist["margin_rub"] = (hist["sales_spp_rub"] - hist["sebes_rub"] - hist["comp_sebes"] - comis_after - hist["logist"]
                          - hist["storage"] - hist["accept"] - hist["other"] - 0.05 * hist["sales_spp_rub"]
                          - hist["reclama"] - ext_total)
    out = pd.concat([hist, daily], ignore_index=True).sort_values(["month", "mp"]).reset_index(drop=True)
    out["is_partial"] = out["days"] < out["month"].dt.days_in_month
    return out


# --------------------------------------------------------------------------- journal engine
class Ledger:
    """Posts operations and keeps running account balances; tops accounts up from donors when funds are short."""

    DONORS = {
        MAIN: [OOO2, D_DEV, OOO3, D_TAX],
        IP_T: [IP_S, IP_V, IP_O, D_IP, MAIN],
        IP_S: [IP_T, D_IP, MAIN],
        F_TAX: [MAIN, D_TAX, OOO2, OOO3, IP_T], F_NDS: [MAIN, D_DEV, OOO2, OOO3, IP_T, F_DEV],
        F_FOT: [MAIN, D_DEV, OOO2, OOO3, IP_T, F_DEV], F_DEV: [MAIN, OOO2],
        F_TAX_IP: [IP_T, IP_S, D_IP], JOINT: [MAIN], CARD_USD: [MAIN], OOO2: [MAIN], OOO3: [MAIN],
    }

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.rows: list[dict] = []
        self.bal = {a[0]: float(a[3]) for a in ACCOUNTS}
        self.counter = 0

    def total_cash(self) -> float:
        return sum(self.bal.values())

    def post(self, dds, pl, article, acc, inc=None, exp=None, comment="", mp="", cabinet=None, sku="", size="", qty=None):
        amount = (inc or 0.0) - (exp or 0.0)
        self.bal[acc] += amount
        self.rows.append({
            "Дата ДДС": dds, "Дата P&L": pl, "Приход": inc, "Расход": exp, "Статья операции": article, "МП": mp,
            "Кабинет": cabinet if cabinet is not None else ENTITY_OF[acc], "Артикул поставщика": sku, "Р-р": size,
            "Кол-во": qty, "Касса / Счет": acc, "Комментарий": comment,
        })

    def transfer(self, date, amount, src, dst, article="Перевод между счетами", comment=""):
        amount = float(round(amount))
        if amount <= 0 or src == dst:
            return
        self.post(date, date, article, src, exp=amount, comment=comment)
        self.post(date, date, article, dst, inc=amount, comment=comment)

    def ensure(self, acc: str, amount: float, date, depth: int = 0) -> None:
        """Move money to `acc` from its donors until it can pay `amount`."""
        need = amount - self.bal[acc]
        if need <= 0:
            return
        need = np.ceil(need / 1000.0) * 1000.0 + 1000.0
        for donor in self.DONORS.get(acc, [MAIN]):
            if need <= 0:
                break
            if donor in DEPOSITS:
                give = min(self.bal[donor], need)
            else:
                spare = self.bal[donor] - (1_200_000 if donor == MAIN else 150_000)
                give = min(max(spare, 0.0), need)
            if give < 1000:
                continue
            art = "Возврат с депозита" if donor in DEPOSITS else "Перевод между счетами"
            self.transfer(date, give, donor, acc, art)
            need -= give
        if need > 0 and depth == 0 and acc != MAIN:
            self.ensure(MAIN, need, date, depth + 1)
            self.transfer(date, min(need, max(self.bal[MAIN], 0)), MAIN, acc)

    def pay(self, dds, pl, article, amount, acc, comment="", **kw):
        amount = float(round(amount))
        if amount <= 0:
            return
        self.ensure(acc, amount, dds)
        self.post(dds, pl, article, acc, exp=amount, comment=comment, **kw)

    def receive(self, dds, pl, article, amount, acc, comment="", **kw):
        amount = float(round(amount))
        if amount > 0:
            self.post(dds, pl, article, acc, inc=amount, comment=comment, **kw)


def business_day(d: pd.Timestamp) -> pd.Timestamp:
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def month_end(d: pd.Timestamp) -> pd.Timestamp:
    return d + pd.offsets.MonthEnd(0)


MODELS_FOR_PURCHASE = ["Vuki", "Moon", "Wendy", "Ruby", "Joy", "Audrey", "Lana", "Moon2", "Vuki2", "Valery", "Bella", "Alice"]
MODEL_W = np.array([0.20, 0.06, 0.15, 0.17, 0.11, 0.09, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03])
SIZES = ["XS", "S", "M", "L", "XL", "S/M", "L/XL"]


def build_journal(fm: pd.DataFrame, creators: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    led = Ledger(rng)
    months = pd.date_range(START, AS_OF, freq="MS")
    by = {(r.month, r.mp): r for r in fm.itertuples(index=False)}
    rev_total = fm.groupby("month")["sales_rub"].sum()
    ref_rev = float(rev_total.loc[pd.Timestamp("2026-03-01")])
    size_idx = lambda m: float(np.clip((rev_total.get(m, ref_rev) / ref_rev) ** 0.9, 0.2, 1.35))  # noqa: E731

    def payout(month: pd.Timestamp, mp: str) -> float:
        r = by.get((month, mp))
        if r is None:
            return 0.0
        comis_after = r.comis_spp - (r.sales_rub - r.sales_spp_rub)
        val = r.sales_spp_rub - comis_after - r.logist - r.storage - r.accept - r.other - r.reclama
        return float(val) * (r.days / month.days_in_month if r.days < month.days_in_month else 1.0)

    events: list[tuple[pd.Timestamp, int, callable]] = []

    def add(date, order, fn):
        if date is not None and START <= date <= AS_OF:
            events.append((date, order, fn))

    planned_tax_rows: list[tuple] = []
    indirect_pl: dict[pd.Timestamp, float] = {}  # accrual-month indirect costs, to size income tax

    for month in months:
        k = size_idx(month) * 0.8
        p_ooo = float(np.clip(0.60 + 0.007 * ((month.year - 2024) * 12 + month.month), 0.6, 0.82))
        mend = month_end(month)
        prev = month - pd.offsets.MonthBegin(1)
        indirect = 0.0
        f = 0.4 + 0.6 * k  # fixed overheads grow with the company

        # --- marketplace payouts: 82% in the month, the rest settles in the following month
        for mp, article, weekday in (("WB", "Продажи Wildberries", (0, 1)), ("Ozon", "Продажи OZON", (2,))):
            cash = 0.82 * payout(month, mp) + 0.18 * payout(prev, mp)
            if cash <= 0:
                continue
            days = [d for d in pd.date_range(month, mend) if d.weekday() in weekday]
            weights = rng.dirichlet(np.full(len(days), 6.0))
            for d, w in zip(days, weights):
                for entity, share in ((ENTITY_OOO, p_ooo), (ENTITY_IP, 1 - p_ooo)):
                    amt = cash * w * share * rng.uniform(0.96, 1.04)
                    if mp == "WB":
                        acc = MAIN if entity == ENTITY_OOO else IP_S
                    else:
                        acc = MAIN if entity == ENTITY_OOO else IP_T
                    text = "Выручка ООО" if entity == ENTITY_OOO else "Выручка ИП"
                    if mp == "Ozon":
                        text = f"Оплата за товары по договору, реестр №{int(rng.integers(10**8, 10**9))}"
                    add(d, 0, lambda d=d, amt=amt, acc=acc, art=article, text=text, mp=mp, e=entity:
                        led.receive(d, d, art, amt, acc, text, mp="WB" if mp == "WB" else "OZON", cabinet=e))

        # --- purchases of goods: next month's cost of sales, 2-3 invoices
        nxt = month + pd.offsets.MonthBegin(1)
        cogs_next = sum((by[(nxt, mp)].sebes_rub if (nxt, mp) in by and not by[(nxt, mp)].is_partial else by[(month, mp)].sebes_rub * 1.04)
                        for mp in ("WB", "Ozon") if (month, mp) in by)
        landed = cogs_next * rng.uniform(0.97, 1.08)  # cost of sales is landed cost: goods + freight + customs + marking
        purchases = landed * 0.88
        for part in rng.dirichlet(np.full(int(rng.integers(2, 4)), 4.0)):
            d = business_day(month + pd.Timedelta(days=int(rng.integers(2, 27))))
            model = rng.choice(MODELS_FOR_PURCHASE, p=MODEL_W)
            amt = purchases * part
            qty = int(amt / rng.uniform(300, 380))
            invoice = int(rng.integers(20, 990))
            add(d, 2, lambda d=d, amt=amt, model=model, qty=qty, inv=invoice, e=(ENTITY_OOO if rng.random() < 0.93 else ENTITY_IP):
                led.pay(d, d, "Закупка товара", amt, MAIN if e == ENTITY_OOO else IP_T,
                        f"Счёт на оплату №{inv} от {d:%d.%m.%Y}, за ткани и пошив, в т.ч. НДС 5%",
                        sku=model, size=str(rng.choice(SIZES)), qty=qty))
        for art, share, lo, hi, text in (
            ("Логистика из Китая", 0.075 / 0.88, 1, 3, "Доставка груза из Китая, поставка"),
            ("Таможенные платежи", 0.035 / 0.88, 0, 1, "Таможенные платежи по декларации"),
            ("Маркировка", 0.01 / 0.88, 1, 1, "Оплата кодов маркировки"),
        ):
            n = int(rng.integers(lo, hi + 1))
            if art == "Таможенные платежи" and month.month % 2:
                n = 0
            for _ in range(n):
                d = business_day(month + pd.Timedelta(days=int(rng.integers(3, 27))))
                amt = purchases * share * (2.0 if art == "Таможенные платежи" else 1.0) / max(n, 1) * rng.uniform(0.7, 1.3)
                add(d, 2, lambda d=d, amt=amt, art=art, text=text: led.pay(d, d, art, amt, MAIN,
                                                                           f"{text} №{int(rng.integers(100, 999))}"))

        # --- advertising: bloggers / external channels (cash ~ spend of the same month), creators
        b_total = sum(by[(month, mp)].b_rub for mp in ("WB", "Ozon") if (month, mp) in by)
        x_total = sum(by[(month, mp)].ads_vk_c_rub + by[(month, mp)].ads_vk_s_rub + by[(month, mp)].ads_s_c_rub
                      + by[(month, mp)].ads_ya_c_rub for mp in ("WB", "Ozon") if (month, mp) in by)
        for art, total, n, text in (("Размещение у блогеров", b_total, 7, "Рекламное размещение у блогера"),
                                    ("Оплата внешних рекламных каналов", x_total, 3, "Рекламный кабинет, пополнение")):
            if total <= 0:
                continue
            for w in rng.dirichlet(np.full(n, 3.0)):
                d = business_day(month + pd.Timedelta(days=int(rng.integers(0, month.days_in_month))))
                adv_date = d - pd.Timedelta(days=int(rng.integers(0, 8)))  # prepayment: accrued a bit later
                add(d, 2, lambda d=d, amt=total * w, art=art, text=text, pl=max(adv_date, month):
                    led.pay(d, pl, art, amt, MAIN if rng.random() < 0.85 else IP_T, text,
                            mp="WB" if rng.random() < 0.8 else "OZON"))
        cr = creators[(creators["week_start"] >= month) & (creators["week_start"] <= mend)]["cr_spend"].sum()
        if month < pd.Timestamp("2026-03-01"):
            cr = rev_total.get(month, ref_rev) * 0.0045 * rng.uniform(0.7, 1.3)
        for w in rng.dirichlet(np.full(3, 4.0)):
            d = business_day(month + pd.Timedelta(days=int(rng.integers(3, 27))))
            add(d, 2, lambda d=d, amt=cr * w: led.pay(d, d, "Услуги контент-креаторов", amt, MAIN, "Оплата услуг контент-креаторов за интеграции"))
        indirect += cr

        # --- payroll: three articles; salary of month M is paid on the 5th/20th of M+1 and M
        for art, base, rows in (("ФОТ управление", 850_000, 7), ("ФОТ Маркетинг", 600_000, 5), ("ФОТ Склад и логистика", 330_000, 4)):
            amount = base * k * rng.uniform(0.96, 1.05)
            indirect += amount
            for i, w in enumerate(rng.dirichlet(np.full(rows, 5.0))):
                advance = i % 2 == 0
                d = business_day(month + pd.Timedelta(days=19)) if advance else business_day(mend + pd.Timedelta(days=5))
                text = "Аванс за месяц" if advance else "Заработная плата за месяц"
                add(d, 2, lambda d=d, amt=amount * w, art=art, text=text, pl=mend, acc=F_FOT:
                    led.pay(d, pl, art, amt, acc, f"{text} {pl:%m.%Y}"))
        fot = sum(base for base in (850_000, 600_000, 330_000)) * k
        for i, w in enumerate(rng.dirichlet(np.full(3, 6.0))):
            d = business_day(mend + pd.Timedelta(days=5 + 10 * (i % 2)))
            add(d, 2, lambda d=d, amt=fot * 0.136 * w: led.pay(d, mend, "Налог на фонд оплаты труда", amt, F_FOT, f"ЕНП (СВ {mend:%m.%Y})"))
        indirect += fot * 0.136

        # --- fixed administrative / production items
        rent = 300_000 * f * (1.05 ** (month.year - 2024)) if month >= pd.Timestamp("2024-02-01") else 0.0
        fixed = [("Аренда помещения", rent, 1, 15, "Арендная плата за склад и офис"),
                 ("РКО", 38_000 * k, 3, 4, "Комиссия за обслуживание счёта"),
                 ("Бухгалтерские услуги", 70_000 * f, 1, 8, "Бухгалтерское обслуживание"),
                 ("IT разработка", 110_000 * f * rng.uniform(0.3, 1.8), 1, 12, "Разработка и сопровождение IT-систем"),
                 ("Юридические услуги", 90_000 * f if month.month % 2 else 0, 1, 18, "Юридическое сопровождение"),
                 ("HR, найм персонала", 50_000 * f * rng.uniform(0, 2.4), 1, 22, "Подбор персонала"),
                 ("Обучение и развитие сотрудников", 25_000 * f * rng.uniform(0, 2.2), 1, 14, "Обучение сотрудников"),
                 ("Корпоративы и дни рождения", 90_000 if month.month % 3 == 0 else 8_000, 1, 25, "Корпоративное мероприятие"),
                 ("Подрядчики", 60_000 * k * rng.uniform(0.6, 1.4), 2, 10, "Услуги подрядчиков по договору"),
                 ("Маркетинговые услуги", 55_000 * k * rng.uniform(0.5, 1.5), 1, 12, "Маркетинговое сопровождение"),
                 ("Проведение фотосъемок", 75_000 * k * rng.uniform(0.4, 1.6), 2, 9, "Предметная и студийная фотосъёмка"),
                 ("Содержание склада", 15_000 * k * rng.uniform(0, 2), 1, 20, "Содержание склада"),
                 ("Закупка расходных материалов", 55_000 * k * rng.uniform(0.6, 1.4), 3, 7, "Упаковка и расходные материалы"),
                 ("Разовый персонал", 30_000 * k * rng.uniform(0, 2), 2, 11, "Оплата разового персонала"),
                 ("Закупка образцов", 30_000 * rng.uniform(0, 2.2), 1, 6, "Закупка образцов"),
                 ("Логистика до WB", 55_000 * k * rng.uniform(0.7, 1.3), 3, 5, "Доставка поставки до склада WB"),
                 ("Логистика до Озон", 7_000 * k * rng.uniform(0, 2), 1, 15, "Доставка поставки до склада Озон"),
                 ("Логистика прочая", 30_000 * k * rng.uniform(0.5, 1.5), 3, 13, "Прочая логистика"),
                 ("Фулфилмент", 60_000 if rng.random() < 0.12 else 0, 1, 21, "Услуги фулфилмента"),
                 ("Самовыкупы", 10_000 * rng.uniform(0, 2.2), 1, 16, "Самовыкупы"),
                 ("Бартерные интеграции", 15_000 if rng.random() < 0.2 else 0, 1, 19, "Бартерная интеграция"),
                 ("Подарки для блогеров", 6_000 if rng.random() < 0.3 else 0, 1, 23, "Подарки блогерам"),
                 ("Кэшбек покупателю", 1_000 * rng.uniform(0, 2.5), 1, 17, "Кэшбек покупателю"),
                 ("Прочие расходы на сотрудников", 10_000 * rng.uniform(0, 2), 1, 24, "Расходы на сотрудников"),
                 ("Прочие операционные расходы", 12_000 * rng.uniform(0, 2.2), 1, 26, "Прочие операционные расходы")]
        for art, amount, n, day, text in fixed:
            if amount <= 0:
                continue
            indirect += amount
            for w in rng.dirichlet(np.full(n, 8.0)):
                d = business_day(month + pd.Timedelta(days=min(day + int(rng.integers(-2, 3)), month.days_in_month - 1)))
                pl = month if art in ("Аренда помещения", "Программное обеспечение") else d
                add(d, 2, lambda d=d, amt=amount * w, art=art, text=text, pl=pl:
                    led.pay(d, pl, art, amt, MAIN if rng.random() < 0.85 else IP_T, text))
        # software subscriptions: many small card payments
        soft = 115_000 * k
        indirect += soft
        for w in rng.dirichlet(np.full(11, 2.0)):
            d = business_day(month + pd.Timedelta(days=int(rng.integers(0, month.days_in_month))))
            tool = rng.choice(["Сервис аналитики МП", "Таск-трекер", "CRM", "Облачное хранилище", "Сервис этикеток",
                               "Корпоративная почта", "Сервис рассылок", "Графический редактор"])
            add(d, 2, lambda d=d, amt=soft * w, tool=tool, pl=month + pd.Timedelta(days=int(rng.integers(0, 3))), acc=(JOINT if rng.random() < 0.3 else MAIN):
                led.pay(d, pl, "Программное обеспечение", amt, acc, f"Подписка: {tool}"))
        if month >= pd.Timestamp("2025-10-01"):
            d = business_day(month + pd.Timedelta(days=int(rng.integers(5, 25))))
            amt = 8_000 * rng.uniform(0.7, 1.4)
            indirect += amt
            add(d, 2, lambda d=d, amt=amt: led.pay(d, d, "ИИ сервис", amt, CARD_USD, "Подписка на ИИ-сервис"))
        # incomes booked against expense categories / after EBITDA
        for art, amount, prob, text in (("Возмещения от клиентов", 4_000, 0.6, "Возмещение от клиента"),
                                        ("Возврат оплаты", 8_000, 0.35, "Возврат оплаты поставщиком"),
                                        ("Возврат маркетинговых расходов", 25_000, 0.2, "Возврат рекламного бюджета"),
                                        ("Продажи через сайт", 1_500, 0.15, "Оплата заказа на сайте")):
            if rng.random() < prob:
                d = business_day(month + pd.Timedelta(days=int(rng.integers(2, 26))))
                amt = amount * rng.uniform(0.5, 2.0)
                indirect -= amt
                add(d, 1, lambda d=d, amt=amt, art=art, text=text: led.receive(d, d, art, amt, MAIN, text))
        indirect_pl[month] = indirect

        # --- credit service
        if month >= pd.Timestamp("2024-02-01"):
            debt_factor = max(0.2, 1 - 0.011 * ((month.year - 2024) * 12 + month.month))
            d10 = business_day(month + pd.Timedelta(days=9))
            add(d10, 2, lambda d=d10, amt=258_000: led.pay(d, d, "Оплаты по кредитам и займам", amt, MAIN, "Погашение основного долга по кредитному договору"))
            add(d10, 2, lambda d=d10, amt=95_000 * debt_factor, m=month: led.pay(d, d, "Погашение процентов по кредитам и займам", amt, MAIN, "Уплата процентов по кредитному договору"))
        if month in (pd.Timestamp("2024-03-01"), pd.Timestamp("2025-05-01")):
            d = business_day(month + pd.Timedelta(days=13))
            amt = 1_500_000.0 if month.year == 2024 else 1_650_000.0
            add(d, 0, lambda d=d, amt=amt: led.receive(d, d, "Получение кредитов и займов", amt, MAIN, "Поступление кредитных средств по договору"))

        # --- VAT: a third of the previous quarter's accrual on the 28th of each following month
        quarter_start = pd.Timestamp(year=month.year, month=3 * ((month.month - 1) // 3) + 1, day=1) - pd.DateOffset(months=3)
        q_months = [quarter_start + pd.DateOffset(months=i) for i in range(3)]
        nds_q = sum(0.05 * by[(qm, mp)].sales_spp_rub for qm in q_months for mp in ("WB", "Ozon") if (qm, mp) in by)
        d28 = business_day(month + pd.Timedelta(days=27))
        if nds_q > 0:
            add(d28, 2, lambda d=d28, amt=nds_q / 3, q=q_months[-1]: led.pay(d, d, "Налог на добавленную стоимость", amt, F_NDS,
                                                                           f"Единый налоговый платеж (ЕНП) (1/3 НДС за {((q.month - 1) // 3) + 1} квартал {q.year}). без НДС"))

    # --- income tax (USN): accrual by month, paid quarterly on the 28th of the month after the quarter
    margin_m = fm.groupby("month")["margin_rub"].sum()
    for month in months:
        if month.normalize() > pd.Timestamp("2026-08-01"):
            continue
        ebitda = margin_m.get(month, 0.0) - indirect_pl.get(month, 0.0)
        accrual = max(0.0, 0.35 * ebitda) * rng.uniform(0.9, 1.1)
        pay_month = pd.Timestamp(year=month.year, month=3 * ((month.month - 1) // 3) + 1, day=1) + pd.DateOffset(months=3)
        pay_date = business_day(pay_month + pd.Timedelta(days=27))
        q = (month.month - 1) // 3 + 1
        mend = month_end(month)
        for entity, share, acc in ((ENTITY_OOO, 0.88, F_TAX), (ENTITY_IP, 0.12, F_TAX_IP)):
            amt = accrual * share
            if pay_date <= AS_OF:
                add(pay_date, 2, lambda d=pay_date, amt=amt, acc=acc, pl=mend, q=q, y=month.year:
                    led.pay(d, pl, "Налог на прибыль", amt, acc, f"Единый налоговый платеж (ЕНП) (УСН за {q} квартал {y}). без НДС"))
            else:
                planned_tax_rows.append((pay_date, mend, round(amt), acc))

    # --- weekly treasury: fund allocation, sweeps between accounts and deposits, interest, dividends
    for friday in pd.date_range(START, AS_OF, freq="W-FRI"):
        add(friday, 1, lambda d=friday: _treasury(led, d))
    for month in months:
        d = business_day(month + pd.Timedelta(days=13))
        add(d, 3, lambda d=d: _dividends(led, d, rng))
        mend = month_end(month)
        if mend <= AS_OF:
            add(mend, 3, lambda d=mend: _interest(led, d))
        # personal cards are funded for card subscriptions
        add(business_day(month + pd.Timedelta(days=1)), 1, lambda d=business_day(month + pd.Timedelta(days=1)):
            led.transfer(d, 25_000 + rng.integers(0, 4) * 5_000, MAIN, JOINT, comment="Пополнение карты для подписок"))

    for date, _, fn in sorted(events, key=lambda e: (e[0], e[1])):
        fn()

    journal = pd.DataFrame(led.rows)
    # planned (not yet paid) tax rows: they feed the P&L by accrual date and the cash-flow forecast by payment date
    for pay_date, pl, amt, acc in planned_tax_rows:
        journal.loc[len(journal)] = {
            "Дата ДДС": pay_date, "Дата P&L": pl, "Приход": np.nan, "Расход": amt, "Статья операции": "Налог на прибыль",
            "МП": "", "Кабинет": ENTITY_OF[acc], "Артикул поставщика": "", "Р-р": "", "Кол-во": np.nan, "Касса / Счет": acc,
            "Комментарий": "НАЛОГ УСН ПЛАНОВО",
        }
    journal = journal.sort_values(["Дата ДДС", "Дата P&L"], kind="stable").reset_index(drop=True)
    return journal, led.bal


def _treasury(led: Ledger, d: pd.Timestamp) -> None:
    rng = led.rng
    # fund allocation from the main accounts: part of the week's income is reserved
    base = max(led.bal[MAIN] - 1_500_000, 0.0)
    for fund, share in ((F_NDS, 0.10), (F_FOT, 0.14), (F_TAX, 0.07), (F_DEV, 0.05)):
        amount = min(base * share * rng.uniform(0.8, 1.2), 900_000)
        if amount > 20_000:
            led.transfer(d, amount, MAIN, fund, comment="Резервирование средств по фонду")
    led.transfer(d, max(led.bal[IP_S] - 150_000, 0) * 0.95, IP_S, IP_T, comment="Перевод с расчётного счёта ИП")
    led.transfer(d, max(led.bal[IP_T] - 500_000, 0) * 0.9, IP_T, MAIN, comment="Перевод средств ИП на счёт ООО по договору займа")
    # excess cash goes to deposits
    excess = led.bal[MAIN] - 3_000_000
    if excess > 500_000:
        led.transfer(d, excess * 0.7, MAIN, D_DEV, "Перечисление на депозит", "Размещение свободных средств на депозит")


def _interest(led: Ledger, d: pd.Timestamp) -> None:
    for dep in DEPOSITS:
        led.receive(d, d, "Прочие поступл. от фин. операций", led.bal[dep] * 0.11 / 12, dep, "Проценты по депозиту")


def _dividends(led: Ledger, d: pd.Timestamp, rng: np.random.Generator) -> None:
    """Free cash above the target level is distributed to the owners (1-3 payments)."""
    target = 4_500_000 + 700_000 * max((d - START).days / 365.0 - 1, 0)
    free = led.total_cash() - target
    if free < 600_000 or d < pd.Timestamp("2024-09-01"):
        return
    total = min(free * 0.6, 5_000_000)
    parts = rng.dirichlet(np.full(int(rng.integers(1, 4)), 5.0))
    for i, w in enumerate(parts):
        dd = business_day(d + pd.Timedelta(days=2 * i))
        text = f"Выплата дивидендов по итогам периода учредителю по решению от {d:%d.%m.%Y}"
        led.pay(dd, dd, "Выплата дивидендов", round(total * w, -3), MAIN, text)


# --------------------------------------------------------------------------- other sheets
def build_balance_1c(fm: pd.DataFrame, journal: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Items delivered by the accounting system ("1C"): only marketplace-related positions."""
    months = pd.date_range("2023-12-01", "2026-08-01", freq="MS")
    tot = fm.groupby("month").agg(sebes=("sebes_rub", "sum"), spp=("sales_spp_rub", "sum"))
    j = journal[journal["Дата ДДС"] <= AS_OF].copy()
    j["m"] = j["Дата ДДС"].dt.to_period("M").dt.to_timestamp()
    purchases = j[j["Статья операции"] == "Закупка товара"].groupby("m")["Расход"].sum()
    tax_acc = fm.groupby("month")["sales_spp_rub"].sum() * 0.05
    fot_month = 1_780_000.0
    rows = []
    debt = 9_000_000.0
    paid = j[j["Статья операции"] == "Оплаты по кредитам и займам"].groupby("m")["Расход"].sum()
    got = j[j["Статья операции"] == "Получение кредитов и займов"].groupby("m")["Приход"].sum()
    for m in months:
        s = float(tot["sebes"].get(m, tot["sebes"].iloc[0]))
        sales_spp = float(tot["spp"].get(m, tot["spp"].iloc[0]))
        scale = (s / float(tot["sebes"].iloc[-1])) ** 0.6 if m >= pd.Timestamp("2024-01-01") else 0.3
        if m >= pd.Timestamp("2024-01-01"):
            debt += float(got.get(m, 0.0)) - float(paid.get(m, 0.0))
        items = {
            "Дебиторская задолженность маркетплейсов": sales_spp * 0.34 * 0.35 * rng.uniform(0.9, 1.1),
            "Товары в пути и на таможне": float(purchases.get(m, s)) * 0.45 * rng.uniform(0.8, 1.2),
            "Предоплаты поставщикам": float(purchases.get(m, s)) * 0.22 * rng.uniform(0.7, 1.3),
            "Кредиторская задолженность поставщикам": float(purchases.get(m, s)) * 0.20 * rng.uniform(0.8, 1.2),
            "Задолженность по налогам": float(tax_acc.get(m, 0.0)) * 1.4 * rng.uniform(0.9, 1.1),
            "Задолженность по оплате труда и взносам": fot_month * 1.136 * scale * 0.45,
            "Кредиты и займы": max(debt, 0.0),
            "Уставный капитал": 100_000.0,
        }
        for name, val in items.items():
            rows.append({"month": m, "item": name, "amount": round(val)})
    return pd.DataFrame(rows)


def build_payment_plan(rng: np.random.Generator) -> pd.DataFrame:
    """Known payments after the as-of date: supplier invoices, logistics, services (manual plan)."""
    base = AS_OF
    rows = [
        (-2, "Закупка товара", "Поставщик тканей №1", ENTITY_OOO, 2_450_000, "Критичный", "Просрочен", "Счёт на оплату ткани, срок истёк"),
        (3, "Закупка товара", "Поставщик тканей №2", ENTITY_OOO, 3_100_000, "Критичный", "Согласован", "Предоплата 50% по счёту"),
        (9, "Закупка товара", "Швейная фабрика", ENTITY_OOO, 1_850_000, "Критичный", "Запланирован", "Оплата пошива партии"),
        (16, "Закупка товара", "Поставщик фурнитуры", ENTITY_OOO, 640_000, "Обязательный", "Запланирован", "Фурнитура и этикетки"),
        (6, "Логистика из Китая", "Карго-компания", ENTITY_OOO, 780_000, "Обязательный", "Согласован", "Доставка партии из Китая"),
        (24, "Таможенные платежи", "Таможенный брокер", ENTITY_OOO, 1_150_000, "Критичный", "Запланирован", "Пошлины по декларации"),
        (11, "Услуги контент-креаторов", "Агентство инфлюенсеров", ENTITY_OOO, 420_000, "Можно отложить", "Запланирован", "Пакет интеграций на октябрь"),
        (13, "Размещение у блогеров", "Рекламная сеть блогеров", ENTITY_OOO, 560_000, "Можно отложить", "Запланирован", "Рекламное размещение"),
        (19, "IT разработка", "IT-подрядчик", ENTITY_OOO, 300_000, "Можно отложить", "Запланирован", "Разработка аналитической системы"),
        (28, "Проведение фотосъемок", "Фотостудия", ENTITY_OOO, 240_000, "Можно отложить", "Запланирован", "Съёмка новой коллекции"),
        (35, "Закупка товара", "Поставщик тканей №1", ENTITY_OOO, 2_800_000, "Критичный", "Запланирован", "Оплата второй партии"),
        (42, "Фулфилмент", "Фулфилмент-оператор", ENTITY_OOO, 180_000, "Обязательный", "Запланирован", "Услуги фулфилмента"),
        (49, "Юридические услуги", "Юридическая фирма", ENTITY_OOO, 150_000, "Можно отложить", "Запланирован", "Юридическое сопровождение"),
    ]
    out = pd.DataFrame(rows, columns=["offset", "Статья операции", "Контрагент", "Юрлицо", "Сумма", "Приоритет", "Статус", "Комментарий"])
    out.insert(0, "Дата", [business_day(base + pd.Timedelta(days=int(o))) if o >= 0 else base + pd.Timedelta(days=int(o)) for o in out["offset"]])
    out.insert(0, "ID", [f"PP-{i + 1:03d}" for i in range(len(out))])
    return out.drop(columns="offset")


def main() -> None:
    rng = np.random.default_rng(SEED)
    creators = pd.read_excel(DB_PATH, sheet_name="creators_weekly")
    creators["week_start"] = pd.to_datetime(creators["week_start"])
    fm = build_fact_monthly(rng)
    journal, balances = build_journal(fm, creators, rng)
    articles = pd.DataFrame(ARTICLES, columns=["Статья операции", "Для P&L", "Категория P&L", "Для ДДС"])
    accounts = pd.DataFrame(ACCOUNTS, columns=["Касса / Счет", "Группа счетов", "Юрлицо", "Входящий остаток"])
    unknown = set(journal["Статья операции"]) - set(articles["Статья операции"])
    if unknown:
        raise ValueError(f"Journal uses articles missing in the reference: {sorted(unknown)}")
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as xw:
        journal.to_excel(xw, sheet_name="journal", index=False)
        articles.to_excel(xw, sheet_name="articles", index=False)
        accounts.to_excel(xw, sheet_name="accounts", index=False)
        fm.assign(month=fm["month"]).round({c: 2 for c in fm.select_dtypes("number").columns}).to_excel(xw, sheet_name="fact_monthly", index=False)
        build_balance_1c(fm, journal, rng).to_excel(xw, sheet_name="balance_1c", index=False)
        build_payment_plan(rng).to_excel(xw, sheet_name="payment_plan", index=False)
        pd.DataFrame(columns=["Маркетплейс", "Рост YoY, %", "Выкуп, %", "Расходы МП, %"]).to_excel(xw, sheet_name="plan_mp", index=False)
        pd.DataFrame(columns=["Параметр", "Значение"]).to_excel(xw, sheet_name="plan_params", index=False)
        pd.DataFrame(columns=["Название", "Маркетплейс", "С месяца", "По месяц", "GMV в месяц, ₽", "Комментарий"]).to_excel(
            xw, sheet_name="plan_extra", index=False)
        pd.DataFrame(columns=["Статья", "С месяца", "По месяц", "Добавить, ₽/мес", "Комментарий"]).to_excel(xw, sheet_name="plan_adj", index=False)
        pd.DataFrame({"key": ["as_of"], "value": [AS_OF.strftime("%Y-%m-%d")]}).to_excel(xw, sheet_name="meta", index=False)
    neg = {a: round(b) for a, b in balances.items() if b < 0}
    print(f"{OUT_PATH.name}: journal={len(journal)} fact_monthly={len(fm)}; cash={sum(balances.values()):,.0f}; negative={neg}")


if __name__ == "__main__":
    main()
