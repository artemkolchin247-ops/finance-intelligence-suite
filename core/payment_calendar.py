"""Payment calendar: 12 weeks of the cash-flow schedule by day and week, cash-gap detection and deferral advice."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.cashflow import business_day

CAL_WEEKS = 12
CAL_GROUPS = ("Налоги и взносы", "ФОТ", "Поставщики и логистика", "Маркетинг", "Аренда и сервисы", "Кредиты и дивиденды")
_GROUP_MAP = {
    "Налоги": "Налоги и взносы", "ФОТ": "ФОТ", "Запасы / Складские операции": "Поставщики и логистика",
    "Логистика и ФФ": "Поставщики и логистика", "Маркетинг": "Маркетинг", "Кредиты": "Кредиты и дивиденды",
    "% по кредитам": "Кредиты и дивиденды", "Вывод дивидендов": "Кредиты и дивиденды",
}
MIN_MOVE = 50_000.0  # smallest payment worth proposing for deferral, rub
PRIORITY_ORDER = ("Критичный", "Обязательный", "Можно отложить")


def calendar_group(group: str) -> str:
    return _GROUP_MAP.get(group, "Аренда и сервисы")


def calendar_start(as_of: pd.Timestamp) -> pd.Timestamp:
    """First Monday after the as-of date (the as-of date itself if the day after is a Monday)."""
    d = as_of + pd.Timedelta(days=1)
    return d + pd.Timedelta(days=(7 - d.weekday()) % 7)


def window(schedule: pd.DataFrame, start: pd.Timestamp, weeks: int = CAL_WEEKS) -> pd.DataFrame:
    """Schedule rows of the calendar window with their signed amount; earlier items fall on the first day."""
    end = start + pd.Timedelta(days=7 * weeks - 1)
    s = schedule[schedule["date"] <= end].copy()
    s["date"] = s["date"].where(s["date"] >= start, start)
    s["signed"] = np.where(s["direction"] == "in", s["amount"], -s["amount"])
    s["cal_group"] = s["group"].map(calendar_group)
    return s


def daily_table(schedule: pd.DataFrame, opening: float, start: pd.Timestamp, weeks: int = CAL_WEEKS) -> pd.DataFrame:
    """One row per calendar day: inflow, outflow, net and the closing balance."""
    s = window(schedule, start, weeks)
    days = pd.date_range(start, periods=7 * weeks)
    inflow = s[s["direction"] == "in"].groupby("date")["amount"].sum().reindex(days, fill_value=0.0)
    outflow = s[s["direction"] == "out"].groupby("date")["amount"].sum().reindex(days, fill_value=0.0)
    out = pd.DataFrame({"date": days, "inflow": inflow.values, "outflow": outflow.values})
    out["net"] = out["inflow"] - out["outflow"]
    out["close"] = opening + out["net"].cumsum()
    out["open"] = out["close"] - out["net"]
    out["week"] = (out["date"] - start).dt.days // 7
    return out


@dataclass
class WeekRow:
    week: int
    start: pd.Timestamp
    end: pd.Timestamp
    opening: float
    inflow: dict  # by article
    outflow: dict  # by calendar group
    by_priority: dict
    net: float
    closing: float
    min_balance: float
    status: str  # "ok" | "below" | "gap"


def weekly_summary(schedule: pd.DataFrame, opening: float, start: pd.Timestamp, buffer: float, weeks: int = CAL_WEEKS) -> list[WeekRow]:
    s = window(schedule, start, weeks)
    daily = daily_table(schedule, opening, start, weeks)
    out = []
    for w in range(weeks):
        d = daily[daily["week"] == w]
        sw = s[(s["date"] >= d["date"].min()) & (s["date"] <= d["date"].max())]
        lo = float(d["close"].min())
        status = "gap" if lo < 0 else ("below" if lo < buffer else "ok")
        out.append(WeekRow(
            w + 1, d["date"].min(), d["date"].max(), float(d["open"].iloc[0]),
            sw[sw["direction"] == "in"].groupby("article")["amount"].sum().to_dict(),
            sw[sw["direction"] == "out"].groupby("cal_group")["amount"].sum().to_dict(),
            sw[sw["direction"] == "out"].groupby("priority")["amount"].sum().to_dict(),
            float(d["net"].sum()), float(d["close"].iloc[-1]), lo, status))
    return out


def apply_moves(schedule: pd.DataFrame, moves: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Shift scheduled payments to new dates (`moves`: row id -> date); the status becomes 'Отложен'."""
    if not moves or schedule.empty:
        return schedule
    s = schedule.copy()
    for row_id, new_date in moves.items():
        m = s["id"] == row_id
        if m.any():
            s.loc[m, "date"] = business_day(pd.Timestamp(new_date))
            s.loc[m, "status"] = "Отложен"
    s["month"] = s["date"].dt.to_period("M").dt.to_timestamp()
    return s


def suggest_deferrals(schedule: pd.DataFrame, opening: float, start: pd.Timestamp, buffer: float,
                      weeks: int = CAL_WEEKS, max_moves: int = 25, exclude: frozenset | set = frozenset()) -> list[dict]:
    """Advice for closing periods when the balance is below the buffer: defer 'Можно отложить' payments.

    Each payment is proposed at most once and moved to just after the shortfall window (at least a week). For the
    first day below the buffer the smallest single payment covering the shortfall is taken, otherwise the largest;
    the balance is then recomputed and the next shortfall is handled. Payments in `exclude` (rejected by the user)
    are never proposed. Payments below max(50 тыс. ₽, 10% of the shortfall) are skipped, so the advice may not
    close the whole shortfall — callers should check the balance after applying it. Returns [{'id','article','amount','from','to','week','shortfall'}] in the order of use.
    """
    advice: list[dict] = []
    used = set(exclude)
    skip_until = start - pd.Timedelta(days=1)  # shortfalls up to this date cannot be helped and are left alone
    current = schedule
    horizon_end = start + pd.Timedelta(days=7 * weeks - 1)
    for _ in range(max_moves):
        daily = daily_table(current, opening, start, weeks)
        below = daily[(daily["close"] < buffer) & (daily["date"] > skip_until)]
        if below.empty:
            break
        d1 = below["date"].iloc[0]
        episode = daily[(daily["date"] >= d1) & (daily["date"] <= d1 + pd.Timedelta(days=6))]
        shortfall = float(buffer - episode["close"].min())
        cand = current[(current["direction"] == "out") & (current["priority"] == "Можно отложить") & (current["date"] >= start)
                       & (current["date"] <= d1) & ~current["id"].isin(used)].copy()
        cand = cand[cand["amount"] >= max(MIN_MOVE, 0.1 * shortfall)]  # micro-payments are not worth advising
        if len(cand):
            cand["target"] = [business_day(max(d + pd.Timedelta(days=7), d1 + pd.Timedelta(days=7))) for d in cand["date"]]
            cand = cand[cand["target"] <= horizon_end]
        if cand.empty:
            skip_until = d1 + pd.Timedelta(days=6)
            continue
        covering = cand[cand["amount"] >= shortfall]
        pick = (covering.sort_values("amount").iloc[0] if len(covering) else cand.sort_values("amount", ascending=False).iloc[0])
        advice.append({"id": pick["id"], "article": pick["article"], "amount": float(pick["amount"]), "from": pick["date"],
                       "to": pick["target"], "week": int((d1 - start).days // 7) + 1, "shortfall": shortfall})
        used.add(pick["id"])
        current = apply_moves(current, {pick["id"]: pick["target"]})
    return advice
