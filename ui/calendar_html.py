"""Heat-map calendar: weeks x weekdays with the day's net flow, closing balance and the payments of the day."""
from __future__ import annotations

from html import escape

import pandas as pd

WD = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def _color(close: float, buffer: float) -> str:
    if close < 0:
        return "#F9C9CC"
    if close < buffer:
        return "#FFE6B3"
    return "#DDF3E6"


def render_calendar(daily: pd.DataFrame, schedule_win: pd.DataFrame, buffer: float, today: pd.Timestamp) -> str:
    by_day = {d: g for d, g in schedule_win.groupby("date")}
    weeks = []
    for w, wk in daily.groupby("week"):
        cells = []
        for r in wk.itertuples():
            g = by_day.get(r.date)
            tip = ""
            if g is not None:
                items = g.sort_values("amount", ascending=False).head(8)
                tip = "&#10;".join(escape(f"{'+' if x.direction == 'in' else '−'}{x.amount / 1000:,.0f} тыс. — {x.article}", quote=True)
                                   for x in items.itertuples())
            wk_end = " we" if r.date.weekday() >= 5 else ""
            net = f"<div class='n {'p' if r.net >= 0 else 'm'}'>{'+' if r.net >= 0 else '−'}{abs(r.net) / 1000:,.0f}</div>" if abs(r.net) > 500 else "<div class='n'>&nbsp;</div>"
            cells.append(f"<td class='c{wk_end}' style='background:{_color(r.close, buffer)}' title='{tip}'>"
                         f"<div class='d'>{r.date:%d.%m}</div>{net}<div class='b'>{r.close / 1e6:.1f}</div></td>")
        label = f"Нед. {int(w) + 1}<br><small>{wk['date'].min():%d.%m}</small>"
        weeks.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th>{d}</th>" for d in WD)
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>"
            f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(weeks)}</tbody></table>"
            "<div class='lg'><i style='background:#DDF3E6'></i>выше минимального остатка <i style='background:#FFE6B3'></i>ниже минимального "
            "<i style='background:#F9C9CC'></i>кассовый разрыв &nbsp;·&nbsp; в ячейке: дата, ЧДП дня (тыс. ₽), остаток на конец дня (млн ₽)</div></body></html>")


_CSS = """
body{margin:0;font:12px -apple-system,Segoe UI,Inter,sans-serif;color:#1B2437}
table{border-collapse:separate;border-spacing:4px;width:100%}th{font-weight:650;color:#6B7690;text-align:center;font-size:11px}
td.c{border-radius:8px;padding:5px 6px;text-align:center;height:54px;min-width:92px;cursor:help;vertical-align:top}
td.we{opacity:.72}.d{font-size:10px;color:#4A556F}.n{font-size:12px;font-weight:700;line-height:1.3}.n.p{color:#14935F}.n.m{color:#D6404A}
.b{font-size:11px;color:#1B2437;font-weight:600}.lg{margin-top:6px;color:#6B7690;font-size:11px}.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin:0 4px 0 8px;vertical-align:-1px}
"""
