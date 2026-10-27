"""Interactive financial statement table (P&L, cash flow, balance, payment calendar) rendered as HTML + JS.

Rows form a hierarchy: clicking a row folds its children. Columns can be facts, plan, the current month, totals or
deltas; negative figures are red, changes are coloured by whether the direction is favourable for the line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Mapping, Sequence

from core.formatting import format_value

KIND_FMT = {"amount": "k0", "qty": "int", "pct": "pct", "unit": "rub0", "days": "days", "memo": "k0"}
STATE_CLASS = {"Факт": "fact", "Факт+План": "curr", "План": "plan", "Итого": "tot", "Δ": "delta", "": "fact"}


@dataclass
class Col:
    key: str
    label: str
    sub: str = ""  # second header line, e.g. "Факт" / "План"
    state: str = ""
    fmt: str | None = None  # overrides the row format (e.g. 'pct' for a relative delta column)
    delta: bool = False  # colour by favourable direction


@dataclass
class TRow:
    id: str
    label: str
    values: Mapping
    level: int = 0
    kind: str = "item"  # title | group | item | total | balance | memo | check
    fmt: str = "k0"
    parent: str | None = None
    good: str | None = None  # "up"/"down" favourable direction (for delta colouring)
    alert_below: float | None = None  # cells below this value are highlighted
    hint: str = ""


def _fmt(v, fmt: str) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if fmt in ("k0", "int") and abs(v) < (500 if fmt == "k0" else 0.5):
        return "–"
    return format_value(v, fmt)


def render_statement(cols: Sequence[Col], rows: Sequence[TRow], *, height: int = 640, folded: Sequence[str] = (),
                     first_col: str = "Показатель") -> str:
    parents = {r.id: r.parent for r in rows}

    def ancestors(rid: str) -> list[str]:
        out, p = [], parents.get(rid)
        while p:
            out.append(p)
            p = parents.get(p)
        return out

    has_children = {r.parent for r in rows if r.parent}
    head = [f"<th class='lab'>{escape(first_col)}</th>"]
    for c in cols:
        cls = STATE_CLASS.get(c.state, "fact")
        sub = f"<small>{escape(c.sub)}</small>" if c.sub else ""
        head.append(f"<th class='{cls}'>{escape(c.label)}<br>{sub or '&nbsp;'}</th>")
    body = []
    for r in rows:
        anc = " ".join(escape(a, quote=True) for a in ancestors(r.id))
        hidden = " folded" if any(a in folded for a in ancestors(r.id)) else ""
        toggle = " has" if r.id in has_children else ""
        caret = "<span class='caret'>▾</span> " if r.id in has_children else ""
        tds = []
        for c in cols:
            v = r.values.get(c.key)
            fmt = c.fmt or r.fmt
            if c.fmt == "pct" and r.fmt == "pct":
                text = ""
            else:
                text = _fmt(v, fmt)
            cls = [STATE_CLASS.get(c.state, "fact")]
            if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v):
                if c.delta and r.good and abs(v) > 1e-9:
                    cls.append("up" if (v > 0) == (r.good == "up") else "dn")
                elif v < -0.5 and fmt in ("k0", "int", "rub0") and not c.delta:
                    cls.append("neg")
                if r.alert_below is not None and v < r.alert_below and not c.delta and c.state != "Итого":
                    cls.append("alert")
            tds.append(f"<td class='{' '.join(cls)}'>{escape(text)}</td>")
        hint = f" title='{escape(r.hint, quote=True)}'" if r.hint else ""
        body.append(f"<tr class='r-{r.kind} l{r.level}{toggle}{hidden}' data-id='{escape(r.id, quote=True)}' data-fold='{1 if r.id in folded else 0}' "
                    f"data-anc='{anc}' data-label='{escape(r.label.lower(), quote=True)}'>"
                    f"<td class='lab'{hint}>{caret}{escape(r.label)}</td>{''.join(tds)}</tr>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>"
            "<div class='bar'><button id='ex'>Развернуть всё</button><button id='co'>Свернуть до итогов</button>"
            "<input id='q' placeholder='Поиск статьи…' autocomplete='off'></div>"
            f"<div class='wrap' style='height:{height - 50}px'><table id='t'><thead><tr>{''.join(head)}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div><script>{_JS}</script></body></html>")


_CSS = """
*{box-sizing:border-box}body{margin:0;font:12px/1.3 -apple-system,Segoe UI,Inter,sans-serif;color:#1B2437;background:#fff}
.bar{display:flex;gap:8px;padding:6px 2px 8px}.bar button{border:1px solid #D5DBE8;background:#fff;border-radius:8px;padding:5px 11px;font-size:12px;cursor:pointer;color:#2F3B57}
.bar button:hover{background:#F0F4FF;border-color:#2F5BEA}.bar input{border:1px solid #D5DBE8;border-radius:8px;padding:5px 10px;font-size:12px;width:190px}
.wrap{overflow:auto;border:1px solid #E4E8F0;border-radius:12px}
table{border-collapse:separate;border-spacing:0;min-width:100%;width:max-content}
th,td{padding:5px 10px;border-bottom:1px solid #EEF1F6;border-right:1px solid #F3F5F9;text-align:right;white-space:nowrap}
thead th{position:sticky;top:0;z-index:5;background:#2F5BEA;color:#fff;font-weight:650;height:42px;line-height:1.15;text-align:center}
thead th small{font-weight:500;opacity:.85}thead th.plan{background:#5B7FD6}thead th.curr{background:#C27A0E}thead th.tot{background:#3c4a63}thead th.delta{background:#6B7690}
.lab{position:sticky;left:0;text-align:left;min-width:300px;max-width:300px;overflow:hidden;text-overflow:ellipsis;background:#fff;z-index:3;border-right:1px solid #E4E8F0}
thead th.lab{z-index:7;background:#2F5BEA;text-align:left}
td.plan{background:#F4F7FF}td.curr{background:#FFF8EC}td.tot{background:#F1F3F8;font-weight:650}td.delta{background:#FAFAFC}
td.neg{color:#D6404A}td.up{color:#14935F;font-weight:650}td.dn{color:#D6404A;font-weight:650}td.alert{background:#FDE8EA!important;color:#B3261E;font-weight:700}
tr.l1 td.lab{padding-left:22px}tr.l2 td.lab{padding-left:40px;color:#46506A}tr.l3 td.lab{padding-left:58px}
tr.r-title td{background:#EAF0FF;color:#233A8C;font-weight:750;letter-spacing:.02em}tr.r-title td.lab{background:#EAF0FF}
tr.r-total td{font-weight:700;border-top:1px solid #C9D1E3;background:#FBFCFE}tr.r-total td.lab{background:#FBFCFE}
tr.r-balance td{font-weight:750;background:#EEF6F0;border-top:1px solid #BFD8C6;border-bottom:1px solid #BFD8C6}tr.r-balance td.lab{background:#EEF6F0}
tr.r-group td{font-weight:650;background:#FAFBFD}tr.r-group td.lab{background:#FAFBFD}
tr.r-memo td{color:#7A859C;font-style:italic;font-size:11px}tr.r-check td{color:#9AA5BD;font-size:10px}
tr.has{cursor:pointer}.caret{display:inline-block;transition:transform .15s;font-size:10px;color:#6B7690}tr.has.collapsed .caret{transform:rotate(-90deg)}
tr.folded,tr.q-hide{display:none}tr:hover td.lab{background:#F1F5FF}
"""

_JS = """
const rows=[...document.querySelectorAll('#t tbody tr')];
const folded=new Set(rows.filter(r=>r.dataset.fold==='1').map(r=>r.dataset.id));
function refresh(){rows.forEach(r=>{const anc=r.dataset.anc?r.dataset.anc.split(' '):[];r.classList.toggle('folded',anc.some(a=>folded.has(a)));
  if(r.classList.contains('has')) r.classList.toggle('collapsed',folded.has(r.dataset.id));});}
rows.forEach(r=>{if(r.classList.contains('has')) r.addEventListener('click',()=>{const id=r.dataset.id; folded.has(id)?folded.delete(id):folded.add(id); refresh();});});
document.getElementById('ex').onclick=()=>{folded.clear();refresh();};
document.getElementById('co').onclick=()=>{rows.filter(r=>r.classList.contains('has')&&r.classList.contains('l1')||r.classList.contains('has')&&r.classList.contains('l0')&&!r.classList.contains('r-title')).forEach(r=>folded.add(r.dataset.id));refresh();};
document.getElementById('q').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();
  rows.forEach(r=>r.classList.toggle('q-hide',q&&!r.dataset.label.includes(q)&&!r.classList.contains('r-title')));if(q){folded.clear();refresh();}});
refresh();
"""
