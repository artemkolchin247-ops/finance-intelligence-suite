"""Interactive РНП table: self-contained HTML + a little JS, rendered in an iframe.

Interactions: click a block header to fold it, click "Итого за неделю" to fold that week's days,
quick presets (key rows only, expand/collapse all), row search, formula hints, hypothesis popups.
Colouring follows the reporting rules: row-wise red-white-green scale on day columns (direction depends
on the metric), red fill for negative profit, green/red arrows for the change vs the previous column.
"""
from __future__ import annotations

import math
from html import escape
from typing import Mapping, Sequence

import numpy as np

from core import rnp_calc as R
from core.formatting import format_value

RED, WHITE, GREEN = (230, 124, 115), (255, 255, 255), (87, 187, 138)
SECTION_COLORS = {
    "Заказы": "#2F5BEA", "Продажи": "#14A38B", "Воронка": "#8B5CF6", "Реклама итого": "#E5484D",
    "Внутренняя реклама": "#F2A33A", "Внешняя реклама": "#0EA5E9", "Блогеры": "#EC4899", "Креаторы": "#A855F7",
    "ВК (подрядчик)": "#3B82F6", "Посевы ВК": "#6366F1", "Посевы (подрядчик)": "#14B8A6",
    "Яндекс (подрядчик)": "#F59E0B", "Маржа": "#16A34A", "Склад и оборачиваемость": "#64748B", "Прогноз": "#7C3AED",
}
WEEKDAY_RU = {"пн": "", "вт": "", "ср": "", "чт": "", "пт": "", "сб": "wk", "вс": "wk"}


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else float("nan")


def _lerp(c0, c1, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))


def _scale_colors(values: Sequence[float], direction: str) -> list[str | None]:
    """Red-white-green gradient by value rank within the row (white at the median)."""
    vals = [v for v in values if not math.isnan(v)]
    if len(vals) < 3:
        return [None] * len(values)
    lo, mid, hi = min(vals), float(np.median(vals)), max(vals)
    low_c, high_c = (RED, GREEN) if direction == "up" else (GREEN, RED)
    out: list[str | None] = []
    for v in values:
        if math.isnan(v):
            out.append(None)
        elif v <= mid:
            out.append(_lerp(low_c, WHITE, (v - lo) / (mid - lo)) if mid > lo else None)
        else:
            out.append(_lerp(WHITE, high_c, (v - mid) / (hi - mid)) if hi > mid else None)
    return out


def _delta(cur: float, prev: float, is_pp: bool, good: str | None) -> tuple[str, str] | None:
    if math.isnan(cur) or math.isnan(prev):
        return None
    if is_pp:
        diff, text = cur - prev, None
        if abs(diff) < 0.0005:
            return None
        text = f"{abs(diff) * 100:.1f} п.п.".replace(".", ",")
    else:
        if prev == 0:
            return None
        diff = (cur - prev) / abs(prev)
        if abs(diff) < 0.005:
            return None
        text = f"{abs(diff) * 100:.0f}%" if abs(diff) >= 0.1 else f"{abs(diff) * 100:.1f}%".replace(".", ",")
    cls = "nt" if good is None else ("up" if (diff > 0) == (good == "up") else "dn")
    return f"{'▲' if diff > 0 else '▼'} {text}", cls


def render_table(report: R.Report, hyps: Mapping, *, delta_mode: str = "key", sections: Sequence[str] | None = None,
                 folded: Sequence[str] = (), height: int = 720) -> str:
    """Return the full HTML document. delta_mode: 'off' | 'key' | 'all'."""
    cols = report.columns
    sections = list(sections) if sections is not None else R.SECTIONS
    day_cols = [c for c in cols if c.kind == "day"]

    # week id for every column (days inherit the id of the preceding week column)
    week_of: dict[int, str] = {}
    wid = -1
    for i, c in enumerate(cols):
        if c.kind == "week":
            wid += 1
        if c.kind in ("week", "day"):
            week_of[i] = f"w{wid}"

    # previous column for change arrows: previous day for days, previous week for weeks
    prev_idx: dict[int, int | None] = {}
    last_day = last_week = None
    for i, c in enumerate(cols):
        if c.kind == "day":
            prev_idx[i], last_day = last_day, i
        elif c.kind == "week":
            prev_idx[i], last_week = last_week, i

    head1 = ['<th class="lab">Показатель</th>']
    head2 = ['<th class="lab hyp-lab">💡 Гипотезы и изменения</th>']
    for i, c in enumerate(cols):
        cls = {"total": "tot", "week": "wk", "day": "day"}[c.kind]
        if c.kind == "day" and WEEKDAY_RU.get(c.name1):
            cls += " weekend"
        attrs = f' data-w="{week_of[i]}"' if c.kind == "day" else ""
        if c.kind == "week":
            head1.append(f'<th class="{cls} fold" data-toggle-week="{week_of[i]}" title="Свернуть/развернуть дни недели">'
                         f'<span class="caret">▾</span> {escape(c.name1)}<br><small>{escape(c.name2)}</small></th>')
        else:
            head1.append(f'<th class="{cls}"{attrs}>{escape(c.name1)}<br><small>{escape(c.name2)}</small></th>')
        items = hyps.get(c.start) if c.kind == "day" else None
        if items:
            payload = "&#10;".join(escape(f"{h['model']}: {h['text']} [{h['status']}]", quote=True) for h in items)
            head2.append(f'<th class="{cls} hyp has" data-w="{week_of[i]}" data-pop="{payload}">💡 {len(items)}</th>')
        else:
            head2.append(f'<th class="{cls} hyp"{attrs}></th>')

    body: list[str] = []
    for sec in sections:
        metrics = [m for m in R.METRICS if m.section == sec]
        if not metrics:
            continue
        color = SECTION_COLORS.get(sec, "#2F5BEA")
        hidden_cls = " folded" if sec in folded else ""
        body.append(f'<tr class="sec{hidden_cls}" data-sec="{escape(sec)}" style="--sc:{color}">'
                    f'<td colspan="{len(cols) + 1}"><span class="caret">▾</span> {escape(sec)}'
                    f'<span class="cnt">{len(metrics)}</span></td></tr>')
        for m in metrics:
            nums = [_num(c.values.get(m.key)) for c in cols]
            day_nums = [nums[i] if cols[i].kind == "day" else float("nan") for i in range(len(cols))]
            colors = _scale_colors(day_nums, m.grad) if m.grad else [None] * len(cols)
            is_pp = m.kind == "pct"
            tds = []
            for i, c in enumerate(cols):
                v = c.values.get(m.key)
                attrs = f' data-w="{week_of[i]}"' if c.kind == "day" else ""
                base = {"total": "tot", "week": "wk", "day": "day"}[c.kind]
                if m.weekly_only and c.kind == "day" or (m.key not in c.values and m.weekly_only):
                    tds.append(f'<td class="{base} blank"{attrs}></td>')
                    continue
                text = format_value(v, m.kind) if not isinstance(v, str) else v
                num = nums[i]
                cls, style = [base], ""
                if text in ("—", R.NO_DATA):
                    cls.append("na")
                if m.neg_red and not math.isnan(num) and num < 0:
                    cls.append("negfill")
                elif colors[i]:
                    style = f' style="background:{colors[i]}"'
                if c.kind == "day" and WEEKDAY_RU.get(c.name1):
                    cls.append("weekend")
                dl = ""
                show = delta_mode == "all" or (delta_mode == "key" and m.key_row)
                if show and not m.weekly_only and c.kind != "total" and prev_idx.get(i) is not None:
                    d = _delta(num, nums[prev_idx[i]], is_pp, m.good)
                    if d:
                        dl = f'<span class="dl {d[1]}">{d[0]}</span>'
                tds.append(f'<td class="{" ".join(cls)}"{attrs}{style}><span class="v">{escape(text)}</span>{dl}</td>')
            hint = f' title="{escape(m.help, quote=True)}"' if m.help else ""
            info = ' <span class="info">ⓘ</span>' if m.help else ""
            row_cls = "m key" if m.key_row else "m"
            body.append(f'<tr class="{row_cls}{hidden_cls}" data-sec="{escape(sec)}" data-k="{1 if m.key_row else 0}" '
                        f'data-label="{escape(m.display_label.lower(), quote=True)}" style="--sc:{color}">'
                        f'<td class="lab"{hint}>{escape(m.display_label)}{info}</td>{"".join(tds)}</tr>')

    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{_CSS}</style></head><body>
<div class="bar">
  <button data-act="expand">Развернуть всё</button><button data-act="collapse">Свернуть блоки</button>
  <button data-act="key" id="keyBtn">Только ключевые строки</button>
  <button data-act="days" id="daysBtn">Скрыть дни</button>
  <input id="q" placeholder="Поиск показателя…" autocomplete="off">
  <span class="legend"><i class="g"></i>лучше <i class="r"></i>хуже &nbsp;·&nbsp; ▲▼ к предыдущему дню/неделе</span>
</div>
<div class="wrap" style="height:{height - 52}px"><table id="t"><thead><tr>{"".join(head1)}</tr><tr class="hrow">{"".join(head2)}</tr></thead>
<tbody>{"".join(body)}</tbody></table></div>
<div id="pop"></div>
<script>{_JS}</script></body></html>"""


_CSS = """
*{box-sizing:border-box} body{margin:0;font:12px/1.35 -apple-system,Segoe UI,Inter,sans-serif;color:#1B2437;background:#fff}
.bar{display:flex;gap:8px;align-items:center;padding:8px 2px 10px;flex-wrap:wrap}
.bar button{border:1px solid #D5DBE8;background:#fff;border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer;color:#2F3B57}
.bar button:hover{background:#F0F4FF;border-color:#2F5BEA} .bar button.on{background:#2F5BEA;color:#fff;border-color:#2F5BEA}
.bar input{border:1px solid #D5DBE8;border-radius:8px;padding:6px 10px;font-size:12px;width:200px}
.legend{margin-left:auto;color:#6B7690;font-size:11px} .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 6px;vertical-align:-1px}
.legend .g{background:#57bb8a}.legend .r{background:#e67c73}
.wrap{overflow:auto;border:1px solid #E4E8F0;border-radius:12px}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}
th,td{padding:4px 9px;border-bottom:1px solid #EEF1F6;border-right:1px solid #F3F5F9;text-align:center;white-space:nowrap}
thead th{position:sticky;top:0;z-index:5;background:#1a73e8;color:#fff;font-weight:650;height:40px;line-height:1.15}
thead th small{font-weight:500;opacity:.85}
thead th.tot{background:#3c4a63} thead th.wk{background:#2f62c4} thead th.weekend{background:#3f86ee}
thead th.fold{cursor:pointer} .caret{display:inline-block;transition:transform .15s;font-size:10px}
tr.hrow th{top:40px;height:26px;background:#fff8e1;color:#7a5b00;font-weight:600;z-index:4;border-bottom:1px solid #F0E3B5}
tr.hrow th.has{cursor:pointer;background:#ffecb3} tr.hrow th.has:hover{background:#ffe082}
.lab{position:sticky;left:0;text-align:left;min-width:285px;max-width:285px;background:#fff;z-index:3;border-right:1px solid #E4E8F0}
thead th.lab{z-index:7;background:#1a73e8;text-align:left} tr.hrow th.lab{background:#fff8e1;color:#7a5b00;z-index:6}
td.tot,td.wk{font-weight:650} td.tot{background:#e8eaed;position:sticky;left:285px;z-index:2;border-right:1px solid #CBD2E1}
thead th.tot{position:sticky;left:285px;z-index:6} tr.hrow th.tot{z-index:5;left:285px;background:#fff8e1}
td.wk{background:#e8f0fe} td.day{background:#fff} td.weekend{box-shadow:inset 0 0 0 99px rgba(30,60,140,.035)}
td.blank{background:#FAFBFD} td.na .v{color:#A3ACBF} td.negfill{background:#f4c7c3!important;font-weight:700;color:#8a1c14}
tr.m.key td.lab{font-weight:700;border-left:3px solid var(--sc);background:#F6F8FF} tr.m.key td.day,tr.m.key td.wk,tr.m.key td.tot{font-weight:650}
tr.m td.lab{border-left:3px solid transparent} tr.m:hover td.lab{background:#F1F5FF}
tr.m td .v{display:block} .dl{display:block;font-size:10px;font-weight:650;line-height:1.1}
.dl.up{color:#14935F}.dl.dn{color:#D6404A}.dl.nt{color:#8791A8}
tr.sec td{position:sticky;left:0;background:color-mix(in srgb,var(--sc) 11%,#fff);color:var(--sc);font-weight:750;text-align:left;
  cursor:pointer;border-left:4px solid var(--sc);letter-spacing:.02em;padding:6px 10px}
tr.sec .cnt{margin-left:8px;font-weight:600;font-size:10px;opacity:.7;background:#fff;border-radius:9px;padding:0 6px}
tr.sec.folded .caret,th.folded .caret{transform:rotate(-90deg)} tr.m.folded{display:none} .hid{display:none!important} tr.q-hide{display:none}
.info{color:#9AA5BD;font-size:10px;cursor:help}
#pop{position:fixed;z-index:50;display:none;max-width:420px;background:#fff;border:1px solid #F0C040;border-left:4px solid #F2A33A;border-radius:8px;
  box-shadow:0 8px 24px rgba(20,30,60,.18);padding:10px 12px;font-size:12px;white-space:pre-wrap;line-height:1.45}
"""

_JS = """
const t=document.getElementById('t'), pop=document.getElementById('pop');
const secs=[...t.querySelectorAll('tr.sec')];
const rows=[...t.querySelectorAll('tr.m')];
function setSec(name, fold){
  const hdr=secs.find(s=>s.dataset.sec===name); if(!hdr) return;
  hdr.classList.toggle('folded', fold);
  rows.filter(r=>r.dataset.sec===name).forEach(r=>r.classList.toggle('folded', fold));
}
secs.forEach(s=>s.addEventListener('click',()=>setSec(s.dataset.sec, !s.classList.contains('folded'))));
let weeksHidden={};
function setWeek(w, hide){
  weeksHidden[w]=hide;
  t.querySelectorAll('[data-w="'+w+'"]').forEach(e=>e.classList.toggle('hid', hide));
  const th=t.querySelector('[data-toggle-week="'+w+'"]'); if(th) th.classList.toggle('folded', hide);
}
t.querySelectorAll('[data-toggle-week]').forEach(h=>h.addEventListener('click',()=>{const w=h.dataset.toggleWeek; setWeek(w,!weeksHidden[w]);}));
const weekIds=[...t.querySelectorAll('[data-toggle-week]')].map(h=>h.dataset.toggleWeek);
let keyOnly=false, daysHidden=false;
function applyKey(){ rows.forEach(r=>r.classList.toggle('q-hide', keyOnly && r.dataset.k!=='1')); document.getElementById('keyBtn').classList.toggle('on',keyOnly);}
document.querySelector('[data-act=expand]').onclick=()=>secs.forEach(s=>setSec(s.dataset.sec,false));
document.querySelector('[data-act=collapse]').onclick=()=>secs.forEach(s=>setSec(s.dataset.sec,true));
document.querySelector('[data-act=key]').onclick=()=>{keyOnly=!keyOnly; if(keyOnly) secs.forEach(s=>setSec(s.dataset.sec,false)); applyKey();};
document.querySelector('[data-act=days]').onclick=(e)=>{daysHidden=!daysHidden; weekIds.forEach(w=>setWeek(w,daysHidden));
  e.target.textContent=daysHidden?'Показать дни':'Скрыть дни'; e.target.classList.toggle('on',daysHidden);};
document.getElementById('q').addEventListener('input',e=>{
  const q=e.target.value.trim().toLowerCase();
  rows.forEach(r=>{ r.classList.toggle('q-hide', (q && !r.dataset.label.includes(q)) || (!q && keyOnly && r.dataset.k!=='1')); if(q) r.classList.remove('folded'); });
  if(q) secs.forEach(s=>s.classList.remove('folded'));
});
t.querySelectorAll('th.has').forEach(h=>h.addEventListener('click',e=>{
  e.stopPropagation(); pop.textContent=h.dataset.pop.replace(/\\n/g,'\\n\\n');
  pop.style.display='block'; const r=h.getBoundingClientRect();
  pop.style.left=Math.max(8,Math.min(r.left, window.innerWidth-440))+'px'; pop.style.top=(r.bottom+4)+'px';
}));
document.addEventListener('click',()=>pop.style.display='none');
"""
