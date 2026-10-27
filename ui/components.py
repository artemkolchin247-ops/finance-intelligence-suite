"""Reusable presentation pieces: KPI cards, HTML tables, plotly theme."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Sequence

import plotly.graph_objects as go
import streamlit as st

PALETTE = ["#2F5BEA", "#14A38B", "#F2A33A", "#8B5CF6", "#E5484D", "#5B6B8C", "#0EA5E9", "#A3B1D1"]


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="page-title">{escape(title)}</div><div class="page-sub">{escape(subtitle)}</div>',
                unsafe_allow_html=True)


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{escape(text)}</div>', unsafe_allow_html=True)


@dataclass(frozen=True)
class Kpi:
    label: str
    value: str
    delta: str = ""
    good: bool | None = None  # True = green, False = red, None = neutral
    spark: tuple[float, ...] = ()
    hint: str = ""


def sparkline(values: Sequence[float], color: str = "#2F5BEA", w: int = 120, h: int = 28) -> str:
    """Tiny inline SVG trend line; empty string when there is nothing to draw."""
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pts = " ".join(f"{i * w / (len(vals) - 1):.1f},{h - 3 - (v - lo) / span * (h - 6):.1f}" for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" stroke-linejoin="round" points="{pts}"/></svg>')


def kpi_row(items: Sequence[Kpi]) -> None:
    for col, k in zip(st.columns(len(items)), items):
        cls = "" if k.good is None else ("up" if k.good else "down")
        color = {"up": "#14935F", "down": "#D6404A"}.get(cls, "#2F5BEA")
        delta = f'<div class="d {cls}">{escape(k.delta)}</div>' if k.delta else '<div class="d">&nbsp;</div>'
        hint = f' title="{escape(k.hint, quote=True)}"' if k.hint else ""
        col.markdown(
            f'<div class="kpi"{hint}><div class="l">{escape(k.label)}</div><div class="v">{escape(k.value)}</div>'
            f'{delta}<div class="sp">{sparkline(k.spark, color)}</div></div>',
            unsafe_allow_html=True,
        )


@dataclass(frozen=True)
class Cell:
    text: str
    negative: bool = False
    missing: bool = False
    badge: bool = False


@dataclass(frozen=True)
class Row:
    label: str
    cells: Sequence[Cell] = ()
    section: bool = False


def html_table(headers: Sequence[str], rows: Sequence[Row], total_idx: int | None = None, corner: str = "Показатель") -> str:
    """Scrollable table with sticky header and first column. `total_idx` highlights one data column."""
    head = [f"<th>{escape(corner)}</th>"]
    for i, h in enumerate(headers):
        head.append(f'<th class="{"total" if i == total_idx else ""}">{escape(h)}</th>')
    body = []
    for r in rows:
        if r.section:
            body.append(f'<tr class="sec"><td colspan="{len(headers) + 1}">{escape(r.label)}</td></tr>')
            continue
        tds = [f"<td>{escape(r.label)}</td>"]
        for i, c in enumerate(r.cells):
            cls = " ".join(x for x in ("total" if i == total_idx else "", "neg" if c.negative else "",
                                       "na" if c.missing else "") if x)
            text = f'<span class="badge {escape(c.text)}">{escape(c.text)}</span>' if c.badge and not c.missing else escape(c.text)
            tds.append(f'<td class="{cls}">{text}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{"".join(head)}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def style_figure(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=64), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1B2437"), colorway=PALETTE,
        legend=dict(orientation="h", y=-0.34, x=0), title=dict(font=dict(size=14), x=0.01),
        hoverlabel=dict(bgcolor="#FFFFFF"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E4E8F0")
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    return fig


def chart_card(fig: go.Figure, key: str) -> None:
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
