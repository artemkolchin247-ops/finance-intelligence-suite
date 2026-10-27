"""Plotly figures of the finance pages."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

from ui.components import PALETTE, style_figure

def _cat(fig: go.Figure, height: int = 320) -> go.Figure:
    """Styled figure with a categorical x axis: labels like '03.2026' would otherwise be read as numbers."""
    fig = style_figure(fig, height)
    fig.update_xaxes(type="category")
    return fig


GOOD, BAD, NEUTRAL, PLAN = "#14A38B", "#E5484D", "#2F5BEA", "#9DB3F0"


def pnl_waterfall(v: Mapping[str, float], title: str) -> go.Figure:
    steps = [("Выручка после СПП", v["rev_after"], "absolute"), ("Себестоимость", -(v["cogs"] + v["cogs_comp"]), "relative"),
             ("Прямые расходы", -v["direct"], "relative"), ("Продвижение", -v["promo_total"], "relative"),
             ("НДС", -v["nds"], "relative"), ("Маржа", v["margin"], "total"), ("Косвенные", -v["indirect"], "relative"),
             ("EBITDA", v["ebitda"], "total"), ("Налоги и прочее", -v["taxes"] + v["income_after"] - v["expense_after"], "relative"),
             ("Чистая прибыль", v["net_profit"], "total")]
    fig = go.Figure(go.Waterfall(
        x=[s[0] for s in steps], y=[s[1] / 1000 for s in steps], measure=[s[2] for s in steps],
        text=[f"{s[1] / 1000:,.0f}".replace(",", " ") for s in steps], textposition="outside",
        increasing=dict(marker_color=GOOD), decreasing=dict(marker_color=BAD), totals=dict(marker_color=NEUTRAL),
        connector=dict(line=dict(color="#C9D1E3"))))
    fig.update_layout(title=title, showlegend=False)
    fig.update_xaxes(automargin=True)
    return _cat(fig, 380)


def pnl_trend(labels: Sequence[str], series: Mapping[str, Sequence[float]]) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_scatter(x=list(labels), y=[None if (v is None or (isinstance(v, float) and math.isnan(v))) else v * 100 for v in vals],
                        name=name, mode="lines+markers", line=dict(color=PALETTE[i], width=2.2))
    fig.update_layout(title="Рентабельность, % от выручки до СПП", yaxis_ticksuffix="%")
    return _cat(fig, 380)


def indirect_structure(labels: Sequence[str], cats: Mapping[str, Sequence[float]]) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(cats.items()):
        fig.add_bar(x=list(labels), y=[v / 1000 for v in vals], name=name, marker_color=PALETTE[i])
    fig.update_layout(title="Косвенные расходы по категориям, тыс. ₽", barmode="stack")
    return _cat(fig, 380)


def dds_balance(labels: Sequence[str], states: Sequence[str], closing: Sequence[float], buffer: float) -> go.Figure:
    colors = [NEUTRAL if s == "Факт" else (PLAN if s == "План" else "#F2A33A") for s in states]
    fig = go.Figure()
    fig.add_bar(x=list(labels), y=[c / 1e6 for c in closing], marker_color=colors, name="Остаток на конец месяца",
                hovertemplate="%{x}: %{y:.2f} млн ₽<extra></extra>")
    fig.add_hline(y=buffer / 1e6, line=dict(color=BAD, dash="dash", width=1.5), annotation_text="минимальный остаток",
                  annotation_position="top left")
    fig.update_layout(title="Остаток денег на конец месяца, млн ₽ (синий — факт, светлый — план)", showlegend=False)
    return _cat(fig, 360)


def dds_net(labels: Sequence[str], op: Sequence[float], fin: Sequence[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=list(labels), y=[v / 1e6 for v in op], name="Операционная", marker_color=NEUTRAL)
    fig.add_bar(x=list(labels), y=[v / 1e6 for v in fin], name="Финансовая", marker_color="#F2A33A")
    fig.update_layout(title="Чистый денежный поток по видам деятельности, млн ₽", barmode="relative")
    return _cat(fig, 360)


def dds_structure(labels: Sequence[str], groups: Mapping[str, Sequence[float]]) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(groups.items()):
        fig.add_bar(x=list(labels), y=[v / 1e6 for v in vals], name=name, marker_color=PALETTE[i % len(PALETTE)])
    fig.update_layout(title="Структура платежей, млн ₽", barmode="stack")
    return _cat(fig, 360)


def calendar_balance(daily: pd.DataFrame, buffer: float, big: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=daily["date"], y=daily["close"] / 1e6, mode="lines", line=dict(color=NEUTRAL, width=2.4, shape="hv"),
                    name="Остаток на конец дня", fill="tozeroy", fillcolor="rgba(47,91,234,0.08)")
    fig.add_hline(y=buffer / 1e6, line=dict(color=BAD, dash="dash", width=1.5), annotation_text="минимальный остаток")
    if len(big):
        fig.add_scatter(x=big["date"], y=[daily.set_index("date")["close"].get(d, 0) / 1e6 for d in big["date"]], mode="markers",
                        marker=dict(size=9, color=BAD, symbol="diamond"), name="Крупные платежи",
                        text=[f"{a}: {v / 1e6:.2f} млн ₽" for a, v in zip(big["article"], big["amount"])], hoverinfo="text+x")
    fig.update_layout(title="Прогноз остатка денег по дням, млн ₽")
    return style_figure(fig, 340)


def balance_structure(labels: Sequence[str], parts: Mapping[str, Sequence[float]]) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(parts.items()):
        fig.add_bar(x=list(labels), y=[v / 1e6 for v in vals], name=name, marker_color=PALETTE[i % len(PALETTE)])
    fig.update_layout(title="Структура активов, млн ₽", barmode="stack")
    return _cat(fig, 360)


def equity_chart(labels: Sequence[str], equity: Sequence[float], debt: Sequence[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=list(labels), y=[v / 1e6 for v in equity], name="Капитал", line=dict(color=GOOD, width=2.4), mode="lines")
    fig.add_scatter(x=list(labels), y=[v / 1e6 for v in debt], name="Кредиты и займы", line=dict(color=BAD, width=2.2, dash="dash"), mode="lines")
    fig.update_layout(title="Капитал и долг, млн ₽")
    return _cat(fig, 360)


def plan_vs_base(labels: Sequence[str], base_close: Sequence[float], plan_close: Sequence[float], buffer: float) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=list(labels), y=[v / 1e6 for v in base_close], name="Базовый прогноз", mode="lines+markers",
                    line=dict(color="#9DB3F0", width=2.4, dash="dash"))
    fig.add_scatter(x=list(labels), y=[v / 1e6 for v in plan_close], name="С учётом плана", mode="lines+markers", line=dict(color=NEUTRAL, width=2.8))
    fig.add_hline(y=buffer / 1e6, line=dict(color=BAD, dash="dash", width=1.2), annotation_text="минимальный остаток")
    fig.update_layout(title="Остаток денег на конец месяца: база и план, млн ₽")
    return _cat(fig, 360)
