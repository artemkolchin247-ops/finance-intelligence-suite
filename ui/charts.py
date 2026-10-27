"""Plotly figures for the РНП page. Every figure marks hypothesis dates and can flag anomalous days."""
from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import insights as I
from ui.components import PALETTE, style_figure

GOOD, BAD, NEUTRAL = "#14A38B", "#E5484D", "#2F5BEA"


def _hypothesis_markers(fig: go.Figure, hyps: Mapping[pd.Timestamp, Sequence[dict]]) -> None:
    for day, items in hyps.items():
        fig.add_shape(type="line", x0=day, x1=day, y0=0, y1=1, xref="x", yref="paper",
                      line=dict(color="#F2A33A", width=1.4, dash="dot"))
        hover = "<br>".join(f"<b>{h['model']}</b>: {h['text']} ({h['status']})" for h in items)
        fig.add_annotation(x=day, y=1.0, xref="x", yref="paper", text="💡", showarrow=False, yanchor="bottom",
                           hovertext=hover, font=dict(size=14))


def _anomaly_points(fig: go.Figure, days: pd.DataFrame, key: str, name: str, row: int | None = None) -> None:
    found = [a for a in I.anomalies(days, keys=(key,))]
    if not found:
        return
    kw = dict(row=row, col=1) if row else {}
    fig.add_scatter(x=[a["date"] for a in found], y=[a["value"] for a in found], mode="markers", name="Аномалия дня",
                    marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color="#1B2437", width=2)),
                    hovertemplate=f"{name}: %{{y:,.0f}}<br>отклонение от тренда 14 дн.<extra></extra>", **kw)


def orders_chart(days: pd.DataFrame, hyps) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=days.index, y=days["orders"], name="Заказы, шт", marker_color=NEUTRAL, opacity=0.85)
    fig.add_scatter(x=days.index, y=days["sales"], name="Выкупы (продажи), шт", mode="lines+markers",
                    line=dict(color=GOOD, width=2), marker=dict(size=5))
    _anomaly_points(fig, days, "orders", "Заказы")
    _hypothesis_markers(fig, hyps)
    fig.update_layout(title="Заказы и выкупы по дням, шт", bargap=0.25)
    return style_figure(fig)


def traffic_chart(days: pd.DataFrame, hyps) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06)
    fig.add_bar(x=days.index, y=days["clicks"], name="Клики по карточке", marker_color="#8B5CF6", opacity=0.8, row=1, col=1)
    for key, name, color in (("cr_click_cart", "CR карточка → корзина", "#F2A33A"),
                             ("cr_cart_order", "CR корзина → заказ", GOOD), ("cr_click_order", "CR клик → заказ", NEUTRAL)):
        fig.add_scatter(x=days.index, y=days[key] * 100, name=name, mode="lines", line=dict(color=color, width=2),
                        row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    _hypothesis_markers(fig, hyps)
    fig.update_layout(title="Трафик и воронка: клики (сверху) и конверсии, %")
    return style_figure(fig)


def ads_chart(days: pd.DataFrame, hyps) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
    for i, (key, name) in enumerate((("adv_rub", "Внутренняя"), ("blog_rub", "Блогеры"), ("vk_c_rub", "ВК (подр.)"),
                                     ("vk_s_rub", "Посевы ВК"), ("s_c_rub", "Посевы (подр.)"), ("ya_c_rub", "Яндекс (подр.)"))):
        fig.add_bar(x=days.index, y=days[key] / 1000, name=name, marker_color=PALETTE[i], row=1, col=1)
    fig.add_scatter(x=days.index, y=days["drr_total_sales"] * 100, name="ДРР общий от продаж", mode="lines",
                    line=dict(color=BAD, width=2.2), row=2, col=1)
    fig.add_scatter(x=days.index, y=days["adv_drr_s"] * 100, name="ДРР внутр. от продаж", mode="lines",
                    line=dict(color="#F2A33A", width=1.6, dash="dash"), row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_layout(title="Реклама по каналам, тыс. ₽ (сверху) и ДРР, %", barmode="stack")
    _hypothesis_markers(fig, hyps)
    return style_figure(fig, 360)


def margin_chart(days: pd.DataFrame, hyps) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
    colors = [GOOD if v >= 0 else BAD for v in days["margin"].fillna(0)]
    fig.add_bar(x=days.index, y=days["margin"] / 1000, name="Маржа, тыс. ₽", marker_color=colors, row=1, col=1)
    _anomaly_points(fig, days.assign(margin=days["margin"] / 1000), "margin", "Маржа, тыс. ₽", row=1)
    fig.add_scatter(x=days.index, y=days["margin_pct"] * 100, name="Маржа, %", mode="lines", line=dict(color=NEUTRAL, width=2),
                    row=2, col=1)
    fig.add_scatter(x=days.index, y=days["margin_before_ads_pct"] * 100, name="Маржа до рекламы, %", mode="lines",
                    line=dict(color="#8B5CF6", width=1.6, dash="dash"), row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    fig.update_layout(title="Маржинальная прибыль, тыс. ₽ (сверху) и маржа, %")
    _hypothesis_markers(fig, hyps)
    return style_figure(fig, 360)


def funnel_chart(total: Mapping) -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=["Клики", "Корзины", "Заказы", "Выкупы"], x=[total["clicks"], total["carts"], total["orders"], total["sales"]],
        texttemplate="%{value:,.0f}<br>%{percentPrevious:.0%}", marker=dict(color=["#2F5BEA", "#4F78F0", "#14A38B", "#0E8A74"])))
    fig.update_layout(title="Воронка периода, шт (% — от предыдущего шага)", showlegend=False)
    return style_figure(fig, 360)


def bridge_chart(bridge: Mapping[str, float]) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        x=["Было", "Объём продаж", "Маржинальность до рекламы", "Реклама", "Стало"],
        y=[bridge["start"] / 1000, bridge["volume"] / 1000, bridge["unit_margin"] / 1000, bridge["ads"] / 1000,
           bridge["end"] / 1000],
        measure=["absolute", "relative", "relative", "relative", "total"],
        increasing=dict(marker_color=GOOD), decreasing=dict(marker_color=BAD), totals=dict(marker_color=NEUTRAL),
        text=[f"{v / 1000:,.0f}".replace(",", " ") for v in
              (bridge["start"], bridge["volume"], bridge["unit_margin"], bridge["ads"], bridge["end"])],
        textposition="outside", connector=dict(line=dict(color="#C9D1E3"))))
    levels = [bridge["start"]]
    for k in ("volume", "unit_margin", "ads"):
        levels.append(levels[-1] + bridge[k])
    lo, hi = min(levels + [bridge["end"]]) / 1000, max(levels + [bridge["end"]]) / 1000
    pad = (hi - lo) * 0.25 or abs(hi) * 0.1 or 1.0
    fig.update_yaxes(range=[lo - pad if lo > 0 else lo - pad, hi + pad])  # zoom to the changes, not the base level
    fig.update_layout(title="Что изменило прогнозную маржу к предыдущему периоду, тыс. ₽", showlegend=False)
    fig.update_xaxes(automargin=True)
    return style_figure(fig, 360)
