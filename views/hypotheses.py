"""Hypotheses log linked to the РНП: filter, edit, and see what changed after each hypothesis date."""
from __future__ import annotations

from datetime import date, timedelta
from html import escape

import pandas as pd
import streamlit as st

from core import hypotheses as H
from core.data_repo import HYPOTHESIS_COLUMNS, HYPOTHESIS_STATUSES, MARKETPLACES
from core.formatting import format_value
from ui import charts
from ui.components import chart_card, page_header, section_title
from ui.state import cached_report, get_repo, load_dataset, load_hypotheses

CHIP = {"В работе": "work", "Подтверждено": "ok", "Опровергнуто": "no"}


def _fmt_effect(v: float, kind: str) -> str:
    return format_value(v, kind)


def _effect_cards(rows: list[H.EffectRow], days_after: int) -> None:
    cards = []
    for r in rows:
        unit = " / день" if r.kind in ("int", "k") else ""
        sign = "+" if r.delta >= 0 else "−"
        num = f"{abs(r.delta) * 100:.1f}".replace(".", ",")
        delta = f"{sign}{num} п.п." if r.is_pp else f"{sign}{num}%"
        arrow = "▲" if r.delta >= 0 else "▼"
        cls = {"good": "good", "bad": "bad"}.get(r.sentiment, "")
        label = r.label.replace(", ₽", ", тыс. ₽") if r.kind == "k" and ", тыс." not in r.label else r.label
        cards.append(
            f'<div class="c {cls}"><div class="n">{escape(label)}{unit}</div>'
            f'<div class="a">{_fmt_effect(r.after, r.kind)}</div>'
            f'<div class="b">было {_fmt_effect(r.before, r.kind)}</div><div class="p">{arrow} {delta}</div></div>')
    st.markdown(f'<div class="fx">{"".join(cards)}</div>', unsafe_allow_html=True)


def _label(r: pd.Series) -> str:
    text = str(r["Гипотеза"])
    return f"{r['Дата']:%d.%m.%Y} · {r['Маркетплейс']} · {r['Модель']} — {text[:70]}{'…' if len(text) > 70 else ''}"


def _save(df: pd.DataFrame) -> bool:
    try:
        get_repo().save_hypotheses(df)
    except (OSError, ValueError) as exc:
        st.error(f"Не удалось сохранить журнал гипотез: {exc}")
        return False
    return True


page_header("Гипотезы", "Журнал изменений и экспериментов: что сделали, в какой день и как изменились показатели РНП")

try:
    ds = load_dataset()
    hyp = load_hypotheses()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

models = sorted(ds.dim["model"].unique(), key=str.lower)
data_max = ds.fact["date"].max().normalize()

# ---- summary + filters
counts = hyp["Статус"].value_counts()
cols = st.columns(4)
for col, (name, key) in zip(cols, (("Всего", None), *((s, s) for s in HYPOTHESIS_STATUSES))):
    n = len(hyp) if key is None else int(counts.get(key, 0))
    cls = "" if key is None else CHIP[key]
    col.markdown(f'<div class="kpi"><div class="l">{name}</div><div class="v">{n}</div>'
                 f'<div class="d {"up" if cls == "ok" else "down" if cls == "no" else ""}">&nbsp;</div></div>',
                 unsafe_allow_html=True)

with st.container(border=True):
    f1, f2, f3, f4 = st.columns(4)
    f_mp = f1.multiselect("Маркетплейс", MARKETPLACES, key="hy_mp", placeholder="Все")
    f_model = f2.multiselect("Модель", sorted(hyp["Модель"].unique(), key=str.lower), key="hy_model", placeholder="Все")
    f_status = f3.multiselect("Статус", list(HYPOTHESIS_STATUSES), key="hy_status", placeholder="Все")
    f_owner = f4.multiselect("Ответственный", sorted(x for x in hyp["Ответственный"].unique() if x), key="hy_owner",
                             placeholder="Все")

view = hyp
if f_mp:
    view = view[view["Маркетплейс"].str.lower().isin([m.lower() for m in f_mp])]
if f_model:
    view = view[view["Модель"].isin(f_model)]
if f_status:
    view = view[view["Статус"].isin(f_status)]
if f_owner:
    view = view[view["Ответственный"].isin(f_owner)]
view = view.sort_values("Дата", ascending=False)

# ---- list
section_title("Журнал")
if view.empty:
    st.info("Нет гипотез под выбранные фильтры.")
for _, r in view.iterrows():
    chip = CHIP.get(r["Статус"], "work")
    links = " · ".join(f'<a href="{escape(r[c], quote=True)}" target="_blank">{c}</a>'
                       for c in ("Ссылка Bitrix", "Ссылка на результат") if str(r[c]).startswith("http"))
    due = f" · срок {r['Сроки выполнения']:%d.%m.%Y}" if pd.notna(r["Сроки выполнения"]) else ""
    problem = f"<br><b>Проблема:</b> {escape(r['Проблема'])}" if r["Проблема"].strip() not in ("", "-") else ""
    result = f"<br><b>Результат:</b> {escape(r['Результат']).replace(chr(10), '<br>')}" if r["Результат"].strip() else ""
    st.markdown(
        f'<div class="hcard"><div class="h">{escape(r["Гипотеза"])}<span class="chip {chip}">{escape(r["Статус"])}</span></div>'
        f'<div class="m">{r["Дата"]:%d.%m.%Y} · {escape(r["Маркетплейс"])} · <b>{escape(r["Модель"])}</b> · '
        f'{escape(r["Ответственный"])}{due}{(" · " + links) if links else ""}'
        f'<div style="margin-top:6px;color:#1B2437">{problem}{result}</div></div></div>', unsafe_allow_html=True)

# ---- effect
section_title("Что изменилось после гипотезы")
if hyp.empty:
    st.info("Журнал пуст — добавьте первую гипотезу ниже.")
else:
    pool = view if not view.empty else hyp
    labels = {idx: _label(row) for idx, row in pool.iterrows()}
    e1, e2 = st.columns([3, 1])
    chosen = e1.selectbox("Гипотеза", list(labels), format_func=labels.get, key="hy_pick")
    window = e2.slider("Окно сравнения, дней", 3, 14, 7, key="hy_window",
                       help="Сравниваются N дней до даты гипотезы и N дней, начиная с неё")
    row = hyp.loc[chosen]
    rows = H.effect(ds.fact, ds.stocks_ms, ds.creators, row["Маркетплейс"], row["Модель"], row["Дата"], window, window,
                    last_date=data_max)
    if rows is None:
        st.warning("Недостаточно данных вокруг даты гипотезы (нет периода «до» или «после»).")
    else:
        st.caption(f"{row['Модель']} · {row['Маркетплейс']} · {window} дн. до {row['Дата']:%d.%m.%Y} → "
                   f"{window} дн. с {row['Дата']:%d.%m.%Y}. Потоковые показатели — в среднем за день. "
                   "Сравнение по заказам и прогнозной марже: выкупы отстают от заказов на несколько дней.")
        _effect_cards(rows, window)
        models_sel = () if row["Модель"].strip().lower() in H.TOTAL_MODEL_NAMES else (row["Модель"],)
        mp_key = "WB" if str(row["Маркетплейс"]).lower() == "wb" else "Ozon"
        w_start = max(row["Дата"] - timedelta(days=14), ds.fact["date"].min().normalize())
        w_end = min(row["Дата"] + timedelta(days=14), data_max)
        rep = cached_report(mp_key, str(w_start.date()), str(w_end.date()), models_sel, (), "ИТОГО")
        if rep is not None:
            marker = {row["Дата"]: [{"model": row["Модель"], "text": row["Гипотеза"], "status": row["Статус"]}]}
            chart_card(charts.orders_chart(rep.day_frame(), marker), "hy_chart")
        if st.button("Открыть в РНП", key="hy_open"):
            st.session_state["_rnp_goto"] = {
                "rnp_mp": mp_key, "rnp_models": list(models_sel), "rnp_skus": [], "rnp_status": "ИТОГО",
                "rnp_period": (max(row["Дата"] - timedelta(days=7), ds.fact["date"].min().normalize()).date(),
                               min(row["Дата"] + timedelta(days=20), data_max).date()),
            }
            st.switch_page("views/rnp.py")

# ---- add / edit
section_title("Добавить гипотезу")
with st.form("hy_add", clear_on_submit=True):
    a1, a2, a3, a4 = st.columns(4)
    n_date = a1.date_input("Дата", value=date.today() if date.today() <= data_max.date() else data_max.date(),
                           format="DD.MM.YYYY")
    n_mp = a2.selectbox("Маркетплейс", MARKETPLACES)
    n_model = a3.selectbox("Модель", ["ИТОГО"] + models)
    n_owner = a4.text_input("Ответственный")
    n_text = st.text_input("Гипотеза", placeholder="Что именно меняем")
    n_problem = st.text_input("Проблема", placeholder="Какую проблему решаем")
    b1, b2, b3 = st.columns(3)
    n_due = b1.date_input("Срок выполнения", value=date.today() + timedelta(days=7), format="DD.MM.YYYY")
    n_status = b2.selectbox("Статус", HYPOTHESIS_STATUSES)
    n_link = b3.text_input("Ссылка Bitrix")
    if st.form_submit_button("Добавить в журнал"):
        if not n_text.strip():
            st.error("Заполните поле «Гипотеза».")
        else:
            new = pd.DataFrame([{
                "Дата": pd.Timestamp(n_date), "Маркетплейс": n_mp, "Модель": n_model, "Гипотеза": n_text.strip(),
                "Проблема": n_problem.strip(), "Результат": "", "Ответственный": n_owner.strip(),
                "Сроки выполнения": pd.Timestamp(n_due), "Статус": n_status, "Ссылка Bitrix": n_link.strip(),
                "Ссылка на результат": "",
            }])
            if _save(pd.concat([hyp, new], ignore_index=True)):
                st.success("Гипотеза добавлена. Она появится в РНП в день, на который назначена.")
                st.rerun()

with st.expander("Редактировать журнал целиком"):
    edited = st.data_editor(
        hyp, key="hy_editor", num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "Дата": st.column_config.DateColumn("Дата", format="DD.MM.YYYY", required=True),
            "Маркетплейс": st.column_config.SelectboxColumn("Маркетплейс", options=list(MARKETPLACES), required=True),
            "Модель": st.column_config.SelectboxColumn("Модель", options=["ИТОГО"] + models, required=True),
            "Гипотеза": st.column_config.TextColumn("Гипотеза", width="large", required=True),
            "Проблема": st.column_config.TextColumn("Проблема", width="medium"),
            "Результат": st.column_config.TextColumn("Результат", width="large"),
            "Сроки выполнения": st.column_config.DateColumn("Сроки выполнения", format="DD.MM.YYYY"),
            "Статус": st.column_config.SelectboxColumn("Статус", options=list(HYPOTHESIS_STATUSES), required=True),
            "Ссылка Bitrix": st.column_config.LinkColumn("Ссылка Bitrix"),
            "Ссылка на результат": st.column_config.LinkColumn("Ссылка на результат"),
        },
    )
    st.caption("Изменения сохраняются в рабочий файл данных. При запуске в облаке файл сбрасывается при перезапуске "
               "приложения — для постоянного хранения журнал нужно вести в базе данных.")
    if st.button("Сохранить изменения", key="hy_save"):
        bad = edited[edited[["Дата", "Маркетплейс", "Модель", "Гипотеза", "Статус"]].isna().any(axis=1)
                     | (edited["Гипотеза"].astype(str).str.strip() == "")]
        if len(bad):
            st.error(f"В {len(bad)} строках не заполнены обязательные поля (дата, маркетплейс, модель, гипотеза, статус).")
        elif _save(edited):
            st.success("Журнал сохранён.")
            st.rerun()
