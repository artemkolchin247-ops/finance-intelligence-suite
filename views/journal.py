"""Журнал операций: upload operations in the accounting layout (everything recalculates), view and filter them."""
from __future__ import annotations

from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st

from core import journal_import as J
from core.ledger import prepare_journal
from ui.components import Kpi, kpi_row, page_header, section_title
from ui.export import to_excel
from ui.state import FinanceData, get_finance, get_ledger_repo, load_finance

page_header("Журнал операций", "Источник ДДС, ОПиУ, платёжного календаря и баланса: загрузите операции из Excel — все отчёты пересчитаются")
try:
    fin = get_finance()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"Не удалось загрузить данные: {exc}")
    st.stop()

override = st.session_state.get("ledger_override")
j = fin.journal
period = f"{j['dds_date'].min():%d.%m.%Y} – {j['dds_date'].max():%d.%m.%Y}"
kpi_row([
    Kpi("Источник", "Загруженный файл" if override else "Рабочий файл данных", override["name"] if override else ""),
    Kpi("Операций", f"{len(j):,}".replace(",", " "), f"из них плановых: {int(j['planned'].sum())}"),
    Kpi("Период по дате ДДС", period),
    Kpi("Данные на дату", f"{fin.as_of:%d.%m.%Y}", "факт до этой даты"),
    Kpi("Статей / счетов", f"{j['article'].nunique()} / {j['account'].nunique()}"),
])

# ------------------------------------------------------------------ upload
section_title("Загрузка операций")
with st.expander("Загрузить журнал из Excel или CSV", expanded=override is None):
    st.markdown(
        "Файл в формате учётного журнала. Обязательные столбцы: **Дата ДДС** (когда получены или потрачены деньги), **Дата P&L** "
        "(когда начислен доход или расход), **Приход**, **Расход**, **Статья операции**, **Касса / Счет**. Необязательные: МП, Кабинет, "
        "Артикул поставщика, Р-р, Кол-во, Комментарий. Названия столбцов распознаются по началу (текст в скобках и после переноса строки "
        "игнорируется). Статьи должны быть из справочника — новые можно классифицировать ниже.")
    t1, t2, t3 = st.columns(3)
    t1.download_button("Скачать шаблон", to_excel({"Журнал операций": J.to_template()}), file_name="journal_template.xlsx", key="jr_tpl")
    t2.download_button("Скачать текущий журнал", to_excel({"Журнал операций": J.to_ru_columns(j)}), file_name="journal.xlsx", key="jr_cur")
    mode = st.radio("Что сделать с загруженными операциями", ["Заменить журнал", "Добавить к текущему"], horizontal=True, key="jr_mode",
                    help="При добавлении полностью совпадающие операции пропускаются")
    upload = st.file_uploader("Файл журнала", type=["xlsx", "xls", "csv"], key="jr_file")
    if upload is not None:
        try:
            if upload.name.lower().endswith(".csv"):
                raw = pd.read_csv(upload, sep=None, engine="python")
            else:
                book = pd.ExcelFile(upload)
                sheet = book.sheet_names[0] if len(book.sheet_names) == 1 else st.selectbox("Лист", book.sheet_names, key="jr_sheet")
                raw = book.parse(sheet)
        except Exception as exc:  # noqa: BLE001 - any unreadable file is reported to the user, not raised
            st.error(f"Не удалось прочитать файл «{upload.name}»: {exc}")
            st.stop()
        res = J.parse_journal(raw, set(fin.articles["article"]), set(fin.accounts["account"]))
        if res.missing_columns:
            st.error("В файле не найдены обязательные столбцы: " + ", ".join(res.missing_columns) +
                     f". Найдены: {', '.join(map(str, raw.columns))}")
        elif res.journal.empty:
            st.error("В файле нет ни одной корректной операции.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Корректных операций", f"{len(res.journal):,}".replace(",", " "))
            m2.metric("Отклонено строк", len(res.errors))
            m3.metric("Период", f"{res.journal['dds_date'].min():%d.%m.%Y} – {res.journal['dds_date'].max():%d.%m.%Y}")
            m4.metric("Новых статей / счетов", f"{len(res.unknown_articles)} / {len(res.new_accounts)}")
            if len(res.errors):
                with st.expander(f"Отклонённые строки ({len(res.errors)}) — исправьте в файле и загрузите снова"):
                    st.dataframe(res.errors.head(200), hide_index=True, width="stretch")
            new_articles, new_accounts = pd.DataFrame(), pd.DataFrame()
            if res.unknown_articles:
                st.warning("В файле есть статьи, которых нет в справочнике. Укажите, как их учитывать в ОПиУ и ДДС.")
                kinds = sorted(fin.articles["pl_kind"].unique())
                groups = sorted(fin.articles["dds_group"].unique())
                cats = sorted(set(fin.articles["pl_category"]) | {""})
                new_articles = st.data_editor(
                    J.articles_to_add(res.unknown_articles), hide_index=True, width="stretch", key="jr_new_art", disabled=["article"],
                    column_config={"article": "Статья", "pl_kind": st.column_config.SelectboxColumn("Для P&L", options=kinds, required=True),
                                   "pl_category": st.column_config.SelectboxColumn("Категория P&L (для косвенных расходов)", options=cats),
                                   "dds_group": st.column_config.SelectboxColumn("Для ДДС (группа)", options=groups, required=True)})
            if res.new_accounts:
                st.info("Новые счета будут добавлены в справочник. Проверьте группу, юрлицо и входящий остаток на начало журнала.")
                new_accounts = st.data_editor(
                    J.accounts_to_add(res.new_accounts), hide_index=True, width="stretch", key="jr_new_acc", disabled=["account"],
                    column_config={"account": "Счёт", "account_group": st.column_config.SelectboxColumn(
                        "Группа", options=sorted(fin.accounts["account_group"].unique())),
                        "entity": st.column_config.SelectboxColumn("Юрлицо", options=[J.ENTITY_OOO, J.ENTITY_IP]),
                        "opening": st.column_config.NumberColumn("Входящий остаток, ₽", format="%d")})
            as_of = st.date_input("Дата отчёта (операции до неё — факт, после — план)", value=(res.as_of or pd.Timestamp(date.today())).date(),
                                  format="DD.MM.YYYY", key="jr_asof")
            if st.button("Применить к приложению", type="primary", key="jr_apply"):
                base = fin.journal[J.INTERNAL] if mode == "Добавить к текущему" else None
                journal = J.merge_journals(base, res.journal) if base is not None else res.journal
                articles = pd.concat([fin.articles, new_articles], ignore_index=True) if len(new_articles) else fin.articles
                accounts = pd.concat([fin.accounts, new_accounts], ignore_index=True) if len(new_accounts) else fin.accounts
                try:
                    prepared = prepare_journal(journal, articles, pd.Timestamp(as_of))
                except ValueError as exc:
                    st.error(f"Журнал не принят: {exc}")
                else:
                    base_fin = load_finance()
                    version = (override["version"] if override else 0) + 1
                    st.session_state["ledger_override"] = {
                        "fin": FinanceData(prepared, articles, accounts, base_fin.fact_monthly, base_fin.balance_1c, pd.Timestamp(as_of)),
                        "raw": journal, "articles": articles, "accounts": accounts, "name": f"{upload.name} ({len(journal)} операций)",
                        "version": version}
                    for key in ("cal_moves", "cal_rejected"):
                        st.session_state.pop(key, None)
                    st.rerun()

if override:
    st.info(f"Применён загруженный журнал: {override['name']}. Все отчёты (ОПиУ, ДДС, календарь, баланс) считаются по нему — только в вашей сессии.")
    s1, s2, _ = st.columns([1.3, 1.3, 2])
    if s1.button("Сохранить как основной журнал", key="jr_save",
                 help="Записывает журнал и справочники в файл данных; изменение увидят все пользователи"):
        try:
            get_ledger_repo().save_ledger(override["raw"], override["articles"], override["accounts"], override["fin"].as_of)
        except (OSError, ValueError) as exc:
            st.error(f"Не удалось сохранить журнал: {exc}")
        else:
            load_finance.clear()
            del st.session_state["ledger_override"]
            st.success("Журнал сохранён в рабочий файл данных.")
            st.rerun()
    if s2.button("Вернуться к рабочему файлу", key="jr_reset"):
        del st.session_state["ledger_override"]
        for key in ("cal_moves", "cal_rejected"):
            st.session_state.pop(key, None)
        st.rerun()

# ------------------------------------------------------------------ viewer
section_title("Операции")
with st.container(border=True):
    f1, f2, f3, f4 = st.columns([1.3, 1.4, 1.4, 1])
    lo, hi = j["dds_date"].min().date(), j["dds_date"].max().date()
    rng = f1.date_input("Дата ДДС", value=(max(lo, (fin.as_of - pd.Timedelta(days=60)).date()), hi), min_value=lo, max_value=hi,
                        format="DD.MM.YYYY", key="jr_range")
    arts = f2.multiselect("Статья", sorted(j["article"].unique()), key="jr_art", placeholder="Все статьи")
    accs = f3.multiselect("Счёт", sorted(j["account"].unique()), key="jr_acc", placeholder="Все счета")
    internal = f4.checkbox("Показывать переводы", value=False, key="jr_int")
    q = st.text_input("Поиск в комментарии", key="jr_q", placeholder="Например: НДС, счёт №")
view = j
if isinstance(rng, (tuple, list)) and len(rng) == 2:
    view = view[view["dds_date"].between(pd.Timestamp(rng[0]), pd.Timestamp(rng[1]))]
if arts:
    view = view[view["article"].isin(arts)]
if accs:
    view = view[view["account"].isin(accs)]
if not internal:
    view = view[view["activity"] != "Внутренние переводы"]
if q:
    view = view[view["comment"].str.contains(q, case=False, regex=False)]
view = view.sort_values("dds_date", ascending=False)
c1, c2, c3 = st.columns(3)
c1.metric("Операций", f"{len(view):,}".replace(",", " "))
c2.metric("Приход, тыс. ₽", f"{view['income'].sum() / 1000:,.0f}".replace(",", " "))
c3.metric("Расход, тыс. ₽", f"{view['expense'].sum() / 1000:,.0f}".replace(",", " "))
shown = J.to_ru_columns(view.head(2000))
st.dataframe(shown, hide_index=True, width="stretch", height=480, column_config={
    "Дата ДДС": st.column_config.DateColumn(format="DD.MM.YYYY"), "Дата P&L": st.column_config.DateColumn(format="DD.MM.YYYY"),
    "Приход": st.column_config.NumberColumn(format="%d"), "Расход": st.column_config.NumberColumn(format="%d")})
if len(view) > 2000:
    st.caption("Показаны первые 2000 операций, уточните фильтры.")
