"""Cached data loading for the UI (keeps core/ free of streamlit)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from core import cashflow, rnp_calc
from core.data_repo import ExcelRepository
from core.ledger import ExcelLedgerRepository, prepare_journal

# default view: the weeks around the logged hypotheses
DEFAULT_START = pd.Timestamp("2026-04-27")
DEFAULT_END = pd.Timestamp("2026-05-24")


@dataclass
class Dataset:
    fact: pd.DataFrame
    stocks_ms: pd.DataFrame
    creators: pd.DataFrame
    dim: pd.DataFrame


@st.cache_resource
def get_repo() -> ExcelRepository:
    return ExcelRepository()


@st.cache_resource(show_spinner="Загрузка данных…")
def load_dataset() -> Dataset:
    repo = get_repo()
    dim = repo.get_dim_sku()
    fact = rnp_calc.prepare_fact(repo.get_fact_daily(), repo.get_fact_ext_daily(), dim)
    return Dataset(fact, repo.get_stocks_ms(), repo.get_creators_weekly(), dim)


def load_hypotheses() -> pd.DataFrame:
    return get_repo().get_hypotheses()


@st.cache_data(show_spinner=False, max_entries=64)
def cached_report(mp: str, start: str, end: str, models: tuple, skus: tuple, status: str):
    """Report for a selection; arguments are plain values so Streamlit can hash them."""
    ds = load_dataset()
    sel = rnp_calc.Selection(mp, pd.Timestamp(start), pd.Timestamp(end), models, skus, status)
    return rnp_calc.build_report(ds.fact, ds.stocks_ms, ds.creators, sel)


# ---------------------------------------------------------------- finance (ledger) data
@dataclass
class FinanceData:
    journal: pd.DataFrame  # classified operations
    articles: pd.DataFrame
    accounts: pd.DataFrame
    fact_monthly: pd.DataFrame
    balance_1c: pd.DataFrame
    as_of: pd.Timestamp


@st.cache_resource
def get_ledger_repo() -> ExcelLedgerRepository:
    return ExcelLedgerRepository()


@st.cache_resource(show_spinner="Загрузка финансовых данных…")
def load_finance() -> FinanceData:
    repo = get_ledger_repo()
    as_of = repo.get_as_of()
    articles = repo.get_articles()
    return FinanceData(prepare_journal(repo.get_journal(), articles, as_of), articles, repo.get_accounts(),
                       repo.get_fact_monthly(), repo.get_balance_1c(), as_of)


def get_finance() -> FinanceData:
    """Finance data of this session: an uploaded journal if one was applied, otherwise the stored workbook."""
    override = st.session_state.get("ledger_override")
    return override["fin"] if override else load_finance()


def finance_key() -> int:
    """Changes whenever the session's journal changes (cache key for derived results)."""
    override = st.session_state.get("ledger_override")
    return override["version"] if override else 0


def load_payment_plan() -> pd.DataFrame:
    return get_ledger_repo().get_payment_plan()


def scenario() -> cashflow.Assumptions:
    """Forecast assumptions of the current session; defaults are derived from the history."""
    fin = get_finance()
    base = cashflow.default_assumptions(fin.journal, fin.as_of)
    ss = st.session_state
    return cashflow.Assumptions(
        growth_monthly=ss.get("sc_growth", base.growth_monthly * 100) / 100, seasonality=ss.get("sc_season", True),
        purchase_scale=ss.get("sc_purchase", 100) / 100, payroll_index=ss.get("sc_payroll", 0.0) / 100,
        dividends_month=ss.get("sc_div", base.dividends_month / 1000) * 1000, buffer=ss.get("sc_buffer", 2000) * 1000.0)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_schedule(growth: float, season: bool, purchase: float, payroll: float, dividends: float, plan: pd.DataFrame,
                    version: int, _fin: FinanceData):
    fin = _fin
    a = cashflow.Assumptions(growth, season, purchase, payroll, dividends)
    return cashflow.build_schedule(fin.journal, fin.balance_1c, plan, fin.as_of, a)


def get_schedule() -> pd.DataFrame:
    a = scenario()
    return cached_schedule(a.growth_monthly, a.seasonality, a.purchase_scale, a.payroll_index, a.dividends_month, load_payment_plan(),
                           finance_key(), get_finance())
