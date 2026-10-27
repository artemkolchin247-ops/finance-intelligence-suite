"""Cached data loading for the UI (keeps core/ free of streamlit)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from core import cashflow, planning, rnp_calc
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
    """Forecast assumptions of the current session (see ui.scenario)."""
    from ui.scenario import scenario_for
    return scenario_for(get_finance())


def get_plan() -> planning.PlanInputs:
    """The plan in use: the session's edited one if present, otherwise the saved one."""
    live = st.session_state.get("plan_live")
    if live is not None:
        return live
    saved = st.session_state.get("plan_saved")
    if saved is None:
        saved = planning.plan_from_sheets(get_ledger_repo().get_plan_sheets())
        st.session_state["plan_saved"] = saved
    return saved


@st.cache_data(show_spinner=False, max_entries=32)
def cached_schedule(assumptions: cashflow.Assumptions, payment_plan: pd.DataFrame, version: int, plan_key: tuple, _plan, _fin: FinanceData):
    return cashflow.build_schedule(_fin.journal, _fin.balance_1c, payment_plan, _fin.as_of, assumptions, plan=_plan)


def get_schedule() -> pd.DataFrame:
    plan = get_plan()
    return cached_schedule(scenario(), load_payment_plan(), finance_key(), plan.key(), plan, get_finance())
