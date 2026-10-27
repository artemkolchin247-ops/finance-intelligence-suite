"""Cached data loading for the UI (keeps core/ free of streamlit)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from core import rnp_calc
from core.data_repo import ExcelRepository

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
