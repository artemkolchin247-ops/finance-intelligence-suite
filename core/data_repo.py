"""Data access layer. The Excel workbook stands in for the PostgreSQL warehouse.

To move to PostgreSQL implement `DataRepository` with SQL queries returning the same
columns; nothing above this layer needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "finance_db.xlsx"
MARKETPLACES = ("WB", "Ozon")

HYPOTHESIS_COLUMNS = [
    "Дата", "Маркетплейс", "Модель", "Гипотеза", "Проблема", "Результат", "Ответственный",
    "Сроки выполнения", "Статус", "Ссылка Bitrix", "Ссылка на результат",
]
HYPOTHESIS_STATUSES = ("В работе", "Подтверждено", "Опровергнуто")


class DataRepository(ABC):
    @abstractmethod
    def get_dim_sku(self) -> pd.DataFrame:
        """sku, model, status."""

    @abstractmethod
    def get_fact_daily(self) -> pd.DataFrame:
        """One row per date x mp x sku with raw daily metrics (orders_count, sales_qty, reclama, ...)."""

    @abstractmethod
    def get_fact_ext_daily(self) -> pd.DataFrame:
        """One row per date x mp x model with raw external-advertising metrics."""

    @abstractmethod
    def get_stocks_ms(self) -> pd.DataFrame:
        """date, sku, qty: own-warehouse stock shared by both marketplaces."""

    @abstractmethod
    def get_creators_weekly(self) -> pd.DataFrame:
        """week_start, week_end, mp, model and raw creator metrics."""

    @abstractmethod
    def get_hypotheses(self) -> pd.DataFrame:
        """Hypothesis log with HYPOTHESIS_COLUMNS."""

    @abstractmethod
    def save_hypotheses(self, df: pd.DataFrame) -> None:
        """Replace the hypothesis log."""


class ExcelRepository(DataRepository):
    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Data workbook not found: {path}")
        self._path = Path(path)
        self._sheets: dict[str, pd.DataFrame] | None = None

    def _load(self) -> dict[str, pd.DataFrame]:
        # one pass over the workbook: parsing xlsx is the slow part
        if self._sheets is None:
            self._sheets = pd.read_excel(self._path, sheet_name=None)
        return self._sheets

    def _sheet(self, name: str) -> pd.DataFrame:
        try:
            return self._load()[name].copy()
        except KeyError as exc:
            raise ValueError(f"Sheet {name!r} is missing in {self._path.name}") from exc

    def get_dim_sku(self) -> pd.DataFrame:
        return self._sheet("dim_sku")

    def get_fact_daily(self) -> pd.DataFrame:
        df = self._sheet("fact_daily")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_fact_ext_daily(self) -> pd.DataFrame:
        df = self._sheet("fact_ext_daily")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_stocks_ms(self) -> pd.DataFrame:
        df = self._sheet("stocks_ms")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_creators_weekly(self) -> pd.DataFrame:
        df = self._sheet("creators_weekly")
        for c in ("week_start", "week_end"):
            df[c] = pd.to_datetime(df[c])
        return df

    def get_hypotheses(self) -> pd.DataFrame:
        df = self._sheet("hypotheses")
        for c in ("Дата", "Сроки выполнения"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in HYPOTHESIS_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        text_cols = [c for c in HYPOTHESIS_COLUMNS if c not in ("Дата", "Сроки выполнения")]
        for c in text_cols:  # empty Excel cells arrive as float NaN
            df[c] = df[c].astype("string").fillna("").astype(str)
        return df[HYPOTHESIS_COLUMNS]

    def save_hypotheses(self, df: pd.DataFrame) -> None:
        missing = [c for c in HYPOTHESIS_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Hypothesis table lacks columns: {missing}")
        try:
            with pd.ExcelWriter(self._path, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
                df[HYPOTHESIS_COLUMNS].to_excel(xw, sheet_name="hypotheses", index=False)
        except OSError as exc:
            raise OSError(f"Cannot write hypotheses to {self._path.name}: {exc}") from exc
        if self._sheets is not None:
            self._sheets["hypotheses"] = df[HYPOTHESIS_COLUMNS].copy()
