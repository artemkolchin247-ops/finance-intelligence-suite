"""Operations journal: repository, validation and classification (cash-flow vs P&L views of the same operations)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LEDGER_PATH = Path(__file__).resolve().parent.parent / "data" / "finance_ledger.xlsx"

JOURNAL_RENAME = {
    "Дата ДДС": "dds_date", "Дата P&L": "pl_date", "Приход": "income", "Расход": "expense",
    "Статья операции": "article", "МП": "mp", "Кабинет": "cabinet", "Артикул поставщика": "sku", "Р-р": "size",
    "Кол-во": "qty", "Касса / Счет": "account", "Комментарий": "comment",
}
ARTICLE_RENAME = {"Статья операции": "article", "Для P&L": "pl_kind", "Категория P&L": "pl_category", "Для ДДС": "dds_group"}
ACCOUNT_RENAME = {"Касса / Счет": "account", "Группа счетов": "account_group", "Юрлицо": "entity", "Входящий остаток": "opening"}
PLAN_COLUMNS = ["ID", "Дата", "Статья операции", "Контрагент", "Юрлицо", "Сумма", "Приоритет", "Статус", "Комментарий"]
PRIORITIES = ("Критичный", "Обязательный", "Можно отложить")
PLAN_STATUSES = ("Запланирован", "Согласован", "Оплачен", "Просрочен", "Отложен")

# P&L classification values (as in the article reference)
PL_INDIRECT = "Косвенные расходы (до EBITDA)"
PL_TAX = "Налоги"
PL_AFTER_INC = "Доходы после EBITDA"
PL_AFTER_EXP = "Расходы после EBITDA"
PL_DIVIDENDS = "Вывод дивидендов"
TRANSFER_GROUP = "Перевод между счетами"
ACT_OPERATING, ACT_FINANCING, ACT_INTERNAL = "Операционная", "Финансовая", "Внутренние переводы"
_FINANCING_GROUPS = {"Кредиты", "% по кредитам", "Вывод дивидендов"}


def activity_of(group: str) -> str:
    """Cash-flow activity derived from the cash-flow group of an article."""
    if group == TRANSFER_GROUP:
        return ACT_INTERNAL
    return ACT_FINANCING if group in _FINANCING_GROUPS else ACT_OPERATING


class LedgerRepository(ABC):
    @abstractmethod
    def get_journal(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_articles(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_accounts(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_fact_monthly(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_balance_1c(self) -> pd.DataFrame: ...

    @abstractmethod
    def get_payment_plan(self) -> pd.DataFrame: ...

    @abstractmethod
    def save_payment_plan(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def get_as_of(self) -> pd.Timestamp: ...

    @abstractmethod
    def save_ledger(self, journal: pd.DataFrame, articles: pd.DataFrame, accounts: pd.DataFrame, as_of: pd.Timestamp) -> None:
        """Replace the journal (internal column names), the article and account references and the as-of date."""


class ExcelLedgerRepository(LedgerRepository):
    def __init__(self, path: Path = DEFAULT_LEDGER_PATH) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Ledger workbook not found: {path}")
        self._path = Path(path)
        self._sheets: dict[str, pd.DataFrame] | None = None

    def _sheet(self, name: str) -> pd.DataFrame:
        if self._sheets is None:
            self._sheets = pd.read_excel(self._path, sheet_name=None)
        try:
            return self._sheets[name].copy()
        except KeyError as exc:
            raise ValueError(f"Sheet {name!r} is missing in {self._path.name}") from exc

    def get_journal(self) -> pd.DataFrame:
        df = self._sheet("journal").rename(columns=JOURNAL_RENAME)
        for c in ("dds_date", "pl_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in ("income", "expense"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ("mp", "cabinet", "sku", "size", "comment"):
            df[c] = df[c].astype("string").fillna("").astype(str)
        return df

    def get_articles(self) -> pd.DataFrame:
        df = self._sheet("articles").rename(columns=ARTICLE_RENAME)
        df["pl_category"] = df["pl_category"].astype("string").fillna("").astype(str)
        return df

    def get_accounts(self) -> pd.DataFrame:
        return self._sheet("accounts").rename(columns=ACCOUNT_RENAME)

    def get_fact_monthly(self) -> pd.DataFrame:
        df = self._sheet("fact_monthly")
        df["month"] = pd.to_datetime(df["month"])
        return df

    def get_balance_1c(self) -> pd.DataFrame:
        df = self._sheet("balance_1c")
        df["month"] = pd.to_datetime(df["month"])
        return df

    def get_payment_plan(self) -> pd.DataFrame:
        df = self._sheet("payment_plan")
        for c in PLAN_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce")
        df["Сумма"] = pd.to_numeric(df["Сумма"], errors="coerce")
        for c in ("ID", "Статья операции", "Контрагент", "Юрлицо", "Приоритет", "Статус", "Комментарий"):
            df[c] = df[c].astype("string").fillna("").astype(str)
        return df[PLAN_COLUMNS]

    def get_as_of(self) -> pd.Timestamp:
        meta = self._sheet("meta")
        return pd.Timestamp(meta.loc[meta["key"] == "as_of", "value"].iloc[0])

    def save_ledger(self, journal: pd.DataFrame, articles: pd.DataFrame, accounts: pd.DataFrame, as_of: pd.Timestamp) -> None:
        inv_j = {v: k for k, v in JOURNAL_RENAME.items()}
        inv_a = {v: k for k, v in ARTICLE_RENAME.items()}
        inv_c = {v: k for k, v in ACCOUNT_RENAME.items()}
        try:
            with pd.ExcelWriter(self._path, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
                journal.rename(columns=inv_j)[list(JOURNAL_RENAME)].to_excel(xw, sheet_name="journal", index=False)
                articles.rename(columns=inv_a)[list(ARTICLE_RENAME)].to_excel(xw, sheet_name="articles", index=False)
                accounts.rename(columns=inv_c)[list(ACCOUNT_RENAME)].to_excel(xw, sheet_name="accounts", index=False)
                pd.DataFrame({"key": ["as_of"], "value": [as_of.strftime("%Y-%m-%d")]}).to_excel(xw, sheet_name="meta", index=False)
        except OSError as exc:
            raise OSError(f"Cannot write the ledger to {self._path.name}: {exc}") from exc
        self._sheets = None

    def save_payment_plan(self, df: pd.DataFrame) -> None:
        missing = [c for c in PLAN_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Payment plan lacks columns: {missing}")
        try:
            with pd.ExcelWriter(self._path, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
                df[PLAN_COLUMNS].to_excel(xw, sheet_name="payment_plan", index=False)
        except OSError as exc:
            raise OSError(f"Cannot write payment plan to {self._path.name}: {exc}") from exc
        if self._sheets is not None:
            self._sheets["payment_plan"] = df[PLAN_COLUMNS].copy()


@dataclass(frozen=True)
class Ledger:
    """Classified journal plus reference tables; everything the finance statements read."""
    journal: pd.DataFrame
    accounts: pd.DataFrame
    as_of: pd.Timestamp


def prepare_journal(journal: pd.DataFrame, articles: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Validate the journal and attach the classification of every operation.

    Derived columns: cash (income - expense), activity, dds_group, pl_kind, pl_category, planned (cash date after as-of).
    Raises ValueError with the offending rows for missing dates, empty/ambiguous amounts or unknown articles.
    """
    df = journal.copy()
    bad_dates = df[df["dds_date"].isna() | df["pl_date"].isna()]
    if len(bad_dates):
        raise ValueError(f"{len(bad_dates)} journal rows have no cash or P&L date, e.g. index {bad_dates.index[:3].tolist()}")
    empty = df[df["income"].isna() & df["expense"].isna()]
    if len(empty):
        raise ValueError(f"{len(empty)} journal rows have neither income nor expense, e.g. index {empty.index[:3].tolist()}")
    unknown = sorted(set(df["article"]) - set(articles["article"]))
    if unknown:
        raise ValueError(f"Articles missing in the reference: {unknown}")
    df["income"] = df["income"].fillna(0.0)
    df["expense"] = df["expense"].fillna(0.0)
    df = df.merge(articles, on="article", how="left", validate="m:1")
    df["cash"] = df["income"] - df["expense"]
    df["activity"] = df["dds_group"].map(activity_of)
    df["planned"] = df["dds_date"] > as_of
    return df


def pl_amount(df: pd.DataFrame) -> pd.Series:
    """Signed P&L effect: costs positive, so refunds booked on cost articles reduce them.

    For income-type P&L categories (income after EBITDA) the sign is reversed: income positive.
    """
    cost = df["expense"] - df["income"]
    return cost.where(df["pl_kind"] != PL_AFTER_INC, -cost)


def month_start(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M").dt.to_timestamp()
