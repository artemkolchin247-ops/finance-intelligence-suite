"""Import of the operations journal from an Excel/CSV table in the accounting layout.

Expected columns (headers are matched loosely, extra text after a line break or in brackets is ignored):
Дата ДДС, Дата P&L, Приход, Расход, Статья операции, [МП], [Кабинет], [Артикул поставщика], [Р-р], [Кол-во],
Касса / Счет, [Комментарий]. Optional columns may be missing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.ledger import PL_INDIRECT

HEADERS: dict[str, tuple[str, ...]] = {
    "dds_date": ("дата ддс",), "pl_date": ("дата p&l", "дата pl", "дата опиу", "дата п&л"),
    "income": ("приход", "поступление"), "expense": ("расход", "выплата"),
    "article": ("статья операции", "статья"), "mp": ("мп", "маркетплейс"), "cabinet": ("кабинет", "юрлицо"),
    "sku": ("артикул поставщика", "артикул"), "size": ("р-р", "размер"), "qty": ("кол-во", "количество"),
    "account": ("касса / счет", "касса/счет", "касса / счёт", "счет", "счёт", "касса"), "comment": ("комментарий",),
}
REQUIRED = ("dds_date", "pl_date", "income", "expense", "article", "account")
OPTIONAL_TEXT = ("mp", "cabinet", "sku", "size", "comment")
ENTITY_OOO, ENTITY_IP = "ООО Вуки", "ИП Петрова"
JOURNAL_COLUMNS_RU = ["Дата ДДС", "Дата P&L", "Приход", "Расход", "Статья операции", "МП", "Кабинет", "Артикул поставщика",
                      "Р-р", "Кол-во", "Касса / Счет", "Комментарий"]
INTERNAL = ["dds_date", "pl_date", "income", "expense", "article", "mp", "cabinet", "sku", "size", "qty", "account", "comment"]


@dataclass
class ImportResult:
    journal: pd.DataFrame  # valid rows in internal column names
    errors: pd.DataFrame  # rejected rows: row number in the file and the reason
    missing_columns: list[str] = field(default_factory=list)
    unknown_articles: list[str] = field(default_factory=list)
    new_accounts: list[str] = field(default_factory=list)
    as_of: pd.Timestamp | None = None

    @property
    def ok(self) -> bool:
        return not self.missing_columns and len(self.journal) > 0


def _header_key(value: object) -> str:
    text = str(value).split("\n")[0]
    text = re.sub(r"\(.*?\)", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def match_headers(columns: list) -> dict[str, str]:
    """Map the file's column names to internal names; returns {original: internal}."""
    found: dict[str, str] = {}
    for col in columns:
        key = _header_key(col)
        for internal, patterns in HEADERS.items():
            if internal not in found.values() and key in patterns:
                found[col] = internal
                break
    return found


def _to_number(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace(" ", "", regex=False).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
        s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan, "<NA>": np.nan})
    return pd.to_numeric(s, errors="coerce")


def guess_group(account: str) -> str:
    low = account.lower()
    if "фонд" in low and "депозит" not in low:
        return "Фонды"
    if "депозит" in low:
        return "Депозиты"
    if any(w in low for w in ("карта", "совместный", "накопительный", "$")):
        return "Карты и личные счета"
    return "Расчётные счета"


def guess_entity(account: str) -> str:
    return ENTITY_IP if account.strip().lower().startswith("ип") else ENTITY_OOO


def parse_journal(raw: pd.DataFrame, known_articles: set[str], known_accounts: set[str]) -> ImportResult:
    """Validate a raw table and convert it to the internal journal layout (see module docstring)."""
    mapping = match_headers(list(raw.columns))
    missing = [HEADERS[k][0].capitalize() if k != "pl_date" else "Дата P&L" for k in REQUIRED if k not in mapping.values()]
    if missing:
        return ImportResult(pd.DataFrame(columns=INTERNAL), pd.DataFrame(columns=["Строка", "Причина"]), missing_columns=missing)
    df = raw.rename(columns=mapping)[list(mapping.values())].copy()
    for c in INTERNAL:
        if c not in df.columns:
            df[c] = "" if c in OPTIONAL_TEXT else np.nan
    df["row"] = np.arange(len(df)) + 2  # Excel row number (header is row 1)
    df = df.dropna(how="all", subset=[c for c in REQUIRED if c in df.columns])
    df["dds_date"] = pd.to_datetime(df["dds_date"], errors="coerce", dayfirst=True)
    df["pl_date"] = pd.to_datetime(df["pl_date"], errors="coerce", dayfirst=True)
    df["income"], df["expense"], df["qty"] = _to_number(df["income"]), _to_number(df["expense"]), _to_number(df["qty"])
    for c in ("article", "account") + OPTIONAL_TEXT:
        df[c] = df[c].astype("string").fillna("").astype(str).str.strip()

    reasons = pd.Series("", index=df.index)

    def flag(mask: pd.Series, text: str) -> None:
        nonlocal reasons
        reasons = reasons.where(~mask | (reasons != ""), text)

    flag(df["dds_date"].isna(), "не указана или не распознана Дата ДДС")
    flag(df["pl_date"].isna(), "не указана или не распознана Дата P&L")
    flag(df["article"] == "", "не указана статья операции")
    flag(df["account"] == "", "не указана касса / счёт")
    inc, exp = df["income"].fillna(0), df["expense"].fillna(0)
    flag((inc < 0) | (exp < 0), "отрицательная сумма")
    flag((inc <= 0) & (exp <= 0), "не заполнены ни приход, ни расход")
    flag((inc > 0) & (exp > 0), "заполнены и приход, и расход")
    bad = reasons != ""
    errors = pd.DataFrame({"Строка": df.loc[bad, "row"], "Причина": reasons[bad]}).reset_index(drop=True)
    good = df[~bad].copy()
    good["cabinet"] = np.where(good["cabinet"] == "", good["account"].map(guess_entity), good["cabinet"])
    good["income"], good["expense"] = good["income"].where(good["income"] > 0), good["expense"].where(good["expense"] > 0)
    unknown = sorted(set(good["article"]) - known_articles)
    new_acc = sorted(set(good["account"]) - known_accounts)
    plain = good[~good["comment"].str.contains("план", case=False, regex=False)]
    as_of = (plain if len(plain) else good)["dds_date"].max() if len(good) else None
    return ImportResult(good[INTERNAL].reset_index(drop=True), errors, [], unknown, new_acc, as_of)


def to_template() -> pd.DataFrame:
    """Empty journal in the expected layout with two example rows."""
    return pd.DataFrame([
        ["10.01.2026", "10.01.2026", 1500000, None, "Продажи Wildberries", "WB", "ООО Вуки", None, None, None, "ООО Вуки Точка ₽", "Выручка"],
        ["12.01.2026", "31.01.2026", None, 85000, "Аренда помещения", None, "ООО Вуки", None, None, None, "ООО Вуки Точка ₽", "Аренда за январь"],
    ], columns=JOURNAL_COLUMNS_RU)


def merge_journals(current: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append `new` to `current`, dropping operations that already exist (same dates, amounts, article, account, comment)."""
    both = pd.concat([current[INTERNAL], new[INTERNAL]], ignore_index=True)
    key = ["dds_date", "pl_date", "income", "expense", "article", "account", "comment"]
    return both.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def default_article_class(article: str) -> tuple[str, str, str]:
    """Starting classification for an unknown article (to be corrected by the user)."""
    return ("Не учитывается в P&L или учитывается автоматически", "", "Прочее")


def articles_to_add(names: list[str]) -> pd.DataFrame:
    rows = [(a, *default_article_class(a)) for a in names]
    return pd.DataFrame(rows, columns=["article", "pl_kind", "pl_category", "dds_group"])


def accounts_to_add(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"account": names, "account_group": [guess_group(n) for n in names],
                         "entity": [guess_entity(n) for n in names], "opening": 0.0})


def to_ru_columns(journal: pd.DataFrame) -> pd.DataFrame:
    out = journal[INTERNAL].copy()
    out.columns = JOURNAL_COLUMNS_RU
    return out
