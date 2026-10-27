import numpy as np
import pandas as pd
import pytest

from core import journal_import as J
from core.ledger import ExcelLedgerRepository, prepare_journal

ARTICLES = {"Продажи Wildberries", "Аренда помещения", "РКО"}
ACCOUNTS = {"ООО Вуки Точка ₽"}


def _raw(**over):
    row = {"Дата ДДС\n(когда получены или потрачены деньги)": "10.01.2026", "Дата P&L\n(когда начислен доход или расход )": "31.01.2026",
           "Приход": None, "Расход": "85 000,50", "Статья операции": "Аренда помещения", "Касса / Счет": "ООО Вуки Точка ₽",
           "Комментарий": "Аренда за январь"}
    row.update(over)
    return pd.DataFrame([row])


def test_headers_with_line_breaks_and_brackets_are_recognised():
    m = J.match_headers(list(_raw().columns))
    assert set(m.values()) == {"dds_date", "pl_date", "income", "expense", "article", "account", "comment"}


def test_valid_row_is_converted_with_dates_numbers_and_entity():
    res = J.parse_journal(_raw(), ARTICLES, ACCOUNTS)
    assert res.ok and res.errors.empty and not res.unknown_articles and not res.new_accounts
    r = res.journal.iloc[0]
    assert r["dds_date"] == pd.Timestamp("2026-01-10") and r["pl_date"] == pd.Timestamp("2026-01-31")  # day first
    assert r["expense"] == pytest.approx(85000.5) and pd.isna(r["income"])
    assert r["cabinet"] == "ООО Вуки" and r["mp"] == ""  # optional columns are filled


def test_missing_required_column_is_reported():
    res = J.parse_journal(_raw().drop(columns=["Касса / Счет"]), ARTICLES, ACCOUNTS)
    assert not res.ok and res.missing_columns


def test_bad_rows_are_rejected_with_reasons_and_good_rows_kept():
    raw = pd.concat([_raw(), _raw(**{"Расход": None}), _raw(Приход="100", Расход="50"), _raw(**{"Дата ДДС\n(когда получены или потрачены деньги)": "не дата"}),
                     _raw(**{"Расход": "-5"})], ignore_index=True)
    res = J.parse_journal(raw, ARTICLES, ACCOUNTS)
    assert len(res.journal) == 1
    reasons = dict(zip(res.errors["Строка"], res.errors["Причина"]))
    assert reasons[3] == "не заполнены ни приход, ни расход" and reasons[4] == "заполнены и приход, и расход"
    assert "Дата ДДС" in reasons[5] and reasons[6] == "отрицательная сумма"


def test_unknown_articles_accounts_and_as_of():
    raw = pd.concat([_raw(), _raw(**{"Статья операции": "Новая статья", "Касса / Счет": "ИП Депозит Фонд", "Комментарий": "НАЛОГ ПЛАНОВО",
                                    "Дата ДДС\n(когда получены или потрачены деньги)": "01.12.2026"})], ignore_index=True)
    res = J.parse_journal(raw, ARTICLES, ACCOUNTS)
    assert res.unknown_articles == ["Новая статья"] and res.new_accounts == ["ИП Депозит Фонд"]
    assert res.as_of == pd.Timestamp("2026-01-10")  # planned rows do not move the as-of date
    acc = J.accounts_to_add(res.new_accounts).iloc[0]
    assert acc["entity"] == J.ENTITY_IP and acc["account_group"] == "Депозиты"
    assert J.guess_group("Фонд НДС") == "Фонды" and J.guess_group("$ Карта") == "Карты и личные счета"


def test_merge_skips_duplicates_and_prepared_journal_is_usable():
    res = J.parse_journal(pd.concat([_raw(), _raw(Расход="1000")], ignore_index=True), ARTICLES, ACCOUNTS)
    merged = J.merge_journals(res.journal.iloc[:1], res.journal)
    assert len(merged) == 2
    arts = pd.DataFrame({"article": ["Аренда помещения"], "pl_kind": ["Косвенные расходы (до EBITDA)"],
                         "pl_category": ["Общепроизводственные расходы"], "dds_group": ["Аренда и коммунальные платежи"]})
    prepared = prepare_journal(merged, arts, pd.Timestamp("2026-09-06"))
    assert prepared["cash"].sum() == pytest.approx(-86000.5)


def test_exported_journal_can_be_imported_back():
    repo = ExcelLedgerRepository()
    j = prepare_journal(repo.get_journal(), repo.get_articles(), repo.get_as_of())
    raw = J.to_ru_columns(j)
    res = J.parse_journal(raw, set(repo.get_articles()["article"]), set(repo.get_accounts()["account"]))
    assert res.errors.empty and len(res.journal) == len(j) and not res.unknown_articles and not res.new_accounts
    assert res.journal["income"].sum() == pytest.approx(j["income"].sum())
