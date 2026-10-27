# Finance Intelligence Suite

Streamlit report platform for marketplace (WB / Ozon): РНП, hypotheses log, ОПиУ, ДДС (12-month forecast), payment calendar (12 weeks), management balance.
Sales data lives in `data/finance_db.xlsx` behind `DataRepository`, finance data in `data/finance_ledger.xlsx` behind `LedgerRepository` (both drop-in replaceable by PostgreSQL).

## Commands
- Run: `streamlit run app.py`
- Tests: `pytest -q`
- Regenerate data (in this order): `python data/generate_data.py` (sales, seeded; bakes hypothesis effects in), then `python data/generate_ledger.py` (journal etc., derived from the sales data)

## Architecture rules
- `core/` = pure logic (pandas/numpy only). No `streamlit` imports there.
- `ui/` and `views/` = presentation only; no formulas inline.
- Data access only through `core/data_repo.py` (`DataRepository`).
- Every division goes through `core.formatting.safe_div` (zero denominator -> NaN, rendered as "—").
- Ratios are recomputed from summed raw fields, never averaged.
- Forecast (`core/forecast.py`): GMV = same month last year × fading YoY (scenario λ), receipts = GMV × buyout × (1 − marketplace costs); expenses = 3-month average + corrections. Plan inputs are edited ONLY via the Excel template (no editing widgets in the app); the sole in-app control is the scenario switch.
- ДДС and calendar come from ONE schedule (`core/cashflow.py: build_schedule`); never compute them separately.
- ОПиУ marketplace lines come from `fact_monthly` (same raw fields as the РНП), the rest from the journal by accrual date; keep `margin == sum(margin_rub)`.
- Real source files (`Журнал операций.xlsx`, `P&L.xlsx`, `rnp_daily.gs`) are git-ignored: never copy their contents (names, INN, card masks, credentials) into code, data or docs.
- The report logic is a port of the production script `rnp_daily.gs` (local file, git-ignored: it contains DB credentials,
  never copy anything from it into code, docs or commits). Formulas in `core/rnp_calc.py` must stay in line with README.

## Conventions
- Code, comments, identifiers in English; user-facing UI text in Russian.
- Money in KPIs/tables: тыс. ₽ (kind "k"). Per-unit values (check, CPC, CPO): ₽.
- UI must never mention "demo", "synthetic", "test data".
- Change a formula => update tests and README.
- Hypotheses are stored in the `hypotheses` sheet; the РНП reads them by date + marketplace + model.
