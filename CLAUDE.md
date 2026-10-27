# Finance Intelligence Suite

Streamlit report platform for marketplace (WB / Ozon) sales: РНП (daily performance table) and a hypotheses log linked to it.
Data lives in `data/finance_db.xlsx` behind `DataRepository` (drop-in replaceable by PostgreSQL).

## Commands
- Run: `streamlit run app.py`
- Tests: `pytest -q`
- Regenerate data: `python data/generate_data.py` (seeded, deterministic; bakes hypothesis effects into the data)

## Architecture rules
- `core/` = pure logic (pandas/numpy only). No `streamlit` imports there.
- `ui/` and `views/` = presentation only; no formulas inline.
- Data access only through `core/data_repo.py` (`DataRepository`).
- Every division goes through `core.formatting.safe_div` (zero denominator -> NaN, rendered as "—").
- Ratios are recomputed from summed raw fields, never averaged.
- The report logic is a port of the production script `rnp_daily.gs` (local file, git-ignored: it contains DB credentials,
  never copy anything from it into code, docs or commits). Formulas in `core/rnp_calc.py` must stay in line with README.

## Conventions
- Code, comments, identifiers in English; user-facing UI text in Russian.
- Money in KPIs/tables: тыс. ₽ (kind "k"). Per-unit values (check, CPC, CPO): ₽.
- UI must never mention "demo", "synthetic", "test data".
- Change a formula => update tests and README.
- Hypotheses are stored in the `hypotheses` sheet; the РНП reads them by date + marketplace + model.
