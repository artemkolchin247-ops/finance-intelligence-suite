"""Excel export of report tables."""
from __future__ import annotations

from io import BytesIO
from typing import Mapping

import pandas as pd


def to_excel(sheets: Mapping[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def to_excel_styled(sheets: Mapping[str, tuple[pd.DataFrame, set[str]]], blank_rows: int = 30) -> bytes:
    """Template workbook: bold headers, input columns filled yellow (with spare rows below), readable widths."""
    from openpyxl.styles import Alignment, Font, PatternFill

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, (df, _) in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        yellow = PatternFill("solid", fgColor="FFF2B3")
        head = PatternFill("solid", fgColor="E8EEFB")
        for name, (df, inputs) in sheets.items():
            ws = xw.sheets[name[:31]]
            for ci, col in enumerate(df.columns, start=1):
                cell = ws.cell(row=1, column=ci)
                cell.font, cell.fill = Font(bold=True), head
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(60)]) + 2
                ws.column_dimensions[cell.column_letter].width = min(max(width, 12), 90 if name == "Как заполнять" else 40)
                if col in inputs:
                    for r in range(2, max(len(df) + 1, blank_rows if name in ("Доп. выручка", "Корректировки расходов") else len(df) + 1) + 1):
                        ws.cell(row=r, column=ci).fill = yellow
            if name == "Как заполнять":
                for r in range(2, len(df) + 2):
                    ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    return buf.getvalue()
