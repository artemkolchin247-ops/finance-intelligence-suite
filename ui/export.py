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
