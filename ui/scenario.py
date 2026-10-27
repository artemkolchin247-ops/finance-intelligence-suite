"""Scenario switch of the forecast (the only control of the plan inside the application)."""
from __future__ import annotations

import streamlit as st

from core.forecast import SCENARIOS


def scenario_select(prefix: str) -> str:
    """Segmented switch: pessimistic / base / optimistic growth fading (lambda 0.45 / 0.70 / 0.90)."""
    store = st.session_state.setdefault("scn", {})
    names = list(SCENARIOS)
    key = f"{prefix}_scenario"
    st.session_state.setdefault(key, store.get("scenario", "Базовый"))  # widget values are dropped when their page is not shown
    st.session_state[key] = st.session_state[key] if st.session_state[key] in names else store.get("scenario", "Базовый")
    choice = st.segmented_control("Сценарий", names, key=key,
                                  help="Насколько сохранится нынешний рост через год: пессимистичный 45%, базовый 70%, оптимистичный 90%")
    store["scenario"] = choice or store.get("scenario", "Базовый")
    return store["scenario"]
