"""Entry point: navigation shell for the Finance Intelligence Suite."""
import streamlit as st

from ui.styles import inject_styles

st.set_page_config(page_title="Finance Intelligence Suite", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_styles()
st.logo("assets/logo.svg", size="large")

navigation = st.navigation(
    {
        "Продажи": [
            st.Page("views/rnp.py", title="РНП", icon=":material/table_chart:", default=True),
            st.Page("views/hypotheses.py", title="Гипотезы", icon=":material/lightbulb:"),
        ],
        "Финансы": [
            st.Page("views/journal.py", title="Журнал операций", icon=":material/receipt_long:"),
            st.Page("views/pnl.py", title="ОПиУ", icon=":material/account_balance_wallet:"),
            st.Page("views/planning.py", title="Планирование", icon=":material/edit_calendar:"),
            st.Page("views/dds.py", title="ДДС", icon=":material/swap_vert:"),
            st.Page("views/payments.py", title="Платёжный календарь", icon=":material/calendar_month:"),
            st.Page("views/balance.py", title="Баланс", icon=":material/balance:"),
        ],
    }
)
navigation.run()
