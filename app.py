"""Entry point: navigation shell for the Finance Intelligence Suite."""
import streamlit as st

from ui.styles import inject_styles

st.set_page_config(page_title="Finance Intelligence Suite", page_icon="📊", layout="wide")
inject_styles()
st.logo("assets/logo.svg", size="large")

navigation = st.navigation(
    [
        st.Page("views/rnp.py", title="РНП", icon=":material/table_chart:", default=True),
        st.Page("views/hypotheses.py", title="Гипотезы", icon=":material/lightbulb:"),
    ]
)
navigation.run()
