"""Global CSS: light business theme shared by all modules."""
import streamlit as st

CSS = """
<style>
:root {
  --bg:#F5F7FB; --card:#FFFFFF; --line:#E4E8F0; --ink:#1B2437; --muted:#6B7690;
  --primary:#2F5BEA; --good:#14935F; --bad:#D6404A; --warn:#C27A0E;
}
#MainMenu, footer, [data-testid="stToolbar"] {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}  /* keep the header: it holds the button that reopens the sidebar */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {visibility:visible !important;}
.block-container {padding-top:1.4rem; padding-bottom:2rem; max-width:1500px;}
section[data-testid="stSidebar"] {background:#FFFFFF; border-right:1px solid var(--line); min-width:300px !important;}
h1, h2, h3 {color:var(--ink); letter-spacing:-0.01em;}

.page-title {font-size:1.6rem; font-weight:700; color:var(--ink); margin:0;}
.page-sub {color:var(--muted); font-size:0.9rem; margin:2px 0 14px 0;}
.section-title {font-size:1.02rem; font-weight:650; color:var(--ink); margin:18px 0 8px 0;}

.kpi {background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 12px;
      height:100%; box-shadow:0 1px 2px rgba(20,30,60,0.04);}
.kpi .l {font-size:0.76rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; font-weight:600;}
.kpi .v {font-size:clamp(1.05rem, 1.35vw, 1.3rem); font-weight:700; color:var(--ink); margin:4px 0 2px 0; overflow-wrap:anywhere;}
.kpi .d {font-size:0.8rem; font-weight:600; color:var(--muted);}
.kpi .d.up {color:var(--good);} .kpi .d.down {color:var(--bad);}

div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:12px;}

.tbl-wrap {background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:auto;
           max-height:640px; box-shadow:0 1px 2px rgba(20,30,60,0.04);}
table.tbl {border-collapse:separate; border-spacing:0; width:100%; font-size:0.8rem; color:var(--ink);}
table.tbl th, table.tbl td {padding:5px 10px; border-bottom:1px solid #EEF1F6; white-space:nowrap; text-align:right;}
table.tbl thead th {position:sticky; top:0; background:#F0F3F9; z-index:3; font-weight:650; color:#3A4560;
                    border-bottom:1px solid var(--line);}
table.tbl th:first-child, table.tbl td:first-child {position:sticky; left:0; text-align:left; background:#fff;
                    z-index:2; min-width:260px; border-right:1px solid var(--line);}
table.tbl thead th:first-child {background:#F0F3F9; z-index:4;}
table.tbl td.total, table.tbl th.total {background:#F7F9FF; font-weight:650;}
table.tbl tr.sec td {background:#EAF0FF; color:#233A8C; font-weight:700; text-align:left; letter-spacing:0.02em;}
table.tbl tr.sec td:first-child {background:#EAF0FF;}
table.tbl tbody tr:hover td {background:#FAFBFE;}
table.tbl tbody tr:hover td:first-child {background:#FAFBFE;}
table.tbl td.neg {color:var(--bad);}
table.tbl td.na {color:#A3ACBF;}
.badge {display:inline-block; padding:1px 9px; border-radius:20px; font-weight:650; font-size:0.74rem;}
.badge.Да {background:#FDE8EA; color:var(--bad);}
.badge.Снизить {background:#FFF3DD; color:var(--warn);}
.badge.Нет {background:#E3F6EC; color:var(--good);}
.kpi .sp {margin-top:6px; opacity:.9;}
.story {background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 16px;
        box-shadow:0 1px 2px rgba(20,30,60,0.04);}
.story .t {font-size:0.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; font-weight:650; margin-bottom:6px;}
.story .it {display:flex; gap:10px; align-items:flex-start; padding:4px 0; font-size:0.9rem; color:var(--ink);}
.story .dot {width:9px; height:9px; border-radius:50%; margin-top:6px; flex:none; background:#A3ACBF;}
.story .dot.good {background:var(--good);} .story .dot.bad {background:var(--bad);}
.hcard {background:var(--card); border:1px solid var(--line); border-left:4px solid #F2A33A; border-radius:10px; padding:10px 14px; margin-bottom:8px;}
.hcard .h {font-weight:650; color:var(--ink);} .hcard .m {color:var(--muted); font-size:0.8rem; margin-top:2px;}
.chip {display:inline-block; padding:1px 9px; border-radius:20px; font-weight:650; font-size:0.74rem; margin-left:6px;}
.chip.work {background:#E8F0FE; color:#1a56c4;} .chip.ok {background:#E3F6EC; color:var(--good);} .chip.no {background:#FDE8EA; color:var(--bad);}
.fx {display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:10px;}
.fx .c {background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px;}
.fx .c .n {font-size:0.74rem; color:var(--muted);} .fx .c .a {font-size:1.05rem; font-weight:700; margin-top:2px;}
.fx .c .b {font-size:0.76rem; color:var(--muted);} .fx .c .p {font-weight:700; font-size:0.82rem;}
.fx .c.good .p {color:var(--good);} .fx .c.bad .p {color:var(--bad);}
.note {color:var(--muted); font-size:0.78rem; margin-top:6px;}
</style>
"""


def inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
