import sys
from pathlib import Path

import streamlit as st
import plotly.express as px

# Ensure repo root is on PYTHONPATH so `from src...` imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sec.sec_client import SECClient
from src.sec.ticker_map import SEC_TICKER_MAP_URL, parse_ticker_map
from src.sec.filings import SEC_SUBMISSIONS_URL, cik_to_10, extract_filings_table
from src.sec.xbrl_facts import SEC_COMPANYFACTS_URL, build_kpi_table


st.set_page_config(page_title="AI Earnings Engine (v0)", layout="wide")
st.title("AI Earnings Intelligence Engine (v0)")
st.caption("Live SEC filings + KPIs. Next: NLP + ML predictions + explainability.")

with st.sidebar:
    email = st.text_input("Email for SEC User-Agent", "akhetuamhen@gmail.com")
    throttle = st.slider("Throttle (seconds)", 0.10, 1.50, 0.50, 0.05)
    ua = f"AI Finance Suite (Thomas Selassie) {email}"

client = SECClient(ua, throttle_s=throttle)

@st.cache_data(ttl=24*3600)
def get_ticker_df() -> "pd.DataFrame":
    data = client.get_json(SEC_TICKER_MAP_URL)
    return parse_ticker_map(data)

tickers = get_ticker_df()

ticker = st.selectbox("Ticker", tickers["ticker"].tolist(), index=0)
row = tickers[tickers["ticker"] == ticker].iloc[0]
cik = int(row["cik"])
cik10 = cik_to_10(cik)

sub = client.get_json(SEC_SUBMISSIONS_URL.format(cik10=cik10))
facts = client.get_json(SEC_COMPANYFACTS_URL.format(cik10=cik10))

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Latest Filings")
    fdf = extract_filings_table(sub, cik)
    if fdf.empty:
        st.warning("No filings found.")
    else:
        show = fdf.copy()
        show["Open"] = show["url"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) else "")
        st.dataframe(show.head(20), use_container_width=True)

with right:
    st.subheader("KPIs (XBRL facts)")
    kpi = build_kpi_table(facts)
    if kpi.empty:
        st.warning("No KPI facts found for this ticker with the default tag set.")
    else:
        st.dataframe(kpi.tail(12), use_container_width=True)

if not kpi.empty and "Revenue" in kpi.columns:
    st.subheader("Revenue Trend")
    fig = px.line(kpi.dropna(subset=["Revenue"]), x="end_dt", y="Revenue")
    st.plotly_chart(fig, use_container_width=True)
