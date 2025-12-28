import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
import plotly.express as px


# ----------------------------
# SEC API settings (IMPORTANT)
# ----------------------------
# SEC asks for a descriptive User-Agent with contact info for scripted requests.
# Use your real email.
SEC_USER_AGENT = "SEC KPI Tracker (Thomas Selassie) akhetuamhen@gmail.com"
HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
    "Accept": "application/json",
}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(ttl=24 * 3600)
def load_ticker_map() -> pd.DataFrame:
    """
    Loads SEC ticker->CIK mapping.
    """
    r = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    # company_tickers.json is a dict keyed by integer-like strings
    rows = []
    for _, v in data.items():
        rows.append(
            {
                "ticker": str(v.get("ticker", "")).upper(),
                "cik": int(v.get("cik_str")),
                "title": v.get("title", ""),
            }
        )
    df = pd.DataFrame(rows).dropna()
    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"])
    return df.sort_values("ticker").reset_index(drop=True)


def cik_to_10(cik: int) -> str:
    return str(cik).zfill(10)


@st.cache_data(ttl=6 * 3600)
def get_submissions(cik10: str) -> Dict[str, Any]:
    url = SUBMISSIONS_URL.format(cik10=cik10)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=6 * 3600)
def get_companyfacts(cik10: str) -> Dict[str, Any]:
    url = COMPANYFACTS_URL.format(cik10=cik10)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_recent_filings(sub_json: Dict[str, Any], forms: Tuple[str, ...] = ("10-Q", "10-K", "8-K")) -> pd.DataFrame:
    """
    Build a table from submissions->filings->recent.
    """
    recent = sub_json.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if df.empty:
        return df

    # Keep key columns (some may not exist for every issuer)
    keep_cols = [c for c in ["filingDate", "reportDate", "form", "accessionNumber", "primaryDocument"] if c in df.columns]
    df = df[keep_cols].copy()

    df["form"] = df["form"].astype(str)
    df = df[df["form"].isin(forms)].copy()

    # Accession link helper (to SEC Archives)
    # Example archive URL pattern uses CIK (no leading zeros) + accession without dashes.
    return df.sort_values("filingDate", ascending=False).reset_index(drop=True)


def safe_get_fact_units(companyfacts: Dict[str, Any], taxonomy: str, tag: str) -> Dict[str, List[Dict[str, Any]]]:
    return (
        companyfacts.get("facts", {})
        .get(taxonomy, {})
        .get(tag, {})
        .get("units", {})
    )


def facts_to_series(companyfacts: Dict[str, Any], taxonomy: str, tag: str, unit_preference: Tuple[str, ...]) -> pd.DataFrame:
    """
    Convert SEC companyfacts for a tag into a tidy DataFrame.
    We prefer rows with a 'frame' like CY2024Q3 for clean quarterly labeling.
    """
    units = safe_get_fact_units(companyfacts, taxonomy, tag)

    chosen_unit = None
    for u in unit_preference:
        if u in units:
            chosen_unit = u
            break

    if not chosen_unit:
        return pd.DataFrame()

    rows = units[chosen_unit]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep columns we care about
    for col in ["end", "start", "fy", "fp", "form", "filed", "frame", "val"]:
        if col not in df.columns:
            df[col] = None

    # Filter for quarterly/annual forms we want
    df["form"] = df["form"].astype(str)
    df = df[df["form"].isin(["10-Q", "10-K"])].copy()

    # Frame is best; fallback to end date
    df["period"] = df["frame"].fillna(df["end"])
    df["end_dt"] = pd.to_datetime(df["end"], errors="coerce")

    # Deduplicate by period keeping latest filed
    df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.sort_values(["period", "filed_dt"], ascending=[True, False])
    df = df.drop_duplicates(subset=["period"], keep="first")

    df = df[["period", "end_dt", "fy", "fp", "form", "filed_dt", "val"]].copy()
    df = df.rename(columns={"val": tag})
    return df.sort_values("end_dt").reset_index(drop=True)


def build_kpi_table(companyfacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a quarterly KPI table from common US-GAAP tags.
    """
    # Core tags (work across many issuers)
    tag_map = {
        "Revenue": ("us-gaap", "Revenues", ("USD",)),
        "Gross Profit": ("us-gaap", "GrossProfit", ("USD",)),
        "Operating Income": ("us-gaap", "OperatingIncomeLoss", ("USD",)),
        "Net Income": ("us-gaap", "NetIncomeLoss", ("USD",)),
        # Shares/EPS can be messy across companies; included but may be missing.
        "EPS Diluted": ("us-gaap", "EarningsPerShareDiluted", ("USD/shares", "USD / shares", "USD")),
    }

    dfs = []
    for label, (tax, tag, units) in tag_map.items():
        df_tag = facts_to_series(companyfacts, tax, tag, units)
        if not df_tag.empty:
            df_tag = df_tag.rename(columns={tag: label})
            dfs.append(df_tag[["period", "end_dt", "fy", "fp", "form", "filed_dt", label]])

    if not dfs:
        return pd.DataFrame()

    # Merge on period/end_dt (outer merge to keep periods even if some tags missing)
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on=["period", "end_dt", "fy", "fp", "form", "filed_dt"], how="outer")

    out = out.sort_values("end_dt").reset_index(drop=True)

    # Compute margins when possible
    if "Revenue" in out.columns and "Gross Profit" in out.columns:
        out["Gross Margin"] = out["Gross Profit"] / out["Revenue"]
    if "Revenue" in out.columns and "Operating Income" in out.columns:
        out["Operating Margin"] = out["Operating Income"] / out["Revenue"]
    if "Revenue" in out.columns and "Net Income" in out.columns:
        out["Net Margin"] = out["Net Income"] / out["Revenue"]

    # QoQ and YoY changes (simple)
    for col in ["Revenue", "Gross Profit", "Operating Income", "Net Income"]:
        if col in out.columns:
            out[f"{col} QoQ %"] = out[col].pct_change(1)
            out[f"{col} YoY %"] = out[col].pct_change(4)

    # Nice date label
    out["Period End"] = out["end_dt"].dt.date
    return out


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="SEC Filings + KPI Tracker", layout="wide")

st.title("SEC Filings + Earnings KPI Tracker (Live)")
st.caption("Pulls latest filings + XBRL facts from SEC EDGAR APIs and computes FP&A KPIs.")

with st.sidebar:
    st.header("Company")
    ticker_df = load_ticker_map()
    ticker = st.selectbox("Select ticker", ticker_df["ticker"].tolist(), index=0)
    company_row = ticker_df[ticker_df["ticker"] == ticker].iloc[0]
    cik10 = cik_to_10(int(company_row["cik"]))
    st.write(f"**Company:** {company_row['title']}")
    st.write(f"**CIK:** {cik10}")

    st.header("Refresh")
    if st.button("Clear cache + refresh"):
        st.cache_data.clear()
        st.experimental_rerun()

# Pull data
with st.spinner("Loading filings and XBRL facts from SEC..."):
    submissions = get_submissions(cik10)
    facts = get_companyfacts(cik10)

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Latest Filings (10-Q / 10-K / 8-K)")
    filings_df = extract_recent_filings(submissions)
    if filings_df.empty:
        st.warning("No recent filings found for this ticker via submissions endpoint.")
    else:
        st.dataframe(filings_df.head(25), use_container_width=True)

with col2:
    st.subheader("KPI Table (Quarterly/Annual from XBRL Facts)")
    kpi = build_kpi_table(facts)

    if kpi.empty:
        st.warning("No KPI facts found (or tags missing). Try another company or expand tag mapping.")
    else:
        # Display a compact view
        display_cols = ["Period End", "form", "Revenue", "Gross Profit", "Operating Income", "Net Income",
                        "Gross Margin", "Operating Margin", "Net Margin",
                        "Revenue QoQ %", "Revenue YoY %"]
        display_cols = [c for c in display_cols if c in kpi.columns]
        st.dataframe(kpi[display_cols].tail(12), use_container_width=True)

st.divider()

if not kpi.empty and "Revenue" in kpi.columns:
    st.subheader("Trends")
    tcol1, tcol2, tcol3 = st.columns([1, 1, 1])

    with tcol1:
        fig = px.line(kpi.dropna(subset=["Revenue"]), x="end_dt", y="Revenue", title="Revenue")
        st.plotly_chart(fig, use_container_width=True)

    if "Operating Income" in kpi.columns:
        with tcol2:
            fig = px.line(kpi.dropna(subset=["Operating Income"]), x="end_dt", y="Operating Income", title="Operating Income")
            st.plotly_chart(fig, use_container_width=True)

    if "Net Income" in kpi.columns:
        with tcol3:
            fig = px.line(kpi.dropna(subset=["Net Income"]), x="end_dt", y="Net Income", title="Net Income")
            st.plotly_chart(fig, use_container_width=True)

if not kpi.empty:
    st.subheader("Export")
    csv_bytes = kpi.to_csv(index=False).encode("utf-8")
    st.download_button("Download KPI CSV", data=csv_bytes, file_name=f"{ticker}_kpis.csv", mime="text/csv")
