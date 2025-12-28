import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

import streamlit as st
import plotly.express as px


# =========================
# Config
# =========================
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"

SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_THROTTLE_SECONDS = 0.20  # be gentle with SEC


@dataclass(frozen=True)
class KPIDefinition:
    label: str
    taxonomy: str
    tag: str
    # list of unit strings you prefer, in order
    unit_preference: Tuple[str, ...]
    # if True, allow non-quarterly/annual facts (we keep False for stability)
    allow_other_forms: bool = False


# Core KPI set for v1 (reliable across many issuers)
KPI_DEFS: List[KPIDefinition] = [
    KPIDefinition("Revenue", "us-gaap", "Revenues", ("USD",)),
    KPIDefinition("Gross Profit", "us-gaap", "GrossProfit", ("USD",)),
    KPIDefinition("Operating Income", "us-gaap", "OperatingIncomeLoss", ("USD",)),
    KPIDefinition("Net Income", "us-gaap", "NetIncomeLoss", ("USD",)),
    KPIDefinition("EPS Diluted", "us-gaap", "EarningsPerShareDiluted", ("USD/shares", "USD / shares", "USD")),
]


# =========================
# HTTP + SEC etiquette
# =========================
def make_session(user_agent: str) -> requests.Session:
    """
    Create a requests Session with retries/backoff for stability.
    """
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return s


def sec_get_json(session: requests.Session, url: str, throttle_s: float) -> Dict[str, Any]:
    """
    SEC GET helper with polite throttling and strong error messages.
    """
    time.sleep(throttle_s)
    r = session.get(url, timeout=30)
    if r.status_code != 200:
        # show a helpful error
        raise RuntimeError(f"SEC request failed ({r.status_code}) for URL: {url}\nResponse: {r.text[:300]}")
    return r.json()


# =========================
# Helpers
# =========================
def cik_to_10(cik: int) -> str:
    return str(int(cik)).zfill(10)


def accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def build_filing_url(cik: int, accession: str, primary_doc: str) -> str:
    """
    Build the SEC Archives URL to the primary filing doc.
    """
    cik_int = int(cik)
    acc = accession_no_dashes(accession)
    return f"{SEC_ARCHIVES_BASE}/{cik_int}/{acc}/{primary_doc}"


def to_datetime_safe(x) -> pd.Timestamp:
    return pd.to_datetime(x, errors="coerce")


def label_period(row: pd.Series) -> str:
    """
    Prefer SEC frame when present. Else build label from end date:
    2024-09-28 -> 2024Q3 (calendar quarter label).
    """
    frame = row.get("frame")
    if isinstance(frame, str) and frame.strip():
        return frame.strip()

    end_dt = row.get("end_dt")
    if pd.isna(end_dt):
        return "Unknown"
    q = ((end_dt.month - 1) // 3) + 1
    return f"{end_dt.year}Q{q}"


def pick_best_unit(units_dict: Dict[str, Any], unit_pref: Tuple[str, ...]) -> Optional[str]:
    for u in unit_pref:
        if u in units_dict:
            return u
    # fallback: if only one unit exists, use it
    if len(units_dict) == 1:
        return next(iter(units_dict.keys()))
    return None


# =========================
# SEC data loaders
# =========================
@st.cache_data(ttl=24 * 3600)
def load_ticker_map(session_headers_key: str) -> pd.DataFrame:
    """
    Cached ticker map. session_headers_key is just to avoid caching across UAs.
    """
    # We'll fetch with plain requests here because it's a public SEC file,
    # but still use the User-Agent via Streamlit's cached dependency pattern.
    # (We pass 'session_headers_key' only to separate caches.)
    s = requests.Session()
    s.headers.update({"User-Agent": session_headers_key, "Accept": "application/json"})
    r = s.get(SEC_TICKER_MAP_URL, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for _, v in data.items():
        rows.append(
            {
                "ticker": str(v.get("ticker", "")).upper(),
                "cik": int(v.get("cik_str")),
                "title": v.get("title", ""),
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["ticker"].notna() & (df["ticker"] != "")]
    df = df.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    return df


@st.cache_data(ttl=6 * 3600)
def fetch_submissions(user_agent: str, cik10: str, throttle_s: float) -> Dict[str, Any]:
    session = make_session(user_agent)
    return sec_get_json(session, SEC_SUBMISSIONS_URL.format(cik10=cik10), throttle_s)


@st.cache_data(ttl=6 * 3600)
def fetch_companyfacts(user_agent: str, cik10: str, throttle_s: float) -> Dict[str, Any]:
    session = make_session(user_agent)
    return sec_get_json(session, SEC_COMPANYFACTS_URL.format(cik10=cik10), throttle_s)


# =========================
# Transformations
# =========================
def extract_filings_table(sub_json: Dict[str, Any], cik: int) -> pd.DataFrame:
    recent = sub_json.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if df.empty:
        return df

    keep = [c for c in ["filingDate", "reportDate", "form", "accessionNumber", "primaryDocument"] if c in df.columns]
    df = df[keep].copy()
    df = df[df["form"].isin(["10-Q", "10-K", "8-K"])].copy()

    df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce").dt.date
    df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce").dt.date

    # Build clickable url
    df["url"] = df.apply(
        lambda r: build_filing_url(cik, r["accessionNumber"], r["primaryDocument"])
        if pd.notna(r.get("accessionNumber")) and pd.notna(r.get("primaryDocument"))
        else None,
        axis=1,
    )

    return df.sort_values("filingDate", ascending=False).reset_index(drop=True)


def companyfacts_to_tag_df(
    facts_json: Dict[str, Any],
    kpi: KPIDefinition,
    keep_forms: Tuple[str, ...] = ("10-Q", "10-K"),
) -> pd.DataFrame:
    """
    Convert a KPI tag into tidy data:
    - Chooses best unit
    - Filters to 10-Q / 10-K
    - Keeps latest filed per period end (restatement-safe-ish)
    """
    tag_obj = (
        facts_json.get("facts", {})
        .get(kpi.taxonomy, {})
        .get(kpi.tag, {})
    )
    if not tag_obj:
        return pd.DataFrame()

    units_dict = tag_obj.get("units", {})
    if not units_dict:
        return pd.DataFrame()

    unit = pick_best_unit(units_dict, kpi.unit_preference)
    if not unit:
        return pd.DataFrame()

    rows = units_dict.get(unit, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure columns exist
    for col in ["end", "start", "fy", "fp", "form", "filed", "frame", "val"]:
        if col not in df.columns:
            df[col] = None

    df["form"] = df["form"].astype(str)
    df = df[df["form"].isin(keep_forms)].copy()

    df["end_dt"] = to_datetime_safe(df["end"])
    df["filed_dt"] = to_datetime_safe(df["filed"])
    df["period"] = df.apply(label_period, axis=1)

    # Keep latest filed per end date (best-effort restatement handling)
    df = df.sort_values(["end_dt", "filed_dt"], ascending=[True, False])
    df = df.drop_duplicates(subset=["end_dt"], keep="first")

    out = df[["period", "end_dt", "fy", "fp", "form", "filed_dt", "val"]].copy()
    out = out.rename(columns={"val": kpi.label})
    out["unit"] = unit
    out["tag"] = f"{kpi.taxonomy}:{kpi.tag}"
    return out.sort_values("end_dt").reset_index(drop=True)


def build_kpi_panel(facts_json: Dict[str, Any], kpis: List[KPIDefinition]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      (kpi_wide_df, audit_df)
    - wide table for analysis
    - audit table showing which tag/unit populated each KPI
    """
    pieces = []
    audit_rows = []

    for k in kpis:
        d = companyfacts_to_tag_df(facts_json, k)
        if d.empty:
            audit_rows.append({"KPI": k.label, "Status": "Missing", "Tag": f"{k.taxonomy}:{k.tag}", "Unit Used": None})
            continue

        audit_rows.append({"KPI": k.label, "Status": "OK", "Tag": d["tag"].iloc[0], "Unit Used": d["unit"].iloc[0]})
        pieces.append(d.drop(columns=["unit", "tag"]))

    audit_df = pd.DataFrame(audit_rows)

    if not pieces:
        return pd.DataFrame(), audit_df

    # Outer merge on end_dt/period etc. so we don't lose rows
    base = pieces[0]
    for p in pieces[1:]:
        base = base.merge(p, on=["period", "end_dt", "fy", "fp", "form", "filed_dt"], how="outer")

    base = base.sort_values("end_dt").reset_index(drop=True)
    base["Period End"] = base["end_dt"].dt.date

    # Derived metrics
    if "Revenue" in base.columns and "Gross Profit" in base.columns:
        base["Gross Margin"] = base["Gross Profit"] / base["Revenue"]
    if "Revenue" in base.columns and "Operating Income" in base.columns:
        base["Operating Margin"] = base["Operating Income"] / base["Revenue"]
    if "Revenue" in base.columns and "Net Income" in base.columns:
        base["Net Margin"] = base["Net Income"] / base["Revenue"]

    # QoQ/YoY (use end_dt order)
    for col in ["Revenue", "Gross Profit", "Operating Income", "Net Income"]:
        if col in base.columns:
            base[f"{col} QoQ %"] = base[col].pct_change(1)
            base[f"{col} YoY %"] = base[col].pct_change(4)

    # Separate quarterly vs annual views
    # SEC 'fp' is usually Q1/Q2/Q3/FY; not always present but helpful.
    return base, audit_df


def split_quarterly_annual(kpi_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if kpi_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    q = kpi_df[kpi_df["form"] == "10-Q"].copy()
    a = kpi_df[kpi_df["form"] == "10-K"].copy()
    return q.reset_index(drop=True), a.reset_index(drop=True)


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="SEC Filings + KPI Tracker", layout="wide")

st.title("SEC Filings + Earnings KPI Tracker (Live)")
st.caption("Live SEC EDGAR filings + extracted XBRL facts → FP&A KPI trends, margins, and variance views.")

with st.sidebar:
    st.header("SEC Access")
    email = st.text_input("Contact email (for SEC User-Agent)", value="akhetuamhen@gmail.com")
    throttle = st.slider("Request throttle (seconds)", 0.10, 1.00, DEFAULT_THROTTLE_SECONDS, 0.05)

    user_agent = f"SEC KPI Tracker (Thomas Selassie) {email}"

    st.header("Company")
    ticker_map = load_ticker_map(user_agent)
    ticker = st.selectbox("Ticker", ticker_map["ticker"].tolist(), index=0)

    row = ticker_map[ticker_map["ticker"] == ticker].iloc[0]
    cik = int(row["cik"])
    cik10 = cik_to_10(cik)

    st.write(f"**Company:** {row['title']}")
    st.write(f"**CIK:** {cik10}")

    if st.button("Clear cache + refresh"):
        st.cache_data.clear()
        st.rerun()

# Fetch data
try:
    with st.spinner("Fetching filings + XBRL facts from SEC..."):
        submissions = fetch_submissions(user_agent, cik10, throttle)
        facts = fetch_companyfacts(user_agent, cik10, throttle)
except Exception as e:
    st.error(str(e))
    st.stop()

# Filings
filings_df = extract_filings_table(submissions, cik)

c1, c2 = st.columns([1.1, 0.9])

with c1:
    st.subheader("Latest Filings")
    if filings_df.empty:
        st.warning("No recent filings found.")
    else:
        show = filings_df[["filingDate", "reportDate", "form", "accessionNumber", "primaryDocument", "url"]].copy()
        # Make URLs clickable by showing as markdown in a new column
        show["Open"] = show["url"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) else "")
        show = show.drop(columns=["url"])
        st.dataframe(show.head(25), use_container_width=True)

with c2:
    st.subheader("Company Snapshot")
    st.write(f"**Name:** {submissions.get('name', row['title'])}")
    st.write(f"**SIC:** {submissions.get('sic', '—')}")
    st.write(f"**Industry:** {submissions.get('sicDescription', '—')}")
    st.write(f"**Fiscal Year End:** {submissions.get('fiscalYearEnd', '—')}")

st.divider()

# KPIs
kpi_df, audit_df = build_kpi_panel(facts, KPI_DEFS)
q_df, a_df = split_quarterly_annual(kpi_df)

tabs = st.tabs(["Quarterly KPIs", "Annual KPIs", "Trends", "Data Audit", "Export"])

with tabs[0]:
    st.subheader("Quarterly (10-Q)")
    if q_df.empty:
        st.warning("No quarterly KPI rows (10-Q) were found for the current KPI set.")
    else:
        cols = ["Period End", "period", "Revenue", "Gross Profit", "Operating Income", "Net Income",
                "Gross Margin", "Operating Margin", "Net Margin",
                "Revenue QoQ %", "Revenue YoY %"]
        cols = [c for c in cols if c in q_df.columns]
        st.dataframe(q_df[cols].tail(16), use_container_width=True)

with tabs[1]:
    st.subheader("Annual (10-K)")
    if a_df.empty:
        st.warning("No annual KPI rows (10-K) were found for the current KPI set.")
    else:
        cols = ["Period End", "period", "Revenue", "Gross Profit", "Operating Income", "Net Income",
                "Gross Margin", "Operating Margin", "Net Margin"]
        cols = [c for c in cols if c in a_df.columns]
        st.dataframe(a_df[cols].tail(10), use_container_width=True)

with tabs[2]:
    st.subheader("KPI Trends")
    if kpi_df.empty:
        st.warning("No KPI facts found. Try another company.")
    else:
        metric_options = [k.label for k in KPI_DEFS if k.label in kpi_df.columns] + \
                         [c for c in ["Gross Margin", "Operating Margin", "Net Margin"] if c in kpi_df.columns]
        metric = st.selectbox("Metric", metric_options, index=0)

        plot_df = kpi_df.dropna(subset=["end_dt", metric]).copy()
        fig = px.line(plot_df, x="end_dt", y=metric, title=f"{ticker} — {metric}")
        st.plotly_chart(fig, use_container_width=True)

        # Optional: show QoQ/YoY if available
        yoy_col = f"{metric} YoY %"
        qoq_col = f"{metric} QoQ %"
        extra_cols = [c for c in ["Period End", metric, qoq_col, yoy_col] if c in kpi_df.columns]
        st.dataframe(kpi_df[extra_cols].tail(12), use_container_width=True)

with tabs[3]:
    st.subheader("Data Audit (Tags + Units)")
    st.caption("Shows which US-GAAP tags populated each KPI (audit-friendly).")
    st.dataframe(audit_df, use_container_width=True)

with tabs[4]:
    st.subheader("Export")
    if kpi_df.empty:
        st.warning("Nothing to export yet.")
    else:
        csv = kpi_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download KPI table (CSV)", csv, f"{ticker}_sec_kpis.csv", "text/csv")
