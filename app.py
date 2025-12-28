import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

import streamlit as st
import plotly.express as px


# =========================
# SEC Endpoints
# =========================
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


# =========================
# KPI Definitions (v1)
# =========================
@dataclass(frozen=True)
class KPIDefinition:
    label: str
    taxonomy: str
    tag: str
    unit_preference: Tuple[str, ...]


KPI_DEFS: List[KPIDefinition] = [
    KPIDefinition("Revenue", "us-gaap", "Revenues", ("USD",)),
    KPIDefinition("Gross Profit", "us-gaap", "GrossProfit", ("USD",)),
    KPIDefinition("Operating Income", "us-gaap", "OperatingIncomeLoss", ("USD",)),
    KPIDefinition("Net Income", "us-gaap", "NetIncomeLoss", ("USD",)),
    KPIDefinition("EPS Diluted", "us-gaap", "EarningsPerShareDiluted", ("USD/shares", "USD / shares", "USD")),
]


# =========================
# Utilities
# =========================
def cik_to_10(cik: int) -> str:
    return str(int(cik)).zfill(10)


def accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def build_filing_url(cik: int, accession: str, primary_doc: str) -> str:
    # Archives path uses cik WITHOUT leading zeros
    return f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=0.7,
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
    time.sleep(throttle_s)
    r = session.get(url, timeout=30)

    if r.status_code == 403:
        raise RuntimeError(
            "SEC returned 403 Forbidden.\n\n"
            "Fix:\n"
            "1) Make sure your User-Agent includes a real email in the sidebar.\n"
            "2) Deployments must also send the same User-Agent.\n"
            f"URL: {url}"
        )
    if r.status_code == 429:
        raise RuntimeError(
            "SEC returned 429 Too Many Requests.\n\n"
            "Fix:\n"
            "1) Increase the throttle slider (try 0.5–1.0s)\n"
            "2) Refresh less often\n"
            f"URL: {url}"
        )
    if r.status_code != 200:
        raise RuntimeError(f"SEC request failed ({r.status_code}). URL: {url}\nResponse: {r.text[:250]}")

    return r.json()


def parse_ticker_map(data: Any) -> pd.DataFrame:
    """
    SEC ticker map sometimes appears as a dict keyed by number-like strings,
    and sometimes as a list. Handle both.
    """
    rows = []

    if isinstance(data, dict):
        iterable = data.values()
    elif isinstance(data, list):
        iterable = data
    else:
        iterable = []

    for v in iterable:
        try:
            ticker = str(v.get("ticker", "")).upper()
            cik = v.get("cik_str", None)
            title = v.get("title", "")
            if ticker and cik is not None:
                rows.append({"ticker": ticker, "cik": int(cik), "title": title})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def pick_best_unit(units_dict: Dict[str, Any], unit_pref: Tuple[str, ...]) -> Optional[str]:
    for u in unit_pref:
        if u in units_dict:
            return u
    if len(units_dict) == 1:
        return next(iter(units_dict.keys()))
    return None


def label_period(end_dt: pd.Timestamp, frame: Optional[str]) -> str:
    if isinstance(frame, str) and frame.strip():
        return frame.strip()
    if pd.isna(end_dt):
        return "Unknown"
    q = ((end_dt.month - 1) // 3) + 1
    return f"{end_dt.year}Q{q}"


# =========================
# Cached fetchers
# =========================
@st.cache_data(ttl=24 * 3600)
def load_ticker_df(user_agent: str, throttle_s: float) -> pd.DataFrame:
    session = make_session(user_agent)
    data = sec_get_json(session, SEC_TICKER_MAP_URL, throttle_s)
    return parse_ticker_map(data)


@st.cache_data(ttl=6 * 3600)
def load_submissions(user_agent: str, cik10: str, throttle_s: float) -> Dict[str, Any]:
    session = make_session(user_agent)
    return sec_get_json(session, SEC_SUBMISSIONS_URL.format(cik10=cik10), throttle_s)


@st.cache_data(ttl=6 * 3600)
def load_companyfacts(user_agent: str, cik10: str, throttle_s: float) -> Dict[str, Any]:
    session = make_session(user_agent)
    return sec_get_json(session, SEC_COMPANYFACTS_URL.format(cik10=cik10), throttle_s)


# =========================
# Transforms
# =========================
def filings_table(sub_json: Dict[str, Any], cik: int) -> pd.DataFrame:
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

    df["url"] = df.apply(
        lambda r: build_filing_url(cik, r["accessionNumber"], r["primaryDocument"])
        if pd.notna(r.get("accessionNumber")) and pd.notna(r.get("primaryDocument"))
        else None,
        axis=1,
    )

    df = df.sort_values("filingDate", ascending=False).reset_index(drop=True)
    return df


def tag_series(companyfacts: Dict[str, Any], kpi: KPIDefinition) -> pd.DataFrame:
    tag_obj = (
        companyfacts.get("facts", {})
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

    # Keep only 10-Q / 10-K for stability
    if "form" in df.columns:
        df["form"] = df["form"].astype(str)
        df = df[df["form"].isin(["10-Q", "10-K"])].copy()
    else:
        df["form"] = None

    df["end_dt"] = pd.to_datetime(df.get("end"), errors="coerce")
    df["filed_dt"] = pd.to_datetime(df.get("filed"), errors="coerce")
    df["frame"] = df.get("frame")
    df["period"] = df.apply(lambda r: label_period(r["end_dt"], r.get("frame")), axis=1)

    # Keep latest filing for each end date (handles amendments/restatements best-effort)
    df = df.sort_values(["end_dt", "filed_dt"], ascending=[True, False])
    df = df.drop_duplicates(subset=["end_dt"], keep="first")

    out = df[["period", "end_dt", "form", "filed_dt", "val"]].copy()
    out = out.rename(columns={"val": kpi.label})
    return out.sort_values("end_dt").reset_index(drop=True)


def build_kpis(companyfacts: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    audit = []

    for k in KPI_DEFS:
        d = tag_series(companyfacts, k)
        if d.empty:
            audit.append({"KPI": k.label, "Status": "Missing", "Tag": f"{k.taxonomy}:{k.tag}"})
            continue
        audit.append({"KPI": k.label, "Status": "OK", "Tag": f"{k.taxonomy}:{k.tag}"})
        pieces.append(d)

    audit_df = pd.DataFrame(audit)

    if not pieces:
        return pd.DataFrame(), audit_df

    base = pieces[0]
    for p in pieces[1:]:
        base = base.merge(p, on=["period", "end_dt", "form", "filed_dt"], how="outer")

    base = base.sort_values("end_dt").reset_index(drop=True)
    base["Period End"] = base["end_dt"].dt.date

    # margins
    if "Revenue" in base.columns and "Gross Profit" in base.columns:
        base["Gross Margin"] = base["Gross Profit"] / base["Revenue"]
    if "Revenue" in base.columns and "Operating Income" in base.columns:
        base["Operating Margin"] = base["Operating Income"] / base["Revenue"]
    if "Revenue" in base.columns and "Net Income" in base.columns:
        base["Net Margin"] = base["Net Income"] / base["Revenue"]

    # qoq / yoy
    for col in ["Revenue", "Gross Profit", "Operating Income", "Net Income"]:
        if col in base.columns:
            base[f"{col} QoQ %"] = base[col].pct_change(1)
            base[f"{col} YoY %"] = base[col].pct_change(4)

    return base, audit_df


# =========================
# UI
# =========================
st.set_page_config(page_title="SEC Filings + KPI Tracker", layout="wide")

st.title("SEC Filings + Earnings KPI Tracker (Live)")
st.caption("Live EDGAR filings + XBRL facts → FP&A KPIs and trends")

with st.sidebar:
    st.header("SEC Access")
    email = st.text_input("Email for SEC User-Agent", value="akhetuamhen@gmail.com")
    throttle = st.slider("Throttle (seconds)", 0.10, 1.50, 0.35, 0.05)

    user_agent = f"SEC KPI Tracker (Thomas Selassie) {email}"

    if st.button("Clear cache + refresh"):
        st.cache_data.clear()
        st.rerun()

# Load ticker map
try:
    ticker_df = load_ticker_df(user_agent, throttle)
    if ticker_df.empty:
        st.error("Ticker map loaded but parsed empty. (Unexpected SEC format)")
        st.stop()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Company")
    ticker = st.selectbox("Ticker", ticker_df["ticker"].tolist(), index=0)
    row = ticker_df[ticker_df["ticker"] == ticker].iloc[0]
    cik = int(row["cik"])
    cik10 = cik_to_10(cik)
    st.write(f"**Company:** {row['title']}")
    st.write(f"**CIK:** {cik10}")

# Fetch SEC JSON
try:
    with st.spinner("Fetching SEC filings + XBRL facts..."):
        sub = load_submissions(user_agent, cik10, throttle)
        facts = load_companyfacts(user_agent, cik10, throttle)
except Exception as e:
    st.error(str(e))
    st.stop()

# Filings + KPIs
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Latest Filings (10-Q / 10-K / 8-K)")
    fdf = filings_table(sub, cik)
    if fdf.empty:
        st.warning("No filings returned in submissions->recent.")
    else:
        show = fdf.copy()
        show["Open"] = show["url"].apply(lambda u: f"[Open]({u})" if isinstance(u, str) else "")
        show = show.drop(columns=["url"])
        st.dataframe(show.head(25), use_container_width=True)

with right:
    st.subheader("Company Snapshot")
    st.write(f"**Name:** {sub.get('name', row['title'])}")
    st.write(f"**SIC:** {sub.get('sic', '—')}")
    st.write(f"**Industry:** {sub.get('sicDescription', '—')}")
    st.write(f"**Fiscal Year End:** {sub.get('fiscalYearEnd', '—')}")

st.divider()

kpi_df, audit_df = build_kpis(facts)
tabs = st.tabs(["KPIs", "Trends", "Data Audit", "Export"])

with tabs[0]:
    st.subheader("KPI Table")
    if kpi_df.empty:
        st.warning("No KPI facts found for this company with the current tag set. Try another ticker.")
    else:
        cols = [
            "Period End", "period", "form",
            "Revenue", "Gross Profit", "Operating Income", "Net Income",
            "Gross Margin", "Operating Margin", "Net Margin",
            "Revenue QoQ %", "Revenue YoY %",
        ]
        cols = [c for c in cols if c in kpi_df.columns]
        st.dataframe(kpi_df[cols].tail(20), use_container_width=True)

with tabs[1]:
    st.subheader("Trends")
    if kpi_df.empty:
        st.warning("No KPI data to chart.")
    else:
        metric_options = [k.label for k in KPI_DEFS if k.label in kpi_df.columns] + \
                         [m for m in ["Gross Margin", "Operating Margin", "Net Margin"] if m in kpi_df.columns]
        metric = st.selectbox("Metric", metric_options, index=0)
        plot_df = kpi_df.dropna(subset=["end_dt", metric]).copy()
        fig = px.line(plot_df, x="end_dt", y=metric, title=f"{ticker} — {metric}")
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Data Audit (Tags Used)")
    st.dataframe(audit_df, use_container_width=True)

with tabs[3]:
    st.subheader("Export")
    if kpi_df.empty:
        st.warning("Nothing to export yet.")
    else:
        st.download_button(
            "Download KPI CSV",
            data=kpi_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_sec_kpis.csv",
            mime="text/csv",
        )
