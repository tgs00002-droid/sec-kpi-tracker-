import time
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

import streamlit as st
import plotly.express as px

# =========================
# CONFIG
# =========================
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

DEFAULT_THROTTLE = 0.5

# =========================
# SEC CLIENT
# =========================
def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def sec_get_json(session: requests.Session, url: str, throttle: float) -> Dict[str, Any]:
    time.sleep(throttle)
    r = session.get(url, timeout=30)

    if r.status_code == 403:
        raise RuntimeError("403 Forbidden from SEC. Use a real email in User-Agent.")
    if r.status_code == 429:
        raise RuntimeError("429 Too Many Requests. Increase throttle.")
    if r.status_code != 200:
        raise RuntimeError(f"SEC error {r.status_code}: {r.text[:200]}")

    return r.json()

# =========================
# HELPERS
# =========================
def cik_to_10(cik: int) -> str:
    return str(int(cik)).zfill(10)


def accession_no_dashes(acc: str) -> str:
    return acc.replace("-", "")


def filing_url(cik: int, accession: str, doc: str) -> str:
    return f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_no_dashes(accession)}/{doc}"


def parse_ticker_map(data: Any) -> pd.DataFrame:
    rows = []
    iterable = data.values() if isinstance(data, dict) else data

    for v in iterable:
        ticker = str(v.get("ticker", "")).upper()
        cik = v.get("cik_str")
        title = v.get("title", "")
        if ticker and cik is not None:
            rows.append({"ticker": ticker, "cik": int(cik), "title": title})

    df = pd.DataFrame(rows)
    return df.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def label_period(end_dt: pd.Timestamp, frame: Optional[str]) -> str:
    if isinstance(frame, str) and frame:
        return frame
    if pd.isna(end_dt):
        return "Unknown"
    q = ((end_dt.month - 1) // 3) + 1
    return f"{end_dt.year}Q{q}"

# =========================
# DATA LOADERS
# =========================
@st.cache_data(ttl=24 * 3600)
def load_tickers(user_agent: str, throttle: float) -> pd.DataFrame:
    session = make_session(user_agent)
    data = sec_get_json(session, SEC_TICKER_MAP_URL, throttle)
    return parse_ticker_map(data)


def load_filings(session, cik10, throttle):
    return sec_get_json(session, SEC_SUBMISSIONS_URL.format(cik10=cik10), throttle)


def load_companyfacts(session, cik10, throttle):
    return sec_get_json(session, SEC_COMPANYFACTS_URL.format(cik10=cik10), throttle)

# =========================
# TRANSFORMS
# =========================
def filings_table(sub_json: Dict[str, Any], cik: int) -> pd.DataFrame:
    recent = sub_json.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df

    keep = ["filingDate", "reportDate", "form", "accessionNumber", "primaryDocument"]
    df = df[keep]
    df = df[df["form"].isin(["10-Q", "10-K", "8-K"])]

    df["filingDate"] = pd.to_datetime(df["filingDate"]).dt.date
    df["reportDate"] = pd.to_datetime(df["reportDate"]).dt.date

    df["url"] = df.apply(
        lambda r: filing_url(cik, r["accessionNumber"], r["primaryDocument"]),
        axis=1,
    )

    return df.sort_values("filingDate", ascending=False).reset_index(drop=True)


def build_kpis(facts: Dict[str, Any]) -> pd.DataFrame:
    TAGS = {
        "Revenue": "Revenues",
        "Gross Profit": "GrossProfit",
        "Operating Income": "OperatingIncomeLoss",
        "Net Income": "NetIncomeLoss",
    }

    pieces = []

    for label, tag in TAGS.items():
        tag_obj = facts.get("facts", {}).get("us-gaap", {}).get(tag, {})
        units = tag_obj.get("units", {})
        if "USD" not in units:
            continue

        df = pd.DataFrame(units["USD"])
        if df.empty:
            continue

        df["form"] = df["form"].astype(str)
        df = df[df["form"].isin(["10-Q", "10-K"])]

        df["end_dt"] = pd.to_datetime(df["end"], errors="coerce")
        df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce")
        df["period"] = df.apply(lambda r: label_period(r["end_dt"], r.get("frame")), axis=1)

        df = df.sort_values(["end_dt", "filed_dt"])
        df = df.drop_duplicates(subset=["end_dt"], keep="last")

        out = df[["period", "end_dt", "form", "val"]].rename(columns={"val": label})
        pieces.append(out)

    if not pieces:
        return pd.DataFrame()

    base = pieces[0]
    for p in pieces[1:]:
        base = base.merge(p, on=["period", "end_dt", "form"], how="outer")

    base = base.sort_values("end_dt").reset_index(drop=True)
    base["Period End"] = base["end_dt"].dt.date

    if "Revenue" in base and "Gross Profit" in base:
        base["Gross Margin"] = base["Gross Profit"] / base["Revenue"]
    if "Revenue" in base and "Operating Income" in base:
        base["Operating Margin"] = base["Operating Income"] / base["Revenue"]
    if "Revenue" in base and "Net Income" in base:
        base["Net Margin"] = base["Net Income"] / base["Revenue"]

    return base

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="AI Earnings Engine", layout="wide")
st.title("AI Earnings Intelligence Engine")
st.caption("Live SEC filings + KPIs. Next: NLP + ML predictions.")

with st.sidebar:
    email = st.text_input("Email for SEC User-Agent", "your_email@gmail.com")
    throttle = st.slider("Throttle (seconds)", 0.2, 1.5, DEFAULT_THROTTLE, 0.1)
    user_agent = f"AI Earnings Engine ({email})"

session = make_session(user_agent)

tickers = load_tickers(user_agent, throttle)
ticker = st.selectbox("Ticker", tickers["ticker"].tolist(), index=0)

row = tickers[tickers["ticker"] == ticker].iloc[0]
cik = int(row["cik"])
cik10 = cik_to_10(cik)

sub = load_filings(session, cik10, throttle)
facts = load_companyfacts(session, cik10, throttle)

left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("Latest Filings")
    fdf = filings_table(sub, cik)
    fdf["Open"] = fdf["url"].apply(lambda u: f"[Open]({u})")
    st.dataframe(fdf.drop(columns=["url"]).head(20), use_container_width=True)

with right:
    st.subheader("KPIs")
    kpi = build_kpis(facts)
    if kpi.empty:
        st.warning("No KPI data found.")
    else:
        st.dataframe(kpi.tail(12), use_container_width=True)

if not kpi.empty and "Revenue" in kpi:
    st.subheader("Revenue Trend")
    fig = px.line(kpi.dropna(subset=["Revenue"]), x="end_dt", y="Revenue")
    st.plotly_chart(fig, use_container_width=True)
