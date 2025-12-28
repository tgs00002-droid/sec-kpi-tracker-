import pandas as pd

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"


def cik_to_10(cik: int) -> str:
    return str(int(cik)).zfill(10)


def accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def build_filing_url(cik: int, accession: str, primary_doc: str) -> str:
    return f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_no_dashes(accession)}/{primary_doc}"


def extract_filings_table(sub_json: dict, cik: int) -> pd.DataFrame:
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
        if pd.notna(r.get("accessionNumber")) and pd.notna(r.get("primaryDocument")) else None,
        axis=1,
    )

    return df.sort_values("filingDate", ascending=False).reset_index(drop=True)
