from typing import Any
import pandas as pd

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def parse_ticker_map(data: Any) -> pd.DataFrame:
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
