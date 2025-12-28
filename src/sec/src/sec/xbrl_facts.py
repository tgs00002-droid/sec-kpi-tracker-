from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


@dataclass(frozen=True)
class KPIDef:
    label: str
    taxonomy: str
    tag: str
    unit_preference: Tuple[str, ...]


KPI_DEFS: List[KPIDef] = [
    KPIDef("Revenue", "us-gaap", "Revenues", ("USD",)),
    KPIDef("Gross Profit", "us-gaap", "GrossProfit", ("USD",)),
    KPIDef("Operating Income", "us-gaap", "OperatingIncomeLoss", ("USD",)),
    KPIDef("Net Income", "us-gaap", "NetIncomeLoss", ("USD",)),
]


def _pick_unit(units: Dict[str, Any], pref: Tuple[str, ...]) -> Optional[str]:
    for u in pref:
        if u in units:
            return u
    if len(units) == 1:
        return next(iter(units.keys()))
    return None


def _label_period(end_dt: pd.Timestamp, frame: Optional[str]) -> str:
    if isinstance(frame, str) and frame.strip():
        return frame.strip()
    if pd.isna(end_dt):
        return "Unknown"
    q = ((end_dt.month - 1) // 3) + 1
    return f"{end_dt.year}Q{q}"


def tag_series(companyfacts: Dict[str, Any], k: KPIDef) -> pd.DataFrame:
    tag_obj = companyfacts.get("facts", {}).get(k.taxonomy, {}).get(k.tag, {})
    units = tag_obj.get("units", {})
    if not units:
        return pd.DataFrame()

    unit = _pick_unit(units, k.unit_preference)
    if not unit:
        return pd.DataFrame()

    df = pd.DataFrame(units.get(unit, []))
    if df.empty:
        return df

    df["form"] = df.get("form", "").astype(str)
    df = df[df["form"].isin(["10-Q", "10-K"])].copy()

    df["end_dt"] = pd.to_datetime(df.get("end"), errors="coerce")
    df["filed_dt"] = pd.to_datetime(df.get("filed"), errors="coerce")
    df["period"] = df.apply(lambda r: _label_period(r["end_dt"], r.get("frame")), axis=1)

    df = df.sort_values(["end_dt", "filed_dt"], ascending=[True, False])
    df = df.drop_duplicates(subset=["end_dt"], keep="first")

    out = df[["period", "end_dt", "form", "filed_dt", "val"]].copy()
    out = out.rename(columns={"val": k.label})
    return out.sort_values("end_dt").reset_index(drop=True)


def build_kpi_table(companyfacts: Dict[str, Any]) -> pd.DataFrame:
    pieces = []
    for k in KPI_DEFS:
        d = tag_series(companyfacts, k)
        if not d.empty:
            pieces.append(d)

    if not pieces:
        return pd.DataFrame()

    base = pieces[0]
    for p in pieces[1:]:
        base = base.merge(p, on=["period", "end_dt", "form", "filed_dt"], how="outer")

    base = base.sort_values("end_dt").reset_index(drop=True)
    base["Period End"] = base["end_dt"].dt.date

    if "Revenue" in base.columns and "Gross Profit" in base.columns:
        base["Gross Margin"] = base["Gross Profit"] / base["Revenue"]
    if "Revenue" in base.columns and "Operating Income" in base.columns:
        base["Operating Margin"] = base["Operating Income"] / base["Revenue"]
    if "Revenue" in base.columns and "Net Income" in base.columns:
        base["Net Margin"] = base["Net Income"] / base["Revenue"]

    return base
