"""Load, clean and split the Midtown-to-JFK trips (2026-11-08 entries in decisions.md).

The cleaning rules are the same for all three months. Nothing here looks at March
beyond the record-quality flags.
"""

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

import events

DATA = Path(__file__).parent / "data"
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-{m}.parquet"
SHA256 = {
    "01": "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510",
    "02": "c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20",
    "03": "2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823",
}
MONTHS = ("01", "02", "03")
PICKUP_ZONES = [161, 162, 163, 230]
JFK = 132
MAX_MINUTES = 360
COLUMNS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
    "trip_distance", "RatecodeID", "fare_amount", "total_amount",
]
# Pickup date, New York local time as the TLC records it; each start is inclusive, each end exclusive.
SPLIT = {
    "fit": ("2024-01-01", "2024-02-01"),
    "calibrate": ("2024-02-01", "2024-03-01"),
    "test": ("2024-03-01", "2024-04-01"),
}


def load_month(m):
    """One month's Midtown-to-JFK records, downloaded if missing and checked against the documented hash."""
    path = DATA / f"yellow_tripdata_2024-{m}.parquet"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(m=m), path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256[m]:
        raise ValueError(f"{path.name} differs from the documented snapshot")
    raw = pd.read_parquet(path, columns=COLUMNS)
    start = pd.Timestamp(f"2024-{m}-01")
    keep = (
        raw["PULocationID"].isin(PICKUP_ZONES)
        & raw["DOLocationID"].eq(JFK)
        & raw["tpep_pickup_datetime"].ge(start)
        & raw["tpep_pickup_datetime"].lt(start + pd.offsets.MonthBegin())
    )
    return raw.loc[keep].reset_index(drop=True)


def flag(trips):
    """Adds duration_minutes and the cleaning flags; drops nothing."""
    out = trips.copy()
    out["duration_minutes"] = (
        out["tpep_dropoff_datetime"] - out["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    # A reversal is the voided copy of a trip: same times, zones and distance, negative fare or total.
    out["flag_reversal"] = out["fare_amount"].lt(0) | out["total_amount"].lt(0)
    out["flag_nonpositive"] = ~out["duration_minutes"].gt(0)
    # Meters left running: not road time.
    out["flag_over_6h"] = out["duration_minutes"].gt(MAX_MINUTES)
    return out


def records(months=MONTHS):
    """All selected records for the months, flagged, nothing dropped."""
    return flag(pd.concat([load_month(m) for m in months], ignore_index=True))


def clean(trips, drop_over_6h=True):
    """Drops reversal copies and non-positive durations, and over-six-hour meters unless drop_over_6h=False."""
    keep = ~trips["flag_reversal"] & ~trips["flag_nonpositive"]
    if drop_over_6h:
        keep &= ~trips["flag_over_6h"]
    return trips.loc[keep].reset_index(drop=True)


def split_of(trips):
    """'fit', 'calibrate' or 'test' by pickup time (cutoffs in SPLIT)."""
    s = pd.Series(pd.NA, index=trips.index, dtype="string")
    for name, (start, end) in SPLIT.items():
        s[trips["tpep_pickup_datetime"].ge(start) & trips["tpep_pickup_datetime"].lt(end)] = name
    return s


def trips(flagged=None, drop_over_6h=True):
    """Cleaned trips with split, flagged_day and flag_rule columns; pass records() output to avoid reloading."""
    if flagged is None:
        flagged = records()
    t = clean(flagged, drop_over_6h)
    t["split"] = split_of(t)
    return events.flag_days(t)
