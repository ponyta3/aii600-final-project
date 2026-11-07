"""Days the allowance is not meant to cover (issue #3): the rules, every candidate checked, and the sources.

A day is flagged if, by 6 pm the day before, it met one of three rules:

    R1  a US federal holiday;
    R2  a publicly announced event with street closures inside pickup zones 161, 162, 163 or 230, or on FDR Drive;
    R3  a National Weather Service Winter Storm, Blizzard, Ice Storm, Flood or Flash Flood Warning
        for Manhattan or Queens (watches and advisories do not count).

Whole days, by pickup date. The same rules and sources for January, February and March. A day whose status
cannot be confirmed from a source is ordinary. Unannounced disruptions stay ordinary: the 90th percentile is
meant to cover them. Rejected candidates are listed too, so the checks can be audited.
"""

import pandas as pd

IEM = "Iowa Environmental Mesonet NWS warning archive (OKX), zones NYZ072, NYZ176, NYZ178; counties NYC061, NYC081"

CANDIDATES = [
    # date, rule, flagged, what, why / source
    ("2024-01-01", "R1", True, "New Year's Day", "federal holiday (OPM)"),
    ("2024-01-08", "R2", False, "Bridge and tunnel protest, 9:40-11:15 am", "not announced in advance (NBC New York, ABC7)"),
    ("2024-01-09", "R3", False, "Rain and wind: Flood Watch, High Wind Warning, Flood Warning", "watch only; high wind is not an R3 type; flood warning issued 9:34 pm the same day (" + IEM + ")"),
    ("2024-01-10", "R3", False, "Flood Warning overnight from 9 Jan; Coastal Flood Warning (Queens)", "flood warning issued 9:34 pm on 9 Jan, after the 6 pm cut-off; coastal flood is not an R3 type (" + IEM + ")"),
    ("2024-01-13", "R3", False, "Coastal Flood Warning", "coastal flood is not an R3 type (" + IEM + ")"),
    ("2024-01-15", "R1", True, "Martin Luther King Jr. Day", "federal holiday (OPM)"),
    ("2024-01-16", "R3", False, "Snow", "Winter Weather Advisory only (" + IEM + "; NYC Emergency Management)"),
    ("2024-01-19", "R3", False, "Snow", "Winter Weather Advisory only (" + IEM + ")"),
    ("2024-02-13", "R3", True, "Nor'easter", "Winter Storm Warning issued 4:00 pm on 12 Feb, product 202402122100-KOKX-WSWOKX (" + IEM + ")"),
    ("2024-02-17", "R3", False, "Snow", "Winter Weather Advisory only (" + IEM + ")"),
    ("2024-02-19", "R1", True, "Presidents' Day", "federal holiday (OPM)"),
    ("2024-02-25", "R2", False, "Lunar New Year parade, Chinatown", "outside the four zones and FDR Drive"),
    ("2024-03-06", "R3", False, "Heavy rain", "Flood Watch and Flood Advisory only (" + IEM + ")"),
    ("2024-03-07", "R3", False, "Heavy rain", "Flood Watch only (" + IEM + ")"),
    ("2024-03-10", "R3", False, "Coastal Flood Warning (Queens)", "coastal flood is not an R3 type (" + IEM + ")"),
    ("2024-03-16", "R2", True, "St Patrick's Day parade", "Fifth Ave 44th-79th St and Madison Ave from 42nd St closed from 11 am (Fox 5, PIX11, NBC New York)"),
    ("2024-03-17", "R2", True, "NYC Half Marathon", "FDR Drive, 42nd St and Times Square closed (nyc.gov closure list)"),
    ("2024-03-23", "R3", False, "Rain: Flood Watch, Flood Warning", "watch only before the day; flood warning issued 7:50 pm the same day (" + IEM + ")"),
    ("2024-03-28", "R2", True, "Presidential fundraiser, Radio City", "date public weeks ahead (NBC News); 49th-51st St between Fifth and Sixth Aves closed noon-10 pm (NBC New York, CBS New York)"),
    ("2024-03-29", "R1", False, "Good Friday; Auto Show opens at Javits", "not a federal holiday; Javits is outside the four zones"),
]


def candidates():
    """Every day checked against the rules, flagged or not, with its source."""
    df = pd.DataFrame(CANDIDATES, columns=["date", "rule", "flagged", "what", "source"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def flagged_dates():
    c = candidates()
    return dict(zip(c.loc[c.flagged, "date"], c.loc[c.flagged, "rule"]))


def flag_days(trips, date_col="tpep_pickup_datetime"):
    """Adds flagged_day (bool) and flag_rule (R1, R2, R3 or empty) by the trip's pickup date."""
    out = trips.copy()
    rule = pd.to_datetime(out[date_col]).dt.date.map(flagged_dates())
    out["flagged_day"] = rule.notna()
    out["flag_rule"] = rule.fillna("")
    return out
