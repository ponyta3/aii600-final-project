# Proposal

## Hypothesis

For yellow-taxi rides from Midtown Manhattan to JFK Airport, a quantile regression on pickup hour, day of week, and pickup neighborhood will achieve lower held-out quantile loss than a fixed travel-time allowance. The quantity estimated is the conditional 90th percentile of metered trip duration, in minutes, for trips from January through March 2024 (fit on January, calibrated on February, evaluated on March). We will also check whether allowances give about 90% coverage overall and across departure periods.

## Data

Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), yellow-taxi records, January–March 2024. Technology providers supplied the records to TLC. One observation is one recorded trip from Midtown Center, Midtown East, Midtown North, or Times Sq/Theatre District (pickup zones 161, 162, 163, 230) to JFK Airport (zone 132). A preliminary download checked 24 October 2026 found 21,725 positive-duration candidate trips, including duplicate candidates and 37 rides over six hours. We will investigate these, document exclusions, and test sensitivity to questionable records rather than drop long trips because they worsen performance.

Duration is meter-end minus meter-start. The decision needs door-to-curb travel time from a planned departure. We measure only metered road travel, with meter-start standing in for planned pickup. Taxi availability, airport security, and gate access are outside scope. Canceled journeys are absent, and TLC does not guarantee record accuracy.

TLC publishes no license for these files; the NYC.gov terms of use reserve all rights and do not grant redistribution. The raw files therefore stay off GitHub and are not republished on a release; `data/README.md` gives the commands that download them from TLC, plus file hashes.

## Decision affected by the model

A traveler going from Midtown to JFK uses the allowance to decide how many minutes to reserve for the road portion of the trip. With the model, the reserve changes with departure hour, weekday, and pickup zone instead of staying a single fixed number. The comparison is quantile regression against a fixed January 90th-percentile allowance. Predictors use only information available before departure, so trip distance, fare, tips, and drop-off time are excluded. Both methods are fit on January. On February, each is calibrated with an additive offset equal to the 90th percentile of its errors (actual minus predicted minutes). Both are then frozen before March is scored.

Underbudgeting costs more, since running short risks missing the flight. For allowance $B$ and observed duration $T$ in minutes:

$$L(B, T) = 0.9\max(T - B, 0) + 0.1\max(B - T, 0)$$

A minute short is penalized nine times as much as a minute of unused padding. These weights express a planning preference, not financial costs. The 90% target is not a per-ride guarantee, and 95% is a sensitivity analysis. The primary outcome is March mean quantile loss. We also report coverage, average allowance, shortfall and unused minutes, uncertainty, and performance by departure period. The fixed allowance may win. Conclusions apply only to this period and to metered road travel.
