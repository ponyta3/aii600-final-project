# Data

The files the notebook reads are not committed, and there is no `data` release.

## Source and terms

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), yellow-taxi trip records. Technology providers authorized under the Taxicab & Livery Passenger Enhancement Programs collected the records and supplied them to TLC. TLC makes no representations as to their accuracy.

TLC publishes no license for these files. The [NYC.gov terms of use](https://www.nyc.gov/home/terms-of-use.page) reserve all rights and do not grant redistribution. The files are therefore not republished here. Download them from the source.

## Download

No account or key is needed. From the project root:

```bash
for m in 01 02 03; do
  curl -fSL -o data/yellow_tripdata_2024-$m.parquet \
    https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-$m.parquet
done
shasum -a 256 data/yellow_tripdata_2024-*.parquet
```

## Files

Downloaded and hashed 26 October 2026.

| File | Bytes | SHA-256 |
|---|---|---|
| `yellow_tripdata_2024-01.parquet` | 49,961,641 | `c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510` |
| `yellow_tripdata_2024-02.parquet` | 50,349,284 | `c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20` |
| `yellow_tripdata_2024-03.parquet` | 60,078,280 | `2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823` |

If a hash differs, TLC has replaced the file. Record the new hash and a `decisions.md` entry in the same commit.

## Sample

- Unit of observation: one recorded yellow-taxi trip from pickup zone 161, 162, 163, or 230 (Midtown Center, Midtown East, Midtown North, Times Sq/Theatre District) to drop-off zone 132 (JFK Airport).
- Time span: January through March 2024, by meter-start time.
- Sample size: 21,725 positive-duration candidate trips, from a preliminary download checked 24 October 2026. This count includes duplicate candidates and 37 rides over six hours. Exclusions are recorded in `decisions.md`.
- Duration: `tpep_dropoff_datetime` minus `tpep_pickup_datetime`, in minutes.
