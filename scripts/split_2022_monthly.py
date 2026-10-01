#!/usr/bin/env python3.12
"""Split the 2022 portion of the raw MT5 M1 CSV into monthly files.

Replicates xauusd_ml_scratch/dataio.py logic exactly:
  - tab-separated MT5 format <DATE>\t<TIME>\t<OPEN>...<SPREAD>
  - output columns: time,open,high,low,close,volume(=TICKVOL)
  - dedup by index keep=last, sort
  - one file per month: XAUUSD_M1_YYYY-MM.csv
Only writes months before 2023-01 (the existing 43 files cover 2023-01+).
"""
import pandas as pd
from pathlib import Path

SRC = Path("/home/z/my-project/upload/5_extracted/XAUUSDc_M1_202201022305_202606262057.csv")
OUT = Path("/home/z/my-project/download/xauusd_data")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC, sep="\t")
df.columns = [c.strip().strip("<>").lower() for c in df.columns]
t = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                   format="%Y.%m.%d %H:%M:%S")
out = df[["open", "high", "low", "close", "tickvol", "spread"]].copy()
out.index = t
out = out.rename(columns={"tickvol": "volume"})
out = out[~out.index.duplicated(keep="last")].sort_index()
for c in out.columns:
    out[c] = out[c].astype("float64")

sub = out.loc[:"2022-12-31 23:59:59"]
print(f"raw bars total {len(out):,} | 2022 portion {len(sub):,} "
      f"({sub.index[0]} -> {sub.index[-1]})")

n = 0
for ym, g in sub.groupby(sub.index.to_period("M")):
    p = OUT / f"XAUUSD_M1_{ym}.csv"
    if p.exists():
        print(f"skip existing {p.name}")
    else:
        g.rename_axis("time").reset_index()[
            ["time", "open", "high", "low", "close", "volume"]
        ].to_csv(p, index=False)
        print(f"wrote {p.name}: {len(g):,} rows")
    n += 1

files = sorted(OUT.glob("XAUUSD_M1_*.csv"))
print(f"total monthly files now: {len(files)} "
      f"({files[0].name} .. {files[-1].name})")
