"""
Create a representative 2,000-row sample of US_Accidents_March23.csv.

Method
------
1. Pass 1 (streaming, chunked): compute full-dataset statistics for every column
   (category frequencies, numeric mean/std, missing-value rates) and the
   stratum of every row (Severity x Year).
2. Generate many candidate samples using proportional stratified sampling
   (Severity x Year), so severity and time trends are preserved exactly.
3. Pass 2: load only the candidate rows and score each candidate against the
   full-dataset statistics. The candidate with the smallest overall
   distribution distance is kept.
4. All 46 columns are kept and values are written exactly as in the original
   file (read as text, no re-formatting).
"""
import json
import numpy as np
import pandas as pd

SRC = "US_Accidents_March23.csv"
OUT = "US_Accidents_2000_sample.csv"
REPORT = "sample_validation_report.txt"
N = 2000
N_CANDIDATES = 150
CHUNK = 500_000
SEED = 42

NUMERIC = ["Severity", "Start_Lat", "Start_Lng", "End_Lat", "End_Lng", "Distance(mi)",
           "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
           "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
CATEGORICAL = ["Source", "Severity", "State", "Timezone", "Wind_Direction",
               "Weather_Condition", "Amenity", "Bump", "Crossing", "Give_Way",
               "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop",
               "Traffic_Calming", "Traffic_Signal", "Turning_Loop", "Sunrise_Sunset",
               "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight",
               "Year", "Month", "Hour", "Weekday"]


def derive(df):
    """Derived time features used only for stratification / scoring."""
    st = df["Start_Time"].str.slice(0, 19)
    t = pd.to_datetime(st, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    return pd.DataFrame({
        "Year": t.dt.year.astype("Int64").astype(str),
        "Month": t.dt.month.astype("Int64").astype(str),
        "Hour": t.dt.hour.astype("Int64").astype(str),
        "Weekday": t.dt.dayofweek.astype("Int64").astype(str),
    }, index=df.index)


def read_chunks():
    return pd.read_csv(SRC, dtype=str, keep_default_na=False, chunksize=CHUNK)


# ---------------------------------------------------------------- pass 1
print("Pass 1: computing full-dataset statistics ...")
columns = None
cat_counts = {c: {} for c in CATEGORICAL}
num_sum = {c: 0.0 for c in NUMERIC}
num_sq = {c: 0.0 for c in NUMERIC}
num_n = {c: 0 for c in NUMERIC}
missing = None
strata = []
total = 0
for ch in read_chunks():
    if columns is None:
        columns = list(ch.columns)
        missing = {c: 0 for c in columns}
    ch = pd.concat([ch, derive(ch)], axis=1)
    for c in columns:
        missing[c] += int((ch[c] == "").sum())
    for c in CATEGORICAL:
        for k, v in ch[c].value_counts().items():
            cat_counts[c][k] = cat_counts[c].get(k, 0) + int(v)
    for c in NUMERIC:
        x = pd.to_numeric(ch[c], errors="coerce").dropna()
        num_sum[c] += float(x.sum()); num_sq[c] += float((x ** 2).sum()); num_n[c] += len(x)
    strata.append((ch["Severity"] + "_" + ch["Year"]).to_numpy())
    total += len(ch)
    print(f"  {total:,} rows")

strata = np.concatenate(strata)
num_mean = {c: num_sum[c] / num_n[c] for c in NUMERIC}
num_std = {c: np.sqrt(max(num_sq[c] / num_n[c] - num_mean[c] ** 2, 1e-12)) for c in NUMERIC}
miss_rate = {c: missing[c] / total for c in columns}
cat_prop = {c: {k: v / total for k, v in d.items()} for c, d in cat_counts.items()}

# ---------------------------------------------------------------- candidates
print("Generating stratified candidate samples ...")
keys, inv, counts = np.unique(strata, return_inverse=True, return_counts=True)
exact = counts / total * N
alloc = np.floor(exact).astype(int)
rem = N - alloc.sum()
alloc[np.argsort(-(exact - alloc))[:rem]] += 1
groups = [np.flatnonzero(inv == i) for i in range(len(keys))]

rng = np.random.default_rng(SEED)
candidates = []
for _ in range(N_CANDIDATES):
    idx = np.concatenate([rng.choice(g, a, replace=False) for g, a in zip(groups, alloc) if a > 0])
    candidates.append(np.sort(idx))
needed = np.unique(np.concatenate(candidates))

# ---------------------------------------------------------------- pass 2
print(f"Pass 2: loading {len(needed):,} candidate rows ...")
parts, offset = [], 0
for ch in read_chunks():
    lo, hi = offset, offset + len(ch)
    sel = needed[(needed >= lo) & (needed < hi)] - lo
    if len(sel):
        parts.append(ch.iloc[sel])
    offset = hi
pool = pd.concat(parts)
pool.index = needed  # global row positions
pool_d = pd.concat([pool, derive(pool)], axis=1)
pool_num = {c: pd.to_numeric(pool_d[c], errors="coerce") for c in NUMERIC}


def score(idx):
    s = pool_d.loc[idx]
    d = 0.0
    for c in CATEGORICAL:  # total variation distance
        p = s[c].value_counts(normalize=True)
        ref = cat_prop[c]
        allk = set(ref) | set(p.index)
        d += 0.5 * sum(abs(ref.get(k, 0) - p.get(k, 0)) for k in allk)
    for c in NUMERIC:      # standardized mean difference
        x = pool_num[c].loc[idx].dropna()
        if len(x):
            d += abs(x.mean() - num_mean[c]) / num_std[c]
    for c in columns:      # missing-value rate difference
        d += abs((s[c] == "").mean() - miss_rate[c])
    return d


print("Scoring candidates ...")
scores = [score(c) for c in candidates]
best = int(np.argmin(scores))
best_idx = candidates[best]
print(f"  best candidate #{best}, score={scores[best]:.4f} (median {np.median(scores):.4f})")

sample = pool.loc[best_idx, columns]
sample.to_csv(OUT, index=False)
print(f"Saved {len(sample):,} rows x {len(columns)} columns -> {OUT}")

# ---------------------------------------------------------------- report
s = pool_d.loc[best_idx]
lines = [f"Original rows: {total:,}   Sample rows: {len(sample):,}   Columns: {len(columns)} (all kept)", ""]
lines.append("=== Numeric columns: mean (original vs sample) ===")
for c in NUMERIC:
    x = pool_num[c].loc[best_idx].dropna()
    lines.append(f"{c:20s} {num_mean[c]:12.4f}  {x.mean():12.4f}")
lines.append("\n=== Missing-value rate % (original vs sample) ===")
for c in columns:
    lines.append(f"{c:22s} {100*miss_rate[c]:7.2f}  {100*(s[c]=='').mean():7.2f}")
for c in ["Severity", "Year", "Source", "Timezone", "Sunrise_Sunset", "State", "Weather_Condition",
          "Traffic_Signal", "Crossing", "Junction", "Stop"]:
    lines.append(f"\n=== {c}: % (original vs sample) ===")
    p = s[c].value_counts(normalize=True)
    for k, v in sorted(cat_prop[c].items(), key=lambda kv: -kv[1])[:15]:
        lines.append(f"{str(k)[:25]:25s} {100*v:7.2f}  {100*p.get(k, 0):7.2f}")
lines.append(f"\nStates in original: {len(cat_prop['State'])}, in sample: {s['State'].nunique()}")
open(REPORT, "w", encoding="utf-8").write("\n".join(lines))
print(f"Validation report -> {REPORT}")
