import numpy as np
import pandas as pd
from pathlib import Path

"""
This script aggregates cell ID, forest coverage, weather, observation, placename, and week
data in the joint, compressed data/processed/cell_week_dataset.parquet.
"""

# ---------------- Configuration ----------------
WEATHER_PATH = Path("./data/weather/sweden_weekly_weather.parquet")
OBSERVATIONS_PATH = Path("./data/observations/observations.parquet")
PLACENAMES_PATH = Path("./data/placenames/cell_placenames.parquet")
GRID_PATH = Path("./data/sweden_master_grid.csv")
OUTPUT_PATH = Path("./data/processed/cell_week_dataset.parquet")

EXPECTED_CELLS = 4983
EXPECTED_WEEKS = 1617
EXPECTED_ROWS = 8057511

# ---------------- Load data ----------------
weather = pd.read_parquet(WEATHER_PATH)
observations = pd.read_parquet(OBSERVATIONS_PATH, columns = ["cell_id", "year", "week"])
grid = pd.read_csv(GRID_PATH, usecols = ["cell_id", "forest_pct"], dtype = {"cell_id": "str"})
placenames = pd.read_parquet(PLACENAMES_PATH, columns = ["cell_id", "placename_count"])
print(f"Loaded {len(weather):,} weather records.")
print(f"Loaded {len(observations):,} observations.")
print(f"Loaded forest coverage for {len(grid):,} cells.")
print(f"Loaded placename counts for {len(placenames):,} cells.")

# ---------------- Validate data ----------------
keys = ["cell_id", "year", "week"]
assert len(weather) == EXPECTED_ROWS
assert weather["cell_id"].nunique() == EXPECTED_CELLS
assert not weather.duplicated(keys).any()
assert not weather.isna().any().any()

n_weeks = weather[["year", "week"]].drop_duplicates().shape[0]
assert n_weeks == EXPECTED_WEEKS
assert len(weather) == EXPECTED_CELLS * n_weeks

assert len(grid) == EXPECTED_CELLS
assert grid["cell_id"].is_unique
assert grid["forest_pct"].notna().all()
assert grid["forest_pct"].between(0, 100).all()
grid["forest_pct"] = grid["forest_pct"].astype("float32")

assert len(placenames) == EXPECTED_CELLS
assert placenames["cell_id"].is_unique
assert placenames["placename_count"].notna().all()
assert placenames["placename_count"].ge(0).all()

# ---------------- Create observation labels ----------------
# Multiple observations in the same cell-week should produce only one positive label.
# Use int8 to reduce file size. Observations are 1, non-observations are 0.
positive_weeks = (observations[keys].drop_duplicates().assign(observation = np.int8(1)))

# Check which observed cell-weeks have weather data.
matched = positive_weeks[keys].merge(
    weather[keys],
    on = keys,
    how = "left",
    indicator = True,
    validate = "one_to_one"
)
unmatched = (matched["_merge"] == "left_only").sum()
print(f"Unique observed cell-weeks: {len(positive_weeks):,}") # 10,316
# 4 cell-weeks without weather are expected, since we used the last two weeks of 1994
# to calculate data for the first week of 1995.
print(f"Observed cell-weeks without weather: {unmatched:,}") # 4

# Attach binary labels to the weather dataset: 1 for observation, 0 for non-observation.
df = weather.merge(
    positive_weeks,
    on = keys,
    how = "left",
    validate = "one_to_one"
)
# Non-observations are 0.
df["observation"] = (df["observation"].fillna(0).astype("int8"))

# ---------------- Encode weeks cyclically ----------------
# Encode weeks cyclically so week 1 is close to week 52/53 of the preceding year.
# ISO years contain 52 or 53 weeks, and December 28 always belongs to the last
# ISO week of its calendar year.
years = df["year"].to_numpy(dtype = np.int32)

last_iso_week = (
    pd.to_datetime(
        pd.Series(years).astype(str) + "-12-28"
    )
    .dt.isocalendar()
    .week.to_numpy(dtype = np.float32)
)

weeks = df["week"].to_numpy(dtype = np.float32)

# Divide by 52 or 53 depending on the year.
angle = (2 * np.pi * (weeks - 1) / last_iso_week)
df["week_sin"] = np.sin(angle).astype("float32")
df["week_cos"] = np.cos(angle).astype("float32")

# ---------------- Add static cell features ----------------
# Add forest_pct
# validate = many_to_one ensures that each cell-week matches at most one grid record.
df = df.merge(grid, on = "cell_id", how = "left", validate = "many_to_one")
assert df["forest_pct"].notna().all()

# Add placename_count
df = df.merge(placenames, on = "cell_id", how = "left", validate = "many_to_one")
assert len(df) == EXPECTED_ROWS
assert df["placename_count"].notna().all()
df["placename_count"] = (df["placename_count"].astype("int32"))

# ---------------- Finalize ----------------
df = df[
    [
        "cell_id",
        "year",
        "week",
        "week_sin",
        "week_cos",
        "observation",
        "temp_7d",
        "precip_7d",
        "forest_pct",
        "placename_count"
    ]
] 
# Save space by using small datatypes
df["year"] = df["year"].astype("int16")
df["week"] = df["week"].astype("int8")
df["temp_7d"] = df["temp_7d"].astype("float32")
df["precip_7d"] = df["precip_7d"].astype("float32")
df["placename_count"] = df["placename_count"].astype("int16")

# Check that the final data makes sense.
assert len(df) == EXPECTED_ROWS
assert not df.duplicated(keys).any()
assert not df.isna().any().any()
assert df["observation"].isin([0, 1]).all()

# ---------------- Save ----------------
OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
df.to_parquet(OUTPUT_PATH, compression = "zstd", index = False)

# ---------------- Summary ----------------
n_positive = int(df["observation"].sum())
prevalence = 100 * n_positive / len(df)

print("\nFinished!")
print(f"Rows: {len(df):,}") # 8,057,511
print(f"Cells: {df['cell_id'].nunique():,}") #4,983
print(f"Weeks: {n_weeks:,}") # 1,617
print(f"Positive cell-weeks: {n_positive:,}") # 10,312
print(f"Positive prevalence: {prevalence:.4f}%") # 0.1280%

print("\nMissing values:") # 0 across the board
print(df.isna().sum())


print("\nData types:")
# cell_id            str
# year             int16
# week              int8
# week_sin       float32
# week_cos       float32
# observation       int8
# temp_7d        float32
# precip_7d      float32
# forest_pct     float32
print(df.dtypes)

print("\nFirst five rows:")
#        cell_id  year  week  week_sin  week_cos  observation   temp_7d  precip_7d   forest_pct
# 260000_6490000  1995     1  0.000000  1.000000            0  4.039240  27.324800   0.231571
print(df.head().to_string(index=False))

size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"File size: {size_mb:.2f} MiB") # 65.83 MiB
