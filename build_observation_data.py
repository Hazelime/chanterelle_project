import requests
import pandas as pd
import time # Used to pause download when needed
import os
import pyarrow # Needed for Parquet file support
import numpy as np
from pathlib import Path
from pyproj import Transformer

# All the specific numbers in the comments below are from a run on 2026-10-20.
# Where to get the chanterelle data
url = "https://api.gbif.org/v1/occurrence/search"
raw_output_path = "./data/observations/raw_gbif_observations.parquet"
filtered_output_path = "./data/observations/filtered_gbif_observations.parquet"
final_output_path = Path("./data/observations/observations.parquet")
grid_path = Path("./data/sweden_master_grid.csv")
cell_size = 10000 # Meters



# ------------DOWNLOAD DATA------------
# Specify what to download: Swedish chantarelle observations.
# The limit is set to 300, which is the maximum allowed by the API.
params = {
    "scientificName": "Cantharellus cibarius",
    "country": "SE",
    "limit": 300
}

all_observations = []
offset = 0

while True:
    params["offset"] = offset

    print(f"Requesting records {offset}–{offset + 299}...")

    response = requests.get(
        url,
        params = params,
        headers = {"User-Agent": "chanterelle-ml-project"},
        timeout = 30
    )

    # Wait a bit more if we are being rate limited, then try again.
    # The rate limiting seems to kick in at around 10,000 downloaded records.
    # UPDATE: Actually this doesn't seem to work at all, but I'm leaving it in in
    # case it's needed in the future.
    if response.status_code == 429:
        print("Rate limited. Waiting 10 seconds...")
        time.sleep(10)
        continue

    # Checks that the request was successful and raises an exception otherwise
    response.raise_for_status()

    data = response.json()
    # Discard uninteresting data and keep only the results.
    results = data["results"]

    all_observations.extend(results)

    # Print progression.
    print(f"Downloaded {len(all_observations)} / {data['count']}")

    if len(all_observations) >= data["count"] or not results:
        break

    offset += 300

    # Give GBIF's API a little breathing room
    time.sleep(1)

observations = pd.DataFrame(all_observations)

print("\nFinished!")
print("Final shape:", observations.shape) # (15965, 175)

# Save data to a Parquet file for later use.
observations.to_parquet(raw_output_path, index = False)

print("Data saved to ", os.path.exists(raw_output_path))
print("File size:", os.path.getsize(raw_output_path) / 1_000_000, "MB") # 3.552689 MB



# ------------FILTER DATA------------ 
# Remove observations with coordinate uncertainty values above 5 km (5000 m). 
# This conveniently also filters out observations with missing coordinates, since NaN <= 5000 evaluates to False.
observations_clean = observations[
    observations["coordinateUncertaintyInMeters"] <= 5000
].copy()

print("Number of observations")
print("Original:", len(observations))
print("Clean:", len(observations_clean))
print("Removed:", len(observations) - len(observations_clean))

# Clean up the eventDate column, since it contains dates in different formats.
# This line splits up the value on "/" and keeps only the first piece.
observations_clean["date"] = (observations_clean["eventDate"].str.split("/").str[0])
# Using mixed means that pandas will try to infer the format of the date, which is useful
# since the dates are in different formats.
observations_clean["date"] = pd.to_datetime(
    observations_clean["date"],
    format = "mixed",
    utc = True,
    errors = "coerce"
)

print("Missing dates:", observations_clean["date"].isna().sum()) # 0
print("Earliest:", observations_clean["date"].min()) # Earliest: 1896-09-02 00:00:00+00:00
print("Latest:", observations_clean["date"].max()) # Latest: 2026-10-09 00:00:00+00:00

# Check the year distribution across the dataset.
observations_clean["year"] = observations_clean["date"].dt.year
# .to_string() allows us to see the full distribution of years by preventing the ... truncation
# in the terminal 
print(observations_clean["year"].value_counts().sort_index().to_string())

# Filter observations to the years 1995-2025, since:
# * 1995 is when we first start getting 100s of observations per year consistently
# * The 2026 data is still incomplete
# * There are still quite a few observations to work with (12,935)
# * The world has changed quite a lot since 1995
observations_model = observations_clean[observations_clean["year"].between(1995, 2025)].copy()

print("1995-2025 observations:", len(observations_model))
print(observations_model["year"].value_counts().sort_index().to_string())

# Save to parquet file for later use.
observations_model.to_parquet(filtered_output_path, index = False)

print("File exists:", os.path.exists(filtered_output_path))
print("File size:", os.path.getsize(filtered_output_path) / 1_000_000, "MB") # 2.933045 MB

# ------------ADD CELL ID------------ 
#obs = pd.read_parquet(filtered_output_path, columns = ["date", "decimalLongitude", "decimalLatitude"])
obs = observations_model[["date", "decimalLongitude", "decimalLatitude"]]
grid = pd.read_csv(grid_path, usecols = ["cell_id", "cell_x", "cell_y"])

print(f"Loaded {len(obs):,} observations.")
print(f"Loaded {len(grid):,} grid cells.")

# Check that all cell ids are unique before proceeding.
assert grid["cell_id"].is_unique


# ---------------- Validate input ----------------

missing = obs[["date", "decimalLongitude", "decimalLatitude"]].isna().any(axis = 1)
print(f"Observations with missing fields: {missing.sum():,}")

obs = obs.loc[~missing].copy()

valid_coords = (
    obs["decimalLongitude"].between(-180, 180)
    & obs["decimalLatitude"].between(-90, 90)
)

print(f"Invalid coordinates: {(~valid_coords).sum():,}")

obs = obs.loc[valid_coords].copy()


# ---------------- Spatial assignment ----------------
# Transform data to the correct coordinate reference system (CRS): EPSG:3006.
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy = True)

x, y = transformer.transform(
    obs["decimalLongitude"].to_numpy(),
    obs["decimalLatitude"].to_numpy()
)

cell_x = (np.floor(x / cell_size).astype(np.int64) * cell_size)
cell_y = (np.floor(y / cell_size).astype(np.int64) * cell_size)
obs["cell_id"] = [f"{cx}_{cy}" for cx, cy in zip(cell_x, cell_y)]


# ---------------- Temporal assignment ----------------

obs["date"] = pd.to_datetime(obs["date"], utc = True)
iso = obs["date"].dt.isocalendar()
obs["year"] = iso.year.astype("int16")
obs["week"] = iso.week.astype("int8")


# ---------------- Match against master grid ----------------

valid_cell_ids = set(grid["cell_id"].astype(str))

matched = obs["cell_id"].isin(valid_cell_ids)

print(f"Observations in valid grid cells: {matched.sum():,}")
print(f"Observations outside grid: {(~matched).sum():,}")

# Drop the observations outside the grid.
obs = obs.loc[matched].copy()


# ---------------- Finalize ----------------

result = obs[
    ["cell_id", "date", "year", "week"]
].sort_values(
    ["cell_id", "date"]
).reset_index(drop = True)

assert result.notna().all().all()
assert result["cell_id"].isin(valid_cell_ids).all()

final_output_path.parent.mkdir(parents = True, exist_ok = True)

result.to_parquet(final_output_path, compression = "zstd", index = False)


# ---------------- Summary ----------------

print("\nFinished!")
print(f"Rows: {len(result):,}")
print(f"Unique cells: {result['cell_id'].nunique():,}")

print("\nObservations by year:")
print(result.groupby("year").size().to_string())

print("\nUnique positive cell-weeks:")
print(
    result[
        ["cell_id", "year", "week"]
    ].drop_duplicates().shape[0]
)

size_mb = final_output_path.stat().st_size / (1024 ** 2)

print(f"\nSaved to: {final_output_path}")
print(f"File size: {size_mb:.2f} MiB")








