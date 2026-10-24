from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

"""
This script extracts placename data from data/placenames/placenames_raw.gpkg and aligns it with the master
grid, and saves it in data/placenames/cell_placenames.parquet.
"""

# ---------------- Configuration ----------------
# Data originally comes from https://geotorget.lantmateriet.se/geodataprodukter/ortnamn-nedladdning-vektor-api
GPKG_PATH = Path("./data/placenames/placenames_raw.gpkg")
GRID_PATH = Path("./data/sweden_master_grid.csv")
OUTPUT_PATH = Path("./data/placenames/cell_placenames.parquet")

CELL_SIZE = 10000
EXPECTED_CELLS = 4983

# ---------------- Load data ----------------
grid = pd.read_csv(GRID_PATH, usecols = ["cell_id"], dtype = {"cell_id": "str"})

# Column explanations:
# ortnamn = placename (str)
# detaljtyp = (code for) type of place (str, e.g. BEBTÄTTX for a densely developed settlement)
# sprak = language (str)
# lopnummer = identifier (float, not unique!)
names = gpd.read_file(
    GPKG_PATH,
    layer = "ortnamn",
    columns = ["ortnamn", "detaljtyp", "sprak", "lopnummer"]
)

print(f"Loaded {len(names):,} placename records.") # 989,813
print(f"Loaded {len(grid):,} grid cells.") # 4,983

# Check that the data makes sense before continuing.
assert len(grid) == EXPECTED_CELLS
assert grid["cell_id"].is_unique
assert names.crs.to_epsg() == 3006

# ---------------- Clean names ----------------
# Remove missing or empty names.
missing_names = (names["ortnamn"].isna() | names["ortnamn"].astype(str).str.strip().eq(""))
print(f"Missing/empty names: {missing_names.sum():,}") # 0
names = names.loc[~missing_names].copy()

# Remove missing or empty geometries.
invalid_geometry = (names.geometry.isna() | names.geometry.is_empty)
print(f"Missing/empty geometries: {invalid_geometry.sum():,}") # 0
names = names.loc[~invalid_geometry].copy()

# Remove whitespace around names.
names["ortnamn"] = names["ortnamn"].str.strip()

# ---------------- Spatial assignment ----------------
x = names.geometry.x.to_numpy()
y = names.geometry.y.to_numpy()

cell_x = (np.floor(x / CELL_SIZE).astype(np.int64) * CELL_SIZE)
cell_y = (np.floor(y / CELL_SIZE).astype(np.int64) * CELL_SIZE)
names["cell_id"] = [f"{cx}_{cy}" for cx, cy in zip(cell_x, cell_y)]

# Check how many placenames are inside and outside the master grid.
valid_cells = set(grid["cell_id"])
matched = names["cell_id"].isin(valid_cells)
print(f"Placenames inside master grid: {matched.sum():,}") # 989,567
print(f"Placenames outside master grid: {(~matched).sum():,}") # 246

# Drop the placenames outside the master the master grid.
names = names.loc[matched].copy()

# ---------------- Remove exact duplicates ----------------
before = len(names)
names["x"] = names.geometry.x
names["y"] = names.geometry.y
names = names.drop_duplicates(subset = ["ortnamn", "x", "y", "detaljtyp", "sprak"])
print(f"Duplicate records removed: {before - len(names):,}") # 0

# ---------------- Aggregate by cell ----------------
# Create the columns placename_count (int for the total number of placenames in that cell) and
# placenames (comma-separated list with all the placenames in that cell)
aggregated = (names.groupby("cell_id")["ortnamn"].agg(list).reset_index(name = "placenames"))
aggregated["placename_count"] = (aggregated["placenames"].str.len().astype("int32"))

# ---------------- Include all grid cells ----------------
result = grid.merge(aggregated, on = "cell_id", how = "left", validate = "one_to_one")

# Cells without placenames get empty lists.
result["placenames"] = result["placenames"].apply(lambda x: x if isinstance(x, list) else [])
result["placename_count"] = (result["placenames"].str.len().astype("int32"))

# Sort by cell_id.
result = result[["cell_id", "placename_count", "placenames"]
].sort_values("cell_id").reset_index(drop = True)

# ---------------- Validate ----------------
assert len(result) == EXPECTED_CELLS
assert result["cell_id"].is_unique
assert result["placename_count"].ge(0).all()
assert (result["placename_count"] == result["placenames"].str.len()).all()
assert (result["placename_count"].sum() == len(names))

# ---------------- Save ----------------
OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
result.to_parquet(OUTPUT_PATH, compression = "zstd", index = False)

# ---------------- Summary ----------------
print("\nFinished!")
print(f"Rows: {len(result):,}") # 4,983
print(f"Cells with placenames: {(result['placename_count'] > 0).sum():,}") # 4,972
print(f"Cells without placenames: {(result['placename_count'] == 0).sum():,}") # 11
print(f"Total assigned placenames: {result['placename_count'].sum():,}") # 989,567
print("\nPlacename count statistics:")
print(result["placename_count"].describe())
# count    4983.000000
# mean      198.588601
# std       196.020622
# min         0.000000
# 25%        63.000000
# 50%       151.000000
# 75%       289.000000
# max      3968.000000

print("\nFirst five rows:")
print(result.head().to_string(index = False))
#        cell_id    placename_count                                         placenames
# 260000_6540000                  3    [Rödskär, Skagerrak, Kosterhavets nationalpark]

# Save
size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)
print(f"\nSaved to: {OUTPUT_PATH}")
print(f"File size: {size_mb:.2f} MiB") # 5.14 MiB
