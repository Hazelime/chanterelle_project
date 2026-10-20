import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ---------------- Configuration ----------------
GRID_PATH = Path("./data/sweden_master_grid.csv")
DATA_PATH = Path("./data/processed/cell_week_dataset.parquet")
OUTPUT_DIR = Path("./data/splits")

BLOCK_SIZE = 100000  # 100 km
SEEDS = [111, 222, 333, 444, 555]

EXPECTED_CELLS = 4983
EXPECTED_ROWS = 8057511

# ---------------- Load data ----------------
grid = pd.read_csv(GRID_PATH, usecols = ["cell_id", "cell_x", "cell_y"], dtype = {"cell_id": "str"})
data = pd.read_parquet(DATA_PATH, columns = ["cell_id", "year", "week", "observation"])

# Check that the data makes sense before continuing.
assert len(grid) == EXPECTED_CELLS
assert grid["cell_id"].is_unique
assert len(data) == EXPECTED_ROWS
keys = ["cell_id", "year", "week"]
assert not data.duplicated(keys).any()
assert set(grid["cell_id"]) == set(data["cell_id"].unique())

print(f"Loaded {len(grid):,} grid cells.")
print(f"Loaded {len(data):,} cell-weeks.")

# ---------------- Create spatial blocks ----------------
# Cell_x and cell_y are EPSG:3006 coordinates in metres, representing cell origins.
grid["block_x"] = (np.floor(grid["cell_x"] / BLOCK_SIZE).astype(int))
grid["block_y"] = (np.floor(grid["cell_y"] / BLOCK_SIZE).astype(int))
grid["block_id"] = (grid["block_x"].astype(str) + "_"+ grid["block_y"].astype(str))
blocks = np.sort(grid["block_id"].unique())
print(f"\nSpatial blocks: {len(blocks):,}")

# ---------------- Spatial splits ----------------
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
for seed in SEEDS:
    # First do an 80-20 split into training and remaining, then split remaining in
    # half for validation and testing.
    train_blocks, remaining = train_test_split(blocks, test_size = 0.20, random_state = seed)
    val_blocks, test_blocks = train_test_split(remaining, test_size = 0.50, random_state = seed)

    # **{...} unpacks each of the three dictionaries into
    # the joint block_to_split dictionary, where each block
    # key is mapped to one of the values "train", "val", or
    # "test".
    block_to_split = {
        **{b: "train" for b in train_blocks},
        **{b: "val" for b in val_blocks},
        **{b: "test" for b in test_blocks},
    }
    # Assign a block id to every cell.
    spatial = grid[["cell_id", "block_id"]].copy()
    spatial["spatial_split"] = (spatial["block_id"].map(block_to_split))

    # Verify assignments.
    assert spatial["spatial_split"].notna().all()
    # Check that no block belongs to multiple splits.
    assert (spatial.groupby("block_id")["spatial_split"].nunique().max() == 1)

    # Save this seed's assignments.
    output_path = (OUTPUT_DIR / f"spatial_blocks_seed{seed}.parquet")
    spatial.to_parquet(output_path, compression = "zstd", index = False)

    # Print various statistics as a kind of summary.
    summary = (data[["cell_id", "observation"]]
        .merge(spatial[["cell_id", "spatial_split"]], on = "cell_id", validate = "many_to_one")
        .groupby("spatial_split")
        .agg(rows = ("observation", "size"), positives = ("observation", "sum"))
    )

    summary["prevalence_pct"] = (100 * summary["positives"] / summary["rows"])
    block_counts = (spatial.groupby("spatial_split")["block_id"].nunique())
    cell_counts = spatial["spatial_split"].value_counts()
    summary["cells"] = cell_counts
    summary["blocks"] = block_counts
    print(f"\nSpatial split — seed {seed}:")
    print(summary[["blocks", "cells", "rows", "positives", "prevalence_pct"]].to_string())

    print(f"Saved: {output_path}")

# ---------------- Temporal split ----------------
# Excluding 2026
years = pd.DataFrame({"year": np.arange(1995, 2026, dtype = np.int16)})

# train <= 2023, val = 2024, test = 2025 
years["temporal_split"] = np.select(
    [
        years["year"] <= 2023,
        years["year"] == 2024,
        years["year"] == 2025,
    ],
    ["train", "val", "test"],
    default = "unused"
)

# Save
temporal_path = OUTPUT_DIR / "temporal_split.parquet"
years.to_parquet(temporal_path, compression = "zstd", index = False)

# Print various statistics as a kind of summary.
temporal_summary = (data[["year", "observation"]]
    .merge(years, on = "year", validate = "many_to_one")
    .groupby("temporal_split")
    .agg(rows = ("observation", "size"), positives = ("observation", "sum"))
)
temporal_summary["prevalence_pct"] = (100 * temporal_summary["positives"] / temporal_summary["rows"])
print("\nTemporal split:")
print(temporal_summary.to_string())

print(f"Saved: {temporal_path}")
print("\nFinished!")
