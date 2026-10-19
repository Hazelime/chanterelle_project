import numpy as np
import pandas as pd
import xarray as xr

from pyproj import CRS, Transformer

# Paths (use the 1995 temperature data as an arbitrary choice)
GRID_PATH = "./data/sweden_master_grid.csv"
OUTPUT_PATH = "./data/weather/smhi_to_master_grid.parquet"
WEATHER_PATH = "./data/weather/temperature/SMHI_pthbv_tas_1995_1995_daily.nc"

# Load target grid
grid = pd.read_csv(GRID_PATH)

# Open SMHI dataset
ds = xr.open_dataset(WEATHER_PATH)

# ----------Transform to the correct coordinate reference system (CRS)---------
# Read source CRS from NetCDF metadata
source_crs = CRS.from_cf(ds["crs"].attrs) 
# We want EPSG:3006, since that is what is used for Swedish national geodata.
target_crs = CRS.from_epsg(3006)

print("Source CRS:", source_crs.to_string()) # RT90 2.5 gon V, AKA EPSG:3021
print("Target CRS:", target_crs.to_string()) # EPSG:3006

# Create coordinate transformer. Always_xy = True forces PyProj to treat coordinates
# in x, y order for both input and output.
transformer = Transformer.from_crs(source_crs, target_crs, always_xy = True)

# Get SMHI grid coordinates
xs = ds["x"].values
ys = ds["y"].values

# Create a NumPy meshgrid of source grid points.
# Each matching pair (xx[row, col], yy[row, col]) is then one point on the SMHI grid.
xx, yy = np.meshgrid(xs, ys)

# Transform coordinates to EPSG:3006 (SWEREF99 TM)
x_3006, y_3006 = transformer.transform(xx, yy)


# ----------Map each SMHI coordinate to a grid cell---------
# Assign each point to a 10 km master-grid cell
cell_size = 10_000
cell_x = (np.floor(x_3006 / cell_size) * cell_size).astype(np.int64)
cell_y = (np.floor(y_3006 / cell_size) * cell_size).astype(np.int64)
cell_ids = np.char.add(np.char.add(cell_x.astype(str), "_"), cell_y.astype(str))

# Build a flat table with one row for each SMHI grid point.
# np.repeat repeats the row index once for every x-coordinate
# np.tile repeats the full sequence of x-indices for each y-row
# The result looks something like this (for 2 y-values and 3 x-values):
# smhi_y_idx = 0, 0, 0, 1, 1, 1
# smhi_x_idx = 0, 1, 2, 0, 1, 2
mapping = pd.DataFrame({
    "smhi_y_idx": np.repeat(np.arange(len(ys)), len(xs)),
    "smhi_x_idx": np.tile(np.arange(len(xs)), len(ys)),
    "cell_id": cell_ids.ravel()
})

# Keep only cells in the master grid
mapping = mapping[mapping["cell_id"].isin(grid["cell_id"])].copy()


# ----------Address missing SMHI data---------
# Two cells are missing SMHI data, so we replace them with their nearest neighbors.
# (Which cells were closest was calculated in another script.)
replacements = {
    "740000_6470000": (120, 149),
    "750000_6470000": (120, 150)
}

# Remove the existing mapping entries for the two cells, since their assigned SMHI
# points contain only missing values.
mapping = mapping[~mapping["cell_id"].isin(replacements)].copy()

# Add the nearest valid SMHI points.
new_rows = pd.DataFrame([
    {
        "smhi_y_idx": y_idx,
        "smhi_x_idx": x_idx,
        "cell_id": cell_id
    }
    for cell_id, (y_idx, x_idx) in replacements.items()
])
# Ignore_index = True gives the combined DataFrame a fresh index
mapping = pd.concat([mapping, new_rows], ignore_index = True)


# ----------Check statistics, save, and close---------
print("\nMapping rows:", len(mapping)) # 31151
print("Target cells covered:", mapping["cell_id"].nunique()) # 4983
print("Total target cells:", len(grid)) # 4983
print("\nSample:")
print(mapping.head())

mapped_cells = set(mapping["cell_id"])
all_cells = set(grid["cell_id"])
missing_cells = all_cells - mapped_cells
print("Cells without SMHI grid points:", len(missing_cells)) # 0
print("Examples:", sorted(missing_cells)[:20]) # []

# Save the data.
print(f"Mapping saved to {OUTPUT_PATH}")
mapping.to_parquet(OUTPUT_PATH, index = False)
ds.close()
