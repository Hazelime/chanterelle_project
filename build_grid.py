import rasterio # Had to pip install this manually.
import numpy as np
import pandas as pd
from rasterio.windows import Window
from pathlib import Path

# In NMD2023, these are the relevant codes:
# 0 = Utanför karteringsområde (outside mapping area)
# 23 = Låg fjällskog på våtmark (low mountain forest on wetlands)
# 43 = Låg fjällskog på fastmark (low mountain forest on solid ground)
# 61 = Vatten, inland (Inland water)
# 62 = Vatten, hav (Marine water)
# 110 = Skog på fastmark (Alpine trees, on firm ground)
# 120 = Skog på våtmark (Alpine trees, on wetlands)
# See the documentation for the codes: https://geodatakatalogen.naturvardsverket.se/geonetwork/srv/swe/catalog.search#/metadata/8853721d-a466-4c01-afcc-9eae57b17b39
BACKGROUND_VALUE = 0
FOREST_CLASSES = 23, 43, 110, 120
WATER_CLASSES = 61, 62
CELL_SIZE = 10000
CHUNK_SIZE = 4000 # Processing window size in raster pixels, as recommended by ChatGPT Edu
RASTER_PATH = Path("NMD2023och2018_basskikt_v0_3/NMD2023och2018bas_v0_3.tif")
OUTPUT_PATH = Path("./data/sweden_master_grid.csv")


def create_master_grid(raster_path: Path,
                       cell_size: int,
                       chunk_size: int,
                       background_value: int,
                       forest_classes: tuple,
                       water_classes: tuple,
                       epsg_crs: int = 3006,
                       res: float = 10.0,
                       save_to_csv: bool = True,
                       output_path: Path = Path("./data/grid.csv")) -> DataFrame:
    """
    Creates a cell_size x cell_size land and forest grid for a given raster, using a given
    EPSG coordinate reference system, given raster grid values, and a given resolution
    (in meters). The data can also be saved as a CSV file if desired.

    Parameters:
        raster_path: Path
            The path to the raster to turn into a grid.
        cell_size: int
            The size of each cell in the grid (in meters)
        chunk_size: int
            How many raster cells to process at once. Use lower values if RAM is a problem.
        background_value: int
            The raster code for background pixels.
        forest_classes: tuple
            The raster codes for forest pixels.
        water_classes: tuple
            The raster codes for water pixels.
        epsg_crs: int
            The EPSG code for the desired coordinate references system. Default is 3006.
        res: float
            The pixel resolution (in meters).
        save_to_csv: bool
            Whether to save the data as a CSV afterward. Default is True. The columns of the
            CSV file are the same as for the returned DataFrame (see below).
        output_path: Path
            Where to save the CSV if saving at all. Default is "grid.csv".
    
    Returns:
        DataFrame
            A Pandas DataFrame where each cell has these values: cell_id (formatted as x_y),
            cell_x, cell_y, n_valid_pixels, n_land_pixels, n_forest_pixels, land_pct, and
            forest_pct.
    """
    # For Sweden, the whole process takes about 10 mins on my laptop.
    with rasterio.open(RASTER_PATH) as src:
        # If any of these assertions fail, an AssertionError will be raised.
        # Check that the EPSG coordinate reference system is correct
        assert src.crs.to_epsg() == epsg_crs
        # Check that the resolution is correct
        assert src.res == (res, res,)
        # Check that there is no rotation/skew
        assert src.transform.b == 0
        assert src.transform.d == 0

        print("Raster shape:", src.shape) # (157991, 71273) for Sweden
        print("Raster bounds:", src.bounds)
        # BoundingBox(left=208450.0, bottom=6091140.0, right=921180.0, top=7671050.0)
        # for Sweden

        # Calculate the grid extent coordinates in chunks of the desired cell size.
        x_min = int(np.floor(src.bounds.left / cell_size) * cell_size)
        x_max = int(np.ceil(src.bounds.right / cell_size) * cell_size)
        y_min = int(np.floor(src.bounds.bottom / cell_size) * cell_size)
        y_max = int(np.ceil(src.bounds.top / cell_size) * cell_size)

        # Calculate the total number of cells
        nx = (x_max - x_min) // cell_size
        ny = (y_max - y_min) // cell_size
        n_cells = nx * ny
        print(f"Grid dimensions: {nx} x {ny}") # 73 x 159 for Sweden
        print(f"Total possible cells: {n_cells}") # 11607 for Sweden

        # Create one entry of valid/land/forest counts per cell
        valid_counts = np.zeros(n_cells, dtype = np.int64)
        land_counts = np.zeros(n_cells, dtype = np.int64)
        forest_counts = np.zeros(n_cells, dtype = np.int64)

        transform = src.transform
        # Row_off(set) and col_off(set) because we are not using every single row/col.
        for row_off in range(0, src.height, chunk_size):
            for col_off in range(0, src.width, chunk_size):

                # Calculating it this way makes sure that we capture the edges accurately,
                # as they may be smaller than chunk_size x chunk_size.
                height = min(chunk_size, src.height - row_off)
                width = min(chunk_size, src.width - col_off)

                window = Window(col_off, row_off, width, height)

                # Read band/layer 1, which corresponds to the grid codes.
                data = src.read(1, window = window)

                # Get the pixel center coordinates calculated in 1D
                cols = col_off + np.arange(width)
                rows = row_off + np.arange(height)
                # c and f are the origin of the transform.
                # a and e are the pixel size horizontally and vertically. 
                xs = transform.c + (cols + 0.5) * transform.a
                ys = transform.f + (rows + 0.5) * transform.e

                # Create the grid column/row indices
                grid_cols = np.floor((xs - x_min) / cell_size).astype(np.int32)
                grid_rows = np.floor((ys - y_min) / cell_size).astype(np.int32)

                # Create a 2D grid of integer cell indices
                # grid_row[:, None] reshapes the row indices into a column, and vice
                # versa for grid_cols[None, :]
                cell_indices = (grid_rows[:, None] * nx + grid_cols[None, :])

                # Filter bad pixels:
                # 1. Exclude background pixels
                valid = data != background_value
                # 2. Also remove (other) NoData values if there are any. In other words,
                # a pixel is valid only if its exists in both valid AND (data != src.nodata)
                if src.nodata is not None:
                    valid &= data != src.nodata
                # 3. Also respect the GDAL validity mask if there is any.
                # GDAL is a geospatial library that Rasterio uses to read raster files.
                # A validity mask indicates which pixels contain useable data.
                # Basically, only keep pixels that the mask does not consider invalid.
                valid &= src.read_masks(1, window = window) != 0
                is_forest = np.isin(data, forest_classes)
                is_water = np.isin(data, water_classes)
                # Only keep valid pixels.
                is_land = valid & ~is_water
                is_forest &= valid

                # Aggregate all cells in the current window. minlength = n_cells makes each result
                # at least as long as the total number of grid cells, even if some cells have
                # no matching pixels. This keeps the count arrays aligned by cell ID, so += 
                # can accumulate each chunk's counts into the correct cells.
                valid_counts += np.bincount(cell_indices[valid], minlength = n_cells)
                land_counts += np.bincount(cell_indices[is_land], minlength = n_cells)
                forest_counts += np.bincount(cell_indices[is_forest], minlength = n_cells)

            # The commas add thousands separators (e.g. 4,000/157,991).
            print(f"Processed row {row_off + height:,}/{src.height:,}")

    # All the _counts variables are currently very long 1D arrays. 
    # Now we want to convert these flat grid indices back into coordinates.
    # land_counts > 0 means that we only keep cells which contain at least 1 land pixel.
    indices = np.flatnonzero(land_counts > 0)
    grid_rows = indices // nx # The integer part is the row number
    grid_cols = indices % nx # The remainder part is the column number
    # Cell_x and cell_y represent the bottom-left corner of the cell.
    cell_x = x_min + grid_cols * cell_size
    cell_y = y_min + grid_rows * cell_size

    # For example, 10 km x 10 km = 1,000,000 pixels at 10 m resolution
    pixels_per_cell = (cell_size // res) ** 2

    # Create the final dataframe.
    # Land percent is calculated based on ALL pixels in a cell, whereas 
    # forest percent is calculated based on only land pixels in a cell!
    df = pd.DataFrame({
        "cell_id": [f"{x}_{y}" for x, y in zip(cell_x, cell_y)],
        "cell_x": cell_x,
        "cell_y": cell_y,
        "n_valid_pixels": valid_counts[indices],
        "n_land_pixels": land_counts[indices],
        "n_forest_pixels": forest_counts[indices],
        "land_pct": (100.0 * land_counts[indices] / pixels_per_cell),
        "forest_pct": (100.0 * forest_counts[indices] / land_counts[indices])
    })

    # Reset_index(drop = True) makes pandas restart the indexing at 0 after sorting
    # instead of letting each cell keep its original index.
    df = df.sort_values(["cell_y", "cell_x"]).reset_index(drop = True)

    if save_to_csv:
        # Create the directory if it doesn't already exist.
        output_path.parent.mkdir(parents = True, exist_ok = True)
        # Save the data as a CSV file for later use
        df.to_csv(output_path, index = False)
        # For Sweden, there are 4,983 valid grid cells out of 11,607 grid cells in total.
        print(f"Saved {len(df):,} valid grid cells out of {n_cells:,} total grid cells to {output_path}")

    return df



if __name__ == '__main__':
    coverage_df = create_master_grid(raster_path = RASTER_PATH,
                                     cell_size = CELL_SIZE,
                                     chunk_size = CHUNK_SIZE,
                                     background_value = BACKGROUND_VALUE,
                                     forest_classes = FOREST_CLASSES,
                                     water_classes = WATER_CLASSES,
                                     epsg_crs = 3006,
                                     res = 10.0,
                                     save_to_csv = True,
                                     output_path = OUTPUT_PATH)
