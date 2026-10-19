from shapely.geometry import box
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from pandas import DataFrame
from geopandas import GeoDataFrame

"""
This script attaches the observations to the cells in the grid and then plots a heat map of aggregate
observations between 1995 and 2025.
"""

observations_path = "./data/observations/filtered_gbif_observations.parquet"
output_path = "./plots/aggregate_chanterelle_observations.png"

def get_gdf_grid(df: DataFrame) -> GeoDataFrame:
    """
    Creates a 10 km x 10 km grid by converting a Pandas DataFrame with coordinates described in longitude and
    latitude (ESPG:4326) to a GeoPandas GeoDataFrame with coordinates described in meters (ESPG:3006).

    Parameters:
        df: DataFrame
            The Pandas DataFrame to be converted.
    
    Returns:
        GeoDataFrame
            The converted GeoDataFrame. Each cell has grid_x and grid_y coordinates along with a grid_id in the format
            x_y.
    """
    # Initialize a GeoDataFrame. EPSG:4326 means that coordinates are described in longitude and latitude degress.
    gdf = gpd.GeoDataFrame(df,
                           geometry = gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
                           crs = "EPSG:4326")
    # Convert the filtered observations to an appropriate coordinate reference system (i.e. EPSG:3006, which is what
    # SWEREF99 TM uses). Then we have the locations described in meters rather than longitude and latitude degrees,
    # which makes it easier to assign observations to grid cells.
    gdf = gdf.to_crs("EPSG:3006")

    # Assign every observation to a 10 km x 10 km grid cell, using the SWEREF99 TM coordinates. The grid cell ID is a string
    # in the format "x_y", where x and y are the SWEREF99 TM coordinates of the lower left corner of the grid cell,
    # divided by 10,000 (to get the grid cell index).
    gdf["grid_x"] = (gdf.geometry.x // 10000).astype(int)
    gdf["grid_y"] = (gdf.geometry.y // 10000).astype(int)
    gdf["grid_id"] = (gdf["grid_x"].astype(str) + "_" + gdf["grid_y"].astype(str))
    
    return gdf

observations_gdf = get_gdf_grid(df = pd.read_parquet(observations_path))

# Get one row per grid cell. grid_cells is now a regular DataFrame.
grid_cells = observations_gdf[["grid_id", "grid_x", "grid_y"]].drop_duplicates()

# Create a 10 × 10 km square for each cell
# Axis = 1 means that the function should be applied to each row (rather than each column)
grid_cells["geometry"] = grid_cells.apply(
    lambda row: box(
        row["grid_x"] * 10000,
        row["grid_y"] * 10000,
        (row["grid_x"] + 1) * 10000,
        (row["grid_y"] + 1) * 10000
    ),
    axis = 1
)

# Turn the object back into a GeoDataFrame.
grid_gdf = gpd.GeoDataFrame(grid_cells, geometry = "geometry", crs = "EPSG:3006")

# Attach the observation counts to the grid cells. Cells without an associated count are zeroed.
cell_counts = observations_gdf["grid_id"].value_counts()
grid_gdf["observation_count"] = (grid_gdf["grid_id"].map(cell_counts).fillna(0))

#---Add an outline of Sweden for visualization purposes---
# Load country boundaries
world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
# Select Sweden
sweden = world[world["NAME"] == "Sweden"].copy()
# Convert to the same coordinate system as the grid
sweden = sweden.to_crs("EPSG:3006")

# Create a plotting area
# Figure is the whole figure; ax is the object we're drawing on.
fig, ax = plt.subplots(figsize = (10, 14))

# Draw Sweden outline
sweden.boundary.plot(ax = ax, linewidth = 1)

# Draw observation grid cells
grid_gdf.plot(
    column = "observation_count",
    ax = ax,
    legend = True,
    edgecolor = "black",
    linewidth = 0.1
)

ax.set_title("Aggregate 1995-2025 chanterelle observations by 10 × 10 km grid cell")
ax.set_axis_off()

plt.savefig(output_path, dpi = 300, bbox_inches = "tight")
plt.show()
plt.close()

