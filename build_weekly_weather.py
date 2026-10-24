import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

"""
This script calculates average temperature and total precipitation for the preceding
week for all weeks during 1995-2025 using data from data/weather, aligns the result with
the master grid, and saves it as data/sweden_weekly_weather.parquet.
"""

# --------------Configuration-----------------------
DATA_DIR = Path("./data/weather")
# This simply adds "/temperature" to the DATA_DIR path
TEMP_DIR = DATA_DIR / "temperature"
PRECIP_DIR = DATA_DIR / "precipitation"
MAPPING_PATH = DATA_DIR / "smhi_to_master_grid.parquet"
OUTPUT_PATH = DATA_DIR / "sweden_weekly_weather.parquet"

START_YEAR = 1995
END_YEAR = 2025

# -----------Load SMHI-to-grid mapping--------------
mapping = pd.read_parquet(MAPPING_PATH)
# Return_inverse = True makes np.unique additionally return an index of the cell ids
# (groud_idx) for easy future reference.
cell_ids, group_idx = np.unique(mapping["cell_id"].values, return_inverse = True)

n_cells = len(cell_ids)
print(f"Loaded mapping for {n_cells:,} cells.") # 4983

# -----------Daily spatial aggregation--------------
def aggregate_year(year: int) -> DataFrame:
    """
    Collects all the SMHI cells associated with each master-grid cell and calculates
    the mean temperature and mean precipitation for each day of the given year.

    Parameters:
        year : int
            The year for which to calculate daily temperature and precipitation
            averages
    Returns:
        DataFrame
            A Pandas DataFrame where each row is itself a DataFrame containing
            cell_id, date, temp, and precip. 
    """
    temp_path = (TEMP_DIR / f"SMHI_pthbv_tas_{year}_{year}_daily.nc")
    precip_path = (PRECIP_DIR / f"SMHI_pthbv_pr_{year}_{year}_daily.nc")

    with xr.open_dataset(temp_path) as tas_ds, \
         xr.open_dataset(precip_path) as pr_ds:

        # Check compatability of the temperature and precipitation data.
        # Do they cover the same coordinates?
        assert np.array_equal(tas_ds["x"].values, pr_ds["x"].values)
        assert np.array_equal(tas_ds["y"].values, pr_ds["y"].values)
        # Do they cover the same datetimes (after normalization)?
        dates_tas = pd.to_datetime(tas_ds["time"].values).normalize()
        dates_pr = pd.to_datetime(pr_ds["time"].values).normalize()
        assert np.array_equal(dates_tas, dates_pr)

        nx = tas_ds.sizes["x"]

        source_indices = (
            mapping["smhi_y_idx"].to_numpy() * nx
            + mapping["smhi_x_idx"].to_numpy()
        ).astype(np.int64)

        # Aggregate the SMHI data associated with each master-grid cell and average it per cell.
        rows = []
        for day_idx, date in enumerate(dates_tas):
            # The mapping variable links each master-grid cell to one or more SMHI grid point.
            # The first line below selects the temperature grid for the current day in the
            # loop, picks the SMHI points listed in the mapping (as indicated by source_indices)
            # and produces an array with one temperature value per mapping row. Breakdown:
            # tas_ds["tas"] selects the temperature variable from the SMHI dataset
            # .isel(time = day_idx) selects the current day in the loop.
            # .values gets the grid values as a 2D NumPy array.
            # .ravel() flattens the 2D grid into a 1D array.
            # [source_indices] filters for the positions listed in source_indices, which
            #   identify the SMHI grid points that correspond to the cells in mapping.
            temp = (tas_ds["tas"].isel(time = day_idx).values.ravel()[source_indices])
            precip = (pr_ds["pr"].isel(time = day_idx).values.ravel()[source_indices])

            def aggregate_mean(values):
                # Only count finite values.
                valid = np.isfinite(values)
                # Minlength = n_cells ensures that sums and counts have an entry for every cell,
                # even if that cell has no valid values (the sum/count value will then be NaN).
                # We use NumPy functions here to speed up calculations.
                sums = np.bincount(group_idx[valid], weights = values[valid], minlength = n_cells)
                counts = np.bincount(group_idx[valid], minlength = n_cells)
                # The output is an array with the mean for cells with data, and NaN for cells without.
                return np.divide(sums, counts, out = np.full(n_cells, np.nan), where = counts > 0)

            temp_mean = aggregate_mean(temp)
            precip_mean = aggregate_mean(precip)

            rows.append(pd.DataFrame({
                "cell_id": cell_ids,
                "date": date,
                "temp": temp_mean.astype("float32"),
                "precip": precip_mean.astype("float32")
            }))

    return pd.concat(rows, ignore_index = True)

# ---------------Weekly aggregation-----------------
def aggregate_weeks(daily_df: DataFrame) -> DataFrame:
    """
    Aggregates the data by week and cell_id and then returns useful weekly features.
    Weeks without complete data are NaN.

    Parameters:
        daily_df : DataFrame
            A Pandas DataFrame containing a year's worth of daily temperature and
            precipitation averages by cell_id.
    
    Returns:
        DataFrame
            A Pandas DataFrame with one row per cell_id x year x week combination.
            It contains cell_id, year, week, temp_7d, and precip_7d. The two latter
            columns are mean temperature and total precipitation during the
            preceding (!) week.
    """
    # Create a copy so we don't mess up the original data.
    df = daily_df.copy()

    # to_timedelta(...) converts each day to a number (0-6). unit = "D" means days.
    df["week_start"] = (df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit = "D"))

    # Aggregate the data per week and group by cell_id and week.
    weekly = (df.groupby(["cell_id", "week_start"], as_index = False)
        .agg(
            temp_7d = ("temp", "mean"),
            precip_7d = ("precip", "sum"),
            n_days = ("date", "nunique"),
            n_temp = ("temp", "count"),
            n_precip = ("precip", "count")
        )
    )

    # Handle incomplete weeks
    complete = (
        (weekly["n_days"] == 7)
        & (weekly["n_temp"] == 7)
        & (weekly["n_precip"] == 7)
    )
    # Set the weekly temperature and precipitation values to NaN
    # for weeks that don't have complete data.
    weekly.loc[~complete, ["temp_7d", "precip_7d"]] = np.nan

    # Features describe the preceding week.
    weekly["week_start"] += pd.Timedelta(days = 7)
    iso = weekly["week_start"].dt.isocalendar()
    weekly["year"] = iso.year.astype("int16")
    weekly["week"] = iso.week.astype("int8")

    return weekly[["cell_id", "year", "week", "temp_7d", "precip_7d"]]

# ---------------Process all years------------------
if __name__ == "__main__":
    all_weekly = []

    # Keep the final 13 days of the previous year for cross-year weeks.
    previous_tail = None

    for year in range(1994, END_YEAR + 1):
        print(f"\nProcessing {year}...")
        # daily contains the temperature and precipitation averages for
        # each cell_id and day of the current year.
        daily = aggregate_year(year = year)

        # Add the data from the last few days of the previous year
        # so we cover the preceding week even for week 1.
        if previous_tail is not None:
            daily = pd.concat(
                [previous_tail, daily],
                ignore_index = True
            )

        # Aggregate data for complete weeks and calculate average
        # temperature and total precipitation for the preceding (!) week. 
        weekly = aggregate_weeks(daily)

        # ------ Data cleanup----------
        # Only keep target years in the requested range.
        weekly = weekly[
            (weekly["year"] >= START_YEAR)
            & (weekly["year"] <= END_YEAR)
        ]
        
        # Save the weekly data for this year.
        all_weekly.append(weekly)

        # Keep data for the final 13 days of the current year for next year.
        current_year_daily = daily[daily["date"].dt.year == year]
        previous_tail = current_year_daily[
            current_year_daily["date"] >=
            pd.Timestamp(year, 12, 19)
        ].copy()

        print(f"Weekly rows added: {len(weekly):,}")

    # Combine results.
    result = pd.concat(all_weekly, ignore_index = True)

    # Add a temporary _complete column to mark entries that are complete.
    result["_complete"] = result[["temp_7d", "precip_7d"]].notna().all(axis = 1)
    result = result.sort_values("_complete", ascending = False, kind = "stable")
    # Drop duplicates.
    result = result.drop_duplicates(
        subset = ["cell_id", "year", "week"],
        keep = "first"
    )
    # Drop incomplete cell-weeks.
    incomplete = result[~result["_complete"]]
    print(
        "Incomplete cell-week records excluded:",
        f"{len(incomplete):,}",
        f"(temperature missing: {incomplete['temp_7d'].isna().sum():,}; "
        f"precipitation missing: {incomplete['precip_7d'].isna().sum():,})"
    )
    # Remove the temporary _complete column.
    result = result[result["_complete"]].drop(columns = "_complete")

    # Convert to float32 to reduce file size.
    result["temp_7d"] = (result["temp_7d"].astype("float32"))
    result["precip_7d"] = (result["precip_7d"].astype("float32"))

    result = result.sort_values(["cell_id", "year", "week"]).reset_index(drop = True)

    # Save as a compressed Parquet file.
    result.to_parquet(OUTPUT_PATH, compression = "zstd", index = False)

    print("\nFinished!")
    print("Rows:", len(result)) # 8057511
    print("Cells:", result["cell_id"].nunique()) # 4983
    print("Missing values:") # 0 across the board
    print(result.isna().sum())
    print("Saved to:", OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)
    print(f"File size: {size_mb:.1f} MB") # 64.7 MB


    # Validation:
    df = pd.read_parquet("./data/weather/sweden_weekly_weather.parquet")

    print("Shape:", df.shape) # (8057511, 5)
    print("Unique cells:", df["cell_id"].nunique()) # 4983

    print("\nFirst weeks:")
    print(df[["year", "week"]].drop_duplicates().sort_values(["year", "week"]).head())

    print("\nLast weeks:")
    print(df[["year", "week"]].drop_duplicates().sort_values(["year", "week"]).tail())

    print("\nMissing values:")
    print(df.isna().sum()) # 0 across the board

    print("\nDuplicate cell-weeks:")
    print(df.duplicated(["cell_id", "year", "week"]).sum()) # 0


    # Check incomplete weeks
    counts = (df.groupby(["year", "week"])["cell_id"].nunique())

    print("Minimum cells per week:", counts.min()) # 4983
    print("Maximum cells per week:", counts.max()) # 4983

    print("\nWeeks with fewer than 4,983 cells:")
    print(counts[counts < 4983]) # none