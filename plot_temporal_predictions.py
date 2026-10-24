from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import geopandas as gpd
from scipy.special import xlogy

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, ListedColormap, BoundaryNorm, SymLogNorm
from matplotlib.patches import Patch
from shapely.geometry import box

from train_model import load_data, prepare_tfidf, NUMERIC_FEATURES
from tfidf_features import build_documents

"""
This script plots various heatmaps based on the input and results data and saves them
in results/maps.
"""

# ---------------- Configuration ----------------
GRID_PATH = Path("./data/sweden_master_grid.csv")
TFIDF_MODEL_PATH = Path("./results/models/tfidf_temporal.joblib")
LOG_MODEL_PATH = Path("./results/models/placename_count_log_temporal.joblib")
OUTPUT_DIR = Path("./results/maps")

CELL_SIZE = 10000
TEST_YEAR = 2025

# ---------------- Load modelling data ----------------
print("Loading modelling data...")
df = load_data(NUMERIC_FEATURES)

test = df[df["year"] == TEST_YEAR].copy()
assert len(test) == 4983 * 52
assert test["cell_id"].nunique() == 4983

print(f"Loaded {len(test):,} test cell-weeks.")

# ---------------- Historical observations ----------------
historical = (df.groupby("cell_id", as_index = False)
    .agg(historical_observed=("observation", "sum")))
print("Historical positive cell-weeks:", f"{historical['historical_observed'].sum():,}")

# ---------------- Load fitted models ----------------
tfidf_bundle = joblib.load(TFIDF_MODEL_PATH)
log_bundle = joblib.load(LOG_MODEL_PATH)

tfidf_model = tfidf_bundle["model"]
tfidf_vectorizer = tfidf_bundle["vectorizer"]

log_model = log_bundle["model"]
log_features = log_bundle["features"]

# ---------------- Log-count predictions ----------------
print("Generating log-count predictions...")
X_log = test[log_features].to_numpy(dtype = np.float32)
# Get positive predictions (1)
test["p_log"] = log_model.predict_proba(X_log)[:, 1]

# ---------------- TF-IDF predictions ----------------
print("Generating TF-IDF predictions...")

# Reconstruct the same sorted cell order used when fitting the temporal TF-IDF model.
placenames = pd.read_parquet("./data/placenames/cell_placenames.parquet",
                             columns  =["cell_id", "placenames"])

# Sort so cell_index will align with the rows of V later.
documents = build_documents(placenames).sort_values("cell_id").reset_index(drop = True)
# Create a stable mapping from cell_id to TF-IDF feature.
cell_index = pd.Series(np.arange(len(documents),
                       dtype = np.int32),
                       index = documents["cell_id"])
test_cell_idx = (test["cell_id"].map(cell_index).to_numpy(dtype = np.int32))

assert (test_cell_idx >= 0).all()

# Use the saved training-fitted vectorizer (instead of fitting the TF-IDF vectorizer again).
V = tfidf_vectorizer.transform(documents["document"]).tocsr()

# Get positive predictions (1)
X_tfidf = test[NUMERIC_FEATURES].to_numpy(dtype = np.float32)
test["p_tfidf"] = tfidf_model.predict_proba(X_tfidf, test_cell_idx, V)[:, 1]

# ---------------- Cell-week log loss ----------------
def binary_log_loss(y: npt.NDArray, p: npt.NDArray) ->  npt.NDArray:
    """
    Calculate binary cross-entropy per observation.
    
    Parameters:
        y: npt.NDArray
            An array of binary labels (0 or 1) representing the observation predictions.
        p: npt.NDArray
            An array of floats representing the prediction probabilities.
    Returns:
         npt.NDArray
            An array of floats representing the loss for each label/prediction.
    """
    # Clip slightly above 0 and slightly below 1 to avoid log(0) making the loss infinite.
    p = np.clip(p, 1e-15, 1 - 1e-15)
    # xlogy(y, p) computes y * log(p) but returns 0 if both x and p are 0 (instead of undefined).
    return -(xlogy(y, p) + xlogy(1 - y, 1 - p))

test["loss_log"] = binary_log_loss(test["observation"].to_numpy(), test["p_log"].to_numpy())
test["loss_tfidf"] = binary_log_loss(test["observation"].to_numpy(), test["p_tfidf"].to_numpy())

# ---------------- Aggregate by cell ----------------
print("Aggregating predictions by cell...")
cell_data = (test.groupby("cell_id", as_index = False)
    .agg(
        observed = ("observation", "sum"),
        expected_log = ("p_log", "sum"),
        expected_tfidf = ("p_tfidf", "sum"),
        mean_p_log = ("p_log", "mean"),
        mean_p_tfidf = ("p_tfidf", "mean"),
        mean_loss_log = ("loss_log", "mean"),
        mean_loss_tfidf = ("loss_tfidf", "mean"),
        weeks = ("observation", "size")))

cell_data["prediction_difference"] = (cell_data["mean_p_tfidf"] - cell_data["mean_p_log"])
cell_data["loss_difference"] = (cell_data["mean_loss_tfidf"] - cell_data["mean_loss_log"])
cell_data["tfidf_residual"] = (cell_data["expected_tfidf"] - cell_data["observed"])

assert (cell_data["weeks"] == 52).all()
assert len(cell_data) == 4983

# ---------------- Save cell-level data ----------------
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
cell_data.to_parquet(OUTPUT_DIR / "temporal_cell_predictions_2025.parquet", index = False)

# ---------------- Reconstruct grid geometry ----------------
print("Building grid geometries...")
grid = pd.read_csv(GRID_PATH)

assert grid["cell_id"].is_unique

geometry = [box(x, y, x + CELL_SIZE, y + CELL_SIZE)
            for x, y in zip(grid["cell_x"], grid["cell_y"])]
gdf = gpd.GeoDataFrame(grid[["cell_id"]], geometry = geometry, crs = "EPSG:3006")
gdf = gdf.merge(cell_data, on = "cell_id", how = "left", validate = "one_to_one")
gdf = gdf.merge(historical, on = "cell_id", how = "left", validate = "one_to_one")

assert gdf["observed"].notna().all()
assert gdf["historical_observed"].notna().all()

# ---------------- Plotting helpers ----------------
def plot_observations():
    """
    Plots a map of observations on the cell-grid, where cells with more observations being more
    brightly colored.
    """
    # Five discrete colors from viridis.
    colors = plt.get_cmap("viridis")([0.0, 0.25, 0.50, 0.75, 1.0])
    cmap = ListedColormap(colors)

    # Limits for each color.
    boundaries = [-0.5, 0.5, 1.5, 2.5, 4.5, np.inf]
    norm = BoundaryNorm(boundaries, ncolors = cmap.N)

    fig, ax = plt.subplots(figsize = (9, 13))

    gdf.plot(
        column = "observed",
        ax = ax,
        cmap = cmap,
        norm = norm,
        edgecolor = "none",
        legend = False
    )

    # Manual categorical legend.
    labels = ["0", "1", "2", "3–4", "5+"]
    handles = [Patch(facecolor = colors[i], label = labels[i]) for i in range(5)]
    ax.legend(handles = handles,
              title = "Positive cell-weeks",
              loc = "lower right",
              frameon = False)
    ax.set_title("Observed positive cell-weeks (2025)", fontsize = 15, pad = 15)
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "01_observed_2025.png", dpi = 200, bbox_inches = "tight")
    plt.close(fig)


def plot_historical_observations():
    """
    Plots a map of all positive cells on the cell-grid, where cells with more positive
    cell-weeks being more brightly colored.
    """
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, 6))
    cmap = ListedColormap(colors)
    # Somewhat arbitrary color boundaries.
    boundaries = [-0.5, 0.5, 2.5, 5.5, 10.5, 25.5, np.inf]
    norm = BoundaryNorm(boundaries, ncolors = cmap.N)

    fig, ax = plt.subplots(figsize = (9, 13))
    gdf.plot(
        column = "historical_observed",
        ax = ax,
        cmap = cmap,
        norm = norm,
        edgecolor = "none",
        legend = False
    )

    labels = ["0", "1–2", "3–5", "6–10", "11–25", "26+"]
    handles = [Patch(facecolor = colors[i], label = labels[i]) for i in range(len(labels))]
    ax.legend(handles = handles, title = "Positive cell-weeks", loc = "lower right", frameon = False)
    ax.set_title("Historical positive cell-weeks (1995–2025)", fontsize = 15, pad = 15)
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "07_historical_observations.png", dpi = 200, bbox_inches = "tight")
    plt.close(fig)


def plot_map(column: str,
             title: str,
             filename: str,
             cmap: str = "viridis",
             centered: bool = False,
             label: str | None = None,
             vmin: int | None = None,
             vmax: int | None = None,
             percentile: int | None = None,
             norm: Normalize | None = None) -> None:
    """
    Plots a map for the given column using the given color settings.

    Arguments:
        column: str
            The column in gdf to use as data for plotting.
        title: str
            The title of the map.
        filename: str
            The saving path.
        cmap: str ("viridis" by default)
            The matplotlib color map.
        centered: bool (False by default)
            Whether the color-scale should be centered around zero (True) or not (False).
        vmin: int | None (None by default)
            The lower color-scale limit (if not centered).
        vmax: int | None (None by default)
            The upper color-scale limit (if not centered).
        percentile: int | None (None by default)
            The percentile of the data to use when setting a color-scale limit. This 
            reduces the influence of extreme values.
        norm: Normalize | None (None by default)
            A Matplotlib normalization object to determine how to plot values (e.g.
            linearly or logarithmically)
    """
    fig, ax = plt.subplots(figsize = (9, 13))
    values = gdf[column].to_numpy()

    plot_kwargs = {
        "column": column,
        "ax": ax,
        "cmap": cmap,
        "legend": True,
        "edgecolor": "none",
        "legend_kwds": {"label": label or column, "shrink": 0.65},
    }
    if norm is not None:
        # Use explicitly supplied normalization.
        plot_kwargs["norm"] = norm
    
    # Create a scale centered at zero if centered = True; otherwise, use percentile and/or vmin
    # and vmax to set the color-scale limits.
    if centered:
        if percentile is not None:
            limit = np.percentile(np.abs(values), percentile)
        else:
            limit = np.max(np.abs(values))
        if limit > 0:
            plot_kwargs["norm"] = TwoSlopeNorm(vmin = -limit, vcenter = 0, vmax = limit)
    else:
        if percentile is not None:
            vmax = np.percentile(values, percentile)
        if vmin is not None:
            plot_kwargs["vmin"] = vmin
        if vmax is not None:
            plot_kwargs["vmax"] = vmax

    gdf.plot(**plot_kwargs)

    ax.set_title(title, fontsize = 15, pad = 15)
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / filename, dpi = 200, bbox_inches = "tight")
    plt.close(fig)

# ---------------- Generate maps ----------------
print("Generating maps...")

# Map 1: Observations
plot_observations()

# Map 2: Log-count predictions
# Use the same color scale for maps 2 and 3.
all_predictions = np.concatenate([gdf["mean_p_log"].to_numpy(),
                                  gdf["mean_p_tfidf"].to_numpy()])
shared_max = np.percentile(all_predictions, 99)
plot_map(
    "mean_p_log",
    "Log-count predicted probability (2025)",
    "02_log_count_predictions_2025.png",
    vmin = 0,
    vmax = shared_max,
    label = "Mean weekly probability"
)
# Map 3: TF-IDF predictions
plot_map(
    "mean_p_tfidf",
    "TF-IDF predicted probability (2025)",
    "03_tfidf_predictions_2025.png",
    vmin = 0,
    vmax = shared_max,
    label = "Mean weekly probability"
)

# Map 4: Prediction difference between TF-IDF and log-count
values = gdf["prediction_difference"].to_numpy()
limit = np.percentile(np.abs(values), 99)
# Values between -0.0001 and 0.0001 are scaled linearly; the rest are scaled logarithmically
# in base 10. linscale = 1 means the positive and negative halves of the linear region gets
# equal visual space to one decade in the logarithmic regions.
symlog_norm = SymLogNorm(linthresh = 0.0001,
                         linscale = 1,
                         vmin = -limit,
                         vmax = limit,
                         base = 10)
plot_map(
    "prediction_difference",
    "TF-IDF minus log-count prediction (2025)",
    "04_prediction_difference_2025.png",
    cmap = "RdBu_r",
    norm = symlog_norm,
    label = "Difference in mean weekly probability"
)
# Map 5: TF-IDF residuals (expectations minus real observations)
plot_map(
    "tfidf_residual",
    "TF-IDF expected minus observed (2025)",
    "05_tfidf_residuals_2025.png",
    cmap = "RdBu_r",
    centered = True,
    percentile = 99,
    label = "Expected minus observed cell-weeks"
)
# Map 6: Difference in log loss between TF-iDF and log-count
values = gdf["loss_difference"].to_numpy()
# Reuse the symlog norm from map 4.
plot_map(
    "loss_difference",
    "TF-IDF minus log-count log loss (2025)",
    "06_log_loss_difference_2025.png",
    cmap = "RdBu_r",
    norm = symlog_norm,
    label = "Difference in mean weekly log loss",
)
# Map 7: Historical observations 1995-2025
plot_historical_observations()

# ---------------- Summary ----------------
print("\nFinished!")

print(f"Cells: {len(cell_data):,}") # 4,983
print(f"Observed positives: {cell_data['observed'].sum():,}") # 652
print(f"Expected positives (log count): {cell_data['expected_log'].sum():.1f}") # 364.6
print(f"Expected positives (TF-IDF): {cell_data['expected_tfidf'].sum():.1f}") # 314.2

print("\nLog-loss comparison:")
print(f"Mean log loss (log count): {test['loss_log'].mean():.8f}") # 0.01415640
print(f"Mean log loss (TF-IDF): {test['loss_tfidf'].mean():.8f}") # 0.01381554
print(f"Cells where TF-IDF performs better: {(cell_data['loss_difference'] < 0).sum():,}") # 4,118
print(f"Cells where log count performs better: {(cell_data['loss_difference'] > 0).sum():,}") # 865

print("\nLog-loss improvement by cell group:")
for label, mask in [("Cells with observations", cell_data["observed"] > 0),
                    ("Cells without observations", cell_data["observed"] == 0)]:
    subset = cell_data.loc[mask]
    print(f"\n{label}:")
    print(f"  Cells: {len(subset):,}") # 450 vs. 4,533
    # -0.00058597 vs. -0.00031653
    print(f"  Mean loss difference: {subset['loss_difference'].mean():.8f}")
    # 190 vs. 3,928
    print(f"  Cells where TF-IDF is better: {(subset['loss_difference'] < 0).sum():,}")

print("\nLog-loss comparison by observation label:")
for label in [0, 1]:
    subset = test[test["observation"] == label]

    log_loss = subset["loss_log"].mean()
    tfidf_loss = subset["loss_tfidf"].mean()

    print(f"\nObservation = {label}")
    print(f"Cell-weeks: {len(subset):,}") # 258,464 vs. 652
    print(f"Log-count loss: {log_loss:.8f}") # 0.00139331 vs. 5.07366371
    print(f"TF-IDF loss: {tfidf_loss:.8f}") # 0.00118650 vs. 5.02018195
    print(f"Difference: {tfidf_loss - log_loss:+.8f}") # -0.00020681 vs. -0.05348177

print("\nHistorical observations per cell:")
print(historical["historical_observed"].describe(percentiles = [0.5, 0.75, 0.90, 0.95, 0.99]))
# count    4983.000000
# mean        2.069436
# std         7.276331
# min         0.000000
# 50%         0.000000
# 75%         1.000000
# 90%         5.000000
# 95%        10.000000
# 99%        29.000000
# max       195.000000
print(f"Cells with historical observations: {(historical['historical_observed'] > 0).sum():,}")
# Cells with historical observations: 1,734

print(f"\nMaps saved to: {OUTPUT_DIR}")
