from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


# ---------------- Configuration ----------------
RESULT_DIR = Path("./results")
FIGURE_DIR = RESULT_DIR / "figures"

SCHEMES = ["temporal", "spatial", "spatiotemporal"]

MODEL_ORDER = ["seasonality", "environmental", "placename_count", "tfidf", "fasttext"]

COMPARISONS = [
    ("seasonality", "environmental"),
    ("environmental", "placename_count"),
    ("placename_count", "tfidf"),
    ("placename_count", "fasttext")
]


def load_results() -> DataFrame:
    """
    Retrive results from recognized model x scheme combinations.

    Returns:
        DataFrame
            A Pandas DataFrame containing all retrieved results. A run_id column distinguishes 
            results from different runs from each other.
    """
    frames = []

    # Get all results, but only load recognized configurations.
    for path in sorted(RESULT_DIR.glob("*.csv")):
        matched_scheme = None

        # Only load recognized scheme configurations.
        for scheme in SCHEMES:
            suffix = f"_{scheme}"
            if path.stem.endswith(suffix):
                matched_scheme = scheme
                model_name = path.stem[:-len(suffix)]
                break

        if matched_scheme is None:
            continue
        # Only load recognized model configurations.
        if model_name not in MODEL_ORDER:
            continue

        df = pd.read_csv(path)

        # Only return DataFrames that contain expected data.
        required = {"seed", "best_C", "test_ap", "test_prevalence", "test_positives"}
        if not required.issubset(df.columns):
            raise ValueError(f"Unexpected result format: {path}")

        df["model"] = model_name
        df["scheme"] = matched_scheme

        frames.append(df)

    if not frames:
        raise FileNotFoundError("No result files found.")

    results = pd.concat(frames, ignore_index = True)
    # Missing seed values (e.g. for temporal runs) are replaced with -1.
    results["run_id"] = (results["seed"].fillna(-1).astype(int))

    if results.duplicated(["scheme", "model", "run_id"]).any():
        raise ValueError("Duplicate results detected.")

    return results


def create_details(results: DataFrame) -> DataFrame:
    """
    Calculate average precision (AP) delta and lift compared to random chance (specifically
    compared to positive prevalance in the test data), and save the details to a CSV file.

    Parameters:
        results: DataFrame
            A Pandas DataFrame containing raw results.
    Returns:
        DataFrame
            A Pandas DataFrame containing the raw results + additional details.
    """
    details = results[["scheme", "model", "seed", "best_C", "test_ap", "test_prevalence", "test_positives"]].copy()

    details["ap_above_chance"] = (details["test_ap"] - details["test_prevalence"])
    details["ap_lift"] = (details["test_ap"] / details["test_prevalence"])

    # Save
    details.to_csv(RESULT_DIR / "comparison_details.csv", index = False)

    print("\nDetailed results:")
    print(details.round(6).to_string(index = False))

    return details


def create_summary(details: DataFrame) -> DataFrame:
    """
    Create a summary of a given statistics DataFrame and save the summary to a CSV file.

    Parameters:
        details: DataFrame
            A Pandas DataFrame containing detailed statistics.
    Returns:
        DataFrame
            A Pandas DataFrame containing the summary.
    """
    summary = (details.groupby(["scheme", "model"])
        .agg(
            runs = ("test_ap", "size"),
            mean_ap = ("test_ap", "mean"),
            std_ap = ("test_ap", "std"),
            mean_prevalence = ("test_prevalence", "mean"),
            mean_ap_above_chance = ("ap_above_chance", "mean"),
            mean_ap_lift = ("ap_lift", "mean")
        )
        .reset_index()
    )
    # Save
    summary.to_csv(RESULT_DIR / "summary_models.csv", index = False)

    print("\nModel summary:")
    print(summary.round(6).to_string(index = False))

    return summary


def create_incremental(results: DataFrame) -> DataFrame:
    """
    Calculate incremental changes between model configurations and save them to a CSV file.

    Parameters:
        results: DataFrame
            The Pandas DataFrame containing statistics from various model configurations,
            distinguished by run_id.
    Returns:
        DataFrame
            A Pandas DataFrame containing the input data + data about incremental changes.
    """
    records = []
    for scheme in SCHEMES:
        subset = results[results["scheme"] == scheme]

        if subset.empty:
            continue

        # Reshape the data so that each run_id becomes a row, each model becomes a column,
        # and the cell values are that model's test AP score.
        pivot = subset.pivot(index = "run_id", columns = "model", values = "test_ap")

        for simpler, richer in COMPARISONS:
            if not {simpler, richer}.issubset(pivot.columns):
                continue

            # Ignore any run that's missing a score for either model.
            paired = pivot[[simpler, richer]].dropna()

            # Missing seed values (e.g. for temporal runs) are filled with -1 in load_results.
            for run_id, row in paired.iterrows():
                records.append({
                    "scheme": scheme,
                    "seed": (np.nan if run_id == -1 else run_id),
                    "comparison": (f"{richer} - {simpler}"),
                    "delta_ap": (row[richer] - row[simpler])
                })

    incremental = pd.DataFrame(records, columns = ["scheme", "seed", "comparison", "delta_ap"])
    
    # Save
    incremental.to_csv(RESULT_DIR / "comparison_incremental.csv", index = False)

    print("\nIncremental improvements:")
    print(incremental.round(6).to_string(index = False))

    return incremental


# ---------------- Figure 1: Overall AP ----------------
def plot_overall_ap(summary: DataFrame) -> None:
    """
    Plot and save a diagram of overall average precision (AP) for given models and schemes.

    Parameters:
        summary: DataFrame
            A Pandas DataFrame containing the statistics for various models and schemes.
    """
    models = [m for m in MODEL_ORDER if m in summary["model"].unique()]

    # Squeeze = False makes subplots() return an array rather than a single Axes object.
    fig, axes = plt.subplots(
        nrows = 1,
        ncols = len(SCHEMES),
        figsize = (5 * len(SCHEMES), 5),
        sharey = True,
        squeeze = False
    )

    for ax, scheme in zip(axes[0], SCHEMES):
        subset = summary[summary["scheme"] == scheme].set_index("model")

        if subset.empty:
            ax.set_visible(False)
            continue

        available = [m for m in models if m in subset.index]
        labels = ["Prevalence"] + available
        values = [subset["mean_prevalence"].iloc[0]] + [subset.loc[m, "mean_ap"] for m in available]

        ax.bar(labels, values)
        ax.set_title(scheme.capitalize())
        ax.set_ylabel("Test Average Precision")
        ax.tick_params(axis = "x", rotation = 45)

    fig.suptitle("Model performance versus positive prevalence")
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "overall_ap.png", dpi = 200, bbox_inches = "tight")
    plt.close()


# ---------------- Figure 2: Spatial AP by seed ----------------
def plot_spatial_ap(results: DataFrame) ->  None:
    """
    Plot and save a diagram of average precision (AP) differences for spatial validation with various 
    seeds and models.

    Parameters:
        results: DataFrame
            A Pandas DataFrame containing the statistics for various models.
    """
    spatial = results[results["scheme"] == "spatial"].copy()

    if spatial.empty:
        return

    models = [m for m in MODEL_ORDER if m in spatial["model"].unique()]

    # Reshape the data so that each run_id becomes a row, each model becomes a column,
    # and the cell values are that model's test AP score.
    pivot = spatial.pivot(index = "run_id", columns = "model", values = "test_ap")

    prevalence = (spatial.groupby("run_id")["test_prevalence"].first())
    plot_data = pd.DataFrame({"Prevalence": prevalence})

    for model in models:
        plot_data[model] = pivot[model]

    ax = plot_data.plot(kind = "bar", figsize = (11, 5), width = 0.85)
    ax.set_title("Spatial test AP by random seed")
    ax.set_xlabel("Spatial split seed")
    ax.set_ylabel("Average Precision")
    ax.legend(title = "Model")

    plt.xticks(rotation = 0)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "spatial_ap_by_seed.png", dpi = 200)
    plt.close()


# ---------------- Figure 3: Incremental AP ----------------
def plot_incremental(incremental: DataFrame) -> None:
    """
    Plot and save a diagram of mean incremental deltas in mean average precision (AP) for various 
    schemes and models.

    Parameters:
        results: DataFrame
            A Pandas DataFrame containing the statistics for various schemes and models.
    """
    if incremental.empty:
        return

    summary = (incremental.groupby(["scheme", "comparison"])["delta_ap"].mean().unstack("comparison"))
    # Reorder the rows to match SCHEMES
    summary = summary.reindex(SCHEMES)

    ax = summary.plot(kind = "bar", figsize = (11, 5), width = 0.8)
    ax.axhline(0, color = "black", linewidth = 0.8)
    ax.set_title("Mean incremental improvement in test AP")
    ax.set_xlabel("Evaluation scheme")
    ax.set_ylabel("Change in Average Precision")
    ax.legend(title = "Added feature group", fontsize = 9)

    plt.xticks(rotation = 0)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / "incremental_ap.png", dpi = 200)
    plt.close()


# ---------------- Main ----------------
if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents = True, exist_ok = True)

    results = load_results()

    details = create_details(results)
    summary = create_summary(details)

    incremental = create_incremental(results)

    plot_overall_ap(summary)
    plot_spatial_ap(results)
    plot_incremental(incremental)

    print("\nFinished!")
    print(f"Tables saved to: {RESULT_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
