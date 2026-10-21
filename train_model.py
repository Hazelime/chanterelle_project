import argparse # Used for user input in Main
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------- Configuration ----------------
DATA_PATH = Path("./data/processed/cell_week_dataset.parquet")

SPLIT_DIR = Path("./data/splits")
RESULT_DIR = Path("./results")

SEEDS = [111, 222, 333, 444, 555]
C_VALUES = [0.01, 0.1, 1.0] # C = strength of regularization

"""FEATURES = [
    "week_sin",
    "week_cos",
    "temp_7d",
    "precip_7d",
    "forest_pct"
]"""
FEATURES = ["week_sin", "week_cos"]

MODEL_FEATURES = {
    "seasonality": [
        "week_sin",
        "week_cos"
    ],
    "environmental": [
        "week_sin",
        "week_cos",
        "temp_7d",
        "precip_7d",
        "forest_pct"
    ]
}


def load_data(features: list) -> DataFrame:
    """
    Load the columns of the given features from the master dataset.

    Parameters:
        features: list
            The list of desired features.
    Returns:
        DataFrame
            A Pandas DataFrame containing the data for the requested features.
    """
    # The asterisk (*) unpacks features
    columns = ["cell_id", "year", "observation", *features]
    df = pd.read_parquet(DATA_PATH, columns = list(dict.fromkeys(columns)))

    # Make sure the data makes sense before continuing.
    assert df["observation"].isin([0, 1]).all()
    assert not df[FEATURES].isna().any().any()

    return df


def get_splits(df: DataFrame, scheme: str, seed = None) -> DataFrame:
    # Get the assigned temporal split labels for each row in df.
    temporal = pd.read_parquet(SPLIT_DIR / "temporal_split.parquet")
    temporal_labels = df["year"].map(temporal.set_index("year")["temporal_split"])

    if scheme == "temporal":
        labels = temporal_labels

    else:
        if seed is None:
            raise ValueError("A seed is required for spatial evaluation.")

        # Get the assigned spatial split labels for each row in df.
        spatial = pd.read_parquet(SPLIT_DIR /f"spatial_blocks_seed{seed}.parquet")
        spatial_labels = df["cell_id"].map(spatial.set_index("cell_id")["spatial_split"])

        if scheme == "spatial":
            labels = spatial_labels
        elif scheme == "spatiotemporal":
            # First mark every row as "unused"
            labels = pd.Series("unused", index = df.index)
            # Then mark every row that is marked as x for both temporal and spatial
            # validation as x also for spatiotemporal validation.
            for split in ["train", "val", "test"]:
                mask = ((spatial_labels == split) & (temporal_labels == split))
                labels.loc[mask] = split
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    # Check that there are no empty labels before continuing.
    assert labels.notna().all()

    return labels


def build_model(C: float) -> Pipeline:
    """
    A more interepretable shell for scikit-learn's make_pipeline that builds a logistic
    regression model with standardizing and Limited-memory Broyden-Fletcher-Goldfarb-Shanno 
    (LBFGS) optimization to find coefficents. LBFGS is apparently is good general-purpose
    solver.

    Parameters: 
        C: float
            The regularization strength
    Returns:
        A scikit-learn Pipeline object.
    """
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C = C, max_iter = 1000, solver = "lbfgs"))


def evaluate(model: Pipeline, X: DataFrame, y: DataFrame) -> dict:
    """
    Calculate average precision, prevalance, raw number of positives, and number of rows
    for a given model and dataset.

    Parameters:
        model: Pipeline
            The model to evaluate, as constructed by scikit-learn's make_pipeline().
        X: DataFrame
            A Pandas DataFrame containing the features of the dataset.
        y: DataFrame
            A Pandas DataFrame containing the labels of the dataset.
    Returns:
        dict
            A dict containing the average precision, prevalance, raw number of positives,
            and number of rows for the given model and dataset.
    """
    scores = model.predict_proba(X)[:, 1]

    return {
        "ap": average_precision_score(y, scores),
        "prevalence": float(np.mean(y)),
        "positives": int(np.sum(y)),
        "rows": len(y)
    }


# ---------------- Run one experiment ----------------
def run_experiment(df: DataFrame,
                   features: list,
                   scheme: str,
                   seed: int | None = None,
                   fixed_C: float | None = None) -> dict:
    """
    Run an experiment for a given dataset, set of features, scheme, seed, and 
    list of potential regularization strengths (C).

    Parameters:
        df: DataFrame
            The data with which to train, validate, and test a model.
        features: list
            The features to include in the experiment.
        scheme: str
            What type of validation scheme to run (temporal, spatial, or spatiotemporal)
        seed: int | None
            The randomizer seed for spatial and spatiotemporal validation. Temporal
            validation does not require any seed.
        fixed_C: float | None
            A fixed regularization strength (in case you want to skip tuning)
    Returns:
        dict
            A dict with metadata and results from the experiment, including scheme,
            seed, best_C, val_ap, val_prevalence, val_positives, test_ap, test_prevalance,
            and test_positives. 
    """
    print(f"\nRunning {scheme}, seed = {seed}")

    labels = get_splits(df, scheme, seed)

    train_mask = (labels == "train").to_numpy()
    val_mask = (labels == "val").to_numpy()
    test_mask = (labels == "test").to_numpy()

    X = df[features].to_numpy(dtype = np.float32)
    y = df["observation"].to_numpy(dtype = np.int8)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    assert y_train.sum() > 0
    assert y_val.sum() > 0
    assert y_test.sum() > 0

    # ---------------- Hyperparameter selection ----------------
    # Select regularization strength (C) based on validation average precision (AP).
    # (Spatiotemporal evaluation uses C selected by temporal validation.)
    # Tuning can be skipped by giving the function a fixed C.
    if fixed_C is None:
        best_C = None
        best_val_ap = -np.inf

        for C in C_VALUES:
            model = build_model(C)
            model.fit(X_train, y_train)

            val_result = evaluate(model, X_val, y_val)
            print(f"  C = {C} -> validation AP = {val_result['ap']:.6f}")

            if val_result["ap"] > best_val_ap:
                best_val_ap = val_result["ap"]
                best_C = C
    else:
        best_C = fixed_C

    # ---------------- Final model ----------------
    # Fit the selected model. Note that this rebuilds and reevaluates one
    # of the models already built during the hyperparameter selection above.
    model = build_model(best_C)
    model.fit(X_train, y_train)

    val_result = evaluate(model, X_val, y_val)
    test_result = evaluate(model, X_test, y_test)

    result = {
        "scheme": scheme,
        "seed": seed,
        "best_C": best_C,
        "val_ap": val_result["ap"],
        "val_prevalence": val_result["prevalence"],
        "val_positives": val_result["positives"],
        "test_ap": test_result["ap"],
        "test_prevalence": test_result["prevalence"],
        "test_positives": test_result["positives"],
    }

    print(f"  Selected regularization strength (C): {best_C}")
    print(f"  Test AP: {test_result['ap']:.6f}")
    print(f"  Test prevalence: {test_result['prevalence']:.6f}")

    return result


if __name__ == "__main__":
    # Let the user choose what kind of experiment to run by adding flags:
    # --model with one of (seasonality, environmental) to choose which features to include
    # --scheme with one of (temporal, spatial, spatiotemporal) to choose type of experiment
    # Both flags must be included. Sample usage:
    # python3 train_model.py --model environmental --scheme temporal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices = list(MODEL_FEATURES.keys()),
        required = True
    )
    parser.add_argument(
        "--scheme",
        choices = ["temporal", "spatial", "spatiotemporal"],
        required = True
    )
    args = parser.parse_args()
    model_name = args.model
    scheme = args.scheme

    features = MODEL_FEATURES[model_name]
    
    print(f"\nModel: {model_name}")
    print(f"Evaluation: {scheme}")
    print(f"Features: {features}")

    # Create a results directory if it does not already exist.
    RESULT_DIR.mkdir(parents = True, exist_ok = True)

    df = load_data(features)
    results = []

    if args.scheme == "temporal":
        # Takes like 15 seconds on my laptop.
        results.append(run_experiment(df, features, "temporal"))
    elif args.scheme == "spatial":
        # Takes about 2-3 minutes on my laptop.
        for seed in SEEDS:
            results.append(run_experiment(df, features, "spatial", seed))
    elif args.scheme == "spatiotemporal":
        # Takes about 2-3 minutes on my laptop.
        for seed in SEEDS:
            results.append(run_experiment(df, features, "spatiotemporal", seed))

    results_df = pd.DataFrame(results)

    # ---------------- Save results ----------------
    output_path = (RESULT_DIR / f"{model_name}_{scheme}.csv")
    # Index = False ignores the DataFrame index column when writing to the CSV.
    results_df.to_csv(output_path, index = False)

    print("\nResults:")
    print(results_df.to_string(index = False))

    if len(results_df) > 1:
        print("\nMean test AP:", results_df["test_ap"].mean())
        print("Standard deviation:", results_df["test_ap"].std())

    print(f"\nSaved: {output_path}")
