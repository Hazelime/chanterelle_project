import argparse # Used for user input in Main
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


# ---------------- Load data ----------------

def load_data(features):
    # The asterisk (*) unpacks features
    columns = ["cell_id", "year", "observation", *features]
    df = pd.read_parquet(DATA_PATH, columns = list(dict.fromkeys(columns)))

    # Make sure the data makes sense before continuing.
    assert df["observation"].isin([0, 1]).all()
    assert not df[FEATURES].isna().any().any()

    return df

# ---------------- Assign splits ----------------
def get_splits(df, scheme, seed = None):

    temporal = pd.read_parquet(SPLIT_DIR / "temporal_split.parquet")
    temporal_labels = df["year"].map(temporal.set_index("year")["temporal_split"])

    if scheme == "temporal":
        labels = temporal_labels

    else:
        if seed is None:
            raise ValueError("A seed is required for spatial evaluation.")

        spatial = pd.read_parquet(SPLIT_DIR /f"spatial_blocks_seed{seed}.parquet")
        spatial_labels = df["cell_id"].map(spatial.set_index("cell_id")["spatial_split"])

        if scheme == "spatial":
            labels = spatial_labels
        elif scheme == "spatiotemporal":
            labels = pd.Series("unused", index = df.index)

            for split in ["train", "val", "test"]:
                mask = ((spatial_labels == split) & (temporal_labels == split))
                labels.loc[mask] = split
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    # Check that there are no empty labels before continuing.
    assert labels.notna().all()

    return labels


# ---------------- Model ----------------
def build_model(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C = C, max_iter = 1000, solver = "lbfgs"))


def evaluate(model, X, y):
    scores = model.predict_proba(X)[:, 1]

    return {
        "ap": average_precision_score(y, scores),
        "prevalence": float(np.mean(y)),
        "positives": int(np.sum(y)),
        "rows": len(y)
    }


# ---------------- Run one experiment ----------------
def run_experiment(df, features, scheme, seed = None, fixed_C = None):
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
    # Fit the selected model.
    # Note that this rebuilds one of the models already validated above.
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
        # Since spatiotemporal validation reuses the C from the temporal validation,
        # we can't do it unless we have first done temporal validation. Reusing this C
        # means that the spatiotemporal validation has indirectly used observations
        # from the spatially held-out regions, which is a clear flaw.
        temporal_path = (RESULT_DIR / f"{model_name}_temporal.csv")
        if not temporal_path.exists():
            raise FileNotFoundError(f"Run temporal evaluation first.")

        temporal_results = pd.read_csv(temporal_path)
        selected_C = float(temporal_results.iloc[0]["best_C"])
        print(f"Using temporal C = {selected_C}")
        for seed in SEEDS:
            results.append(run_experiment(df, features, "spatiotemporal", seed, fixed_C = selected_C))

    results_df = pd.DataFrame(results)

    # ---------------- Save results ----------------
    output_path = (RESULT_DIR / f"{model_name}_{scheme}.csv")
    results_df.to_csv(output_path, index = False)

    print("\nResults:")
    print(results_df.to_string(index = False))

    if len(results_df) > 1:
        print("\nMean test AP:", results_df["test_ap"].mean())
        print("Standard deviation:", results_df["test_ap"].std())

    print(f"\nSaved: {output_path}")
