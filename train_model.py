import argparse # Used for user input in Main
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.sparse import csr_matrix

from tfidf_features import build_documents, fit_tfidf
from cell_feature_model import CellFeatureLogisticRegression

import joblib


# ---------------- Configuration ----------------
DATA_PATH = Path("./data/processed/cell_week_dataset.parquet")
PLACENAMES_PATH = Path("./data/placenames/cell_placenames.parquet")
FASTTEXT_PATH = Path("./data/placenames/cell_fasttext_embeddings.parquet")
MODEL_DIR = Path("./results/models")
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
    ],
    "placename_count": [
        "week_sin",
        "week_cos",
        "temp_7d",
        "precip_7d",
        "forest_pct",
        "placename_count",
    ],
    "placename_count_log": [
        "week_sin",
        "week_cos",
        "temp_7d",
        "precip_7d",
        "forest_pct",
        "placename_count_log",
    ]
}
NUMERIC_FEATURES = [
    "week_sin",
    "week_cos",
    "temp_7d",
    "precip_7d",
    "forest_pct",
    "placename_count_log",
]


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
    # The log feature is derived from placename_count, so load the original column instead.
    read_columns = ["placename_count" if col == "placename_count_log" else col for col in columns]

    df = pd.read_parquet(DATA_PATH, columns = list(dict.fromkeys(read_columns)))

    # Calculate placename_count_log = ln(1 + placename_count). The +1 handles cell
    # cells with zero placenames.
    if "placename_count_log" in features:
        assert df["placename_count"].ge(0).all()
        df["placename_count_log"] = np.log1p(df["placename_count"]).astype("float32")

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
    # The second column is the probability of at least one recorded observation.
    scores = model.predict_proba(X)[:, 1]

    return {
        "ap": average_precision_score(y, scores),
        "prevalence": float(np.mean(y)),
        "positives": int(np.sum(y)),
        "rows": len(y)
    }


def run_experiment(df: DataFrame,
                   features: list,
                   scheme: str,
                   model_name: str,
                   seed: int | None = None,
                   fixed_C: float | None = None) -> dict:
    """
    Run an experiment for a given dataset, set of features, scheme, and -- optionally --
    a seed and a fixed regularization strengths (C).

    Parameters:
        df: DataFrame
            The data with which to train, validate, and test a model.
        features: list
            The features to include in the experiment.
        scheme: str
            What type of validation scheme to run (temporal, spatial, or spatiotemporal)
        model_name: str
            The type of model used for the experiment (tfidf or fasttext)
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

    # Save the temporal placename_count_log model for plotting later.
    if scheme == "temporal" and model_name == "placename_count_log":
        MODEL_DIR.mkdir(parents = True, exist_ok = True)
        joblib.dump({"model": model, "features": features,},
            MODEL_DIR / "placename_count_log_temporal.joblib")
        print("Saved temporal placename_count_log model.")

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


def prepare_tfidf(df: DataFrame,
                  train_mask: npt.NDArray[np.bool_]) -> tuple[csr_matrix, npt.NDArray[np.int32], TfidfVectorizer]:
    """
    Fit a TF-IDF vectorizer on training data and build TF-IDF features for all cells.
    
    Parameters:
        df: DataFrame
            The data.
        train_mask: npt.NDArray[np.bool_]
            A mask to indicate which data should be used (= training data).
    
    Returns:
        csr_matrix
            The TF-IDF features for all cells (train + val + test).
        npt.NDArray[np.int32]
            A mapping from cell_id to the row index in the CSR matrix.
        TfidfVectorizer
            A Scikit-learn TF-IDF vectorizer fitted on the training data.
    """
    placenames = pd.read_parquet(PLACENAMES_PATH, columns = ["cell_id", "placenames"])
    documents = build_documents(placenames) # Custom-made function

    assert documents["cell_id"].is_unique
    assert set(df["cell_id"].unique()) == set(documents["cell_id"])

    # Sort so cell_index will align with the rows of V later.
    documents = documents.sort_values("cell_id").reset_index(drop = True)
    # Create a stable mapping from cell_id to TF-IDF feature.
    cell_index = pd.Series(np.arange(len(documents),
                                     dtype = np.int32),
                                     index = documents["cell_id"])
    observation_cell_idx = (df["cell_id"].map(cell_index).to_numpy(dtype = np.int32))

    # Fit on unique training cells only.
    training_cell_indices = np.unique(observation_cell_idx[train_mask])
    training_documents = (documents.iloc[training_cell_indices]["document"])
    # Use Scikit-learn's TF-IDF vectorizer.
    vectorizer, _ = fit_tfidf(training_documents)

    # Transform the documents for all cells into TF-IDF features using
    # the vectorizer that was fitted on training data. Save the results
    # in a sparse matrix.
    V = vectorizer.transform(documents["document"]).tocsr()

    return V, observation_cell_idx, vectorizer


def run_linguistic_experiment(df: DataFrame,
                              scheme: str,
                              model_name: str,
                              seed: int = None) -> dict:
    """
    A version of run_experiment() which uses the custom-made CellFeatureLogisticRegression
    for TF-IDF and FastText models.

    Parameters:
        df: DataFrame
            The data with which to train, validate, and test a model.
        scheme: str
            What type of validation scheme to run (temporal, spatial, or spatiotemporal)
        model_name: str
            The type of model used for the experiment (tfidf or fasttext)
        seed: int | None
            The randomizer seed for spatial and spatiotemporal validation. Temporal
            validation does not require any seed.
    Returns:
        dict
            A dict with metadata and results from the experiment, including scheme,
            seed, best_C, val_ap, val_prevalence, val_positives, test_ap, test_prevalance,
            and test_positives. 
    """
    print(f"\nRunning {model_name}: {scheme}, seed = {seed}")

    labels = get_splits(df, scheme, seed)

    train_mask = (labels == "train").to_numpy()
    val_mask = (labels == "val").to_numpy()
    test_mask = (labels == "test").to_numpy()

    X = df[NUMERIC_FEATURES].to_numpy(dtype = np.float32)
    y = df["observation"].to_numpy(dtype = np.int8)

    if model_name == "tfidf":
        V, cell_idx, vectorizer = prepare_tfidf(df = df, train_mask = train_mask)
    elif model_name == "fasttext":
        V, cell_idx = prepare_fasttext(df)
        vectorizer = None
    else:
        raise ValueError(f"Unknown linguistic model: {model_name}")
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    idx_train = cell_idx[train_mask]
    idx_val = cell_idx[val_mask]
    idx_test = cell_idx[test_mask]

    assert y_train.sum() > 0
    assert y_val.sum() > 0
    assert y_test.sum() > 0

    best_C = None
    best_val_ap = -np.inf

    # ---------------- Hyperparameter selection ----------------
    for C in C_VALUES:
        model = CellFeatureLogisticRegression(C = C, max_iter = 1000, tol = 1e-8)
        model.fit(X_train, idx_train, V, y_train)

        if not model.success_:
            raise RuntimeError(f"Model optimizer failed for C = {C}: {model.message_}")

        val_scores = model.predict_proba(X_val, idx_val, V)[:, 1]
        val_ap = average_precision_score(y_val, val_scores)
        print(f"  C = {C}: validation AP = {val_ap:.6f}")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_C = C
    
    # ---------------- Final model ----------------
    # Refit the best model and use it as the final model.
    model = CellFeatureLogisticRegression(C = best_C, max_iter = 1000, tol = 1e-8)
    model.fit(X_train, idx_train, V, y_train)

    if not model.success_:
        raise RuntimeError(f"Final model fit failed: {model.message_}")

    val_scores = model.predict_proba(X_val, idx_val, V)[:, 1]
    test_scores = model.predict_proba(X_test, idx_test, V)[:, 1]

    # Save the TF-IDF temporal model (which covers all cells) for coefficient analysis.
    if model_name == "tfidf":
        if scheme == "temporal":
            MODEL_DIR.mkdir(parents = True, exist_ok = True)
        joblib.dump({"model": model,
                    "vectorizer": vectorizer,
                    "features": TFIDF_NUMERIC_FEATURES},
                    MODEL_DIR / "tfidf_temporal.joblib")
        print("Saved temporal TF-IDF model.")

    result = {
        "scheme": scheme,
        "seed": seed,
        "best_C": best_C,
        "val_ap": average_precision_score(y_val, val_scores),
        "val_prevalence": float(y_val.mean()),
        "val_positives": int(y_val.sum()),
        "test_ap": average_precision_score(y_test, test_scores),
        "test_prevalence": float(y_test.mean()),
        "test_positives": int(y_test.sum()),
    }

    print("Converged:", model.success_)
    print("Iterations:", model.n_iter_)
    print(f"Final loss: {model.final_loss_:.8f}")
    print(f"Gradient infinity norm: {model.final_gradient_norm_:.2e}")
    print(f"Selected C: {best_C}")
    print(f"Test AP: {result['test_ap']:.6f}")

    return result


def prepare_fasttext(df: DataFrame) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """
    Build FastText features for all cells.
    
    Parameters:
        df: DataFrame
            The data.
    Returns:
        NDArray
            The FastText features for all cells (train + val + test).
        npt.NDArray[np.int32]
            A mapping from cell_id to the row index in the CSR matrix.
    """
    embeddings = pd.read_parquet(FASTTEXT_PATH)

    assert embeddings["cell_id"].is_unique
    assert set(df["cell_id"].unique()) == set(embeddings["cell_id"])

    # Sort so cell_index will align with the rows of V later.
    embeddings = embeddings.sort_values("cell_id").reset_index(drop = True)
    # Create a stable mapping from cell_id to FastText feature.
    cell_index = pd.Series(np.arange(len(embeddings),
                                     dtype = np.int32),
                                     index = embeddings["cell_id"])
    observation_cell_idx = (df["cell_id"].map(cell_index).to_numpy(dtype = np.int32))
    
    # Put the FastText features in a dense NumPy array.
    V = np.stack(embeddings["embedding"].to_numpy()).astype(np.float32)

    assert V.shape == (4983, 300)
    assert np.isfinite(V).all()

    return V, observation_cell_idx


if __name__ == "__main__":
    # Let the user choose what kind of experiment to run by adding flags:
    # --model [seasonality, environmental, placenames, placenames_log, tfidf, fasttext]
    #   to choose which features to include.
    # --scheme (temporal, spatial, spatiotemporal) to choose type of experiment.
    # Both flags must be included. Sample usage:
    # python3 train_model.py --model environmental --scheme temporal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices = [*MODEL_FEATURES.keys(), "tfidf", "fasttext"],
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

    if model_name in ["tfidf", "fasttext"]:
        features = NUMERIC_FEATURES
    else:
        features = MODEL_FEATURES[model_name]
    
    print(f"\nModel: {model_name}")
    print(f"Evaluation: {scheme}")
    print(f"Features: {features}")

    # Create a results directory if it does not already exist.
    RESULT_DIR.mkdir(parents = True, exist_ok = True)

    df = load_data(features)
    results = []

    runner = (run_linguistic_experiment if model_name in ["tfidf", "fasttext"] else None)
    if scheme == "temporal":
        # Takes like 15 seconds for numerical-only models and 2 min for lingusitic models.
        if runner: results.append(runner(df, "temporal", model_name))
        else: results.append(run_experiment(df, features, "temporal", model_name))
    elif scheme in ["spatial", "spatiotemporal"]:
        # Takes about 2 minutes for numerical-only models and 10 min for linguistic models.
        for seed in SEEDS:
            if runner: results.append(runner(df, scheme, model_name, seed))
            else: results.append(run_experiment(df, features, scheme, model_name, seed))

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
