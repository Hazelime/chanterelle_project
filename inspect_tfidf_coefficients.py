from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def inspect_placename_patterns(patterns: list, placenames_path: Path, top_n: int = 15) -> None:
    """
    Find actual placenames containing given patterns and report number of matching name
    records, number of cells containing matches, and the most frequent matching placenames.

    Parameters:
        patterns: list
            The list of patterns to investigate.
        placenames_path: Path
            A pathlib Path to the placename database.
        top_n: int (default = 15)
            How many of the most frequent matching placenames to report.
    """
    placenames = pd.read_parquet(placenames_path, columns = ["cell_id", "placenames"])
    # explode() turns each list in the placenames column into separate rows,
    # repeating the other columns' values for each item.
    names = (placenames[["cell_id", "placenames"]].explode("placenames")
        .dropna(subset = ["placenames"]).copy())

    for pattern in patterns:
        # Missing values are treated as not matching.
        matches = names[names["placenames"].str.contains(
                pattern,
                case = False,
                regex = False,
                na = False)]

        print(f"\n{'=' * 60}")
        print(f"Pattern: {pattern!r}")
        print(f"{'=' * 60}")

        print(f"Matching name records: {len(matches):,}")
        print(f"Cells containing pattern: {matches['cell_id'].nunique():,}")

        if matches.empty:
            continue

        print("\nMost frequent matching placenames:")
        print(matches["placenames"].value_counts().head(top_n).to_string())


# ---------------- Configuration ----------------
MODEL_PATH = Path("./results/models/tfidf_temporal.joblib")
OUTPUT_DIR = Path("./results/tfidf_analysis")
PLACENAMES_PATH = Path("./data/placenames/cell_placenames.parquet")
# Handpicked patterns that are investigated more closely.
PATTERNS = ["mad", "kull", "myr", "jávr", "lund", "holm"]

TOP_N = 20

# ---------------- Load model ----------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
vectorizer = bundle["vectorizer"]
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_tfidf_

assert len(feature_names) == len(coefficients)

print(f"Loaded {len(feature_names):,} TF-IDF features.")

# ---------------- Document frequencies ----------------
# For the temporal model, all 4,983 cells are included in the training period.
# Get TF-IDF features.
V = model.V_

# V has one row per cell/document and one column per TF-IDF feature.
# A feature's value is positive when that n-gram appears in that document.
# ravel() flattens the array into a 1D array aligned with feature_names. 
document_frequency = np.asarray((V > 0).sum(axis = 0)).ravel()

assert len(document_frequency) == len(feature_names)

# ---------------- Create coefficient table ----------------
df = pd.DataFrame({
    "ngram": feature_names,
    "coefficient": coefficients,
    "abs_coefficient": np.abs(coefficients),
    "document_frequency": document_frequency,
    "idf": vectorizer.idf_,
})

df = df.sort_values("coefficient", ascending = False).reset_index(drop = True)

# ---------------- Placename pattern inspection ----------------
inspect_placename_patterns(patterns = PATTERNS,
                           placenames_path = PLACENAMES_PATH,
                           top_n = 15)

# ---------------- Save results ----------------
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

# utf-8-sig writes the CSV with a byte-order mark (BOM) at the beginning, which
# helps some program (*cough* Excel *cough*) reognize the file as UTF-8.
df.to_csv(OUTPUT_DIR / "tfidf_coefficients.csv", index = False, encoding = "utf-8-sig")

positive = df.nlargest(TOP_N, "coefficient").sort_values("coefficient")
negative = df.nsmallest(TOP_N, "coefficient").sort_values("coefficient")

# ---------------- Print results ----------------
print("\nTop positive n-grams:")
print(positive.sort_values("coefficient", ascending = False).to_string(index = False))

print("\nTop negative n-grams:")
print(negative.to_string(index = False))

# ---------------- Plotting ----------------
def plot_coefficients(data: DataFrame, title: str, filename: str) -> None:
    """
    Plot diagrams of n-ngrams vs. TF-IDF coefficients.

    Parameters:
        data: DataFrame
            A Pandas DataFrame containing the columns "ngram" and "coefficient".
        title: str
            The title of the diagram.
        filename: str
            Where to save the diagram.
    """
    fig, ax = plt.subplots(figsize = (9, 7))
    ax.barh(data["ngram"], data["coefficient"])
    ax.axvline(0, color = "black", linewidth = 0.8)
    ax.set_xlabel("Logistic regression coefficient")
    ax.set_title(title)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / filename, dpi = 200, bbox_inches = "tight")
    plt.close()

plot_coefficients(positive, "Top positive TF-IDF n-grams", "top_positive_ngrams.png")
plot_coefficients(negative, "Top negative TF-IDF n-grams", "top_negative_ngrams.png")

# ---------------- Summary ----------------
print("\nCoefficient statistics:")
print(df["coefficient"].describe())

print("\nFinished!")
print(f"Results saved to: {OUTPUT_DIR}")
