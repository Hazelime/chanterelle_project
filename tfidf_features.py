import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

"""
This script provides high-level functions for ftting a TF-IDF vectorizer and building
TF-IDF features.
"""

# ---------------- Configuration ----------------
# Character-level TF-IDF with n-grams of length 3-5 within word boundaries,
# excluding n-grams occuring only in one cell and treating uppercase and
# lowercase forms as the same.
TFIDF_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_features": 5000,
    "lowercase": True,
    "dtype": np.float32,
}


def build_documents(cell_placenames: DataFrame) -> DataFrame:
    """
    Convert placename lists into one text document per cell. Duplicate name strings
    within each cell are removed, but the original data is unchanged.

    Parameters:
        cell_placenames: DataFrame
            A Pandas DataFrame containing cell_id and placenames columns.
    Returns:
        DataFrame
            A Pandas DataFrame containing cell_id and document columns, where
            document is a string joining the unique placenames in a cell.
    """
    documents = cell_placenames[["cell_id", "placenames"]].copy()

    def join_names(names: list[str]) -> str:
        """
        Parameters:
            names: list
                A list of placenames (str).
        Returns:
            str
                A string joining the placenames with spaces, or "" if there 
                are no names.
        """
        if names is None or len(names) == 0:
            return ""
        # Remove repeated name strings while preserving their original order.
        unique_names = dict.fromkeys(
            str(name).strip()
            for name in names
            if name is not None and str(name).strip()
        )
        return " ".join(unique_names)
    documents["document"] = (documents["placenames"].apply(join_names))

    return documents[["cell_id", "document"]]


def fit_tfidf(train_documents: list[str]) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Fit a TF-IDF vectorizer using training-cell documents only.

    Parameters:
        train_documents: list[str]
            A list of strings, each of which contains space-separated placenames.
    Returns:
        vectorizer: TfidfVectorizer
            A fitted Scikit-learn TfidfVectorizer built according to TFIDF_PARAMS
        X_train: csr_matrix
            A SciPy sparse matrix of TF-IDF features for the training documents.
    """
    # ** unpacks TFIDF_PARAMS.
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    X_train = vectorizer.fit_transform(train_documents)
    return vectorizer, X_train


def transform_tfidf(vectorizer: TfidfVectorizer, documents: list[str]) -> csr_matrix:
    """
    Transform validation or test documents using an already fitted vectorizer.
    Parameters:
        vectorizer: TfidfVectorizer
            An already fitted Scikit-learn TF-IDF vectorizer.
        documents: list[str]
            A list of string, each of which contains space-separated placenames.
    Returns:
        csr_matrix
            A SciPy sparse matrix of TF-IDF features for the given documents.
    """
    return vectorizer.transform(documents)
