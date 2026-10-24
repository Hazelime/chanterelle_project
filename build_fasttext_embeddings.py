from pathlib import Path
from functools import lru_cache
from collections.abc import Sequence

import fasttext
import numpy as np
import numpy.typing as npt
import pandas as pd

"""
This script generates cell-level FastText placename embeddings based on
data/placenames/fasttext/cc.sv.300.bin and data/placenames/cell_placenames.parquet and
then saves them as data/placenames/cell_fasttext_embeddings.parquet.
"""

# ---------------- Configuration ----------------
MODEL_PATH = Path("./data/placenames/fasttext/cc.sv.300.bin")
PLACENAMES_PATH = Path("./data/placenames/cell_placenames.parquet")
OUTPUT_PATH = Path("./data/placenames/cell_fasttext_embeddings.parquet")

EXPECTED_CELLS = 4983
EMBEDDING_DIM = 300

# ---------------- Load data ----------------
print("Loading Swedish FastText model...")
model = fasttext.load_model(str(MODEL_PATH))
placenames = pd.read_parquet(PLACENAMES_PATH, columns = ["cell_id", "placenames"])

assert model.get_dimension() == EMBEDDING_DIM
assert len(placenames) == EXPECTED_CELLS
assert placenames["cell_id"].is_unique

print(f"Loaded {len(placenames):,} cells.")

# ---------------- Token embeddings ----------------
# Cache vectorized tokens so we save some time and computation.
@lru_cache(maxsize = 200000)
def get_token_vector(token : str) -> npt.NDArray:
    """
    Return a pretrained FastText vector for a lowercased token.
    
    Parameters:
        token: str
            The token to vectorize.
    Returns:
        npt.NDArray
            The vector for the token.
    """
    return model.get_word_vector(token).astype(np.float32)

# ---------------- Placename embeddings ----------------
def embed_name(name: str) -> npt.NDArray | None:
    """
    Average token vectors within one placename.
    
    Parameters:
        name: str
            The placename for which to average token vectors.
    Returns:
        npt.NDArray | None
            The aveage vector or None if the placename does not contain any tokens.
    """
    tokens = str(name).lower().split()
    if not tokens:
        return None
    vectors = [get_token_vector(token) for token in tokens]
    return np.mean(vectors, axis = 0, dtype = np.float32)

# ---------------- Cell embeddings ----------------
def embed_cell(names: Sequence[str | None] | None) -> npt.NDArray:
    """
    Average unique placename vectors within a cell and L2-normalize the resulting embedding.
    
    Parameters:
        names: Sequence[str | None] | None
            A sequence of unique placenames within a cell.
    Returns:
        npt.NDArray
            The L2-normalized average FastText embedding for the unique placenames.
    """
    if names is None or len(names) == 0:
        return np.zeros(EMBEDDING_DIM, dtype = np.float32)

    unique_names = dict.fromkeys(
        str(name).strip()
        for name in names
        if name is not None and str(name).strip())

    if not unique_names:
        return np.zeros(EMBEDDING_DIM, dtype = np.float32)

    # Accumulate vectors without creating a large intermediate matrix.
    total = np.zeros(EMBEDDING_DIM, dtype = np.float64)
    count = 0
    for name in unique_names:
        vector = embed_name(name)
        if vector is not None:
            total += vector
            count += 1

    if count == 0:
        return np.zeros(EMBEDDING_DIM, dtype = np.float32)

    # Average placename vectors within one cell.
    cell_vector = (total / count).astype(np.float32)

    # L2-normalize.
    norm = np.linalg.norm(cell_vector)
    if norm > 0:
        cell_vector /= norm

    return cell_vector

# ---------------- Generate embeddings ----------------
embeddings = []
for i, row in enumerate(placenames.itertuples(index = False), start = 1):
    embeddings.append(embed_cell(row.placenames))
    if i % 500 == 0:
        print(f"Processed {i:,}/{EXPECTED_CELLS:,} cells.")

# ---------------- Save ----------------
result = pd.DataFrame({
    "cell_id": placenames["cell_id"],
    "embedding": embeddings,})

OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
result.to_parquet(OUTPUT_PATH, compression = "zstd", index = False)

# ---------------- Diagnostics ----------------
matrix = np.stack(result["embedding"].to_numpy())
norms = np.linalg.norm(matrix, axis = 1)

print("\nFinished!")
print(f"Rows: {len(result):,}") # 4,983
print(f"Embedding shape: {matrix.shape}") # (4983, 300)
print(f"Nonzero vectors: {(norms > 0).sum():,}") # 4,972
print(f"Zero vectors: {(norms == 0).sum():,}") # 11 (11 cells have no placenames)

print("\nVector norm statistics:")
print(pd.Series(norms).describe())
# count    4983.000000
# mean        0.997792
# std         0.046937
# min         0.000000
# 25%         1.000000
# 50%         1.000000
# 75%         1.000000
# max         1.000000

size_mb = OUTPUT_PATH.stat().st_size / (1024 ** 2)

print(f"\nSaved to: {OUTPUT_PATH}")
print(f"File size: {size_mb:.2f} MiB") # 5.89 MiB
