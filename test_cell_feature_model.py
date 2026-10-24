import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cell_feature_model import CellFeatureLogisticRegression

"""
This script tests whether the custom-made logistic-regression model produces similar results
to Scikit-Learn's LogisticRegression model.
"""

# Create a synthetic dataset for testing whether my custom-made logistic regression
# model produces similar results to Scikit-learn's logistic regression model.
# If the maximum difference in prediction probabilities is < 1e-4, I consider the
# models equivalent.
rng = np.random.default_rng(317)

n_cells = 30
n_rows = 3000
n_numeric = 6
n_tfidf = 20

X = rng.normal(size = (n_rows, n_numeric))
cell_idx = rng.integers(0, n_cells, size = n_rows).astype(np.int32)
V = csr_matrix(rng.random((n_cells, n_tfidf)))
y = rng.binomial(1, 0.1, size = n_rows)

C = 0.1

# Test the custom-made model. Note that this tol is ftol, i.e. tolerance based on
# relative improvement in the objective function.
custom = CellFeatureLogisticRegression(C = C, max_iter = 1000, tol = 1e-8)
custom.fit(X, cell_idx, V, y)
p_custom = custom.predict_proba(X, cell_idx, V)[:, 1]
custom_coef = np.concatenate([custom.coef_numeric_, custom.coef_linguistic_,])

# Build an expanded matrix for fair comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_full = hstack([csr_matrix(X_scaled), V[cell_idx]], format = "csr")

# Test Scikit-learn's model. Note that this tol is gtol, i.e. tolerance based on a
# gradient-based convergence criterion. Using a higher tolerance than the default
# 1e-5 is fine here since it is the performance of the custom model we are interested in.
reference = LogisticRegression(C = C, solver = "lbfgs", max_iter = 1000, tol = 1e-8)
reference.fit(X_full, y)
p_reference = reference.predict_proba(X_full)[:, 1]
reference_coef = reference.coef_.ravel()

max_difference = np.max(np.abs(p_custom - p_reference))

print("Custom convergence:", custom.success_)
print("Maximum probability difference:", max_difference)

print("Custom iterations:", custom.n_iter_)
print("Reference iterations:", reference.n_iter_[0])

print("Maximum coefficient difference:", np.max(np.abs(custom_coef - reference_coef)))
print("Mean probability difference:", np.mean(np.abs(p_custom - p_reference)))

assert max_difference < 1e-4
print("Test passed.")
