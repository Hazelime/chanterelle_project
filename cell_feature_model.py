import numpy as np
import numpy.typing as npt

from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import StandardScaler

"""
This script provides a custom-made logistic-regression model class that handles recurring
data more efficiently than Scikit-Learn's LogisticRegression.

We want a custom logistic regression model for TF-IDF and FastText because Scikit-Learn's
LogisticRegression works inefficiently when many of the 300+ features can be calculated once
per cell instead of once per cell-week. For example, we can't just append 5,000 TF-iDF
columns to all the 8 million cell-weeks or we'd end up with an enormous matrix -- especially
since my tests show that the cell TF-IDF vectors actually are relatively dense (22-32%
nonzero entries in each cell).
"""

class CellFeatureLogisticRegression:
    """
    Logistic regression where cell-level feature vectors (TF-IDF, FastText) are not
    repeated across observations. Uses L2 regularization and SciPy L-BFGS-B.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-8):
        self.C = C # Regularization strength
        # For the default max_iter and tol, I also tested (300, 1e-5) and (1500, 1e-10).
        # (300, 1e-5) was too permissive as the optimizer stopped after only 15 iterations,
        #   likely because objective values themselves are quite small (0.006-0.007).
        #   (1000, 1e-08) also scored +54.1% temporal validation AP in comparison.
        # (1500, 1e-10) produced identical results to (1000, 1e-8) in temporal validation AP.
        # I therefore settled on (1000, 1e-8) as the default.
        self.max_iter = max_iter # Maximum number of optimizatoin iterations.
        self.tol = tol # Relative objective-change tolerance used by L-BFGS-B for convergence.

    def fit(self,
            X_num: list,
            cell_idx: npt.NDArray[np.int32],
            V: csr_matrix,
            y: npt.NDArray[np.int32]):
        """
        Fits a logistic regression model in an efficient way by utilizing the fact
        that the linguistic features are static for each cell.

        Parameters:
            self: CellFeatureLogisticRegression
                The model to fit.
            X_num: list
                The numerical features for each every cell-week (week_sin, week_cos,
                temp_7d, etc.)
            cell_idx: npt.NDArray[np.int32]
                An integer array index mapping each row in V to a cell-week.
            V: csr_matrix
                A sparse matrix containing one linguistic feature vector per grid cell.
            y: npt.NDArray[np.int32]
                An integer array of binary observation labels (0 or 1), where each
                value corresponds to one row of X_num and one entry in cell_idx.
                1 means that at least one chanterelle was observation in that
                cell-week; 0 means no observation was recorded.
        Returns:
            CellFeatureLogisticRegression
                The fitted model.
        """
        # Convert inputs to efficient (int32) and precise (float64) NumPy arrays.
        X_num = np.asarray(X_num, dtype = np.float64)
        cell_idx = np.asarray(cell_idx, dtype = np.int32)
        y = np.asarray(y, dtype = np.float64)

        # Check various fail cases before continuing.
        if self.C <= 0:
            raise ValueError("C must be positive.")
        # Every cell-week must have numerical features (X_num), a cell index (cell_idx),
        # and a target label (y).
        if len(y) != len(cell_idx) or len(y) != len(X_num):
            raise ValueError("Inconsistent observation counts.")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Target must be binary.")
        if cell_idx.min() < 0 or cell_idx.max() >= V.shape[0]:
            raise ValueError("Cell index outside linguistic matrix.")
        # Sum = 0 means that all labels are 0; sum = len(y) means that all labels are 1.
        if y.sum() == 0 or y.sum() == len(y):
            raise ValueError("Both classes are required.")

        # Use Scikit-learn's standardizer to standardize the numerical features.
        # Since this happens in fit(), it standardizes based on training data only.
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X_num)

        # Use a sparse, precise matrix.
        self.V_ = (V.tocsr().astype(np.float64) if issparse(V) else csr_matrix(V, dtype = np.float64))
        # Record how many features there are of each kind so we can separate them
        # later during optimization.
        self.n_numeric_ = X.shape[1] # Number of numerical features (6)
        # Number of linguistic features (5,000 for TF-IDF, 300 for FastText)
        self.n_linguistic_ = V.shape[1] 

        # Introduce shorthand to make the rest of the code more readable:
        n = len(y) # Number of training cell-weeks (millions)
        p = self.n_numeric_ # Number of numerical features
        q = self.n_linguistic_ # Number of linguistic features

        # Pre-compute the regularization parameter so we don't have to do it over and over
        # in objective(). This calculation is the same as for the standard L2 penalty used by
        # Scikit-learn's unweighted logistic regression when the loss is expressed as a mean.
        # This is important since we want to mirror the logistic regression used for experiment
        # runs without linguistic features.
        regularization = 1.0 / (self.C * n)

        # Now we want to set the optimizer's initial guesses for each one of the 5,007 or 307
        # parameters: 1 intercept + 6 numerical features + 5,000 or 300 linguistic features
        # Start with 0 for all:
        initial = np.zeros(1 + p + q, dtype = np.float64)
        # y.mean() is the prevalence since it is the fraction of training examples labeled 1.
        # Clip slightly above 0 and slightly below 1 to avoid division by and logarithm of 0 later.
        prevalence = np.clip(y.mean(), 1e-10, 1 - 1e-10)
        # Set the intercept to the log-odds of the training prevalence. This makes the model's
        # initial predicted probability equal to the training prevalence, which is a sensible
        # starting point.
        initial[0] = np.log(prevalence / (1 - prevalence))

        def objective(params: npt.NDArray) -> tuple[float, npt.NDArray]:
            """
            Compute loss and gradient for the model in an efficient way.
            
            Parameters:
                params: npt.NDArray
                    An array of the model's current parameters (intercept,
                    numerical coefficients, and linguistic coefficients)
            Returns:
                float
                    Calculated regularizad loss (loss + penalty)
                npt.NDArray
                    Calculated gradient of that loss with respect to every parameter.
            """
            # Split up the parameters into...
            intercept = params[0] # intercept
            beta = params[1:1 + p] # numerical coefficients
            gamma = params[1 + p:] # linguistic coefficients

            # Multiply matrices (@) to get one linguistic contribution per cell.
            # ravel() flattens the array so it works like a list.
            # This is the first major memory-saving optimization!
            cell_scores = np.asarray(self.V_ @ gamma).ravel() 

            # Calculate one prediction per cell-week.
            # Doing X @ gamma last would have required lots of computation! Instead, we simply
            # look up the appropriate value in the cell_scores array.
            z = (intercept + X @ beta + cell_scores[cell_idx])

            # Use numerically stable Binary Cross-Entropy (BCE) loss to avoid overflow when
            # calculating e^z for large z. (Note that logadexp(0, z) = log(e^0 + e^z).)
            # A full derivation of the formula is below.
            loss = np.mean(np.logaddexp(0, z) - y * z)
            # 1. First, note that BCE loss = -[y * ln(p) + (1 - y) * ln(1 - p)].
            # 2. Now, note that the sigmoid function gives us p = 1 / (1 + e^-z), so
            # ln(p) = ln(1 / (1 + e^-z)) = ln(1) - ln(1 + e^-z) = 0 - ln((e^z + 1) / e^z)
            # = -(ln(e^z + 1) - ln(e^z)) = z - ln(1 + e^z).
            # In other words, ln(p) = z - ln(1 + e^z).
            # 3. Similarly, we can derive ln(p - 1) = -ln(1 + e^z)
            # 4. Substituting these back into the original BCE formula, we get:
            # -[y * (z - ln(1 + e^z)) + (1 - y)(-ln(1 + e^z))]
            # = -[yz - y * ln(1 + e^z) - ln(1 + e^z) + y * ln(1 + e^z)]
            # = -[yz - ln(1 + e^z)] = ln(1 + e^z) - y * z, which is what we have above.

            # L2 regularization penalty. (regularization was defined above as 1 / (C * n))
            penalty = (0.5 * regularization * (beta @ beta + gamma @ gamma))

            # Expit() is SciPy's numerically stable implementation of the sigmoid function,
            # so expit(z) produces the predicted probabilities. Residuals are the
            # differences between the predicted probabilities and the actual labels.
            residual = (expit(z) - y) / n

            # The intercept contributes 1 to every observation's linear score, so its
            # gradient is just the sum of residuals.
            grad_intercept = residual.sum()

            # Gradient for the numeric coefficients.
            # X has shape (n, 6), so its transpose has shape (6, n). The residual vector
            # has shape (n,), so we get one gradient value for each numerical feature.
            # regularization * beta is the derivative of the L2 penalty:
            # d/d(beta)[lambda * beta^2 / 2] = lambda * beta.
            grad_beta = (X.T @ residual + regularization * beta)

            # Aggregate residuals by cell to avoid having to calculate the linguistic gradients
            # for every cell-week. Basically, instead of summing over cell-weeks directly, we first
            # sum the residuals within each cell and then multiply them by the linguistic values.
            # This is the second major memory-saving optimization!
            # bincount() counts the number of occurrences of each cell index.
            cell_residual = np.bincount(cell_idx, weights = residual, minlength = self.V_.shape[0])

            # Gradient for the linguistic coefficients.
            # V has shape (4983, 5000) if there are 5000 linguistic features, so its transpose
            # has shape (5000, 4983). Because we calculated cell_residual so cleverly, it has
            # shape (4983,) instead of ([millions],), so the result of the multiplication is
            # one gradient value for each linguistic feature.
            grad_gamma = (np.asarray(self.V_.T @ cell_residual).ravel()
                          + regularization * gamma)

            # Combine all the gradients:
            # 1 intercept + 6 numerical coefficients + 5000 or 300 linguistic coefficients = 
            # 5007 or 307 total
            gradient = np.concatenate([[grad_intercept], grad_beta, grad_gamma])

            return loss + penalty, gradient

        # minimize() is SciPy's optimizer. Breakdown:
        # objective is the function to minimize.
        # initial is the initial guess for the 5,007 parameters (i.e. positive prevalence
        #   for the intercept, 0 for the rest).
        # L-BFGS-B is the optimization algorithm: Limited-memory Broyden-Fletcher-Goldfarb-
        #   Shanno with Bounds: https://en.wikipedia.org/wiki/Limited-memory_BFGS#L-BFGS-B
        #   (I don't really understand how this works, but as far as I can tell it also
        #   uses a subset of the second derivatives to improve optimization.)
        # jac = True means that objective() should return both the gradient and the loss.
        #   (Jac stands for Jacobian.)
        # maxiter is the maximum number of optimization iterations (default = 300).
        # ftol is the relative objective-change tolerance for stopping. Together with
        #   maxiter, this decides when the optimization should stop.
        result = minimize(
            fun = objective,
            x0 = initial,
            method = "L-BFGS-B",
            jac = True,
            options = {
                "maxiter": self.max_iter,
                "ftol": self.tol,
            }
        )

        # The trailing underscore is apparently a Scikit-learn convention indicating that
        # something was learned during fitting.
        self.intercept_ = result.x[0]
        self.coef_numeric_ = result.x[1:1 + p]
        self.coef_linguistic_ = result.x[1 + p:]

        self.success_ = result.success # True if we reached convergence.
        self.message_ = result.message
        self.n_iter_ = result.nit

        self.final_loss_ = result.fun
        self.final_gradient_norm_ = np.linalg.norm(result.jac, ord = np.inf)

        if not result.success:
            print("Warning: optimizer did not converge:", result.message)

        return self

    def predict_proba(self : CellFeatureLogisticRegression,
                      X_num: list,
                      cell_idx: npt.NDArray[np.int32],
                      V: csr_matrix = None) -> npt.NDArray:
        """
        Predict probabilities based on the fitted model. # The equations are essentially
        the same as in objective(), but now we're not learning anything.

        Paramters:
            self: CellFeatureLogisticRegression
                The fitted model
            X_num: list
                The numerical features for each every cell-week (week_sin, week_cos,
                temp_7d, etc.)
            cell_idx: npt.NDArray[np.int32]
                An integer array index mapping each row in V to a cell-week.
            V: csr_matrix
                A sparse matrix containing one linguistic (TF-IDF/FastText) vector per grid cell.
        Returns:
              npt.NDArray
                An array that contains two probabilities for each cell-week: 1) the probability
                of no recorded observation, and 2) the probability of at least one.
        """
        # Convert inputs to efficient (int32) and precise (float64) NumPy arrays.
        X_num = np.asarray(X_num, dtype = np.float64)
        cell_idx = np.asarray(cell_idx, dtype = np.int32)

        # Get the fitted linguistic (TF-IDF/FastText) matrix if none is supplied.
        if V is None:
            V = self.V_

        # Use transform() rather than fit_transform() to reuse training data means and std.
        X = self.scaler_.transform(X_num)

        # One score per cell instead of one per cell-week.
        cell_scores = np.asarray(V @ self.coef_linguistic_).ravel()
        # One score per cell-week.
        z = (self.intercept_ + X @ self.coef_numeric_ + cell_scores[cell_idx])

        # Convert scores into probabilities. (Expit() is SciPy's numerically stable
        # implementation of the sigmoid function.)
        positive = expit(z)

        # For each cell-week, return 1) the probability of no recorded observation, and
        # 2) the probability of at least one recorded observation. 
        return np.column_stack([1 - positive, positive])
