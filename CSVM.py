"""
Author: Ethan Rajkumar
Editor: 

The purpose of this code is to implement a custom C-SVM model/classification for multi-label classification problems.
"""


import numpy as np

def manual_k_fold_split(X, y, n_splits=3, shuffle=True, random_state=None):
    """
    Manual implementation of k-fold splitting, similar to sklearn.model_selection.KFold
    but without using sklearn.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples, ...)
    n_splits : int
        Number of folds for splitting.
    shuffle : bool
        Whether to shuffle the data before splitting.
    random_state : int or None
        Seed for reproducible shuffling.

    Yields
    ------
    (train_indices, val_indices) : tuple of numpy arrays
        Indices for training and validation subsets.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    # Shuffle if needed
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        np.random.shuffle(indices)
    
    # Compute fold sizes
    fold_size = n_samples // n_splits
    for fold_i in range(n_splits):
        # Determine the start/end of this fold
        start = fold_i * fold_size
        # Last fold takes remaining samples if n_samples not divisible by n_splits
        end = (fold_i + 1) * fold_size if (fold_i < n_splits - 1) else n_samples
        
        val_indices = indices[start:end]
        # Everything not in val_indices is train
        train_indices = np.concatenate([indices[:start], indices[end:]])
        yield (train_indices, val_indices)

class CSVM:
    """
    Custom C-SVM implementation for multi-label classification problems.
    Each instance can have 1-3 labels.

    Parameters (for initialization):
    --------------------------------
    C : float, default=1.0
        Regularization parameter. Trades off margin size and training error.
    kernel : str, default='linear'
        Kernel type to be used. Options: 'linear', 'poly', 'rbf', 'matern'
    gamma : float, default=0.1
        Kernel coefficient for 'rbf', 'poly', and 'matern'. (Not used in linear training.)
    degree : int, default=3
        Degree for 'poly' kernel. (Not used in linear training.)
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    max_iter : int, default=1000
        Maximum number of iterations.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.

    Methods:
    --------
    fit(X, y_multilabel):
        Fits the SVM model to the provided multi-label data.
    predict(X):
        Predicts labels for input samples.
    score(X, y_true):
        Returns an average F1 score for the predictions vs. the true labels.
    """
    
    def __init__(
        self, 
        C=1.0, 
        kernel='linear', 
        gamma=0.1, 
        degree=3,
        tol=1e-3, 
        max_iter=1000, 
        learning_rate=0.01
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.models = {}
        self.classes_ = None
    
    def _kernel_function(self, x1, x2):
        """
        Compute the kernel function between two vectors.
        (Not truly used by the gradient-based training below.)
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        elif self.kernel == 'matern':
            diff = x1 - x2
            r = np.linalg.norm(diff) * np.sqrt(self.gamma)
            return (1.0 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_gradient(self, X, y, weights, bias):
        """
        Compute the gradient of the SVM hinge-loss function (plus L2 regularization).
        """
        n_samples = X.shape[0]
        predictions = np.dot(X, weights) + bias
        margins = y * predictions
        
        # Start with regularization term
        weight_gradient = weights.copy()
        bias_gradient = 0
        
        # Hinge-loss gradient for misclassified points
        for i in range(n_samples):
            if margins[i] < 1:
                weight_gradient -= self.C * y[i] * X[i]
                bias_gradient -= self.C * y[i]
        
        return weight_gradient, bias_gradient
    
    def _train_binary_classifier(self, X, y):
        """
        Train a binary SVM classifier for (1 vs -1).
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0.0
        
        for _ in range(self.max_iter):
            w_grad, b_grad = self._compute_gradient(X, y, weights, bias)
            
            # Gradient descent update
            weights -= self.learning_rate * w_grad
            bias -= self.learning_rate * b_grad
            
            # Check for early stopping
            predictions = np.dot(X, weights) + bias
            margins = y * predictions
            hinge_loss = np.mean(np.maximum(0, 1 - margins))
            reg = 0.5 * np.sum(weights ** 2)
            total_loss = reg + self.C * hinge_loss
            
            if total_loss < self.tol:
                break
        
        return weights, bias
    
    def fit(self, X, y_multilabel):
        """
        Fit the SVM model using one-vs-rest strategy for multi-label data.
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y_multilabel, list):
            y_multilabel = np.array(y_multilabel)
        
        if len(y_multilabel.shape) == 1:
            y_multilabel = y_multilabel.reshape(-1, 1)
        
        # Identify unique classes
        unique_classes = set()
        for row in y_multilabel:
            for label in row:
                if label != 0:
                    unique_classes.add(label)
        self.classes_ = sorted(unique_classes)
        
        # Train one binary classifier per class
        for cls in self.classes_:
            binary_y = np.array([1 if cls in row else -1 for row in y_multilabel])
            weights, bias = self._train_binary_classifier(X, binary_y)
            self.models[cls] = (weights, bias)
        
        return self
    
    def predict(self, X):
        """
        Predict up to 3 labels for each sample in X.
        """
        if isinstance(X, list):
            X = np.array(X)
        
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(n_samples):
            scores = {}
            # Linear scoring with learned weights
            for cls in self.classes_:
                weights, bias = self.models[cls]
                scores[cls] = np.dot(X[i], weights) + bias
            
            # Sort classes by descending score
            sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Keep classes with positive scores, up to 3
            pos_classes = [cls for cls, val in sorted_classes if val > 0][:3]
            if not pos_classes and sorted_classes:
                # If none is positive, take the top one anyway
                pos_classes = [sorted_classes[0][0]]
            
            predictions.append(pos_classes)
        
        # Pad to produce a consistent 2D output
        max_labels = max(len(pred) for pred in predictions) if predictions else 0
        padded_predictions = np.zeros((n_samples, max_labels))
        for i, pred in enumerate(predictions):
            for j, cls in enumerate(pred):
                padded_predictions[i, j] = cls
        
        return padded_predictions
    
    def score(self, X, y_true):
        """
        Compute the average F1 score over all samples.
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        y_pred = self.predict(X)
        f1_scores = []
        
        for i in range(len(y_true)):
            true_set = set(y_true[i][y_true[i] != 0])
            pred_set = set(y_pred[i][y_pred[i] != 0])
            
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores)


def hyperparameter_tuning(X, y, param_grid, cv=3, shuffle=True, random_state=42):
    """
    Simple grid search to find the best hyperparameters for CSVM 
    based on F1 score across manual k-fold cross-validation (no sklearn).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples, n_labels)
        Multi-label targets.
    param_grid : dict
        Dictionary where each key is a hyperparameter name (str), and
        each value is a list of possible values.
        e.g. {
          "C": [5, 10, 100],
          "kernel": ["linear", "rbf", "poly"],
          "gamma": [1, 0.1, 0.01, 0.001],
          "degree": [1, 2, 3, 4, 5, 6]
        }
    cv : int, default=3
        Number of cross-validation folds.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int or None, default=42
        Random seed for reproducible splits.

    Returns
    -------
    best_params : dict
        Best combination of hyperparameters found.
    best_score : float
        Best F1 score obtained with those hyperparameters.
    """
    best_score = float("-inf")
    best_params = None
    
    # Extract hyperparameter lists (with defaults if not provided)
    c_values = param_grid.get("C", [1.0])
    kernels = param_grid.get("kernel", ["linear"])
    gammas = param_grid.get("gamma", [0.1])
    degrees = param_grid.get("degree", [3])
    tols = param_grid.get("tol", [1e-3])
    max_iters = param_grid.get("max_iter", [1000])
    lrs = param_grid.get("learning_rate", [0.01])
    
    # We'll run our manual k-fold cross-validation
    # for each combination of hyperparameters
    for c in c_values:
        for kernel in kernels:
            for gamma in gammas:
                for degree in degrees:
                    for tol in tols:
                        for max_iter in max_iters:
                            for lr in lrs:
                                # For each combination, accumulate CV scores
                                cv_scores = []
                                
                                # Perform manual KFold splits
                                for train_indices, val_indices in manual_k_fold_split(
                                    X, y, n_splits=cv, shuffle=shuffle, random_state=random_state
                                ):
                                    X_train, X_val = X[train_indices], X[val_indices]
                                    y_train, y_val = y[train_indices], y[val_indices]
                                    
                                    model = CSVM(
                                        C=c, 
                                        kernel=kernel,
                                        gamma=gamma,
                                        degree=degree,
                                        tol=tol, 
                                        max_iter=max_iter,
                                        learning_rate=lr
                                    )
                                    model.fit(X_train, y_train)
                                    fold_score = model.score(X_val, y_val)
                                    cv_scores.append(fold_score)
                                
                                # Compute average score over folds
                                avg_score = np.mean(cv_scores)
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_params = {
                                        "C": c,
                                        "kernel": kernel,
                                        "gamma": gamma,
                                        "degree": degree,
                                        "tol": tol,
                                        "max_iter": max_iter,
                                        "learning_rate": lr
                                    }
    
    return best_params, best_score
