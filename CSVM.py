"""
Author: Ethan Rajkumar
Editor: 

The purpose of this code is to implement a custom C-SVM model/classification for multi-label classification problems.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

class LinearSVM:
    """
    A basic linear SVM classifier optimized with gradient descent for the
    squared hinge loss and L2 regularization.
    """
    def __init__(self, penalty='l2', loss='squared_hinge', tol=1e-4, 
                 C=1.0, fit_intercept=True, intercept_scaling=1,
                 verbose=0, random_state=None, max_iter=1000):
        self.penalty = penalty           # Only 'l2' is supported.
        self.loss = loss                 # Only 'squared_hinge' is supported.
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling  # Not actively used.
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter

        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the model to the binary data X and y.
        Expects y to be in {0, 1} and converts 0 -> -1.
        """
        # Convert binary labels: 0 becomes -1
        y = np.where(y == 0, -1, y)
        n_samples, n_features = X.shape

        # Initialize weights (and bias)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.w = np.zeros(n_features)
        self.b = 0.0 if self.fit_intercept else 0.0

        # Fixed learning rate; in practice, an adaptive scheme might be preferable.
        learning_rate = 1e-3

        for iteration in range(self.max_iter):
            # Compute decision values: w^T x + b
            decision = np.dot(X, self.w) + (self.b if self.fit_intercept else 0)
            # Compute margins: 1 - y * decision
            margins = 1 - y * decision

            # Identify samples that contribute to the loss (margin > 0)
            active = margins > 0

            # Compute gradient for weights:
            # Regularization gradient: w
            grad_w = self.w.copy()
            if np.any(active):
                # Loss gradient: sum_{i active} -2 * C * y_i * (1 - y_i * decision_i) * x_i
                grad_w -= 2 * self.C * np.dot(X[active].T, y[active] * margins[active])

            # Compute gradient for bias if using an intercept:
            grad_b = 0.0
            if self.fit_intercept and np.any(active):
                grad_b = -2 * self.C * np.sum(y[active] * margins[active])

            # Check for convergence (combined norm of gradients)
            grad_norm = np.linalg.norm(np.append(grad_w, grad_b))
            if self.verbose:
                print(f"Iteration {iteration}, grad_norm: {grad_norm:.6f}")
            if grad_norm < self.tol:
                break

            # Update parameters
            self.w -= learning_rate * grad_w
            if self.fit_intercept:
                self.b -= learning_rate * grad_b

        return self

    def decision_function(self, X):
        """
        Compute the decision function (w^T x + b) for each sample in X.
        """
        return np.dot(X, self.w) + (self.b if self.fit_intercept else 0)

class CSVM:
    """
    Multi-label linear SVM classifier.
    
    This implementation trains one binary LinearSVM classifier per column of the
    label matrix Y in parallel using threading.
    """
    def __init__(self, penalty='l2', loss='squared_hinge', tol=1e-4, 
                 C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                 verbose=0, random_state=None, max_iter=1000, n_jobs=-1):
        # Parameters for each LinearSVM instance.
        self.params = {
            'penalty': penalty,
            'loss': loss,
            'tol': tol,
            'C': C,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'verbose': verbose,
            'random_state': random_state,
            'max_iter': max_iter
        }
        self.n_jobs = n_jobs
        self.models = None

    def fit(self, X, Y):
        """
        Fit the multi-label classifier.
        
        Parameters:
        - X: Feature matrix of shape (n_samples, n_features).
        - Y: Binary label matrix of shape (n_samples, n_classes); each column
             is assumed to contain labels in {0, 1}.
        """
        n_classes = Y.shape[1]

        # Helper function to train one classifier for class i.
        def train_one_class(i):
            model = LinearSVM(**self.params)
            model.fit(X, Y[:, i])
            return model

        # Determine the number of worker threads.
        n_jobs = os.cpu_count() if self.n_jobs == -1 else self.n_jobs

        # Train each binary classifier in parallel.
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(train_one_class, i) for i in range(n_classes)]
            self.models = [future.result() for future in futures]
        return self

    def decision_function(self, X):
        """
        Compute the decision function for each classifier.
        
        Returns:
        - A matrix of shape (n_samples, n_classes) with decision scores.
        """
        decisions = [model.decision_function(X) for model in self.models]
        return np.vstack(decisions).T

    def predict(self, X):
        """
        Predict binary labels for each class.
        
        Returns:
        - A binary matrix of shape (n_samples, n_classes), where an entry is 1
          if the corresponding decision score is >= 0, and 0 otherwise.
        """
        decisions = self.decision_function(X)
        return (decisions >= 0).astype(int)
