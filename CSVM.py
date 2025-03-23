"""
Author: Ethan Rajkumar
Editor: 
The purpose of this code is to implement a custom C-SVM model/classification for multi-label classification problems.
"""

import numpy as np
from collections import defaultdict

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
    
    def __init__(self, C=1.0, kernel='linear', tol=1e-3, max_iter=1000, learning_rate=0.01):
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.models = {}  # Dict to store binary classifiers
        self.classes_ = None
    
    def _kernel_function(self, x1, x2):
        """
        Compute the kernel function between two vectors.
        
        Currently supported:
        - linear
        - polynomial (with degree=2)
        - rbf (Gaussian)
        - matern (3/2)
        """
        if self.kernel == 'linear':
            # Linear kernel
            return np.dot(x1, x2)
        
        elif self.kernel == 'poly':
            # Polynomial kernel (degree=2 for simplicity)
            return (np.dot(x1, x2) + 1) ** 2
        
        elif self.kernel == 'rbf':
            # Radial Basis Function (Gaussian) kernel
            gamma = 1.0 / x1.shape[0]
            return np.exp(-gamma * np.sum((x1 - x2) ** 2))
        
        elif self.kernel == 'matern':
            # Matern (3/2) kernel
            # k(r) = (1 + sqrt(3)*r) * exp(-sqrt(3)*r), where r = ||x1 - x2|| / length_scale
            # We hardcode length_scale = 1.0 for simplicity; adjust as needed.
            length_scale = 1.0
            diff = x1 - x2
            r = np.linalg.norm(diff) / length_scale
            return (1.0 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_gradient(self, X, y, weights, bias):
        """
        Compute the gradient of the SVM hinge-loss function (plus L2 regularization).
        
        NOTE: This code is effectively a linear SVM approach.
              The _kernel_function is not utilized here.
        """
        n_samples = X.shape[0]
        
        # Predictions and margins
        predictions = np.dot(X, weights) + bias
        margins = y * predictions
        
        # Start with the gradient from the regularization term
        weight_gradient = weights.copy()
        bias_gradient = 0
        
        # Accumulate gradient from misclassified points (margins < 1)
        for i in range(n_samples):
            if margins[i] < 1:
                weight_gradient -= self.C * y[i] * X[i]
                bias_gradient -= self.C * y[i]
        
        return weight_gradient, bias_gradient
    
    def _train_binary_classifier(self, X, y):
        """
        Train a binary SVM classifier for a single class (1 vs -1).
        
        Currently uses a linear approach: w·x + b
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        weights = np.zeros(n_features)
        bias = 0.0
        
        for _ in range(self.max_iter):
            weight_gradient, bias_gradient = self._compute_gradient(X, y, weights, bias)
            
            # Update rule (gradient descent)
            weights -= self.learning_rate * weight_gradient
            bias -= self.learning_rate * bias_gradient
            
            # Check convergence based on total loss
            predictions = np.dot(X, weights) + bias
            margins = y * predictions
            
            hinge_loss = np.mean(np.maximum(0, 1 - margins))
            regularization = 0.5 * np.sum(weights ** 2)
            total_loss = regularization + self.C * hinge_loss
            
            if total_loss < self.tol:
                break
        
        return weights, bias
    
    def fit(self, X, y_multilabel):
        """
        Fit the SVM model using one-vs-rest strategy for multi-label data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y_multilabel : array-like of shape (n_samples, n_labels)
            Each row can have between 1 and 3 labels. Zeros denote "no label".
        """
        # Convert lists to numpy arrays if needed
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y_multilabel, list):
            y_multilabel = np.array(y_multilabel)
        
        # Ensure 2D shape in y
        if len(y_multilabel.shape) == 1:
            y_multilabel = y_multilabel.reshape(-1, 1)
        
        # Identify unique classes (non-zero)
        unique_classes = set()
        for row in y_multilabel:
            for label in row:
                if label != 0:
                    unique_classes.add(label)
        
        self.classes_ = sorted(unique_classes)
        
        # Train one binary classifier (1 vs -1) per class
        for cls in self.classes_:
            # Binary labels for this class
            binary_y = np.array([
                1 if cls in row else -1 
                for row in y_multilabel
            ])
            
            weights, bias = self._train_binary_classifier(X, binary_y)
            self.models[cls] = (weights, bias)
        
        return self
    
    def predict(self, X):
        """
        Predict the top 1-3 labels for each sample in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        
        Returns:
        --------
        padded_predictions : np.ndarray of shape (n_samples, <= 3)
            Each row has up to 3 predicted labels (0 used for padding if needed).
        """
        if isinstance(X, list):
            X = np.array(X)
        
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(n_samples):
            scores = {}
            # Compute score for each known class (linearly, ignoring kernel function)
            # If you wanted kernel-based classification, you'd do something more like:
            # scores[cls] = sum(alpha_j * y_j * K(x_j, X[i])) + bias
            # for each support vector x_j. This code is strictly linear.
            for cls in self.classes_:
                weights, bias = self.models[cls]
                scores[cls] = np.dot(X[i], weights) + bias
            
            # Sort classes by descending score
            sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select classes with positive scores, up to 3
            pos_classes = [cls for cls, score in sorted_classes if score > 0][:3]
            
            # If no positive scores, choose the top one by raw score
            if not pos_classes and sorted_classes:
                pos_classes = [sorted_classes[0][0]]
            
            predictions.append(pos_classes)
        
        # Determine the max number of labels predicted for padding
        max_labels = max(len(pred) for pred in predictions) if predictions else 0
        padded_predictions = np.zeros((n_samples, max_labels))
        
        for i, pred in enumerate(predictions):
            for j, cls in enumerate(pred):
                padded_predictions[i, j] = cls
        
        return padded_predictions
    
    def score(self, X, y_true):
        """
        Compute the average F1 score over all samples.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
        y_true : array-like of shape (n_samples, n_labels)
        
        Returns:
        --------
        mean_f1 : float
            Mean F1 score across all samples.
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        
        # Ensure 2D shape
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
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall   = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
