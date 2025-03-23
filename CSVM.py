import numpy as np
from collections import defaultdict

class CSVM:
    """
    Custom C-SVM implementation for multi-label classification problems.
    Each instance can have 1-3 labels.
    """
    
    def __init__(self, X, y, C=1.0, kernel='linear', tol=1e-3, max_iter=1000, learning_rate=0.01):
        """
        Initialize the C-SVM.
        
        Parameters:
        -----------
        C : float, default=1.0
            Regularization parameter. Trades off margin size and training error.
        kernel : str, default='linear'
            Kernel type to be used. Options: 'linear', 'poly', 'rbf'
        tol : float, default=1e-3
            Tolerance for stopping criterion.
        max_iter : int, default=1000
            Maximum number of iterations.
        learning_rate : float, default=0.01
            Learning rate for gradient descent.
        """
        self.X = X
        self.y = y 
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
        
        Parameters:
        -----------
        x1, x2 : array-like
            Input vectors.
            
        Returns:
        --------
        float : Kernel value
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** 2
        elif self.kernel == 'rbf':
            gamma = 1.0 / x1.shape[0]
            return np.exp(-gamma * np.sum((x1 - x2) ** 2))
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_gradient(self, X, y, weights, bias):
        """
        Compute the gradient of the SVM loss function.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        weights : array-like of shape (n_features,)
            Current weights.
        bias : float
            Current bias.
            
        Returns:
        --------
        tuple : (weight_gradient, bias_gradient)
        """
        n_samples = X.shape[0]
        
        # Calculate predictions
        predictions = np.dot(X, weights) + bias
        
        # Compute margins
        margins = y * predictions
        
        # Calculate the gradient
        weight_gradient = weights  # Regularization term
        bias_gradient = 0
        
        for i in range(n_samples):
            if margins[i] < 1:
                weight_gradient -= self.C * y[i] * X[i]
                bias_gradient -= self.C * y[i]
        
        return weight_gradient, bias_gradient
    
    def _train_binary_classifier(self, X, y):
        """
        Train a binary SVM classifier.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns:
        --------
        tuple : (weights, bias) parameters of the trained model
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        weights = np.zeros(n_features)
        bias = 0.0
        
        # Gradient descent
        for _ in range(self.max_iter):
            weight_gradient, bias_gradient = self._compute_gradient(X, y, weights, bias)
            
            # Update parameters
            weights -= self.learning_rate * weight_gradient
            bias -= self.learning_rate * bias_gradient
            
            # Calculate loss for convergence check
            predictions = np.dot(X, weights) + bias
            margins = y * predictions
            hinge_loss = np.mean(np.maximum(0, 1 - margins))
            regularization = 0.5 * np.sum(weights ** 2)
            
            total_loss = regularization + self.C * hinge_loss
            
            # Check convergence
            if total_loss < self.tol:
                break
        
        return weights, bias
    
    def fit(self, X, y_multilabel):
        """
        Fit the SVM model according to the given training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y_multilabel : array-like of shape (n_samples, n_labels)
            Target values. Each row contains 1-3 labels.
            
        Returns:
        --------
        self : object
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y_multilabel, list):
            y_multilabel = np.array(y_multilabel)
        
        # If y is provided as a 1D array, reshape it
        if len(y_multilabel.shape) == 1:
            y_multilabel = y_multilabel.reshape(-1, 1)
        
        # Get unique classes
        unique_classes = set()
        for row in y_multilabel:
            for label in row:
                if label != 0:  # Assuming 0 means no label
                    unique_classes.add(label)
        
        self.classes_ = sorted(list(unique_classes))
        
        # Train one-vs-rest binary classifiers for each class
        for cls in self.classes_:
            # Create binary labels: 1 if the instance has this class, -1 otherwise
            binary_y = np.array([-1] * len(y_multilabel))
            for i, row in enumerate(y_multilabel):
                if cls in row:
                    binary_y[i] = 1
            
            # Train binary classifier
            weights, bias = self._train_binary_classifier(X, binary_y)
            self.models[cls] = (weights, bias)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples, n_max_labels)
            The predicted classes. Each row contains 1-3 labels.
        """
        if isinstance(X, list):
            X = np.array(X)
        
        n_samples = X.shape[0]
        
        # Store predictions and scores for each class
        predictions = []
        
        for i in range(n_samples):
            # Get scores for each class
            scores = {}
            for cls in self.classes_:
                weights, bias = self.models[cls]
                scores[cls] = np.dot(X[i], weights) + bias
            
            # Sort classes by scores and get top 3
            sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Take only classes with positive scores, up to 3
            pos_classes = [cls for cls, score in sorted_classes if score > 0][:3]
            
            # If no positive scores, take the highest score
            if not pos_classes and sorted_classes:
                pos_classes = [sorted_classes[0][0]]
            
            predictions.append(pos_classes)
        
        # Pad predictions to have consistent shape
        max_labels = max(len(pred) for pred in predictions)
        padded_predictions = np.zeros((n_samples, max_labels))
        
        for i, pred in enumerate(predictions):
            for j, cls in enumerate(pred):
                padded_predictions[i, j] = cls
        
        return padded_predictions
    
    def score(self, X, y_true):
        """
        Calculate the accuracy for the multi-label classification.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y_true : array-like of shape (n_samples, n_labels)
            True labels.
            
        Returns:
        --------
        score : float
            Average F1 score across all instances.
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        
        # If y is provided as a 1D array, reshape it
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            
        y_pred = self.predict(X)
        
        # Calculate F1 score for each instance
        f1_scores = []
        
        for i in range(len(y_true)):
            true_set = set(y_true[i][y_true[i] != 0])
            pred_set = set(y_pred[i][y_pred[i] != 0])
            
            true_positive = len(true_set.intersection(pred_set))
            false_positive = len(pred_set - true_set)
            false_negative = len(true_set - pred_set)
            
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    

if __name__ == "__main__":

    # Initialize and train the model
    model = CSVM(C=1.0, kernel='linear', max_iter=1000, learning_rate=0.01)
    model.fit(X, y)
    
    # Make predictions on your test data
    y_pred = model.predict(X_test)  # Assuming X_test is defined
    
    # Evaluate the model
    score = model.score(X_test, y_test)  # Assuming y_test is defined
    print(f"F1 Score: {score:.4f}")
    
