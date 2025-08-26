import numpy as np
from collections import defaultdict
import random

class DecisionTreeRegressor:
    """Decision Tree Regressor implemented from scratch"""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (n_samples < 2 * self.min_samples_leaf) or \
           (len(np.unique(y)) == 1):
            return np.mean(y)
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return np.mean(y)
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check minimum samples in leaves
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return np.mean(y)
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape

        # Determine how many features to consider
        if self.max_features == 'sqrt':
            features_to_consider = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            features_to_consider = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            features_to_consider = min(self.max_features, n_features)
        elif self.max_features is None:
            features_to_consider = n_features
        else:
            raise ValueError("Invalid value for max_features")

        # Randomly select a subset of features
        feature_indices = np.random.choice(n_features, features_to_consider, replace=False)

        best_mse = float('inf')
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_mse = np.var(y[left_mask]) * np.sum(left_mask)
                right_mse = np.var(y[right_mask]) * np.sum(right_mask)
                weighted_mse = (left_mse + right_mse) / n_samples

                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold
    
    def predict(self, X):
        """Make predictions for input data"""
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def _predict_single(self, x, tree):
        """Make prediction for a single sample"""
        if isinstance(tree, (int, float)):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features='sqrt', bootstrap=True, random_state=None, sample_size=0.63):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.oob_predictions = defaultdict(list)
        self.sample_size = sample_size
        self.oob_score_ = None
        self.oob_mse_ = None
        self.y_train = None
        self.sample_size = sample_size


    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=int(self.sample_size*n_samples), replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        return X[indices], y[indices], oob_indices

    def fit(self, X, y):
            
        self.trees = []
        self.oob_predictions = defaultdict(list)
        self.y_train = y.copy()
        
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample, oob_indices = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
                oob_indices = []

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            for idx in oob_indices:
                pred = tree.predict(X[idx].reshape(1, -1))[0]
                self.oob_predictions[idx].append(pred)

        # Compute OOB score if we have OOB samples
        if len(self.oob_predictions) > 0:
            y_true = []
            y_pred = []
            for idx, preds in self.oob_predictions.items():
                y_true.append(self.y_train[idx])
                y_pred.append(np.mean(preds))

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            self.oob_score_ = r2_score(y_true, y_pred)
            self.oob_mse_ = mean_squared_error(y_true, y_pred)
        
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)

    def get_oob_r2(self):
        if not hasattr(self, 'y_train'):
            raise ValueError("Model not fitted yet")
        if len(self.oob_predictions) == 0:
            return None
            
        y_true = []
        y_pred = []
        for idx, preds in self.oob_predictions.items():
            y_true.append(self.y_train[idx])
            y_pred.append(np.mean(preds))

        return r2_score(np.array(y_true), np.array(y_pred))

    def get_oob_mse(self):
        if not hasattr(self, 'y_train'):
            raise ValueError("Model not fitted yet")
        if len(self.oob_predictions) == 0:
            return None
            
        y_true = []
        y_pred = []
        for idx, preds in self.oob_predictions.items():
            y_true.append(self.y_train[idx])
            y_pred.append(np.mean(preds))

        return mean_squared_error(np.array(y_true), np.array(y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return float('nan')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def r2_score(y_true, y_pred):
    """Calculate R-squared score"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float('nan')
    return 1 - (ss_res / ss_tot)

def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))