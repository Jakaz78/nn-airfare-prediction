import numpy as np
from collections import defaultdict
import random


class DecisionTreeClassifier:
    """Decision Tree Classifier implemented from scratch"""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        """Train the decision tree"""
        # Ensure y is binary
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only binary values (0 or 1) for binary classification.")
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
            return self._get_majority_class(y)

        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:  # No good split found
            return self._get_majority_class(y)

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Check minimum samples in leaves
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self._get_majority_class(y)

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

        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini = self._calculate_gini_impurity(y, left_mask, right_mask)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini_impurity(self, y, left_mask, right_mask):
        n_samples = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        gini_left = 1.0
        if n_left > 0:
            p_left_0 = np.sum(y[left_mask] == 0) / n_left
            p_left_1 = np.sum(y[left_mask] == 1) / n_left
            gini_left = 1 - (p_left_0 ** 2 + p_left_1 ** 2)

        gini_right = 1.0
        if n_right > 0:
            p_right_0 = np.sum(y[right_mask] == 0) / n_right
            p_right_1 = np.sum(y[right_mask] == 1) / n_right
            gini_right = 1 - (p_right_0 ** 2 + p_right_1 ** 2)

        weighted_gini = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
        return weighted_gini

    def _get_majority_class(self, y):
        """Returns the majority class in a given set of labels."""
        if len(y) == 0:
            return 0  # Default to 0 if no samples
        unique_classes, counts = np.unique(y, return_counts=True)
        return unique_classes[np.argmax(counts)]

    def predict(self, X):
        """Make predictions for input data"""
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        """Make prediction for a single sample"""
        if isinstance(tree, (int, float)):  # Leaf node contains the predicted class
            return int(tree)  # Ensure integer output for classification

        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.oob_predictions = defaultdict(list)
        self.oob_accuracy_ = None
        self.oob_recall_ = None
        self.oob_precision_ = None
        self.oob_f1_ = None
        self.y_train = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        return X[indices], y[indices], oob_indices

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        self.trees = []
        self.oob_predictions = defaultdict(list)
        self.y_train = y.copy()  # Store y_train for OOB calculations

        for i in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample, oob_indices = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
                oob_indices = []

            tree = DecisionTreeClassifier(
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

        # Compute OOB scores if we have OOB samples
        if len(self.oob_predictions) > 0:
            y_true_oob = []
            y_pred_oob = []
            for idx, preds in self.oob_predictions.items():
                y_true_oob.append(self.y_train[idx])
                # Majority vote for OOB predictions
                unique_preds, counts = np.unique(preds, return_counts=True)
                y_pred_oob.append(unique_preds[np.argmax(counts)])

            y_true_oob = np.array(y_true_oob)
            y_pred_oob = np.array(y_pred_oob)

            self.oob_accuracy_ = accuracy_score(y_true_oob, y_pred_oob)
            self.oob_recall_ = recall_score(y_true_oob, y_pred_oob)
            self.oob_precision_ = precision_score(y_true_oob, y_pred_oob)
            self.oob_f1_ = f1_score(y_true_oob, y_pred_oob)

        return self

    def predict(self, X):
        # Collect predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # For each sample, perform a majority vote across all tree predictions
        final_predictions = []
        for i in range(tree_predictions.shape[1]):  # Iterate through samples
            predictions_for_sample = tree_predictions[:, i]
            unique_classes, counts = np.unique(predictions_for_sample, return_counts=True)
            final_predictions.append(unique_classes[np.argmax(counts)])

        return np.array(final_predictions)


# --- Classification Metrics ---

def accuracy_score(y_true, y_pred):
    """Calculate Accuracy Score"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, positive_label=1):
    """Calculate Precision Score for binary classification"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positives = np.sum((y_pred == positive_label) & (y_true == positive_label))
    predicted_positives = np.sum(y_pred == positive_label)
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall_score(y_true, y_pred, positive_label=1):
    """Calculate Recall Score for binary classification"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    true_positives = np.sum((y_pred == positive_label) & (y_true == positive_label))
    actual_positives = np.sum(y_true == positive_label)
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def f1_score(y_true, y_pred, positive_label=1):
    """Calculate F1-Score for binary classification"""
    precision = precision_score(y_true, y_pred, positive_label)
    recall = recall_score(y_true, y_pred, positive_label)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and testing sets"""
    if random_state:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# --- Example Usage ---
if __name__ == "__main__":
    # Generate some dummy data for binary classification
    np.random.seed(42)
    X = np.random.rand(200, 5) * 10  # 200 samples, 5 features
    y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Binary target based on two features

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("--- Testing DecisionTreeClassifier ---")
    dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)

    print(f"Decision Tree Test Accuracy: {accuracy_score(y_test, dt_predictions):.4f}")
    print(f"Decision Tree Test Recall: {recall_score(y_test, dt_predictions):.4f}")
    print(f"Decision Tree Test Precision: {precision_score(y_test, dt_predictions):.4f}")
    print(f"Decision Tree Test F1-Score: {f1_score(y_test, dt_predictions):.4f}")

    print("\n--- Testing RandomForestClassifier ---")
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    rf_predictions_test = rf_classifier.predict(X_test)
    rf_predictions_train = rf_classifier.predict(X_train)

    print("\n--- Training Metrics ---")
    print(f"Random Forest Train Accuracy: {accuracy_score(y_train, rf_predictions_train):.4f}")
    print(f"Random Forest Train Recall: {recall_score(y_train, rf_predictions_train):.4f}")
    print(f"Random Forest Train Precision: {precision_score(y_train, rf_predictions_train):.4f}")
    print(f"Random Forest Train F1-Score: {f1_score(y_train, rf_predictions_train):.4f}")

    print("\n--- Test Metrics ---")
    print(f"Random Forest Test Accuracy: {accuracy_score(y_test, rf_predictions_test):.4f}")
    print(f"Random Forest Test Recall: {recall_score(y_test, rf_predictions_test):.4f}")
    print(f"Random Forest Test Precision: {precision_score(y_test, rf_predictions_test):.4f}")
    print(f"Random Forest Test F1-Score: {f1_score(y_test, rf_predictions_test):.4f}")

    print("\n--- Out-of-Bag (OOB) Metrics ---")
    if rf_classifier.oob_accuracy_ is not None:
        print(f"Random Forest OOB Accuracy: {rf_classifier.oob_accuracy_:.4f}")
        print(f"Random Forest OOB Recall: {rf_classifier.oob_recall_:.4f}")
        print(f"Random Forest OOB Precision: {rf_classifier.oob_precision_:.4f}")
        print(f"Random Forest OOB F1-Score: {rf_classifier.oob_f1_:.4f}")
    else:
        print("OOB score not available (perhaps bootstrap was set to False or no OOB samples generated).")