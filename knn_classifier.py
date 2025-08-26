import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        print("🚨 X input shape:", X.shape)
        print("🚨 X_train shape:", self.X_train.shape)

        predictions = []

        for x in X:
            print("🧪 single x shape:", x.shape if hasattr(x, 'shape') else type(x))
            x = np.array(x).reshape(1, -1)  # Ensure shape (1, n_features)
            print("✅ reshaped x shape:", x.shape)

            distances = np.linalg.norm(self.X_train - x, axis=1)
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = self.y_train[neighbor_indices]
            neighbor_distances = distances[neighbor_indices]

            if self.weights == 'uniform':
                prediction = Counter(neighbor_labels).most_common(1)[0][0]

            elif self.weights == 'distance':
                weights = 1 / (neighbor_distances + 1e-8)  # Avoid division by zero
                class_weights = {}
                for label, weight in zip(neighbor_labels, weights):
                    class_weights[label] = class_weights.get(label, 0) + weight
                prediction = max(class_weights, key=class_weights.get)

            else:
                raise ValueError("weights must be 'uniform' or 'distance'")

            predictions.append(prediction)

        return np.array(predictions)
