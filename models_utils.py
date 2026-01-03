import re
import numpy as np
from collections import Counter
from scipy.special import expit

custom_stopwords = set(["the", "and", "is", "of", "in", "to", "a", "an", "and", "are", "as", "at", "be", "but", "by", 
                        "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", 
                        "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])

class DecisionTree:
    def __init__(self, depth=10):
        self.depth = depth
        self.tree = None
        self.feature_importance_ = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
        self.feature_importance_ = self._calculate_feature_importance(X)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))
        if depth >= self.depth or num_labels == 1 or num_samples < 2:
            return {"leaf": True, "value": self._most_common_label(y)}
        feat_idxs = np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left_idxs = left_idxs.astype(int)
        right_idxs = right_idxs.astype(int)
        left_tree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_tree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {"leaf": False, "feature": best_feat, "threshold": best_thresh, "left": left_tree, "right": right_tree}

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.percentile(X_column, np.linspace(0, 100, 10))
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        n, n_l, n_r = len(y), len(left_idxs), len(right_idxs)
        if n_l == 0 or n_r == 0:
            return 0
        child_entropy = (n_l / n) * self._entropy(y[left_idxs]) + (n_r / n) * self._entropy(y[right_idxs])
        return parent_entropy - child_entropy

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0].astype(int)
        right_idxs = np.where(X_column > split_thresh)[0].astype(int)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return np.arange(len(X_column)), np.array([])
        return left_idxs, right_idxs

    def _entropy(self, y):
        if y.dtype.kind == 'f': return np.var(y)
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0: return 0
        if y.dtype.kind == 'f': return np.mean(y)
        return np.bincount(y).argmax()

    def _calculate_feature_importance(self, X):
        feature_importance = np.zeros(X.shape[1])
        self._accumulate_feature_importance(self.tree, feature_importance)
        return feature_importance

    def _accumulate_feature_importance(self, node, feature_importance):
        if node["leaf"]: return
        feature_importance[node["feature"]] += 1
        self._accumulate_feature_importance(node["left"], feature_importance)
        self._accumulate_feature_importance(node["right"], feature_importance)

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.trees = []
        self.initial_prediction = None
        self.feature_importances_ = None

    def fit(self, X, y):
        y = np.array(y)
        self.initial_prediction = np.mean(y)
        current_prediction = self.initial_prediction * np.ones_like(y)
        feature_importance = np.zeros(X.shape[1])
        for _ in range(self.n_estimators):
            if self.subsample < 1.0:
                sample_idxs = np.random.choice(np.arange(len(y)), size=int(self.subsample * len(y)), replace=False)
                X_sample = X[sample_idxs, :]
                y_sample = y[sample_idxs]
            else:
                X_sample = X
                y_sample = y
            if self.subsample < 1.0:
                residual = y_sample - current_prediction[sample_idxs]
            else:
                residual = y_sample - current_prediction
            if len(residual) == 0: residual = np.zeros_like(y_sample)
            tree = DecisionTree(depth=self.max_depth)
            tree.fit(X_sample, residual)
            self.trees.append(tree)
            if self.subsample < 1.0:
                current_prediction[sample_idxs] += self.learning_rate * tree.predict(X_sample)
            else:
                current_prediction += self.learning_rate * tree.predict(X_sample)
            feature_importance += tree.feature_importance_
        self.feature_importances_ = feature_importance / max(1, self.n_estimators)

    def predict(self, X):
        prediction = self.initial_prediction * np.ones((X.shape[0],))
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(X)
        return np.round(prediction).astype(int)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

class SimpleVectorizer:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocab = {}
        self.vocab_size = 0
        self.stopwords = custom_stopwords

    def fit_transform(self, texts):
        word_counts = Counter()
        for text in texts:
            cleaned_text = self.clean_text(text)
            word_counts.update(cleaned_text.split())
        if self.max_features is not None:
            top_n_words = word_counts.most_common(self.max_features)
            self.vocab = {word: i for i, (word, _) in enumerate(top_n_words)}
            self.vocab_size = len(top_n_words)
        else:
            for word in word_counts:
                self.vocab[word] = self.vocab_size
                self.vocab_size += 1
        X = np.zeros((len(texts), self.vocab_size))
        for i, text in enumerate(texts):
            cleaned_text = self.clean_text(text)
            for word in cleaned_text.split():
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

    def transform(self, texts):
        X = np.zeros((len(texts), self.vocab_size))
        for i, text in enumerate(texts):
            cleaned_text = self.clean_text(text)
            for word in cleaned_text.split():
                if word in self.vocab:
                    X[i, self.vocab[word]] += 1
        return X

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stopwords]
        return ' '.join(tokens)
