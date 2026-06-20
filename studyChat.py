import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report

# 1) Toy data (replace with your own X, y)
X, y = make_classification(
    n_samples=5000, n_features=30, n_informative=12, n_redundant=5,
    n_classes=3, random_state=42
)

X_train, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Make a partially-labeled training set:
#    unlabeled points must be marked as -1 in sklearn semi-supervised APIs
rng = np.random.RandomState(42)
y_train = y_train_full.copy()
mask_unlabeled = rng.rand(len(y_train)) < 0.90   # 90% unlabeled, 10% labeled
y_train[mask_unlabeled] = -1

# 3) Base classifier (must support predict_proba or decision_function)
base = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, multi_class="auto"))
])

# 4) Self-training wrapper
ssl = SelfTrainingClassifier(
    estimator=base,
    threshold=0.9,      # confidence cutoff for pseudo-labels
    criterion="threshold",  # or "k_best"
    max_iter=20,
    verbose=True
)

# 5) Fit and evaluate
ssl.fit(X_train, y_train)
y_pred = ssl.predict(X_test)

print("Pseudo-labeled added:", (ssl.transduction_ != -1).sum() - (y_train != -1).sum())
print(classification_report(y_test, y_pred))
