
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

features = [
    "pages_viewed", "session_minutes", "products_viewed",
    "cart_additions", "discount_seen", "previous_orders"
]

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nClass balance:\n", df["target"].value_counts())

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        solver="liblinear", max_iter=1000, random_state=42
    ))
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

scores = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}
print("\nEvaluation:")
for name, score in scores.items():
    print(f"{name}: {score:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"]).plot(values_format="d")
plt.title("Confusion Matrix — Logistic Regression")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=180)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {scores['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=180)
plt.close()

lr = model.named_steps["logistic_regression"]
coef = pd.DataFrame({
    "feature": features,
    "coefficient": lr.coef_[0],
    "odds_ratio_per_1_SD": np.exp(lr.coef_[0])
}).sort_values("coefficient", ascending=False)
print("\nCoefficient interpretation:\n", coef.to_string(index=False))
coef.to_csv(OUT / "coefficient_analysis.csv", index=False)

print("\nClassification report:\n")
print(__import__("sklearn").metrics.classification_report(
    y_test, y_pred, target_names=["No Purchase", "Purchase"]
))
