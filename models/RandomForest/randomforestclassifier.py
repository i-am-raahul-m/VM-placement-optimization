import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib

# Load training data
X_train = pd.read_csv("model_features_train.csv")
y_train = pd.read_csv("model_labels_train.csv").values.ravel()  # flatten in case it's a dataframe

# Load testing data
X_test = pd.read_csv("model_features_test.csv")
y_test = pd.read_csv("model_labels_test.csv").values.ravel()

# -------------------------------
# Random Forest
# -------------------------------
rf_param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = RandomizedSearchCV(
    rf,
    rf_param_grid,
    n_iter=20,             # try 20 random combinations
    cv=3,                  # 3-fold cross validation
    scoring="accuracy",
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
print("Best Random Forest Params:", rf_grid.best_params_)
rf_best = rf_grid.best_estimator_

# Evaluate
y_pred_rf = rf_best.predict(X_test)
print("\n=== Random Forest (Tuned) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
