import pandas as pd
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib

# Load training data
X_train = pd.read_csv("data/model_features_train.csv")
y_train = pd.read_csv("model_labels_train.csv").values.ravel()  # flatten in case it's a dataframe

# Load testing data
X_test = pd.read_csv("data/model_features_test.csv")
y_test = pd.read_csv("data/model_labels_test.csv").values.ravel()

ada_param_grid = {
    "n_estimators": [100, 200, 500],
    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
}

ada = AdaBoostClassifier(random_state=42)
ada_grid = GridSearchCV(
    ada,
    ada_param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1
)

ada_grid.fit(X_train, y_train)
print("Best AdaBoost Params:", ada_grid.best_params_)
ada_best = ada_grid.best_estimator_

# Evaluate
y_pred_ada = ada_best.predict(X_test)
print("\n=== AdaBoost (Tuned) ===")
print("Accuracy:", accuracy_score(y_test, y_pred_ada))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_ada))
print("Classification Report:\n", classification_report(y_test, y_pred_ada))

joblib.dump(ada_best, "adaboost_sla_tuned.pkl")