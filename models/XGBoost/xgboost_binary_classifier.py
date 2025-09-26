import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Train -- X: features, y: target
X_train = pd.read_csv("data/model_features_train.csv")
y_train = pd.read_csv("model_labels_train.csv")

# Test -- X: features, y: target
X_test = pd.read_csv("data/model_features_test.csv")
y_test = pd.read_csv("data/model_labels_test.csv")

# Convert to DMatrix (optimized data format for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters for binary classification
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",  # can also add "auc"
    "eta": 0.05,              # learning rate
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "seed": 42,
}

# Train with early stopping
evals = [(dtrain, "train"), (dtest, "eval")]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=50
)

# Predictions
y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration+1))
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
model.save_model("xgboost_sla_model.json")

# Load later if needed:
# loaded_model = xgb.Booster()
# loaded_model.load_model("xgboost_sla_model.json")
