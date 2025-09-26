from sklearn.linear_model import SGDClassifier
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
X_train = pd.read_csv("../../data/model_features_train.csv")
y_train = pd.read_csv("../../data/model_labels_train.csv").values.ravel()
X_test = pd.read_csv("../../data/model_features_test.csv")
y_test = pd.read_csv("../../data/model_labels_test.csv").values.ravel()

# Train
model = SGDClassifier(
    loss='log_loss',       # for binary classification with probability estimates
    penalty='l2',
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("SGDClassifier Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'sgd_classifier.pkl')