import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load data
X_train = pd.read_csv("model_features_train.csv")
y_train = pd.read_csv("model_labels_train.csv").values.ravel()
X_test = pd.read_csv("model_features_test.csv")
y_test = pd.read_csv("model_labels_test.csv").values.ravel()

# Train
model = RidgeClassifier(alpha=1.0)  # You can tune alpha
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("RidgeClassifier Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'ridge_classifier.pkl')
