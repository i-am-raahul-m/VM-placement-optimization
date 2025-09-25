from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np

# ----------------------------
# Load model
# ----------------------------
model = TabNetClassifier()   # no args needed if you're just loading
model.load_model("tabnet_classifier_model.zip")  # or "tabnet_classifier" (without extension)

# ----------------------------
# Dummy input test
# ----------------------------
# suppose model was trained on 16 features
X_test = np.random.rand(5, 16).astype(np.float32)

# Predict probabilities
proba = model.predict_proba(X_test)
print("Probabilities:\n", proba)

# Predict labels
preds = model.predict(X_test)
print("Predictions:\n", preds)
