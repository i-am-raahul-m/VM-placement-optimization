import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import numpy as np

# Load data
X_train = pd.read_csv("data/model_features_train.csv")
y_train = pd.read_csv("model_labels_train.csv").values.ravel()
X_test = pd.read_csv("data/model_features_test.csv")
y_test = pd.read_csv("model_labels_test.csv").values.ravel()

# Convert to NumPy (TabNet requires this)
X_train_np = X_train.values
X_test_np = X_test.values
y_train_np = y_train.astype(int)
y_test_np = y_test.astype(int)

from pytorch_tabnet.pretraining import TabNetPretrainer

# Initialize and train TabNet
tabnet_model = TabNetClassifier(
    n_d=16, n_a=16,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',  # 'sparsemax' or 'entmax'
    verbose=1,
    seed=42
)

tabnet_model.fit(
    X_train=X_train_np, y_train=y_train_np,
    eval_set=[(X_test_np, y_test_np)],
    eval_metric=['accuracy'],
    max_epochs=200,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Predict
y_pred = tabnet_model.predict(X_test_np)

# Evaluate
print("TabNet Accuracy:", accuracy_score(y_test_np, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred))
print("Classification Report:\n", classification_report(y_test_np, y_pred))

# Save the model and weights
tabnet_model.save_model("tabnet_classifier_model")
