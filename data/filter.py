import pandas as pd

df = pd.read_csv('model_features_test_with_exec_time.csv')
label = pd.read_csv('model_labels_test.csv')
df['sla_violation'] = label
df = df[df["sla_violation"] == 0]

df.to_csv("simulation_tasks.csv", index=False)