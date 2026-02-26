"""
explore_automl.py

Contains the code to for a simple AutoML pipeline to find suitable models for our data.
We ran this with different configs to identify interesing models.

Generated using ChatGPT and then adapted
"""

# Install FLAML if not already installed
# pip install flaml
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

all_results_df = pd.read_parquet("./analysis_data/all_embeddings.parquet")
finger_id = "032"
task = "regression"

df_this_finger = all_results_df[(all_results_df["task"] == task)]
embeddings = np.vstack(df_this_finger["embeddings"].values)
y = df_this_finger["labels"].values


# ---- Step 1: Split your data ----
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42
)
# ---- Step 2: Scale features (important for linear models) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Step 3: Initialize AutoML ----
automl = AutoML()

# ---- Step 4: Define settings ----
settings = {
    "time_budget": 60,  # in seconds, adjust based on how long you want it to run
    "metric": "r2",  # regression metric, can also use 'rmse'
    "task": "regression",
    "log_file_name": "flaml.log",
    "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree", "catboost"],
    # a mix of linear and non-linear models
}

# ---- Step 5: Fit AutoML ----
automl.fit(X_train_scaled, y_train, **settings)

# ---- Step 6: Evaluate ----
y_pred = automl.predict(X_test_scaled)
r2 = automl.score(X_test_scaled, y_test)
print("Best model:", automl.best_estimator)
print("R^2 on test set:", r2)
