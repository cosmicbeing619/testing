import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_excel("preprocessed_data.xlsx")

# Encode categorical columns
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include="object").columns
for col in categorical_cols:
    df_encoded[col] = df_encoded[col].astype(str)
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Define selected features (same as used in Streamlit app)
selected_features = [
    "bulkdensity", "clay(%)", "medium",
    "organiccarbon", "sand(%)", "silt(%)", "veryfine"
]

# Features and target
X = df_encoded[selected_features]
y = df_encoded["ksat_cm_hr"]

# Subset sizes for training analysis
subset_sizes = list(range(len(df_encoded), 2000, -2000)) + [2000]
rmsle_results = []
r2_results = []

# Track best model and predictions
best_model = None
best_r2 = -np.inf
final_predictions_df = None

for subset_size in subset_sizes:
    rmsle_scores = []
    r2_scores = []

    for _ in tqdm(range(50), desc=f"Subset size {subset_size}"):
        sample = df_encoded.sample(n=subset_size, random_state=np.random.randint(10000))
        X_sample = sample[selected_features]
        y_sample = sample["ksat_cm_hr"]

        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

        # Fixed hyperparameter model
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rmsle_scores.append(rmsle)
        r2_scores.append(r2)

        # Save best model and predictions
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            final_predictions_df = pd.DataFrame({
                "Actual_ksat_cm_hr": y_test.values,
                "Predicted_ksat_cm_hr": y_pred
            })

    rmsle_results.append(np.mean(rmsle_scores))
    r2_results.append(np.mean(r2_scores))

# Save best model and feature list
if best_model is not None:
    joblib.dump((best_model, selected_features), "best_rf_model.joblib")

# Save final test predictions
if final_predictions_df is not None:
    final_predictions_df.to_csv("rf_test_predictions.csv", index=False)

# Plot RMSLE
plt.figure(figsize=(12, 6))
plt.plot(subset_sizes, rmsle_results, marker='o', label='RMSLE')
plt.xlabel('Training Sample Size')
plt.ylabel('RMSLE')
plt.title('RMSLE vs Training Sample Size')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rf_rmsle_plot.png")
plt.show()

# Plot R²
plt.figure(figsize=(12, 6))
plt.plot(subset_sizes, r2_results, marker='s', color='green', label='R² Score')
plt.xlabel('Training Sample Size')
plt.ylabel('R² Score')
plt.title('R² Score vs Training Sample Size')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rf_r2_plot.png")
plt.show()
