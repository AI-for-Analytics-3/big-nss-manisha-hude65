prompt_to_claude = """
Use this cleaned dataset to build a classification model where:
- Target variable: DerateFlag
- Predictors: DistanceLtd, FuelLtd, EngineTimeLtd, EngineOilPressure, EngineOilTemperature
- Goal: predict whether DerateFlag is True or False using those five numeric features
- Validation requirement: split the data into train and test sets so the model is evaluated on unseen data, not just the same data it was trained on
- If I am unfamiliar with train/test split, explain what it is in simple terms: a model is trained on one subset and evaluated on a different subset to check how well it generalizes
- Please suggest a model, explain why it is appropriate, and show the code to train it and evaluate it on the test set
- Report the key metrics such as accuracy, precision, recall, F1-score, and a confusion matrix
- Also explain the business meaning of the results in plain language
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path("big_nss.csv")
OUTPUT_DIR = Path("visuals")
OUTPUT_DIR.mkdir(exist_ok=True)

# Read the CSV into a dataframe named big_nss
big_nss = pd.read_csv(DATA_PATH)

# Remove the unnamed index column if present
if big_nss.columns[0].startswith("Unnamed"):
    big_nss = big_nss.drop(columns=[big_nss.columns[0]])

# Clean the derate column so it is a boolean flag
big_nss["Derate"] = big_nss["Derate"].astype(str).str.strip().str.lower()
big_nss["DerateFlag"] = big_nss["Derate"].map({
    "true": True,
    "false": False,
    "1": True,
    "0": False,
}).fillna(False)

# Convert the 5 measurement columns to numeric values
measurement_cols = [
    "EngineOilPressure",
    "EngineOilTemperature",
    "DistanceLtd",
    "FuelLtd",
    "EngineTimeLtd",
]

for col in measurement_cols:
    big_nss[col] = pd.to_numeric(big_nss[col], errors="coerce")

# Drop rows where all oil and LTD measurements are missing.
# This keeps valid rows but removes rows that are incomplete for the key core metrics.
core_measurement_cols = [
    "EngineOilPressure",
    "EngineOilTemperature",
    "DistanceLtd",
    "FuelLtd",
    "EngineTimeLtd",
]

rows_to_drop = big_nss[core_measurement_cols].isna().all(axis=1)
big_nss = big_nss.loc[~rows_to_drop].copy()

# Remove trucks with too little LTD information across the entire equipment history.
# Keep only EquipmentIDs where fewer than 2 of the 3 LTD columns are all-NaN.
ltd_cols = ["DistanceLtd", "FuelLtd", "EngineTimeLtd"]
missing_ltd_by_equipment = big_nss.groupby("EquipmentID")[ltd_cols].apply(
    lambda x: (x.isna().all(axis=0)).sum()
)
ids_to_remove = missing_ltd_by_equipment[missing_ltd_by_equipment >= 2].index
big_nss = big_nss.loc[~big_nss["EquipmentID"].isin(ids_to_remove)].copy()

print("Equipment IDs removed for missing LTD columns:", len(ids_to_remove))
print("Rows removed for fully missing core measurements:", rows_to_drop.sum())

# If a row has at least one known LTD value, use the mean of the known LTD values
# to fill the remaining LTD fields for that same row.
ltd_cols = ["DistanceLtd", "FuelLtd", "EngineTimeLtd"]
row_ltd_mean = big_nss[ltd_cols].mean(axis=1, skipna=True)
big_nss["row_ltd_fill"] = row_ltd_mean
for col in ltd_cols:
    big_nss[col] = big_nss[col].fillna(big_nss["row_ltd_fill"])

big_nss = big_nss.drop(columns=["row_ltd_fill"])

print("Missing values after LTD fill using known LTD values:")
print(big_nss[core_measurement_cols].isna().sum())

# Predict missing EngineOil values using the LTD columns.
# Case 1: both EngineOil columns are missing -> estimate from the LTD columns.
# Case 2: only one EngineOil column is missing -> use the non-missing engine-oil value and LTD values to estimate the missing one.
oil_cols = ["EngineOilPressure", "EngineOilTemperature"]

for idx, row in big_nss.iterrows():
    engine_oil_values = row[oil_cols]
    known_oil = engine_oil_values.dropna()
    missing_oil = engine_oil_values.isna()

    if not missing_oil.any():
        continue

    ltd_values = row[ltd_cols].dropna()
    if ltd_values.empty:
        continue

    if missing_oil.all():
        predicted_value = ltd_values.mean()
        for col in oil_cols:
            big_nss.at[idx, col] = predicted_value
    else:
        observed_oil_value = known_oil.iloc[0]
        predicted_value = observed_oil_value + (ltd_values.mean() - row[ltd_cols].mean())
        for col in oil_cols:
            if pd.isna(row[col]):
                big_nss.at[idx, col] = predicted_value

print("Missing values after EngineOil prediction:")
print(big_nss[core_measurement_cols].isna().sum())

# Print a quick summary for comparison
summary = big_nss.groupby("DerateFlag")[measurement_cols].mean().round(2)
print("\nMean values by derate status:")
print(summary)

# Model: predict DerateFlag using the 5 numeric feature columns.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

features = ["DistanceLtd", "FuelLtd", "EngineTimeLtd", "EngineOilPressure", "EngineOilTemperature"]
X = big_nss[features]
y = big_nss["DerateFlag"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name} metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if name == "Logistic Regression":
        coef_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_[0],
            "OddsRatio": np.exp(model.coef_[0])
        })
        print("\nLogistic Regression Odds Ratios:")
        print(coef_df.round(4).to_string(index=False))
        print("\nInterpretation:")
        print("An odds ratio greater than 1 means that as the feature increases, the odds of a derate increase.")
        print("An odds ratio less than 1 means that as the feature increases, the odds of a derate decrease.")
        print("For example, if EngineOilPressure has an odds ratio of 1.10, then a 1-unit increase in pressure is associated with 10% higher odds of a derate, holding other variables constant.")

# Time-based future trigger prediction: evaluate ROC AUC for multiple future horizons.
# Use vectorized pandas operations instead of loops/iterrows for efficiency.
# We will classify whether a derate occurs within the next lead time window.
lead_times = {
    "12_hours": 12,
    "1_day": 1,
    "3_days": 3,
    "7_days": 7,
    "14_days": 14,
    "21_days": 21,
}

# Convert timestamps to datetime and sort by time.
if "EventTimeStamp" in big_nss.columns:
    big_nss["EventTimeStamp"] = pd.to_datetime(big_nss["EventTimeStamp"], errors="coerce")
    big_nss = big_nss.sort_values("EventTimeStamp").reset_index(drop=True)

# Build the target for each lead time using vectorized time window logic.
# A row is labeled 1 if a derate is observed within the next lead_time days.
future_df = big_nss.copy()

for label, days in lead_times.items():
    time_delta = pd.to_timedelta(days, unit="D")
    future_df[f"future_derate_{label}"] = (
        future_df["EventTimeStamp"].shift(-1).fillna(pd.NaT) <= (future_df["EventTimeStamp"] + time_delta)
    ).astype(int)

# Keep only rows with a valid timestamp for time-based prediction.
future_df = future_df.dropna(subset=["EventTimeStamp"]).copy()

print("\nROC AUC by future trigger window:")
for label, days in lead_times.items():
    target_col = f"future_derate_{label}"
    y_future = future_df[target_col].astype(int)
    if y_future.nunique() < 2:
        print(f"{label}: insufficient variation in target")
        continue

    X_future = future_df[features]
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_future,
        y_future,
        test_size=0.2,
        random_state=42,
        stratify=y_future,
    )

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_f, y_train_f)
    y_score = model.predict_proba(X_test_f)[:, 1]
    auc = roc_auc_score(y_test_f, y_score)
    print(f"{label}: ROC AUC = {auc:.4f}")

# Create a visualization for the 5 numeric measurement columns
fig, axes = plt.subplots(len(measurement_cols), 1, figsize=(12, 18))

for ax, col in zip(axes, measurement_cols):
    derate_values = big_nss.loc[big_nss["DerateFlag"] == True, col].dropna()
    non_derate_values = big_nss.loc[big_nss["DerateFlag"] == False, col].dropna()

    ax.boxplot(
        [derate_values, non_derate_values],
        tick_labels=["Derate", "No Derate"],
        patch_artist=True,
        widths=0.5,
        boxprops={"facecolor": "lightsteelblue", "edgecolor": "black"},
        medianprops={"color": "darkred", "linewidth": 2},
    )
    ax.set_title(f"{col}: Derate vs No Derate")
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "derate_vs_no_derate_measurements.png", dpi=200)
plt.close()

# Bar chart comparing average oil pressure and temperature by derate status.
oil_bar_data = big_nss.groupby("DerateFlag")[["EngineOilPressure", "EngineOilTemperature"]].mean().reindex([True, False]).copy()
oil_bar_data.index = ["Derate", "No Derate"]

plt.figure(figsize=(8, 6))
x = np.arange(len(oil_bar_data.index))
width = 0.35

plt.bar(x - width/2, oil_bar_data["EngineOilPressure"], width=width, color="royalblue", label="Engine Oil Pressure")
plt.bar(x + width/2, oil_bar_data["EngineOilTemperature"], width=width, color="darkorange", label="Engine Oil Temperature")

plt.xticks(x, oil_bar_data.index)
plt.ylabel("Average Value")
plt.title("Average Oil Pressure and Temperature by Derate Status", fontsize=14, fontweight="bold")
plt.legend(frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "pressure_temperature_bar_chart.png", dpi=200)
plt.close()

print("\nSaved chart: visuals/pressure_temperature_bar_chart.png")
