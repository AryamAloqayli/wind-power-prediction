import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

save_dir = "2figures"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv("turbines.csv", encoding="latin1")

df.columns = df.columns.str.strip()
df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")

df = df.dropna(subset=[
    "Date/Time",
    "Wind Speed (m/s)",
    "LV ActivePower (kW)",
    "Wind Direction (°)",
    "Theoretical_Power_Curve (KWh)"
])

df = df[
    (df["Wind Speed (m/s)"] >= 0) &
    (df["LV ActivePower (kW)"] >= 0)
].copy()

df["hour"] = df["Date/Time"].dt.hour
df["month"] = df["Date/Time"].dt.month
df["dayofweek"] = df["Date/Time"].dt.dayofweek

print(df.head())
print(df.info())
print(df.isnull().sum())

wind_speed = df["Wind Speed (m/s)"].values
active_power = df["LV ActivePower (kW)"].values
wind_direction = df["Wind Direction (°)"].values
theoretical_power = df["Theoretical_Power_Curve (KWh)"].values

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(wind_speed, active_power, alpha=0.2)
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("LV Active Power (kW)")
ax.set_title("Real Wind Turbine Power Curve")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "01_real_power_curve.png"), dpi=300)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(wind_speed, theoretical_power, alpha=0.2)
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Theoretical Power Curve (kWh)")
ax.set_title("Theoretical Power Curve vs Wind Speed")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "02_theoretical_power_curve.png"), dpi=300)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(wind_speed, bins=40)
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Wind Speed")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "03_wind_speed_distribution.png"), dpi=300)
plt.show()
plt.close(fig)

phys_mask = (wind_speed > 0) & (active_power > 0)

v_phys = wind_speed[phys_mask]
P_phys = active_power[phys_mask]

k_est = np.sum(P_phys * v_phys**3) / np.sum(v_phys**6)
print("Estimated k =", k_est)

P_pred_physics = k_est * v_phys**3
P_max = np.max(active_power)
P_pred_physics_clipped = np.clip(P_pred_physics, 0, P_max)

sort_idx = np.argsort(v_phys)
v_sorted = v_phys[sort_idx]
P_phys_sorted = P_phys[sort_idx]
P_pred_sorted = P_pred_physics_clipped[sort_idx]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(v_phys, P_phys, alpha=0.2, label="Real Data")
ax.plot(v_sorted, P_pred_sorted, color="red", linewidth=3, label="Physics Model")
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Power (kW)")
ax.set_title("Physics Model vs Real Data")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "04_physics_vs_real.png"), dpi=300)
plt.show()
plt.close(fig)

mae_phys = mean_absolute_error(P_phys, P_pred_physics_clipped)
rmse_phys = np.sqrt(mean_squared_error(P_phys, P_pred_physics_clipped))
r2_phys = r2_score(P_phys, P_pred_physics_clipped)

print("Physics Model Performance")
print("MAE =", mae_phys)
print("RMSE =", rmse_phys)
print("R2 =", r2_phys)

df_ml = df[df["LV ActivePower (kW)"] > 0].copy()

target_col = "LV ActivePower (kW)"
y = df_ml[target_col]

X_ws = df_ml[["Wind Speed (m/s)"]]

X_ws_train, X_ws_test, y_ws_train, y_ws_test = train_test_split(
    X_ws, y, test_size=0.2, random_state=42
)

rf_ws = RandomForestRegressor(
    n_estimators=400,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features=1,
    random_state=42,
    n_jobs=-1
)

rf_ws.fit(X_ws_train, y_ws_train)
y_pred_ws = rf_ws.predict(X_ws_test)

mae_ws = mean_absolute_error(y_ws_test, y_pred_ws)
rmse_ws = np.sqrt(mean_squared_error(y_ws_test, y_pred_ws))
r2_ws = r2_score(y_ws_test, y_pred_ws)

print("Machine Learning Model Performance (Wind Speed Only)")
print("MAE =", mae_ws)
print("RMSE =", rmse_ws)
print("R2 =", r2_ws)

max_val_ws = max(y_ws_test.max(), y_pred_ws.max())

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_ws_test, y_pred_ws, alpha=0.3, label="Predictions")
ax.plot([0, max_val_ws], [0, max_val_ws], "r--", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Power (kW)")
ax.set_ylabel("Predicted Power (kW)")
ax.set_title("ML Model (Wind Speed Only): Predicted vs Actual")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "05_ml_ws_predicted_vs_actual.png"), dpi=300)
plt.show()
plt.close(fig)

residuals_ws = y_ws_test - y_pred_ws

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred_ws, residuals_ws, alpha=0.3)
ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Predicted Power (kW)")
ax.set_ylabel("Residuals")
ax.set_title("ML Model (Wind Speed Only) Residuals")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "06_ml_ws_residuals.png"), dpi=300)
plt.show()
plt.close(fig)

ws_test_plot = X_ws_test.copy()
ws_test_plot["Actual"] = y_ws_test.values
ws_test_plot["Predicted"] = y_pred_ws
ws_test_plot = ws_test_plot.sort_values("Wind Speed (m/s)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(v_phys, P_phys, alpha=0.15, label="Real Data")
ax.plot(ws_test_plot["Wind Speed (m/s)"], ws_test_plot["Predicted"], color="orange", linewidth=2, label="ML Wind Speed Only")
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Power (kW)")
ax.set_title("ML (Wind Speed Only) vs Real Data")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "07_ml_ws_vs_real.png"), dpi=300)
plt.show()
plt.close(fig)

feature_cols = [
    "Wind Speed (m/s)",
    "Wind Direction (°)",
    "Theoretical_Power_Curve (KWh)",
    "hour",
    "month",
    "dayofweek"
]

X_ext = df_ml[feature_cols]

X_ext_train, X_ext_test, y_ext_train, y_ext_test = train_test_split(
    X_ext, y, test_size=0.2, random_state=42
)

rf_ext = RandomForestRegressor(
    n_estimators=400,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_ext.fit(X_ext_train, y_ext_train)
y_pred_ext = rf_ext.predict(X_ext_test)

mae_ext = mean_absolute_error(y_ext_test, y_pred_ext)
rmse_ext = np.sqrt(mean_squared_error(y_ext_test, y_pred_ext))
r2_ext = r2_score(y_ext_test, y_pred_ext)

print("Machine Learning Model Performance (Extended Features)")
print("MAE =", mae_ext)
print("RMSE =", rmse_ext)
print("R2 =", r2_ext)

max_val_ext = max(y_ext_test.max(), y_pred_ext.max())

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_ext_test, y_pred_ext, alpha=0.3, label="Predictions")
ax.plot([0, max_val_ext], [0, max_val_ext], "r--", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Power (kW)")
ax.set_ylabel("Predicted Power (kW)")
ax.set_title("ML Model (Extended Features): Predicted vs Actual")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "08_ml_extended_predicted_vs_actual.png"), dpi=300)
plt.show()
plt.close(fig)

residuals_ext = y_ext_test - y_pred_ext

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred_ext, residuals_ext, alpha=0.3)
ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Predicted Power (kW)")
ax.set_ylabel("Residuals")
ax.set_title("ML Model (Extended Features) Residuals")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "09_ml_extended_residuals.png"), dpi=300)
plt.show()
plt.close(fig)

wind_test_ext = X_ext_test["Wind Speed (m/s)"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(wind_test_ext, residuals_ext, alpha=0.3)
ax.axhline(0, linestyle="--", color="black")
ax.set_xlabel("Wind Speed (m/s)")
ax.set_ylabel("Residuals")
ax.set_title("Extended ML Residuals vs Wind Speed")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "10_residuals_vs_wind_speed.png"), dpi=300)
plt.show()
plt.close(fig)

comparison = pd.DataFrame({
    "Model": [
        "Physics Model",
        "ML (Wind Speed Only)",
        "ML (Extended Features)"
    ],
    "MAE": [mae_phys, mae_ws, mae_ext],
    "RMSE": [rmse_phys, rmse_ws, rmse_ext],
    "R2": [r2_phys, r2_ws, r2_ext]
})

print(comparison)

fig, ax = plt.subplots(figsize=(9, 5))
comparison.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", ax=ax)
ax.set_title("Model Comparison: MAE and RMSE")
ax.set_ylabel("Error")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "11_model_comparison_errors.png"), dpi=300)
plt.show()
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(comparison["Model"], comparison["R2"])
ax.set_title("Model Comparison: R2")
ax.set_ylabel("R2")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "12_model_comparison_r2.png"), dpi=300)
plt.show()
plt.close(fig)

feature_importance = pd.Series(
    rf_ext.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print(feature_importance)

fig, ax = plt.subplots(figsize=(8, 5))
feature_importance.plot(kind="bar", ax=ax)
ax.set_ylabel("Importance")
ax.set_title("Extended ML Feature Importance")
ax.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, "13_feature_importance.png"), dpi=300)
plt.show()
plt.close(fig)

comparison.to_csv(os.path.join(save_dir, "model_comparison.csv"), index=False)
feature_importance.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=True)

print("All figures and tables saved in:", save_dir)