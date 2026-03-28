# Wind Turbine Power Prediction: Physics vs Machine Learning

## Overview
This project investigates wind turbine power generation using both a physics-based model and machine learning approaches. The goal is to compare predictive performance and understand how physical laws, data distribution, and model flexibility influence results.

---

## Physical Background

Wind power follows:

P ∝ v³  
P = k v³  

This assumes dependence only on wind speed and ignores operational constraints such as control systems and rated limits.

In practice, turbines operate in three regimes:

- Low-speed: negligible power (cut-in)  
- Intermediate: approximate cubic growth  
- High-speed: power saturation  

The gap between this ideal behavior and real data is central to the analysis.

---

## Dataset and Features

The dataset includes:

- Wind Speed (m/s)  
- Active Power (kW)  
- Wind Direction (°)  
- Theoretical Power Curve (kWh)  
- Timestamp  

Derived features:

- Hour  
- Month  
- Day of week  

---

## Modeling Strategy

Three models are used:

1. **Physics Model** – cubic relationship using wind speed  
2. **ML (Wind Speed Only)** – Random Forest using only wind speed  
3. **ML (Extended Features)** – Random Forest using all features  

---

## Results and Analysis

### Real Power Curve

The real power curve shows the expected increasing trend with saturation, but with significant scatter. Multiple power values appear at the same wind speed, indicating that wind speed alone is insufficient.  

This motivates the use of a theoretical curve as a structured reference.

01_real_power_curve.png

---
### Theoretical Power Curve

The theoretical curve represents ideal turbine behavior: cut-in, nonlinear growth, and saturation.  

It encodes physical constraints and aligns well with real data, which explains its importance in the ML model. Deviations highlight environmental and operational effects not captured by the ideal model.

02_theoretical_power_curve.png
---

### Wind Speed Distribution

Wind speeds are concentrated in the mid-range.  

This means:
- The model learns mostly from this region  
- Extreme conditions are underrepresented  
- Errors increase at high power levels  

03_wind_speed_distribution.png

---

### Physics Model vs Real Data

The physics model captures the general trend but deviates significantly:

- Overestimates power at high speeds  
- Fails to represent saturation properly  

This shows that a simple cubic model is fundamentally incomplete.  


04_physics_vs_real.png

---

### ML (Wind Speed Only): Predicted vs Actual

The model follows the general trend but shows noticeable spread, especially at high power.  

Because the same wind speed can produce different outputs, the model cannot uniquely determine power, leading to uncertainty.

05_ml_ws_predicted_vs_actual.png

---

### Residuals (Wind Speed Only Model)

Residuals are centered around zero, but error increases with predicted power.  

This is due to missing variables such as turbine control and operational effects, making the model less reliable at higher power levels.

06_ml_ws_residuals.png

---

### ML Prediction Curve (Wind Speed Only)

The predicted curve is noisy and irregular.  

This reflects the non-unique relationship between wind speed and power, where the model tries to fit noisy data without enough information.

07_ml_ws_vs_real.png

---

### ML (Extended Features): Predicted vs Actual

Predictions closely follow the ideal line with reduced spread.  

Including additional features improves accuracy by capturing variations beyond wind speed.

08_ml_extended_predicted_vs_actual.png
---

### Residuals (Extended Features Model)

Residuals remain centered around zero with smaller spread.  

Some large errors persist, especially at high power, likely due to missing operational variables.

09_ml_extended_residuals.png

---

### Residuals vs Wind Speed (Extended Model)

Error is larger in the mid-range, where power is most sensitive to changes.  

At low and high speeds, behavior is more stable, leading to smaller errors. Remaining outliers indicate unmodeled effects.

10_residuals_vs_wind_speed.png

---

### Model Comparison (MAE & RMSE)

The physics model has the highest error.  

The ML (wind speed only) model improves significantly, while the extended model achieves the lowest error, showing the importance of both model flexibility and additional features.
11_model_comparison_errors.png

---

### Model Comparison (R²)

The physics model has low explanatory power.  

Both ML models achieve high R², with further improvement when using extended features.

12_model_comparison_r2.png

---

### Feature Importance

- Wind speed is the dominant factor  
- Theoretical power curve is highly influential  
- Other features have minor impact  

This shows the model mainly learns physical relationships, with additional features refining predictions.

13_feature_importance.png

---

## Key Insights

- A cubic model alone cannot describe turbine behavior  
- Machine learning captures nonlinear and real-world effects  
- Data distribution affects model reliability  
- Physical features dominate predictive performance  
- Residuals reveal hidden system behavior  

---

## Conclusion

Machine learning outperforms the physics model by adapting to real-world complexity and incorporating additional information.  

Even with the same input, ML improves predictions by learning deviations from ideal physics.  

The best approach combines physical understanding with data-driven methods.

---

## Future Work

- Develop a piecewise physics model  
- Build hybrid physics + ML models  
- Model abnormal turbine states  
- Include more environmental variables  
