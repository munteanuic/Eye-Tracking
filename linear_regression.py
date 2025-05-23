import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("scrub_vs_surgery_metrics.csv")

# Filter for surgery phase only
df_surgery = df[df["phase"] == "surgery"].copy()

# Min-max normalization (range [0, 1])
df_surgery["fix_dur_norm"] = (df_surgery["avg_fix_dur"] - df_surgery["avg_fix_dur"].min()) / (df_surgery["avg_fix_dur"].max() - df_surgery["avg_fix_dur"].min())
df_surgery["velocity_norm"] = (df_surgery["avg_velocity"] - df_surgery["avg_velocity"].min()) / (df_surgery["avg_velocity"].max() - df_surgery["avg_velocity"].min())

# Fit linear model with interaction
model = smf.ols("experience ~ fix_dur_norm * velocity_norm", data=df_surgery).fit()

# Save regression summary to file
with open("minmax_regression_summary.txt", "w") as f:
    f.write(model.summary().as_text())

# Predicted vs Actual plot
df_surgery["predicted_experience"] = model.predict(df_surgery)

plt.figure(figsize=(8, 6))
plt.scatter(df_surgery["experience"], df_surgery["predicted_experience"], label="Predicted")
plt.plot([df_surgery["experience"].min(), df_surgery["experience"].max()],
         [df_surgery["experience"].min(), df_surgery["experience"].max()],
         color='red', linestyle='--', label="Ideal")
plt.xlabel("Actual Experience")
plt.ylabel("Predicted Experience")
plt.title("Actual vs Predicted Surgeon Experience (Min-Max Normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_minmax.png")
plt.show()
