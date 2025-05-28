import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices

# Load your data
df = pd.read_csv("scrub_vs_surgery_metrics.csv")

# Filter to surgery phase only
df_surgery = df[df['phase'] == 'surgery']

# Rename columns if needed (optional, but simplifies formula syntax)
df_surgery = df_surgery.rename(columns={
    'avg_fix_dur': 'fix_dur',
    'avg_velocity': 'velocity',
    'experience': 'experience'
})

# Run two-way ANOVA with interaction
model = smf.ols("experience ~ fix_dur * velocity", data=df_surgery).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# Show results
print(anova_table)

# Optionally save to CSV
anova_table.to_csv("anova_results.csv")
print("Saved ANOVA results to anova_results.csv")
