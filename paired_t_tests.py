import pandas as pd
from scipy.stats import ttest_rel

# Load your metrics table
df = pd.read_csv("scrub_vs_surgery_metrics.csv")

# Pivot to wide format: 1 row per participant, 2 columns per measure (scrub/surgery)
pivot = df.pivot(index='participant', columns='phase', values=['avg_fix_dur', 'avg_velocity'])

# Drop participants with missing data (must have both scrub + surgery)
pivot = pivot.dropna()

# Filter out participants with 0.0 scrub-in data (not meaningful)
valid_scrub = (
    (pivot['avg_fix_dur']['scrub'] > 0.0) &
    (pivot['avg_velocity']['scrub'] > 0.0)
)
pivot_filtered = pivot[valid_scrub]

# Optional: Print excluded participant IDs
excluded_ids = pivot[~valid_scrub].index.tolist()
print("Excluded participants due to 0 scrub-in data:", excluded_ids)

# Extract paired data
fix_dur_scrub = pivot_filtered['avg_fix_dur']['scrub']
fix_dur_surgery = pivot_filtered['avg_fix_dur']['surgery']
vel_scrub = pivot_filtered['avg_velocity']['scrub']
vel_surgery = pivot_filtered['avg_velocity']['surgery']

# Run paired t-tests
fix_dur_ttest = ttest_rel(fix_dur_scrub, fix_dur_surgery)
vel_ttest = ttest_rel(vel_scrub, vel_surgery)

# Format results
results = pd.DataFrame([
    {
        "measure": "avg_fix_dur",
        "t_stat": fix_dur_ttest.statistic,
        "p_value": fix_dur_ttest.pvalue,
        "n": len(fix_dur_scrub)
    },
    {
        "measure": "avg_velocity",
        "t_stat": vel_ttest.statistic,
        "p_value": vel_ttest.pvalue,
        "n": len(vel_scrub)
    }
])

# Save results to CSV
results.to_csv("paired_ttest_filtered.csv", index=False)
print("Saved paired t-test results to paired_ttest_filtered.csv")
