"""
Simulation Analyses - Photo Task 2

Calculates deviation metrics for simulated trajectories.
Compares medium and low control conditions against full control baseline.

Metrics:
- Mean deviation (pixels) per participant per condition
- Standard deviation per participant per condition

Output: Formatted table showing results
"""

import numpy as np
import pandas as pd
import pingouin as pg

# Load simulated data
print("Loading simulated trajectories...")
data = np.load('simulated_trajectories.npy', allow_pickle=True).item()

# Extract trajectory data
full_control = data['full_control']      # Shape: (100 participants, 40 trials, 120 frames, 2 coords)
medium_control = data['medium_control']
low_control = data['low_control']

metadata = data['metadata']
n_participants = metadata['n_participants']
n_trials = metadata['trials_per_condition']
n_frames = metadata['n_frames']

print(f"Analyzing {n_participants} participants, {n_trials} trials per condition, {n_frames} frames per trial")
print()

# ============================================================================
# CALCULATE DEVIATIONS
# ============================================================================

def calculate_deviation_from_baseline(perturbed_trajectories, baseline_trajectories):
    """
    Calculate Euclidean deviation between perturbed and baseline trajectories.
    
    Args:
        perturbed_trajectories: (n_participants, n_trials, n_frames, 2)
        baseline_trajectories: (n_participants, n_trials, n_frames, 2)
    
    Returns:
        deviations: (n_participants, n_trials, n_frames) - Euclidean distance per frame
    """
    # Calculate Euclidean distance at each frame
    diff = perturbed_trajectories - baseline_trajectories  # (participants, trials, frames, 2)
    euclidean_dist = np.sqrt(np.sum(diff**2, axis=-1))     # (participants, trials, frames)
    
    return euclidean_dist


# Calculate deviations for medium and low control
print("Calculating deviations from full control baseline...")
medium_deviations = calculate_deviation_from_baseline(medium_control, full_control)
low_deviations = calculate_deviation_from_baseline(low_control, full_control)

# ============================================================================
# AGGREGATE METRICS PER PARTICIPANT
# ============================================================================

# For each participant, calculate mean and SD across all trials and frames
medium_mean_per_participant = medium_deviations.reshape(n_participants, -1).mean(axis=1)
medium_sd_per_participant = medium_deviations.reshape(n_participants, -1).std(axis=1)

low_mean_per_participant = low_deviations.reshape(n_participants, -1).mean(axis=1)
low_sd_per_participant = low_deviations.reshape(n_participants, -1).std(axis=1)

# ============================================================================
# CREATE RESULTS TABLE
# ============================================================================

# Build DataFrame
results = pd.DataFrame({
    'Participant': range(1, n_participants + 1),
    'Medium_Mean_Deviation_px': medium_mean_per_participant,
    'Medium_SD_px': medium_sd_per_participant,
    'Low_Mean_Deviation_px': low_mean_per_participant,
    'Low_SD_px': low_sd_per_participant
})

# Round to 2 decimal places for readability
results = results.round(2)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("="*80)
print("DEVIATION METRICS: Medium and Low Control vs Full Control Baseline")
print("="*80)
print()
print(results.to_string(index=False))
print()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("="*80)
print("SUMMARY STATISTICS (across all participants)")
print("="*80)
print()

summary = pd.DataFrame({
    'Condition': ['Medium Control', 'Low Control'],
    'Mean_Deviation_px': [
        medium_mean_per_participant.mean(),
        low_mean_per_participant.mean()
    ],
    'SD_Deviation_px': [
        medium_mean_per_participant.std(),
        low_mean_per_participant.std()
    ],
    'Min_Deviation_px': [
        medium_mean_per_participant.min(),
        low_mean_per_participant.min()
    ],
    'Max_Deviation_px': [
        medium_mean_per_participant.max(),
        low_mean_per_participant.max()
    ]
})

summary = summary.round(2)
print(summary.to_string(index=False))
print()

# ============================================================================
# STATISTICAL TESTING
# ============================================================================

print("="*80)
print("STATISTICAL TEST: Paired t-test (Medium vs Low Control)")
print("="*80)
print()

# Perform paired t-test using pingouin
# H0: Mean deviation in medium control = Mean deviation in low control
# H1: Mean deviation in medium control ≠ Mean deviation in low control
ttest_results = pg.ttest(medium_mean_per_participant, low_mean_per_participant, paired=True)

print(f"Paired t-test comparing mean deviations:")
print(f"  Medium Control: M = {medium_mean_per_participant.mean():.2f} px, SD = {medium_mean_per_participant.std():.2f} px")
print(f"  Low Control:    M = {low_mean_per_participant.mean():.2f} px, SD = {low_mean_per_participant.std():.2f} px")
print()
print(f"  t-statistic: {ttest_results['T'].values[0]:.4f}")
print(f"  p-value:     {ttest_results['p-val'].values[0]:.4e}")
print(f"  df:          {int(ttest_results['dof'].values[0])}")
print(f"  Cohen's d:   {ttest_results['cohen-d'].values[0]:.4f}")
print(f"  95% CI:      [{ttest_results['CI95%'].values[0][0]:.2f}, {ttest_results['CI95%'].values[0][1]:.2f}]")
print()

# Interpretation
alpha = 0.05
p_value = ttest_results['p-val'].values[0]
cohens_d = ttest_results['cohen-d'].values[0]

if p_value < alpha:
    print(f"  Result: SIGNIFICANT (p < {alpha})")
    print(f"  The mean deviation differs significantly between conditions.")
else:
    print(f"  Result: NOT SIGNIFICANT (p >= {alpha})")
    print(f"  No significant difference in mean deviation between conditions.")

print()

# Effect size interpretation
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"  Effect size interpretation: {effect_interpretation} effect")
print()

print("="*80)
print("Analysis complete!")
print("="*80)

