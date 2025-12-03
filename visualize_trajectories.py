import numpy as np
import matplotlib.pyplot as plt
import random

# Load simulated data
print("Loading simulated trajectories...")
data = np.load('simulated_trajectories.npy', allow_pickle=True).item()

metadata = data['metadata']
start_pos = metadata['start_pos']
target_pos = metadata['target_pos']

# Conditions & titles
conditions = ["full_control", "medium_control", "low_control"]
condition_titles = ["Full Control (100%)", "Medium Control (50%)", "Low Control (20%)"]

# Trial indices to visualize
trial_numbers = [9, 19, 39]   # (10th, 20th, 40th) zero-indexed
trial_titles = ["Trial 10", "Trial 20", "Trial 40"]

# Randomly pick 5 participants
random.seed(18)
selected_participants = random.sample(range(100), 10)
print(f"Selected participants: {selected_participants}")

# Colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create a 3 × 3 grid
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row_idx, trial_idx in enumerate(trial_numbers):
    for col_idx, condition in enumerate(conditions):

        ax = axes[row_idx, col_idx]

        # Plot 5 participants' single-trial trajectories
        for p_idx, participant_id in enumerate(selected_participants):
            # trajectory: (120, 2)
            trajectory = data[condition][participant_id][trial_idx]

            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=colors[p_idx],
                linewidth=1.5,
                alpha=0.9,
                label=f"P{participant_id+1}" if row_idx == 0 else None  # legend only in top row
            )

        # Markers
        ax.plot(start_pos[0], start_pos[1], 'go', markersize=10)
        ax.plot(target_pos[0], target_pos[1], 'r*', markersize=14)

        # Titles
        if row_idx == 0:
            ax.set_title(condition_titles[col_idx], fontsize=13, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(trial_titles[row_idx], fontsize=13, fontweight='bold')

        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

# Create legend outside plot
fig.legend(
    [plt.Line2D([0], [0], color=colors[i], lw=3) for i in range(5)],
    [f"P{selected_participants[i]+1}" for i in range(5)],
    loc='upper center',
    ncol=5,
    fontsize=12
)

fig.suptitle("Single-Trial Trajectories Across Conditions", fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("trajectory_grid_3x3.png", dpi=300)
plt.show()
