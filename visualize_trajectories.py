"""
Visualize Simulated Trajectories

Creates 3 plots (one per condition) showing trajectories from 5 random participants.
Each participant's trajectories are shown in a different color.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Load the simulated data
print("Loading simulated trajectories...")
data = np.load('simulated_trajectories.npy', allow_pickle=True).item()

# Extract metadata
metadata = data['metadata']
start_pos = metadata['start_pos']
target_pos = metadata['target_pos']

# Select 5 random participants
random.seed(42)  # For reproducibility
selected_participants = random.sample(range(100), 5)
print(f"Selected participants: {selected_participants}")

# Define colors for each participant
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create figure with 3 subplots (one per condition)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
conditions = ['full_control', 'medium_control', 'low_control']
titles = ['Full Control (100%)', 'Medium Control (50%)', 'Low Control (20%)']

for idx, (condition, title) in enumerate(zip(conditions, titles)):
    ax = axes[idx]
    
    # Plot trajectories for each selected participant
    for p_idx, participant_id in enumerate(selected_participants):
        # Get all trials for this participant in this condition
        participant_data = data[condition][participant_id]  # Shape: (40, 120, 2)
        
        # Plot each trial
        for trial_id in range(participant_data.shape[0]):
            trajectory = participant_data[trial_id]  # Shape: (120, 2)
            
            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=colors[p_idx], alpha=0.3, linewidth=0.8)
    
    # Plot start and target positions
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=15, 
           label='Start', zorder=10, markeredgecolor='darkgreen', markeredgewidth=2)
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=20, 
           label='Target', zorder=10, markeredgecolor='darkred', markeredgewidth=1.5)
    
    # Create custom legend for participants
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2, label=f'P{selected_participants[i]+1}')
        for i in range(5)
    ]
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
               markersize=10, label='Start', markeredgecolor='darkgreen', markeredgewidth=2),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='r', 
               markersize=12, label='Target', markeredgecolor='darkred', markeredgewidth=1.5)
    ])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (pixels)', fontsize=11)
    ax.set_ylabel('Y Position (pixels)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Invert y-axis (screen coordinates: y increases downward)
    ax.invert_yaxis()

# Overall title
fig.suptitle('Simulated Cursor Trajectories: 5 Participants × 40 Trials per Condition', 
             fontsize=16, fontweight='bold', y=0.98)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_filename = 'trajectory_visualization.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_filename}")

# Show the plot
plt.show()

print("\nVisualization complete!")
print(f"Each plot shows 40 trials × 5 participants = 200 trajectories")
