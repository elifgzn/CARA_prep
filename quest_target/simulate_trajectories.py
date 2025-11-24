import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def min_jerk_trajectory(start, end, duration, num_points, via_point=None):
    """
    Generates a minimum jerk trajectory between start and end points.
    Optionally passes through a via_point to create curvature.
    """
    t = np.linspace(0, duration, num_points)
    
    if via_point is None:
        # Standard point-to-point minimum jerk
        # x(t) = xi + (xf - xi) * (10(t/d)^3 - 15(t/d)^4 + 6(t/d)^5)
        tau = t / duration
        poly = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        traj = np.zeros((num_points, 2))
        traj[:, 0] = start[0] + (end[0] - start[0]) * poly
        traj[:, 1] = start[1] + (end[1] - start[1]) * poly
        return traj
    else:
        # Multi-segment minimum jerk (simplified approximation for demo)
        # We just blend two trajectories: Start->Via and Via->End
        # This is a quick hack to get "curved" looking paths
        
        # Better approach: Add a perpendicular offset to the straight line
        # based on a smooth time function
        
        straight_traj = min_jerk_trajectory(start, end, duration, num_points)
        
        # Calculate perpendicular vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        perp = np.array([-dy, dx])
        perp = perp / (np.linalg.norm(perp) + 1e-9)
        
        # Add a "bump" in the middle
        # Bump shape: sin(pi * t/d)^2
        tau = t / duration
        bump = np.sin(np.pi * tau)**2
        
        # Offset magnitude (randomized curvature amount)
        offset_mag = via_point  # Treat via_point as scalar offset magnitude
        
        curved_traj = straight_traj.copy()
        curved_traj[:, 0] += perp[0] * bump * offset_mag
        curved_traj[:, 1] += perp[1] * bump * offset_mag
        
        return curved_traj

def generate_simulated_dataset(n_trajectories=200):
    start_pos = np.array([0, -300])
    target_pos = np.array([0, 250])
    
    print(f"Generating {n_trajectories} simulated trajectories...")
    
    velocities_list = []
    
    plt.figure(figsize=(6, 8))
    plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    plt.plot(target_pos[0], target_pos[1], 'rx', markersize=10, label='Target')
    
    for i in range(n_trajectories):
        # Randomize curvature:
        # Offset can be positive (curve right) or negative (curve left)
        # Magnitude between -200 and 200 pixels
        curvature = np.random.uniform(-200, 200)
        
        # Randomize duration slightly (speed variability)
        duration = np.random.uniform(0.8, 1.2)
        
        # Generate position trajectory
        # Use 61 points to get 60 velocity frames (standard for 60Hz)
        traj_pos = min_jerk_trajectory(start_pos, target_pos, duration, num_points=61, via_point=curvature)
        
        # Convert to velocities (diff)
        traj_vel = np.diff(traj_pos, axis=0)
        velocities_list.append(traj_vel)
        
        # Plot a subset for visualization
        if i < 20:
            plt.plot(traj_pos[:, 0], traj_pos[:, 1], 'b-', alpha=0.2)
            
    # Convert to numpy array (N, Frames, 2)
    simulated_pool = np.array(velocities_list)
    
    output_npy = Path(__file__).parent / "simulated_pool.npy"
    np.save(output_npy, simulated_pool)
    print(f"Saved dataset to {output_npy}")
    print(f"Shape: {simulated_pool.shape}")
    
    plt.title(f"Simulated Dataset Preview ({n_trajectories} items)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    output_img = Path(__file__).parent / "simulated_pool_preview.png"
    plt.savefig(output_img)
    print(f"Saved preview to {output_img}")

if __name__ == "__main__":
    generate_simulated_dataset(200)
