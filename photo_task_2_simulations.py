"""
Trajectory Simulation Script - Photo Task 2

Simulates 100 virtual participants completing the cursor manipulation task.
Each participant completes 40 trials per condition (full/medium/low control).
Uses minimum jerk trajectories with Perlin noise perturbations.
Perlin noise applied additively!

Output: simulated_trajectories.npy
"""

import numpy as np
import math
import random

# ============================================================================
# PERLIN NOISE IMPLEMENTATION
# ============================================================================
# Copied from photo_task_2.py for consistency

class PerlinNoise:
    def __init__(self, seed=0):
        
        # these are the directions used to generate the noise field
        # more correctly, these are the gradient vectors with a fixed magnitude and direction
        self.grad3 = [
            (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
            (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
            (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)
        ]
        
        # these provide a fixed permutation to create repeateable randomness
        self.p = [
            151,160,137,91,90,15,
            131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
            88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
            77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
            102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
            135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
            5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
            223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
            129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
            251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
            49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        ]
        
        # these are used to store the permutation and gradient values
        self.perm = [0] * 512
        self.gradP = [None] * 512
        
        self.seed(seed)

    # this function sets the seed for the random number generator. 
    # If the seed is fixed we can get the exact kind of randomness (repeateable across participants)
    # what is (pseudo)random is, which vector is used at which point in the grid
    # so here, we determine all future noise
    def seed(self, seed):
        if 0 < seed < 1:
            seed *= 65536
        
        seed = int(math.floor(seed))
        if seed < 256:
            seed |= seed << 8
            
        for i in range(256):
            if i & 1:
                v = self.p[i] ^ (seed & 255)
            else:
                v = self.p[i] ^ ((seed >> 8) & 255)
            
            self.perm[i] = self.perm[i + 256] = v
            self.gradP[i] = self.gradP[i + 256] = self.grad3[v % 12]
            # here, the v comes from the permutation table pseudorandomly based on the seed
            # v%12 gives a number 0-11, selecting one of the 12 gradient vectors (identical in magnitude but can be different in direction)

    def dot2(self, g, x, y):
        return g[0]*x + g[1]*y

    def fade(self, t):
        return t*t*t*(t*(t*6-15)+10)

    def lerp(self, a, b, t):
        return (1-t)*a + t*b

    def perlin2(self, x, y):
        # determine which grid cell the point is in
        X = int(math.floor(x))
        Y = int(math.floor(y))
        
        # Get relative xy coordinates of point within that cell
        x = x - X
        y = y - Y
        
        # Wrap the integer cells at 255
        X = X & 255
        Y = Y & 255
        
        # Calculate noise contributions from each of the four corners
        n00 = self.dot2(self.gradP[X + self.perm[Y]], x, y)
        n01 = self.dot2(self.gradP[X + self.perm[Y + 1]], x, y - 1)
        n10 = self.dot2(self.gradP[X + 1 + self.perm[Y]], x - 1, y)
        n11 = self.dot2(self.gradP[X + 1 + self.perm[Y + 1]], x - 1, y - 1)
        
        # Compute the fade curve value for x
        u = self.fade(x)
        
        # Interpolate the four results
        return self.lerp(
            self.lerp(n00, n10, u),
            self.lerp(n01, n11, u),
            self.fade(y)
        )


# ============================================================================
# MINIMUM JERK TRAJECTORY GENERATOR
# ============================================================================

def minimum_jerk_trajectory(start_pos, target_pos, t, duration):
    """
    Generate human-like reaching trajectory using minimum jerk model.
    
    The minimum jerk model produces smooth, natural movements by minimizing
    the rate of change of acceleration (jerk). This is a well-established
    model of human reaching movements.
    
    Args:
        start_pos: (x, y) starting position
        target_pos: (x, y) target position
        t: current time (0 to duration)
        duration: total movement duration in seconds
    
    Returns:
        (x, y) position at time t
    """
    # Normalize time to range [0, 1]
    tau = t / duration
    
    # Minimum jerk trajectory uses 5th order polynomial
    # s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
    # This ensures smooth start (s=0, ds/dt=0, d²s/dt²=0 at τ=0)
    # and smooth end (s=1, ds/dt=0, d²s/dt²=0 at τ=1)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    
    # Interpolate position between start and target
    x = start_pos[0] + s * (target_pos[0] - start_pos[0])
    y = start_pos[1] + s * (target_pos[1] - start_pos[1])
    
    return (x, y)


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Experimental design
N_PARTICIPANTS = 100
TRIALS_PER_CONDITION = 40

# Conditions (matching photo_task_2.py)
conditions = [
    {'control': 1.0, 'label': 'full_control'},
    {'control': 0.5, 'label': 'medium_control'},
    {'control': 0.2, 'label': 'low_control'}
]

# Noise parameters (matching photo_task_2.py)
NOISE_SCALE = 40.0


# Timing parameters
FRAME_RATE = 60  # Hz (matches pygame clock.tick(60))
MOVEMENT_DURATION = 2.0  # seconds per trial
N_FRAMES = int(FRAME_RATE * MOVEMENT_DURATION)  # 120 frames per trial

# Screen dimensions (matching photo_task_2.py - typical fullscreen)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Start and target positions (matching photo_task_2.py)
START_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT * 5 // 6)  # Lower 1/3, centered
TARGET_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 6)     # Upper 1/3, centered


# ============================================================================
# SIMULATION FUNCTION
# ============================================================================

def simulate_trial(control_level, perlin, noise_t_offset=0.0):
    """
    Simulate a single trial with given control level.
    
    Args:
        control_level: 1.0 (full), 0.5 (medium), or 0.2 (low)
        perlin: PerlinNoise instance
        noise_t_offset: time offset for Perlin noise (for variation)
    
    Returns:
        trajectory: numpy array of shape (N_FRAMES, 2) with [x, y] positions
    """
    trajectory = np.zeros((N_FRAMES, 2))
    
    # Initialize positions
    # Track both intended (ideal) and actual (perturbed) positions
    actual_x, actual_y = START_POS
    prev_ideal_x, prev_ideal_y = START_POS
    trajectory[0] = [actual_x, actual_y]
    
    # Noise time variable
    noise_t = noise_t_offset
    
    # Simulate each frame
    for frame in range(1, N_FRAMES):
        # Current time in seconds
        t = frame / FRAME_RATE
        
        # Calculate ideal position using minimum jerk (what the participant intends)
        ideal_x, ideal_y = minimum_jerk_trajectory(START_POS, TARGET_POS, t, MOVEMENT_DURATION)
        
        # Calculate intended movement based on ideal trajectory
        # This is the key fix: movement is based on the ideal path, not the perturbed position
        mouse_dx = ideal_x - prev_ideal_x
        mouse_dy = ideal_y - prev_ideal_y
        
        # Only apply perturbation if there's movement
        if mouse_dx != 0 or mouse_dy != 0:
            control = control_level
            # Advance noise time
            noise_t += 0.05
            
            # Generate Perlin noise values
            # Use different y-offsets for independent x and y noise
            noise_val_x = perlin.perlin2(noise_t, 0)
            noise_val_y = perlin.perlin2(noise_t, 100)
            
            # Apply perturbation formula: dx = input + (1 - control) * noise * scale
            perturbed_dx = mouse_dx + (1 - control) * noise_val_x * NOISE_SCALE
            perturbed_dy = mouse_dy + (1 - control) * noise_val_y * NOISE_SCALE

            # ---- NEW CORRECTION FORCE so that the simulated cursor also goes to the target position----
            kx = 0.10 * (ideal_x - actual_x)
            ky = 0.10 * (ideal_y - actual_y)

            perturbed_dx += kx
            perturbed_dy += ky
            
            # Update actual position with perturbed movement
            actual_x += perturbed_dx
            actual_y += perturbed_dy
        
        # Update previous ideal position for next iteration
        prev_ideal_x = ideal_x
        prev_ideal_y = ideal_y

        # Force endpoint to exact target
        if frame == N_FRAMES - 1:
            actual_x, actual_y = TARGET_POS

        # Store actual (perturbed) position
        trajectory[frame] = [actual_x, actual_y]
    
    return trajectory


# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

def run_simulation():
    """
    Run the full simulation for all participants and conditions.
    
    Returns:
        data: dictionary containing all trajectories and metadata
    """
    print("Starting trajectory simulation...")
    print(f"Simulating {N_PARTICIPANTS} participants")
    print(f"{TRIALS_PER_CONDITION} trials per condition")
    print(f"{len(conditions)} conditions")
    print(f"Total trials: {N_PARTICIPANTS * TRIALS_PER_CONDITION * len(conditions)}")
    print()
    
    # Initialize data structure
    # Shape: [participant, trial, frame, coordinate]
    data = {
        'full_control': np.zeros((N_PARTICIPANTS, TRIALS_PER_CONDITION, N_FRAMES, 2)),
        'medium_control': np.zeros((N_PARTICIPANTS, TRIALS_PER_CONDITION, N_FRAMES, 2)),
        'low_control': np.zeros((N_PARTICIPANTS, TRIALS_PER_CONDITION, N_FRAMES, 2)),
        'metadata': {
            'n_participants': N_PARTICIPANTS,
            'trials_per_condition': TRIALS_PER_CONDITION,
            'frame_rate': FRAME_RATE,
            'movement_duration': MOVEMENT_DURATION,
            'n_frames': N_FRAMES,
            'start_pos': START_POS,
            'target_pos': TARGET_POS,
            'noise_scale': NOISE_SCALE,
            'screen_width': SCREEN_WIDTH,
            'screen_height': SCREEN_HEIGHT,
            'conditions': conditions
        }
    }
    
    # Simulate each participant
    for participant_id in range(N_PARTICIPANTS):
        if (participant_id + 1) % 10 == 0:
            print(f"Simulating participant {participant_id + 1}/{N_PARTICIPANTS}...")
        
        # Each participant gets a unique base seed
        participant_seed = participant_id * 1000
        
        # Simulate each condition
        for condition in conditions:
            control_level = condition['control']
            label = condition['label']
            
            # Simulate each trial
            for trial_id in range(TRIALS_PER_CONDITION):
                # Create unique seed for this trial
                trial_seed = participant_seed + trial_id
                
                # Initialize Perlin noise with unique seed
                perlin = PerlinNoise(seed=trial_seed)
                
                # Add random offset to noise time for variation
                noise_t_offset = random.uniform(0, 100)
                
                # Simulate the trial
                trajectory = simulate_trial(control_level, perlin, noise_t_offset)
                
                # Store trajectory
                data[label][participant_id, trial_id] = trajectory
    
    print("\nSimulation complete!")
    return data


# ============================================================================
# SAVE DATA
# ============================================================================

def save_trajectories(data, filename='simulated_trajectories.npy'):
    """
    Save simulated trajectories to .npy file.
    
    Args:
        data: dictionary containing trajectories and metadata
        filename: output filename
    """
    print(f"\nSaving trajectories to {filename}...")
    np.save(filename, data, allow_pickle=True)
    print(f"Saved successfully!")
    
    # Print summary
    print("\nData structure:")
    print(f"  full_control: {data['full_control'].shape}")
    print(f"  medium_control: {data['medium_control'].shape}")
    print(f"  low_control: {data['low_control'].shape}")
    print(f"  Shape format: (participants, trials, frames, coordinates)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run simulation
    data = run_simulation()
    
    # Save results
    save_trajectories(data)
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)
    print("\nYou can now analyze the trajectories:")
    print("  - Compare conditions (full vs medium vs low control)")
    print("  - Calculate deviation from full control baseline")
    print("  - Analyze individual participant variability")
    print("  - Examine trial-by-trial patterns")
