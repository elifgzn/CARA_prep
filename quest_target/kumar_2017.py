from psychopy import visual, core, event, data, gui
import numpy as np
import random
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
WINDOW_SIZE = (1024, 768)
START_POS = (0, 0)
TARGET_RADIUS = 20
CURSOR_RADIUS = 10
TARGET_ZONE_X = (-300, 300)
TARGET_ZONE_Y = (-200, 200)
MAX_TRIAL_DURATION = 30  # seconds
TRIALS_PER_CONDITION = 3
NOISE_SCALE = 1.0 # noise strength if 1, has no effect. 

# Control conditions
CONDITIONS = [
    {'control': 1.0, 'label': 'Full Control'},
    {'control': 0.25, 'label': '25% Control'},
    {'control': 0.5, 'label': '50% Control'},
    {'control': 0.75, 'label': '75% Control'}
]

# -----------------------------
# Helper Functions
# -----------------------------
def get_random_target_pos():
    x = random.uniform(*TARGET_ZONE_X)
    y = random.uniform(*TARGET_ZONE_Y)
    return (x, y)

def show_message(win, text, wait_for_key=True):
    stim = visual.TextStim(win, text, color="white", height=30, wrapWidth=900)
    stim.draw()
    win.flip()
    if wait_for_key:
        keys = event.waitKeys(keyList=["space", "escape"])
        if "escape" in keys:
            win.close()
            core.quit()
    else:
        core.wait(1.0)

def distance(pos1, pos2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# -----------------------------
# Main Task
# -----------------------------
def run_trial(win, mouse, condition, trial_num, trial_data):
    control = condition['control']
    label = condition['label']
    
    # Setup visuals
    target_pos = get_random_target_pos()
    target = visual.Circle(win, radius=TARGET_RADIUS, fillColor="yellow", pos=target_pos)
    cursor = visual.Circle(win, radius=CURSOR_RADIUS, fillColor="white", pos=START_POS)
    start_pos_visual = visual.Circle(win, radius=10, fillColor="grey", pos=START_POS)
    
    info_text = visual.TextStim(win, f"{label}\nTrial {trial_num}/{TRIALS_PER_CONDITION}", 
                               color="white", height=24, pos=(0, -320))
    
    # Reset mouse
    win.mouseVisible = False
    mouse.setPos(START_POS)
    core.wait(0.1)
    event.clearEvents()

    clock = core.Clock()

    cursor_pos = list(START_POS)  # Initialize as list [0, 0]
    last_mouse_pos = np.array(mouse.getPos())

    while True:
        # Get current mouse position
        current_mouse_pos = np.array(mouse.getPos())
        
        # Calculate input (i) as the change in mouse position
        i_x = (current_mouse_pos[0] - last_mouse_pos[0]) 
        i_y = (current_mouse_pos[1] - last_mouse_pos[1]) 
      
        # Generate random values centered around 0
    
        rx = (np.random.random() - 0.5) * 2 * NOISE_SCALE
        ry = (np.random.random() - 0.5) * 2 * NOISE_SCALE
        
        # Apply perturbation formula
        # dx = i + (1-control) * rx
        # dy = i + (1-control) * ry
        dx = i_x + (1 - control) * rx
        dy = i_y + (1 - control) * ry
        
        # Update cursor position
        cursor_pos[0] += dx
        cursor_pos[1] += dy
        
        # Keep cursor within window bounds
        window_width = win.size[0] / 2
        window_height = win.size[1] / 2
        cursor_pos[0] = np.clip(cursor_pos[0], -window_width, window_width)
        cursor_pos[1] = np.clip(cursor_pos[1], -window_height, window_height)
        
        # Update crosshair position
        cursor.pos = cursor_pos
        
        # Update last mouse position
        last_mouse_pos = current_mouse_pos.copy()
        
        # Check if cursor is on target
        dist_to_target = distance(cursor_pos, target_pos)
        on_target = dist_to_target < (TARGET_RADIUS + CURSOR_RADIUS)
        
        if on_target:
            hit_achieved = True
            target.fillColor = "green"
            target.draw()
            cursor.draw()
            win.flip()
            core.wait(0.3)
            break

        # Check for key presses
        keys = event.getKeys()
        if 'escape' in keys:
            win.close()
            core.quit()
        
        
        # Draw everything
        start_pos_visual.draw()
        target.draw()
        cursor.draw()
        info_text.draw()
        win.flip()
        
        # Small delay to control frame rate
        core.wait(0.01)
        
    

    # Feedback screen
    if hit_achieved:
        show_message(win, "Success!", wait_for_key=False)
    else:
        show_message(win, "Time's up! Try again.", wait_for_key=False)

    return hit_achieved

# -----------------------------
# Main Experiment
# -----------------------------
def run_experiment():
    # Setup window (fullscreen)
    win = visual.Window(
        size=WINDOW_SIZE,
        units='pix',
        fullscr=True,
        color='black'
    )
    
    mouse = event.Mouse(visible=False, win=win)
    
    # Welcome message
    show_message(win, 
        "Welcome to the Motor Control Experiment!\n\n"
        "Your task is to move the WHITE cursor\n"
        "to the YELLOW target.\n\n"
        "You'll experience different levels of control.\n\n"
        "Press SPACE to begin")
    
    # Prepare all trials
    all_trials = []
    for condition in CONDITIONS:
        for trial in range(TRIALS_PER_CONDITION):
            all_trials.append(condition.copy())
    
    # # Randomize trial order
    # random.shuffle(all_trials)
    
    # Store all data
    experiment_data = []
    
    # Run all trials
    for i, condition in enumerate(all_trials, 1):
        show_message(win, f"Trial {i} of {len(all_trials)}\n\n"
                         f"Condition: {condition['label']}\n\n"
                         f"Press SPACE when ready", wait_for_key=True)
        
        run_trial(win, mouse, condition, i, experiment_data)
    
    # End message
    show_message(win, "Experiment Complete!\n\nThank you for participating.\n\n"
                     "Press SPACE to exit")
    
    win.close()
    core.quit()

# -----------------------------
# Run the experiment
# -----------------------------
if __name__ == "__main__":
    run_experiment()