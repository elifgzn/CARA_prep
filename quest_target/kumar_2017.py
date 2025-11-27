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
NOISE_SCALE = 5.0 # noise strength

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
    
    # Get the initial mouse position
    mouse_pos = mouse.getPos()

    # Initialize previous position with initial position
    x_initial = mouse_pos[0]
    y_initial = mouse_pos[1]

    previous_x_actual = x_initial
    previous_y_actual = y_initial

    # Initialize transformed position
    x_transformed = x_initial
    y_transformed = y_initial   
    
    # Initialize trial variables
    clock = core.Clock()
    start_time = clock.getTime()
    hit_achieved = False
    completion_time = None
    trajectory = []

    while (clock.getTime() - start_time) < MAX_TRIAL_DURATION:

        current_time = clock.getTime()

        # Get the current mouse position
        mouse_pos = mouse.getPos()
        x_current = mouse_pos[0]
        y_current = mouse_pos[1]

        # Calculate deltas for the actual cursor
        mouse_dx_actual = x_current - previous_x_actual
        mouse_dy_actual = y_current - previous_y_actual

        # Check if the actual cursor moved
        if mouse_dx_actual != 0 or mouse_dy_actual != 0:
            
            # Calculate deltas for transformed cursor
            mouse_dx_transformed = mouse_dx_actual
            mouse_dy_transformed = mouse_dy_actual

            # Generate noise
            rx = (np.random.random() * 2 - 1) * NOISE_SCALE
            ry = (np.random.random() * 2 - 1) * NOISE_SCALE
            
            # Update transformed position with movement + noise
            x00 = x_transformed + mouse_dx_transformed + (1 - control) * rx
            y00 = y_transformed + mouse_dy_transformed + (1 - control) * ry

            x_transformed = x00
            y_transformed = y00

        # Check target hit
        dist_to_target = distance((x_transformed, y_transformed), target_pos)
        if dist_to_target < (TARGET_RADIUS + CURSOR_RADIUS):
            hit_achieved = True
            completion_time = current_time

            target.fillColor = "green"
            target.draw()
            cursor.draw()
            win.flip()
            core.wait(0.3)
            break

        # Draw all visuals
        cursor.pos = (x_transformed, y_transformed)

        start_pos_visual.draw()
        target.draw()
        cursor.draw()
        info_text.draw()
        win.flip()

        # Escape key
        if 'escape' in event.getKeys(['escape']):
            win.close()
            core.quit()
        
        # Update previous actual positions for next frame (CRITICAL FIX)
        previous_x_actual = x_current
        previous_y_actual = y_current

    # Trial result storage
    trial_result = {
        'condition': label,
        'control_level': control,
        'trial_num': trial_num,
        'target_x': target_pos[0],
        'target_y': target_pos[1],
        'hit': hit_achieved,
        'completion_time': completion_time if hit_achieved else None,
        'trajectory': trajectory
    }
    trial_data.append(trial_result)

    # Feedback screen
    if hit_achieved:
        show_message(win, f"Success!\nTime: {completion_time:.2f}s", wait_for_key=False)
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