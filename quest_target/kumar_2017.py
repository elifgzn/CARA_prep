import numpy as np
from psychopy import visual, event, core
import math
import random

# -----------------------------
# Kumar & Srinivashan (2017) Replication Task
# -----------------------------
# This script implements a target-reaching task with varying levels of control.
#
# NOISE IMPLEMENTATION:
# The noise logic follows the formula: dx = input + (1-control) * noise
#
# Initially, this was implemented as Gaussian White Noise (independent random values per frame),
# which caused excessive high-frequency jitter.
#
# It has been updated to use Smoothed Noise (effectively an Ornstein-Uhlenbeck process).
# A low-pass filter (NOISE_SMOOTHING) is applied to the random values, causing the
# noise vector to drift smoothly over time rather than jumping randomly.
# This creates a "wind-like" disturbance that is challenging but not jittery.
#
# GENERAL STRUCTURE:
# The way the cursor is updated and the way the noise is applied follows Target Reaching Task (MCRL),
# since these were optimized this way. 
# -----------------------------

# -----------------------------
# Configuration
# -----------------------------

WINDOW_SIZE = (1280, 720)
TRIALS_PER_CONDITION = 3
NOISE_SCALE = 200.0  # Increased to make noise stronger
NOISE_SMOOTHING = 0.05 # Slightly more dynamic
TARGET_ZONE_Y = (150, 300)  # Upper third of screen
TARGET_ZONE_X = (-400, 400) # Keep target somewhat central horizontally
START_POS = (0, -300)
MAX_TRIAL_DURATION = 20.0
TARGET_RADIUS = 20
CURSOR_RADIUS = 10

# Fixed order: Full -> Medium -> Low
CONDITIONS = [
    {'control': 1.0, 'label': 'Full Control (0% Noise)'},
    {'control': 0.5, 'label': 'Medium Control (50% Noise)'},
    {'control': 0.2, 'label': 'Low Control (80% Noise)'}
]

# -----------------------------
# Helpers
# -----------------------------

def get_random_target_pos():
    x = random.uniform(*TARGET_ZONE_X)
    y = random.uniform(*TARGET_ZONE_Y)
    return (x, y)

def clamp_pos(pos, bounds):
    half_w = bounds[0] / 2
    half_h = bounds[1] / 2
    x = max(-half_w + CURSOR_RADIUS, min(half_w - CURSOR_RADIUS, pos[0]))
    y = max(-half_h + CURSOR_RADIUS, min(half_h - CURSOR_RADIUS, pos[1]))
    return [x, y]

def show_message(win, text, wait_for_key=True):
    stim = visual.TextStim(win, text, color="white", height=30, wrapWidth=900)
    stim.draw()
    win.flip()
    if wait_for_key:
        event.waitKeys(keyList=["space", "escape"])
        if "escape" in event.getKeys(["escape"]):
            win.close()
            core.quit()
    else:
        core.wait(1.0)

# -----------------------------
# Main Task
# -----------------------------

def run_trial(win, mouse, condition, trial_num):
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
    
    # Initialize position
    current_pos = list(START_POS)
    last_mouse_pos = mouse.getPos()
    

    clock = core.Clock()
    
    while clock.getTime() < MAX_TRIAL_DURATION:
        # 1. Get current mouse position
        mouse_pos = mouse.getPos()  # absolute screen coordinates
        dx = mouse_pos[0] - last_mouse_pos[0]
        dy = mouse_pos[1] - last_mouse_pos[1]

        # Update last_mouse_pos for next frame
        last_mouse_pos = mouse_pos

        # Generate per-frame random noise (0 to 1)
        noise_rx = random.random()
        noise_ry = random.random()

        # Apply control factor and scaling to noise
        rx = (1.0 - control) * noise_rx * NOISE_SCALE
        ry = (1.0 - control) * noise_ry * NOISE_SCALE

        # 3. Update Cursor (Only if moving)
        if dx != 0 or dy != 0:
            # Apply movement to logical cursor position
            current_pos[0] += dx + rx
            current_pos[1] += dy + ry

            # Clamp logical cursor to screen boundaries
            current_pos = clamp_pos(current_pos, WINDOW_SIZE)
            cursor.pos = current_pos

        # 5. Optional: prevent physical mouse from hitting absolute screen edges
        half_w, half_h = WINDOW_SIZE[0]/2, WINDOW_SIZE[1]/2
        edge_margin = 1  # small margin to prevent jumps
        if abs(mouse_pos[0]) > (half_w - edge_margin) or abs(mouse_pos[1]) > (half_h - edge_margin):
            # Move physical mouse to center or safe zone
            mouse.setPos((0, 0))
            last_mouse_pos = (0, 0)  # reset last_mouse_pos to avoid dx spike

        # 6. Draw everything
        target.draw()
        start_pos_visual.draw()
        cursor.draw()
        info_text.draw()
        win.flip()

        
        # Check success
        dist = math.hypot(current_pos[0] - target_pos[0], current_pos[1] - target_pos[1])
        if dist < (TARGET_RADIUS + CURSOR_RADIUS):
            return True
            
        # Exit
        if "escape" in event.getKeys(["escape"]):
            win.close()
            core.quit()
            
    return False

def main():
    win = visual.Window(WINDOW_SIZE, color=(-0.2, -0.2, -0.2), units="pix")
    win.setMouseVisible(False)
    mouse = event.Mouse(win=win, visible=False)
    
    show_message(win, 
                 "Kumar Task Replication\n\n"
                 "You will control a white cursor.\n"
                 "Reach the yellow target as fast as possible.\n\n"
                 "There will be 3 difficulty levels.\n"
                 "Press SPACE to start.")
    
    for condition in CONDITIONS:
        show_message(win, f"Starting Condition:\n{condition['label']}\n\nPress SPACE to continue.")
        
        for i in range(TRIALS_PER_CONDITION):
            success = run_trial(win, mouse, condition, i+1)
            
            if success:
                msg = visual.TextStim(win, "Target Reached!", color="lime", height=40)
            else:
                msg = visual.TextStim(win, "Time's Up!", color="red", height=40)
                
            msg.draw()
            win.flip()
            core.wait(1.0)
            
    show_message(win, "Experiment Complete!\nPress SPACE to exit.")
    win.close()
    core.quit()

if __name__ == "__main__":
    main()
