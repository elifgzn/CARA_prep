# -----------------------------
# Kumar-like Wolf-Sheep Chasing Task
# (Single wolf, single sheep)
# Heat-sink sheep escape (sampled potential)
# Mouse recentring (fixed-center, recenter every frame) for touchpad support
# -----------------------------

# PROBLEMS:
# 1) finger touches edges of touchpad, stimuli on screen moves very little
# 2) sheep does not escape well. it just goes to one direction in a straight line.


import numpy as np
from psychopy import visual, event, core
import math
import random

# -----------------------------
# Configuration
# -----------------------------
WINDOW_SIZE = (1280, 720)
TRIALS_PER_CONDITION = 2

# Movement speeds (pixels per frame)
WOLF_SPEED = 4.0        # How fast the wolf moves (pixels/frame)
SHEEP_SPEED = 3.0       # How fast the sheep moves when fleeing (pixels/frame)
MOUSE_SENSITIVITY = 2.5 # Scale mouse input (higher = more sensitive)

# Behavioral thresholds
SHEEP_FLEE_DISTANCE = 250.0  # Sheep starts fleeing when wolf is within this distance
CATCH_DISTANCE = 25.0        # Wolf catches sheep when within this distance
MAX_TRIAL_DURATION = 60.0    # Maximum time per trial in seconds

# Visual settings
WOLF_RADIUS = 15
SHEEP_RADIUS = 15
WOLF_COLOR = "blue"
SHEEP_COLOR = "white"

# Control levels: proportion of user control
CONTROL_LEVELS = [
    {'control': 0.25, 'label': '25% User Control'},
    {'control': 0.50, 'label': '50% User Control'},
    {'control': 0.75, 'label': '75% User Control'},
    {'control': 1.00, 'label': '100% User Control (Full Control)'}
]

# Starting positions
WOLF_START_POS = (0, -250)  # Bottom center (in window pixels, center = (0,0))
SHEEP_ZONE_Y = (50, 200)
SHEEP_ZONE_X = (-300, 300)

# Virtual mouse center (we recenter the OS cursor here every frame)
VIRTUAL_CENTER = (0, 0)

# Deadzone for tiny touchpad jitter
MOVE_DEADZONE = 0.5  # pixels of relative movement (after sensitivity) considered "no movement"

# -----------------------------
# Helper Functions
# -----------------------------
def get_random_sheep_pos():
    x = random.uniform(*SHEEP_ZONE_X)
    y = random.uniform(*SHEEP_ZONE_Y)
    return (x, y)

def clamp_pos(pos, bounds, margin=30):
    half_w = bounds[0] / 2
    half_h = bounds[1] / 2
    x = max(-half_w + margin, min(half_w - margin, pos[0]))
    y = max(-half_h + margin, min(half_h - margin, pos[1]))
    return [x, y]

def show_message(win, text, wait_for_key=True):
    stim = visual.TextStim(win, text, color="white", height=30, wrapWidth=900)
    stim.draw()
    win.flip()
    if wait_for_key:
        keys = event.waitKeys(keyList=["space", "escape"])
        if keys and "escape" in keys:
            win.close()
            core.quit()
    else:
        core.wait(1.0)

def calculate_potential(pos, wolf_pos, bounds):
    # Distance to wolf (closer = higher potential)
    dist_to_wolf = math.hypot(pos[0] - wolf_pos[0], pos[1] - wolf_pos[1])
    wolf_potential = 1000.0 / (dist_to_wolf + 1.0)

    # Distance to boundaries (closer = higher potential)
    half_w = bounds[0] / 2
    half_h = bounds[1] / 2

    dist_to_left = abs(pos[0] - (-half_w))
    dist_to_right = abs(pos[0] - half_w)
    dist_to_top = abs(pos[1] - half_h)
    dist_to_bottom = abs(pos[1] - (-half_h))

    min_boundary_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
    boundary_potential = 500.0 / (min_boundary_dist + 1.0)

    return wolf_potential + boundary_potential

def find_best_escape_direction(sheep_pos, wolf_pos, bounds, num_samples=16, look_ahead_dist=60):
    """
    Heat-sink style: sample directions around the sheep and pick the one
    with lowest potential (considering wolf and boundaries).
    """
    best_angle = 0.0
    lowest_potential = float('inf')
    for i in range(num_samples):
        angle = (2 * math.pi * i) / num_samples
        test_x = sheep_pos[0] + look_ahead_dist * math.cos(angle)
        test_y = sheep_pos[1] + look_ahead_dist * math.sin(angle)
        potential = calculate_potential((test_x, test_y), wolf_pos, bounds)
        if potential < lowest_potential:
            lowest_potential = potential
            best_angle = angle
    return best_angle

# -----------------------------
# Main Task
# -----------------------------
def run_trial(win, mouse, condition, trial_num):
    control = condition['control']
    label = condition['label']

    # Visuals
    sheep_pos = list(get_random_sheep_pos())
    sheep = visual.Circle(win, radius=SHEEP_RADIUS, fillColor=SHEEP_COLOR, pos=sheep_pos)
    wolf_pos = list(WOLF_START_POS)
    wolf = visual.Circle(win, radius=WOLF_RADIUS, fillColor=WOLF_COLOR, pos=wolf_pos)
    start_marker = visual.Circle(win, radius=10, fillColor="grey", pos=WOLF_START_POS)
    info_text = visual.TextStim(win, f"{label}\nTrial {trial_num}/{TRIALS_PER_CONDITION}",
                                color="white", height=24, pos=(0, -320))

    # Initialize mouse - hide and recenter
    win.mouseVisible = False
    mouse.setPos(VIRTUAL_CENTER)
    core.wait(0.01)
    event.clearEvents()
    clock = core.Clock()

    # Trial loop
    while clock.getTime() < MAX_TRIAL_DURATION:
        # ---------- 1) READ RELATIVE MOUSE (from center) ----------
        # Note: we use the OS cursor displacement from VIRTUAL_CENTER as the control signal.
        mouse_pos = mouse.getPos()  # relative to center because we recenter every frame
        # Scale by sensitivity
        mouse_dx = mouse_pos[0] * MOUSE_SENSITIVITY
        mouse_dy = mouse_pos[1] * MOUSE_SENSITIVITY
        # Immediately recenter the OS cursor so the user never hits edges
        mouse.setPos(VIRTUAL_CENTER)

        # Apply small deadzone to avoid drift from noisy touchpads
        if abs(mouse_dx) < MOVE_DEADZONE and abs(mouse_dy) < MOVE_DEADZONE:
            # treat as no intentional movement
            moved = False
        else:
            moved = True

        # ---------- 2) COMPUTE COMPUTER DIRECTION (straight-line to sheep) ----------
        dx_to_sheep = sheep_pos[0] - wolf_pos[0]
        dy_to_sheep = sheep_pos[1] - wolf_pos[1]
        computer_angle = math.atan2(dy_to_sheep, dx_to_sheep)

        # ---------- 3) COMPUTE USER DIRECTION (from mouse displacement) ----------
        # If user didn't move (deadzone), we can still let the computer steer (shared control)
        if moved:
            user_angle = math.atan2(mouse_dy, mouse_dx)
        else:
            # If no input, user vector is zero; we'll rely on computer contribution
            user_angle = None

        # ---------- 4) BLEND DIRECTIONS (vector-weighted average) ----------
        omega_user = control
        omega_computer = 1.0 - control

        # ---------- 5) MOVE WOLF (ONLY IF USER MOVED) ----------
        # Critical: wolf should NOT move on its own
        if moved:
            if omega_user == 1.0:
                final_angle = user_angle
            elif omega_user == 0.0:
                final_angle = computer_angle
            else:
                # Vector-weighted average
                ux, uy = math.cos(user_angle), math.sin(user_angle)
                cx, cy = math.cos(computer_angle), math.sin(computer_angle)
                vx = omega_user * ux + omega_computer * cx
                vy = omega_user * uy + omega_computer * cy
                if vx == 0 and vy == 0:
                    final_angle = computer_angle  # fallback
                else:
                    final_angle = math.atan2(vy, vx)

            # Move wolf
            wolf_pos[0] += WOLF_SPEED * math.cos(final_angle)
            wolf_pos[1] += WOLF_SPEED * math.sin(final_angle)
            wolf_pos = clamp_pos(wolf_pos, WINDOW_SIZE)
            wolf.pos = wolf_pos

        # ---------- 6) MOVE SHEEP (heat-sink sampling) ----------
        # Sheep always flees when wolf is moving
        if moved:
            flee_angle = find_best_escape_direction(sheep_pos, wolf_pos, WINDOW_SIZE)
            flee_angle += random.uniform(-0.2, 0.2)  # small jitter
            new_sheep_x = sheep_pos[0] + SHEEP_SPEED * math.cos(flee_angle)
            new_sheep_y = sheep_pos[1] + SHEEP_SPEED * math.sin(flee_angle)
            sheep_pos[0], sheep_pos[1] = clamp_pos((new_sheep_x, new_sheep_y), WINDOW_SIZE, margin=30)
            sheep.pos = sheep_pos

        # ---------- 7) CHECK CATCH ----------
        distance_to_wolf = math.hypot(wolf_pos[0] - sheep_pos[0], wolf_pos[1] - sheep_pos[1])
        if distance_to_wolf < CATCH_DISTANCE:
            return True

        # ---------- 8) DRAW & FLIP ----------
        start_marker.draw()
        sheep.draw()
        wolf.draw()
        info_text.draw()
        win.flip()

        # ---------- 9) ESCAPE KEY ----------
        if "escape" in event.getKeys(["escape"]):
            win.close()
            core.quit()

    # trial timed out
    return False

def main():
    win = visual.Window(WINDOW_SIZE, color=(-0.2, -0.2, -0.2), units="pix")
    win.setMouseVisible(False)
    mouse = event.Mouse(win=win, visible=False)

    show_message(win,
                 "Wolf-Sheep Chasing Task\n\n"
                 "You control the BLUE wolf.\n"
                 "Catch the WHITE sheep as fast as possible.\n\n"
                 "The sheep will try to run away from you.\n"
                 "Your control level will vary across trials.\n\n"
                 "Press SPACE to start.")

    for condition in CONTROL_LEVELS:
        show_message(win, f"Starting Condition:\n{condition['label']}\n\nPress SPACE to continue.")
        for i in range(TRIALS_PER_CONDITION):
            # Recenter mouse before each trial start
            mouse.setPos(VIRTUAL_CENTER)
            core.wait(0.01)
            success = run_trial(win, mouse, condition, i+1)

            if success:
                msg = visual.TextStim(win, "Sheep Caught!", color="lime", height=40)
            else:
                msg = visual.TextStim(win, "Time's Up!", color="red", height=40)
            msg.draw()
            win.flip()
            core.wait(1.5)

    show_message(win, "Experiment Complete!\n\nThank you for participating.\n\nPress SPACE to exit.")
    win.close()
    core.quit()

if __name__ == "__main__":
    main()
