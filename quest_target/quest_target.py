import math
from pathlib import Path

import numpy as np  # type: ignore
from psychopy import visual, event, core  # type: ignore
import traceback
import sys


# -----------------------------
# Utility helpers
# -----------------------------

rng = np.random.default_rng()
SCRIPT_DIR = Path(__file__).parent
# Use the simulated pool in the same directory
CORE_POOL_PATH = SCRIPT_DIR / "simulated_pool.npy"

if not CORE_POOL_PATH.exists():
    raise FileNotFoundError(f"Motion library not found at {CORE_POOL_PATH}")

motion_pool = np.load(CORE_POOL_PATH)


def logit(x):
    x = float(np.clip(x, 1e-6, 1 - 1e-6))
    return float(np.log(x / (1 - x)))


def inv_logit(z):
    return float(1.0 / (1.0 + np.exp(-z)))


def clamp_prop(s):
    return float(np.clip(s, 0.02, 0.90))


class QuestPlusStaircase:
    """
    QUEST+ implementation with the same settings as the main CDT experiment.
    """

    def __init__(self, target_type):
        self.s_grid = np.linspace(logit(0.05), logit(0.90), 61)
        self.alpha_grid = np.linspace(logit(0.05), logit(0.90), 61)
        self.beta_grid = np.geomspace(1.0, 12.0, 25)
        self.lambda_grid = np.array([0.00, 0.01, 0.02, 0.04, 0.06])
        self.gamma = 0.5
        self.target_type = target_type

        if target_type == "high":
            alpha_mu = logit(0.48)
        elif target_type == "low":
            alpha_mu = logit(0.33)
        else:
            alpha_mu = logit(0.40)
        alpha_sd = 1.0

        self.prior_alpha = np.exp(-0.5 * ((self.alpha_grid - alpha_mu) / alpha_sd) ** 2)
        self.prior_alpha /= self.prior_alpha.sum()

        beta_mean = 2.5
        beta_gsd = 2.0
        ln_beta_mean = np.log(beta_mean)
        ln_beta_sd = np.log(beta_gsd)

        self.prior_beta = np.exp(-0.5 * ((np.log(self.beta_grid) - ln_beta_mean) / ln_beta_sd) ** 2)
        self.prior_beta /= self.prior_beta.sum()

        self.prior_lambda = np.ones_like(self.lambda_grid) / len(self.lambda_grid)

        self.post_alpha = self.prior_alpha.copy()
        self.post_beta = self.prior_beta.copy()
        self.post_lambda = self.prior_lambda.copy()

        self.trial_count = 0
        self.responses = []

    def psychometric(self, s_logit, alpha, beta, lapse):
        sigmoid = 1.0 / (1.0 + np.exp(-beta * (s_logit - alpha)))
        return self.gamma + (1.0 - self.gamma - lapse) * sigmoid

    def compute_entropy(self, posterior):
        posterior = posterior + 1e-12
        return -np.sum(posterior * np.log(posterior))

    def select_stimulus_entropy_fast(self):
        s_grid_subset = self.s_grid[::3]

        current_entropy = self.compute_entropy(self.post_alpha)
        best_stimulus = None
        max_info_gain = -np.inf

        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        beta_mean = np.sum(self.beta_grid * self.post_beta)
        lambda_mean = np.sum(self.lambda_grid * self.post_lambda)

        for s_logit in s_grid_subset:
            p_correct = self.psychometric(s_logit, alpha_mean, beta_mean, lambda_mean)
            p_incorrect = 1.0 - p_correct
            if p_correct < 1e-6 or p_incorrect < 1e-6:
                continue

            post_alpha_correct = np.zeros_like(self.post_alpha)
            post_alpha_incorrect = np.zeros_like(self.post_alpha)

            for i, alpha in enumerate(self.alpha_grid):
                like_correct = self.psychometric(s_logit, alpha, beta_mean, lambda_mean)
                like_incorrect = 1.0 - like_correct
                post_alpha_correct[i] = self.post_alpha[i] * like_correct
                post_alpha_incorrect[i] = self.post_alpha[i] * like_incorrect

            post_alpha_correct /= (post_alpha_correct.sum() + 1e-12)
            post_alpha_incorrect /= (post_alpha_incorrect.sum() + 1e-12)

            entropy_correct = self.compute_entropy(post_alpha_correct)
            entropy_incorrect = self.compute_entropy(post_alpha_incorrect)
            expected_entropy = p_correct * entropy_correct + p_incorrect * entropy_incorrect

            info_gain = current_entropy - expected_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_stimulus = s_logit

        if best_stimulus is None:
            best_stimulus = self.s_grid[len(self.s_grid) // 2]
        return clamp_prop(inv_logit(best_stimulus))

    def select_stimulus_entropy(self):
        return self.select_stimulus_entropy_fast()

    def update(self, stimulus_prop, correct):
        s_logit = logit(clamp_prop(stimulus_prop))
        new_post = np.zeros((len(self.alpha_grid), len(self.beta_grid), len(self.lambda_grid)))

        for i, alpha in enumerate(self.alpha_grid):
            for j, beta in enumerate(self.beta_grid):
                for k, lapse in enumerate(self.lambda_grid):
                    prior_weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                    likelihood = (
                        self.psychometric(s_logit, alpha, beta, lapse)
                        if correct
                        else (1.0 - self.psychometric(s_logit, alpha, beta, lapse))
                    )
                    new_post[i, j, k] = prior_weight * likelihood

        new_post /= (new_post.sum() + 1e-12)
        self.post_alpha = new_post.sum(axis=(1, 2))
        self.post_beta = new_post.sum(axis=(0, 2))
        self.post_lambda = new_post.sum(axis=(0, 1))

        self.trial_count += 1
        self.responses.append((s_logit, correct))

    def get_threshold_sd(self):
        alpha_mean = np.sum(self.alpha_grid * self.post_alpha)
        alpha_var = np.sum(self.post_alpha * (self.alpha_grid - alpha_mean) ** 2)
        return float(np.sqrt(alpha_var))

    def threshold_for_target(self, p_target):
        lambda_hat = np.sum(self.lambda_grid * self.post_lambda)
        max_achievable = 1.0 - lambda_hat
        if p_target > max_achievable:
            p_target = min(0.85, max_achievable - 0.02)

        best_diff = float("inf")
        best_s = 0.5
        for s_logit in self.s_grid:
            p_pred = 0.0
            for i, alpha in enumerate(self.alpha_grid):
                for j, beta in enumerate(self.beta_grid):
                    for k, lapse in enumerate(self.lambda_grid):
                        weight = self.post_alpha[i] * self.post_beta[j] * self.post_lambda[k]
                        p_pred += weight * self.psychometric(s_logit, alpha, beta, lapse)

            diff = abs(p_pred - p_target)
            if diff < best_diff:
                best_diff = diff
                best_s = inv_logit(s_logit)
        return clamp_prop(best_s)


# -----------------------------
# Visual setup helpers
# -----------------------------

WINDOW_SIZE = (1280, 720)
CONTROL_RADIUS = 350
TRIAL_DURATION = 4.0
CALIBRATION_TRIALS_PER_STAIRSTEP = 10
OFFSET_X = 300
LOWPASS = 0.5
MAX_MOUSE_SPEED = 20.0
RECT_MARGIN_X = 30
RECT_MARGIN_BOTTOM = 30
RECT_TOP_OFFSET = 220  # matches instruction text position

# Fixed positions for reaching task (matches simulation)
START_POS = np.array([0.0, -300.0])
TARGET_POS = np.array([0.0, 250.0])


def confine(pos, limit=CONTROL_RADIUS):
    r = math.hypot(*pos)
    if r <= limit:
        return tuple(pos)
    scale = limit / r
    return (pos[0] * scale, pos[1] * scale)


def rotate(dx, dy, angle):
    if angle == 0:
        return dx, dy
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a


def apply_consistent_smoothing(trajectory1, trajectory2):
    def smooth_trajectory(traj, window_size=3):
        if len(traj) < window_size:
            return traj
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        return smoothed

    pos1 = np.cumsum(trajectory1, axis=0)
    pos2 = np.cumsum(trajectory2, axis=0)
    smooth_pos1 = smooth_trajectory(pos1)
    smooth_pos2 = smooth_trajectory(pos2)
    vel1 = np.diff(smooth_pos1, axis=0)
    vel2 = np.diff(smooth_pos2, axis=0)
    return vel1, vel2


def sample_snippet_pair():
    idx_target, idx_distractor = rng.choice(len(motion_pool), size=2, replace=False)
    target_snippet = motion_pool[idx_target]
    distractor_snippet = motion_pool[idx_distractor]
    return apply_consistent_smoothing(target_snippet, distractor_snippet)


def get_control_rect(padding=0.0):
    half_w = WINDOW_SIZE[0] / 2
    half_h = WINDOW_SIZE[1] / 2
    left = -half_w + RECT_MARGIN_X
    right = half_w - RECT_MARGIN_X
    bottom = -half_h + RECT_MARGIN_BOTTOM
    top = half_h - RECT_TOP_OFFSET
    left += padding
    right -= padding
    bottom += padding
    top -= padding
    return left, right, bottom, top


def clamp_to_rect(position, rect_bounds):
    left, right, bottom, top = rect_bounds
    x = float(np.clip(position[0], left, right))
    y = float(np.clip(position[1], bottom, top))
    return np.array([x, y], dtype=float)


def inside_rect(position, rect_bounds):
    left, right, bottom, top = rect_bounds
    return left <= position[0] <= right and bottom <= position[1] <= top


def run_calibration_trial(win, mouse, prop, angle_bias, show_distractor=True):
    # Smaller shapes (half the original size)
    square = visual.Rect(win, width=30, height=30, fillColor="deepskyblue", lineColor="deepskyblue")
    dot = visual.Circle(win, radius=15, fillColor="orangered", lineColor="orangered")
    
    # Add target visual to match simulation setup
    target_visual = visual.Circle(win, radius=15, fillColor="yellow", lineColor="yellow", pos=TARGET_POS)
    
    fixation = visual.TextStim(win, "+", color="white", height=36, pos=(0, 120))
    prompt = visual.TextStim(
        win,
        "Move the mouse. Who followed you more?\nA = Square    S = Circle",
        color="white",
        height=24,
        pos=(0, 200),
    )

    target_shape = rng.choice(["square", "circle"])
    target_snippet, distractor_snippet = sample_snippet_pair()

    # Hide system mouse cursor and set to start position
    win.mouseVisible = False
    mouse.setPos(START_POS)
    core.wait(0.1)  # Longer wait to ensure position updates
    event.clearEvents()
    clock = core.Clock()
    response = None

    # Initialize shapes at START_POS
    square_pos = START_POS.copy()
    dot_pos = START_POS.copy()
    square.pos = square_pos
    dot.pos = dot_pos

    vt = np.zeros(2, dtype=float)
    vd = np.zeros(2, dtype=float)
    mag_m_lp = 0.0
    applied_angle = angle_bias
    if angle_bias == 90:
        applied_angle = int(rng.choice([90, -90]))

    last = mouse.getPos()  # Get actual mouse position after setting
    frame = 0

    while clock.getTime() < TRIAL_DURATION and response is None:
        x, y = mouse.getPos()
        dx = x - last[0]
        dy = y - last[1]
        last = (x, y)
        
        # Softer boundary handling - only prevent movement beyond edges
        # Don't re-center, just clamp the position
        if abs(x) > (WINDOW_SIZE[0]/2 - 50) or abs(y) > (WINDOW_SIZE[1]/2 - 50):
            # Clamp position to window bounds
            clamped_x = np.clip(x, -(WINDOW_SIZE[0]/2 - 50), (WINDOW_SIZE[0]/2 - 50))
            clamped_y = np.clip(y, -(WINDOW_SIZE[1]/2 - 50), (WINDOW_SIZE[1]/2 - 50))
            mouse.setPos((clamped_x, clamped_y))
            last = (clamped_x, clamped_y)
            dx = dy = 0  # No movement this frame

        dx, dy = rotate(dx, dy, applied_angle)
        frame += 1

        mag_m = math.hypot(dx, dy)
        if mag_m > MAX_MOUSE_SPEED:
            scale_factor = MAX_MOUSE_SPEED / mag_m
            dx *= scale_factor
            dy *= scale_factor
            mag_m = MAX_MOUSE_SPEED
        if frame == 1:
            mag_m_lp = mag_m
        else:
            mag_m_lp = 0.5 * mag_m_lp + 0.5 * mag_m

        target_ou_dx, target_ou_dy = target_snippet[frame % len(target_snippet)]
        mag_target = math.hypot(target_ou_dx, target_ou_dy)
        if mag_target > 0:
            dir_target_x = target_ou_dx / mag_target
            dir_target_y = target_ou_dy / mag_target
        else:
            dir_target_x = dir_target_y = 0
        target_ou_dx = dir_target_x * mag_m_lp
        target_ou_dy = dir_target_y * mag_m_lp

        distractor_ou_dx, distractor_ou_dy = distractor_snippet[frame % len(distractor_snippet)]
        mag_distractor = math.hypot(distractor_ou_dx, distractor_ou_dy)
        if mag_distractor > 0:
            dir_distractor_x = distractor_ou_dx / mag_distractor
            dir_distractor_y = distractor_ou_dy / mag_distractor
        else:
            dir_distractor_x = dir_distractor_y = 0
        distractor_ou_dx = dir_distractor_x * mag_m_lp
        distractor_ou_dy = dir_distractor_y * mag_m_lp

        tdx = prop * dx + (1 - prop) * target_ou_dx
        tdy = prop * dy + (1 - prop) * target_ou_dy
        ddx = (1 - prop) * dx + prop * distractor_ou_dx
        ddy = (1 - prop) * dy + prop * distractor_ou_dy

        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx, ddy])

        if target_shape == "square":
            square_pos = np.array(confine(square_pos + vt))
            dot_pos = np.array(confine(dot_pos + vd))
        else:
            dot_pos = np.array(confine(dot_pos + vt))
            square_pos = np.array(confine(square_pos + vd))

        square.pos = square_pos
        dot.pos = dot_pos

        # Draw everything
        prompt.draw()
        target_visual.draw()  # Show target
        
        if target_shape == "square":
            square.draw()
            if show_distractor:
                dot.draw()
        else:
            if show_distractor:
                square.draw()
            dot.draw()

        win.flip()

        keys = event.getKeys(["a", "s", "escape"])
        if "escape" in keys:
            win.close()
            core.quit()
        if "a" in keys:
            response = "square"
        elif "s" in keys:
            response = "circle"

    if response is None:
        msg = visual.TextStim(win, "Too slow! Respond faster next time.", color="yellow", height=28)
        msg.draw()
        win.flip()
        core.wait(1.2)
        return None

    correct = int(response == target_shape)
    feedback = visual.TextStim(
        win,
        "Right!" if correct else "Wrong!",
        color="lime" if correct else "red",
        height=32,
    )
    feedback.draw()
    win.flip()
    core.wait(0.8)
    return correct


def run_calibration_phase(win, mouse):
    quests = {
        "0": QuestPlusStaircase("neutral"),
        "90": QuestPlusStaircase("neutral"),
    }
    trials_per = {"0": 0, "90": 0}

    info = visual.TextStim(win, "", color="white", height=26, pos=(0, 250))

    while min(trials_per.values()) < CALIBRATION_TRIALS_PER_STAIRSTEP:
        staircase_id = "0" if trials_per["0"] <= trials_per["90"] else "90"
        q = quests[staircase_id]
        prop = q.select_stimulus_entropy()

        info.text = (
            f"Calibration (angle {staircase_id}°)\n"
            f"Trial {trials_per[staircase_id] + 1} / {CALIBRATION_TRIALS_PER_STAIRSTEP}"
        )
        info.draw()
        win.flip()
        core.wait(0.5)

        angle_bias = int(staircase_id)
        result = run_calibration_trial(win, mouse, prop, angle_bias, show_distractor=True)
        if result is None:
            continue
        q.update(prop, result)
        trials_per[staircase_id] += 1

    return quests


def collect_thresholds(quests):
    hard_levels = []
    easy_levels = []
    for q in quests.values():
        hard_levels.append(q.threshold_for_target(0.60))
        easy_levels.append(q.threshold_for_target(0.80))
    hard_prop = float(np.mean(hard_levels))
    easy_prop = float(np.mean(easy_levels))
    return hard_prop, easy_prop


def run_control_only_phase(win, mouse, prop, label, targets_per_phase=3, max_target_time=20.0):
    dot_radius = 8.75  # quarter of calibration radius (35 / 4)
    stimulus = visual.Circle(win, radius=dot_radius, fillColor="white", lineColor="white")
    text = visual.TextStim(
        win,
        f"{label} phase\nMove the mouse to guide the dot to the target.\nPress ESC to quit.",
        height=28,
        color="white",
        pos=(0, -200),
    )
    
    # Fixed target visual
    target_visual = visual.Circle(win, radius=15, fillColor="yellow", lineColor="yellow", pos=TARGET_POS)
    start_visual = visual.Circle(win, radius=10, fillColor="grey", lineColor="grey", pos=START_POS)
    reached_msg = visual.TextStim(win, "Target reached!", color="lime", height=30, pos=(0, -200))
    
    # Screen bounds (centered at 0,0)
    half_w = WINDOW_SIZE[0] / 2
    half_h = WINDOW_SIZE[1] / 2
    screen_bounds = (-half_w, half_w, -half_h, half_h)

    for target_idx in range(targets_per_phase):
        print(f"DEBUG: Starting target {target_idx+1}/{targets_per_phase}")
        # Fixed target position
        target_pos = TARGET_POS

        # Hide system mouse cursor and set to start position
        win.mouseVisible = False
        mouse.setPos(START_POS)
        core.wait(0.1)  # Longer wait to ensure position updates
        event.clearEvents()
        
        # Start at START_POS
        position = START_POS.copy()
        
        vt = np.zeros(2, dtype=float)
        mag_m_lp = 0.0
        target_snippet, _ = sample_snippet_pair()
        last = mouse.getPos()  # Get actual mouse position after setting
        frame = 0
        clock = core.Clock()

        reached = False
        while not reached and clock.getTime() < max_target_time:
            x, y = mouse.getPos()
            dx = x - last[0]
            dy = y - last[1]
            last = (x, y)
            
            # Softer boundary handling - only prevent movement beyond edges
            # Don't re-center, just clamp the position
            if abs(x) > (WINDOW_SIZE[0]/2 - 50) or abs(y) > (WINDOW_SIZE[1]/2 - 50):
                # Clamp position to window bounds
                clamped_x = np.clip(x, -(WINDOW_SIZE[0]/2 - 50), (WINDOW_SIZE[0]/2 - 50))
                clamped_y = np.clip(y, -(WINDOW_SIZE[1]/2 - 50), (WINDOW_SIZE[1]/2 - 50))
                mouse.setPos((clamped_x, clamped_y))
                last = (clamped_x, clamped_y)
                dx = dy = 0  # No movement this frame
            
            frame += 1

            mag_m = math.hypot(dx, dy)
            if mag_m > MAX_MOUSE_SPEED:
                scale_factor = MAX_MOUSE_SPEED / mag_m
                dx *= scale_factor
                dy *= scale_factor
                mag_m = MAX_MOUSE_SPEED
            if frame == 1:
                mag_m_lp = mag_m
            else:
                mag_m_lp = 0.5 * mag_m_lp + 0.5 * mag_m

            target_ou_dx, target_ou_dy = target_snippet[frame % len(target_snippet)]
            mag_target = math.hypot(target_ou_dx, target_ou_dy)
            if mag_target > 0:
                dir_target_x = target_ou_dx / mag_target
                dir_target_y = target_ou_dy / mag_target
            else:
                dir_target_x = dir_target_y = 0
            target_ou_dx = dir_target_x * mag_m_lp
            target_ou_dy = dir_target_y * mag_m_lp

            tdx = prop * dx + (1 - prop) * target_ou_dx
            tdy = prop * dy + (1 - prop) * target_ou_dy

            # Apply lowpass filter to velocity
            vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
            proposed_pos = position + vt
            
            # Clamp to screen bounds and dampen velocity if we hit a boundary
            if inside_rect(proposed_pos, screen_bounds):
                position = proposed_pos
            else:
                clamped_pos = clamp_to_rect(proposed_pos, screen_bounds)
                # Dampen velocity when hitting boundaries to prevent getting stuck
                # This is especially important in hard condition where OU noise dominates
                if clamped_pos[0] != proposed_pos[0]:  # Hit left or right edge
                    vt[0] = 0
                    vt[1] *= 0.5  # Dampen parallel component too
                if clamped_pos[1] != proposed_pos[1]:  # Hit top or bottom edge
                    vt[1] = 0
                    vt[0] *= 0.5  # Dampen parallel component too
                position = clamped_pos
            
            stimulus.pos = position

            text.draw()
            # rect_shape.draw()
            start_visual.draw()
            target_visual.draw()
            stimulus.draw()
            win.flip()

            if "escape" in event.getKeys(["escape"]):
                win.close()
                core.quit()

            if np.linalg.norm(position - target_pos) <= (dot_radius * 2):
                reached = True

        if reached:
            reached_msg.draw()
            start_visual.draw()
            target_visual.draw()
            stimulus.draw()
            win.flip()
            core.wait(1.0)


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


def main():
    win = visual.Window(WINDOW_SIZE, color=(-0.2, -0.2, -0.2), units="pix")
    win.setMouseVisible(False)
    win.mouseVisible = False  # Force cursor to be invisible
    
    # Try to access Pyglet backend to hide cursor (Windows workaround)
    try:
        win.winHandle.set_mouse_visible(False)
    except:
        pass
    
    mouse = event.Mouse(win=win, visible=False)

    show_message(
        win,
        "Target Reaching Task:\n\n"
        "1) Calibration: Reach for the target. Identify which shape is yours.\n"
        "2) Easy condition: Reach for the target (High Control).\n"
        "3) Hard condition: Reach for the target (Low Control).\n\n"
        "Press SPACE to begin.",
    )

    quests = run_calibration_phase(win, mouse)
    hard_prop, easy_prop = collect_thresholds(quests)
    print(f"DEBUG: Easy Prop (80% acc) = {easy_prop:.4f}")
    print(f"DEBUG: Hard Prop (60% acc) = {hard_prop:.4f}")

    show_message(
        win,
        "Calibration complete!\n"
        f"Easy control level ≈ {easy_prop:.2f}\n"
        f"Hard control level ≈ {hard_prop:.2f}\n\n"
        "Press SPACE to start the easy condition.",
    )
    print("DEBUG: Starting Easy phase with 3 trials")
    run_control_only_phase(win, mouse, easy_prop, "Easy", targets_per_phase=3)

    show_message(win, "Ready for the harder condition?\nPress SPACE to continue.")
    print("DEBUG: Starting Hard phase with 3 trials")
    run_control_only_phase(win, mouse, hard_prop, "Hard", targets_per_phase=3)

    show_message(win, "All done! Press SPACE to exit.")
    win.close()
    core.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open("debug_log.txt", "w") as f:
            f.write(traceback.format_exc())
        traceback.print_exc()
        core.quit()
