from psychopy import visual, core, event
import numpy as np

# Initialize window
win = visual.Window(
    size=[1024, 768],
    units='pix',
    fullscr=False,
    color='gray'
)

# Create crosshair (cursor)
crosshair = visual.ShapeStim(
    win,
    vertices='cross',
    size=20,
    lineColor='red',
    fillColor='red',
    pos=[0, 0]
)

# Create target (optional - something to aim for)
target = visual.Circle(
    win,
    radius=30,
    fillColor='green',
    lineColor='white',
    pos=[200, 150]
)

# Create instruction text
instruction_text = visual.TextStim(
    win,
    text='Use mouse to move crosshair\nPress SPACE to change control level\nPress ESC to quit',
    pos=[0, -300],
    height=20,
    color='white'
)

# Initialize mouse
mouse = event.Mouse(visible=False, win=win)
mouse.setPos([0, 0])

# Control parameters
control_levels = [1.0, 0.75, 0.5, 0.25, 0.0]  # Different control levels to test
current_control_idx = 0
control = control_levels[current_control_idx]

# Cursor position
cursor_pos = np.array([0.0, 0.0])

# Sensitivity factor for mouse input
sensitivity = 0.5

# Display control level
control_display = visual.TextStim(
    win,
    text=f'Control Level: {control:.2f}',
    pos=[0, 320],
    height=25,
    color='yellow'
)

# Main loop
clock = core.Clock()
last_mouse_pos = np.array(mouse.getPos())

while True:
    # Get current mouse position
    current_mouse_pos = np.array(mouse.getPos())
    
    # Calculate input (i) as the change in mouse position
    i_x = (current_mouse_pos[0] - last_mouse_pos[0]) * sensitivity
    i_y = (current_mouse_pos[1] - last_mouse_pos[1]) * sensitivity
    
    # Generate random values between 0 and 1
    rx = np.random.random()
    ry = np.random.random()
    
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
    crosshair.pos = cursor_pos
    
    # Update last mouse position
    last_mouse_pos = current_mouse_pos.copy()
    
    # Check for key presses
    keys = event.getKeys()
    if 'escape' in keys:
        break
    elif 'space' in keys:
        # Cycle through control levels
        current_control_idx = (current_control_idx + 1) % len(control_levels)
        control = control_levels[current_control_idx]
        control_display.text = f'Control Level: {control:.2f}'
    
    # Draw everything
    target.draw()
    crosshair.draw()
    instruction_text.draw()
    control_display.draw()
    win.flip()
    
    # Small delay to control frame rate
    core.wait(0.01)

# Cleanup
win.close()
core.quit()