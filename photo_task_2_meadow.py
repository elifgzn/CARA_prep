"""
Cursor Manipulation Task - Photo Task 2 (Meadow)

Modelled after Kumar & Srinivashan, 2017
Cursor perturbations taken from Kumar & Srinivashan, 2017

A Pygame experiment with three trials featuring different levels of cursor control.
The user must move a rectangle frame (with perturbed cursor control).
Background is a meadow scene.

Trial 1 (Blue): Full control (0% perturbation)
Trial 2 (Orange): Medium control (50% perturbation)
Trial 3 (Green): Low control (80% perturbation)

Press SPACEBAR to advance through trials.
Press ESC to exit.
"""

import pygame
import random
import sys
import os

# ============================================================================
# CONFIGURABLE CONSTANTS
# ============================================================================

# Screen settings
BACKGROUND_COLOR = (200, 200, 200)  # Light grey (for fixation)

# Sizes (easily modifiable)
FRAME_SIZE = 150
FRAME_THICKNESS = 3

# Noise smoothing factor (0.0 = no smoothing, 1.0 = maximum smoothing)
# Increased smoothing means: new noise generated dominates
# Decreased smoothing means: previous noise has more weight, i.e., less "jumpy" cursor
NOISE_ALPHA = 0.2

# Noise magnitude scaling factor (controls how strong the perturbation is)
NOISE_SCALE = 40.0  # Adjust this to make perturbations more/less noticeable

# Colors
FIXATION_COLOR = (0, 0, 0)  # Black
TEXT_COLOR = (0, 0, 0)  # Black

# ============================================================================
# TRIAL CONFIGURATION
# ============================================================================

trials = [
    {
        'color': (0, 0, 255),
        'name': 'Blue',
        'control': 1.0,
        'label': 'Trial 1: Full Control'
    },
    {
        'color': (255, 165, 0),
        'name': 'Orange',
        'control': 0.5,
        'label': 'Trial 2: Medium Control'
    },
    {
        'color': (0, 255, 0),
        'name': 'Green',
        'control': 0.2,
        'label': 'Trial 3: Low Control'
    }
]

# ============================================================================
# PYGAME INITIALIZATION
# ============================================================================

pygame.init()

# Create fullscreen display
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_size()

pygame.display.set_caption("Cursor Manipulation Task - Meadow")
clock = pygame.time.Clock()

# Hide the system cursor
pygame.mouse.set_visible(True)

# Font for text display
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 28)

# Load and scale background image
try:
    meadow_img = pygame.image.load('meadow_scene.svg')
    meadow_img = pygame.transform.scale(meadow_img, (screen_width, screen_height))
except pygame.error as e:
    print(f"Error loading background image: {e}")
    # Fallback surface if image fails
    meadow_img = pygame.Surface((screen_width, screen_height))
    meadow_img.fill((50, 150, 50)) # Green fallback

# ============================================================================
# POSITION CALCULATIONS
# ============================================================================

# Frame initial position (lower 1/3, centered horizontally)
frame_start_x = screen_width // 2
frame_start_y = screen_height * 5 // 6

# Fixation cross position (screen center)
fixation_x = screen_width // 2
fixation_y = screen_height // 2

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def draw_frame(surface, color, center_x, center_y, size, thickness):
    """Draw a rectangle frame (outline only) centered at the given position."""
    rect = pygame.Rect(
        center_x - size // 2,
        center_y - size // 2,
        size,
        size
    )
    pygame.draw.rect(surface, color, rect, thickness)


def draw_fixation_cross(surface, color, center_x, center_y, size=20, thickness=3):
    """Draw a fixation cross at the given position."""
    # Horizontal line
    pygame.draw.line(
        surface,
        color,
        (center_x - size, center_y),
        (center_x + size, center_y),
        thickness
    )
    # Vertical line
    pygame.draw.line(
        surface,
        color,
        (center_x, center_y - size),
        (center_x, center_y + size),
        thickness
    )


def draw_text(surface, text, font, color, center_x, y):
    """Draw text centered horizontally at the given y position."""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(center_x, y))
    surface.blit(text_surface, text_rect)


def apply_boundary_constraints(x, y, frame_size, screen_w, screen_h):
    """Ensure the frame stays within screen bounds."""
    half_frame = frame_size // 2
    x = max(half_frame, min(screen_w - half_frame, x))
    y = max(half_frame, min(screen_h - half_frame, y))
    return x, y


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    """Main experiment loop."""
    
    # State variables
    current_trial = 0
    state = 'fixation'  # 'fixation' or 'trial'
    running = True
    
    # Cursor tracking - start frame at lower 1/3 position
    frame_x = float(frame_start_x)
    frame_y = float(frame_start_y)
    prev_mouse_x, prev_mouse_y = pygame.mouse.get_pos()
    
    # Noise state for smooth perturbation
    noise_rx = 0.0
    noise_ry = 0.0
    
    while running:
        # ====================================================================
        # EVENT HANDLING
        # ====================================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
                elif event.key == pygame.K_SPACE:
                    if state == 'fixation':
                        # Start the trial
                        state = 'trial'
                        # Reset frame position to lower 1/3
                        frame_x = float(frame_start_x)
                        frame_y = float(frame_start_y)
                        # Reset noise state
                        noise_rx = 0.0
                        noise_ry = 0.0
                        # Move cursor to the center of the frame
                        pygame.mouse.set_pos(int(frame_x), int(frame_y))
                        # Set tracking origin to frame position
                        prev_mouse_x = frame_x
                        prev_mouse_y = frame_y
                        
                    elif state == 'trial':
                        # End current trial
                        current_trial += 1
                        
                        if current_trial >= len(trials):
                            # All trials completed
                            running = False
                        else:
                            # Move to fixation for next trial
                            state = 'fixation'
        
        # ====================================================================
        # TRIAL STATE - CURSOR MOVEMENT WITH PERTURBATION
        # ====================================================================
        if state == 'trial':
            # Get current mouse position
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            # Calculate actual mouse displacement from previous frame
            mouse_dx = mouse_x - prev_mouse_x
            mouse_dy = mouse_y - prev_mouse_y
            
            # Only update if there's actual mouse movement
            if mouse_dx != 0 or mouse_dy != 0:
                # Get current trial's control level
                control = trials[current_trial]['control']
                
                # Generate new random values for perturbation
                new_rx = random.uniform(-1, 1)
                if new_rx == 0:
                    new_rx = 1e-12  # tiny nonzero fallback
                new_ry = random.uniform(-1, 1)
                if new_ry == 0:
                    new_ry = 1e-12  # tiny nonzero fallback
                
                # Apply exponential smoothing to prevent jitter
                noise_rx = NOISE_ALPHA * new_rx + (1 - NOISE_ALPHA) * noise_rx
                noise_ry = NOISE_ALPHA * new_ry + (1 - NOISE_ALPHA) * noise_ry
                
                # Apply perturbation formula: dx = input + (1 - control) * noise * scale
                perturbed_dx = mouse_dx + (1 - control) * noise_rx * NOISE_SCALE
                perturbed_dy = mouse_dy + (1 - control) * noise_ry * NOISE_SCALE
                
                # Update frame position
                frame_x += perturbed_dx
                frame_y += perturbed_dy
                
                # Apply boundary constraints
                frame_x, frame_y = apply_boundary_constraints(
                    frame_x, frame_y, FRAME_SIZE, screen_width, screen_height
                )
            
            # Always update previous mouse position
            prev_mouse_x = mouse_x
            prev_mouse_y = mouse_y
        
        # ====================================================================
        # RENDERING
        # ====================================================================
        
        if state == 'fixation':
            screen.fill(BACKGROUND_COLOR)
            # Draw fixation cross
            draw_fixation_cross(screen, FIXATION_COLOR, fixation_x, fixation_y)
            
            # Draw instruction
            if current_trial < len(trials):
                instruction = "Press SPACEBAR to start next trial"
            else:
                instruction = "Experiment complete!"
            draw_text(screen, instruction, font, TEXT_COLOR, screen_width // 2, screen_height - 100)
            
        elif state == 'trial':
            # Draw background image
            screen.blit(meadow_img, (0, 0))
            
            # Draw frame at perturbed cursor position
            trial_color = trials[current_trial]['color']
            draw_frame(screen, trial_color, int(frame_x), int(frame_y), FRAME_SIZE, FRAME_THICKNESS)
            
            # Draw trial label at top
            trial_label = trials[current_trial]['label']
            draw_text(screen, trial_label, font, TEXT_COLOR, screen_width // 2, 50)
            
            # Draw instruction at bottom
            instruction = "Press SPACEBAR when done"
            draw_text(screen, instruction, small_font, TEXT_COLOR, screen_width // 2, screen_height - 50)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        clock.tick(60)
    
    # Cleanup
    pygame.quit()
    sys.exit()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
