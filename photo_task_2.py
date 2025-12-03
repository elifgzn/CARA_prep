"""
Cursor Manipulation Task - Photo Task 2 (Perlin Noise)

Modelled after Kumar & Srinivashan, 2017
Cursor perturbations using Perlin Noise instead of smoothed random noise.

    Perlin Noise: 
    - gradient magnitudes are fixed
    - gradient directions are 12 distinct but fixed directions
    - which gradient vector is used at which point in the grid is pseudorandom

A Pygame experiment with three trials featuring different levels of cursor control.
The user must move a rectangle frame (with perturbed cursor control) to encapsulate
a target square.

Trial 1 (Blue): Full control (0% perturbation)
Trial 2 (Orange): Medium control (50% perturbation)
Trial 3 (Green): Low control (80% perturbation)

Press SPACEBAR to advance through trials.
Press ESC to exit.
"""

import pygame
import random
import sys
import math

# ============================================================================
# PERLIN NOISE IMPLEMENTATION
# ============================================================================
# Ported from the provided JS code (Stefan Gustavson / Joseph Gentle)

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

    # imagine the whole field as a grid, each corner has a gradient vector(like an arrow. how big this is and its direction is the noise that is pseudorandom)
    # this dot product function determines how much influence a given corner has on the final noise value (because values in all corners will be somehow blended for smooth noise) 
    def dot2(self, g, x, y):
        return g[0]*x + g[1]*y

    # this function is used to smooth the noise values
    def fade(self, t):
        return t*t*t*(t*(t*6-15)+10)

    # this function interpolates contributions from each corner of the grid square
    def lerp(self, a, b, t):
        return (1-t)*a + t*b

    # THIS IS THE MAIN NOISE FUNCTION, where a noise value is actually computed for a given p(x,y)

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
        # look up gradient vectors (the arrows on the corners) to shuffle consistently and compute dot products, one for each corner
        n00 = self.dot2(self.gradP[X + self.perm[Y]], x, y)
        n01 = self.dot2(self.gradP[X + self.perm[Y + 1]], x, y - 1)
        n10 = self.dot2(self.gradP[X + 1 + self.perm[Y]], x - 1, y)
        n11 = self.dot2(self.gradP[X + 1 + self.perm[Y + 1]], x - 1, y - 1)
        
        # Compute the fade curve value for x - i.e. smoothing
        u = self.fade(x)
        
        # Interpolate the four results
        # and this is the perlin noise value for the point (x,y)
        return self.lerp(
            self.lerp(n00, n10, u),
            self.lerp(n01, n11, u),
            self.fade(y)
        )

# Overall: if you call:
# value = noise.perlin2(x, y)
# value will be a pseudorandom number between -1 and 1, where:
    # neighboring points have similar values
    # noise changes gradually
    # no sudden jump like in completely random / gaussian noise
    # If you feed time into it you get smooth osscilations
# ============================================================================
# CONFIGURABLE CONSTANTS
# ============================================================================

# Screen settings
BACKGROUND_COLOR = (200, 200, 200)  # Light grey

# Sizes
SQUARE_SIZE = 40
FRAME_SIZE = 100
FRAME_THICKNESS = 3

# Noise magnitude scaling factor
NOISE_SCALE = 40.0
NOISE_SCALE_MULT = 10.0

# Minimum multiplier to prevent cursor reversal when noise opposes movement
MIN_MULTIPLIER = 0.2  # Adjust based on pilot testing (0.1 = harsh, 0.5 = mild)

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

pygame.display.set_caption("Cursor Manipulation Task - Perlin")
clock = pygame.time.Clock()

# Hide the system cursor
pygame.mouse.set_visible(False)

# Font for text display
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 28)

# Initialize Perlin Noise
perlin = PerlinNoise(seed=random.randint(0, 65535))

# ============================================================================
# POSITION CALCULATIONS
# ============================================================================

# Target square position (upper 1/3, centered horizontally)
target_x = screen_width // 2
target_y = screen_height // 6

# Frame initial position (lower 1/3, centered horizontally)
frame_start_x = screen_width // 2
frame_start_y = screen_height * 5 // 6

# Fixation cross position (screen center)
fixation_x = screen_width // 2
fixation_y = screen_height // 2

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def draw_square(surface, color, center_x, center_y, size):
    """Draw a filled square centered at the given position."""
    rect = pygame.Rect(
        center_x - size // 2,
        center_y - size // 2,
        size,
        size
    )
    pygame.draw.rect(surface, color, rect)


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
    
    # Perlin noise time variable
    noise_t = 0.0
    
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
                        # Reset noise time
                        noise_t = 0.0
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
        # TRIAL STATE - CURSOR MOVEMENT WITH ADDITIVE PERTURBATION
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
                
                # Advance noise time -- increasing would make noise evolve smoothly instead of jitter
                noise_t += 0.05
                
                # Generate Perlin noise values
                # Use different y-offsets to get independent noise for x and y (two independent 1D perlin streams instead of a single 2D why?)
                noise_val_x = perlin.perlin2(noise_t, 0)
                noise_val_y = perlin.perlin2(noise_t, 100)
                
                # Apply perturbation formula: dx = input + (1 - control) * noise * scale
                perturbed_dx = mouse_dx + (1 - control) * noise_val_x * NOISE_SCALE
                perturbed_dy = mouse_dy + (1 - control) * noise_val_y * NOISE_SCALE
                
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
        # TRIAL STATE - CURSOR MOVEMENT WITH MULTIPLICATIVE PERTURBATION
        # needs a different (smaller) noise scale!
        # ====================================================================
        # if state == 'trial':
        #     # Get current mouse position
        #     mouse_x, mouse_y = pygame.mouse.get_pos()
            
        #     # Calculate actual mouse displacement from previous frame
        #     mouse_dx = mouse_x - prev_mouse_x
        #     mouse_dy = mouse_y - prev_mouse_y
            
        #     # Only update if there's actual mouse movement
        #     if mouse_dx != 0 or mouse_dy != 0:
        #         # Get current trial's control level
        #         control = trials[current_trial]['control']
                
        #         # Advance noise time -- increasing would make noise evolve smoothly instead of jitter
        #         noise_t += 0.05
                
        #         # Generate Perlin noise values
        #         # Use different y-offsets to get independent noise for x and y (two independent 1D perlin streams instead of a single 2D why?)
        #         noise_val_x = perlin.perlin2(noise_t, 0)
        #         noise_val_y = perlin.perlin2(noise_t, 100)
                
        #         #---------------------------------------------------
        #         # OG multiplied
        #         perturbed_dx = mouse_dx * (1+ (1 - control) * noise_val_x * NOISE_SCALE_MULT)
        #         perturbed_dy = mouse_dy * (1+ (1 - control) * noise_val_y * NOISE_SCALE_MULT)
        #         #---------------------------------------------------

                #---------------------------------------------------
                # # Apply perturbation with clamping to prevent reversal
                # multiplier_x = 1 + (1 - control) * noise_val_x * NOISE_SCALE_MULT
                # multiplier_y = 1 + (1 - control) * noise_val_y * NOISE_SCALE_MULT
                
                # # Clamp multipliers to prevent cursor from reversing direction
                # multiplier_x = max(MIN_MULTIPLIER, multiplier_x)
                # multiplier_y = max(MIN_MULTIPLIER, multiplier_y)
                
                # perturbed_dx = mouse_dx * multiplier_x
                # perturbed_dy = mouse_dy * multiplier_y
                #---------------------------------------------------

               
                #---------------------------------------------------
                # ALTERNATIVE to prevent reversal: 2D Vector Directional Scaling
                # This uses full 2D dot product for more accurate directional detection
                #---------------------------------------------------
                # # Calculate movement magnitude
                # movement_magnitude = math.sqrt(mouse_dx**2 + mouse_dy**2)
                # 
                # if movement_magnitude > 0:
                #     # Normalize movement direction vector
                #     move_dir_x = mouse_dx / movement_magnitude
                #     move_dir_y = mouse_dy / movement_magnitude
                #     
                #     # Normalize noise vector
                #     noise_magnitude = math.sqrt(noise_val_x**2 + noise_val_y**2)
                #     if noise_magnitude > 0:
                #         noise_dir_x = noise_val_x / noise_magnitude
                #         noise_dir_y = noise_val_y / noise_magnitude
                #     else:
                #         noise_dir_x = 0
                #         noise_dir_y = 0
                #     
                #     # Calculate 2D dot product: alignment between movement and noise
                #     # -1 = completely opposite, 0 = perpendicular, 1 = aligned
                #     alignment = move_dir_x * noise_dir_x + move_dir_y * noise_dir_y
                #     
                #     # Scale down noise magnitude when opposing (alignment < 0)
                #     # When aligned or perpendicular, keep full noise magnitude
                #     if alignment < 0:
                #         # Smoothly reduce noise from full strength (alignment=0) to zero (alignment=-1)
                #         noise_scale_factor = max(0, 1 + alignment)  # 0 to 1
                #         scaled_noise_x = noise_val_x * noise_scale_factor
                #         scaled_noise_y = noise_val_y * noise_scale_factor
                #     else:
                #         # Keep full noise when aligned or perpendicular
                #         scaled_noise_x = noise_val_x
                #         scaled_noise_y = noise_val_y
                #     
                #     # Apply perturbation with directionally-scaled noise
                #     multiplier_x = 1 + (1 - control) * scaled_noise_x * NOISE_SCALE_MULT
                #     multiplier_y = 1 + (1 - control) * scaled_noise_y * NOISE_SCALE_MULT
                #     
                #     perturbed_dx = mouse_dx * multiplier_x
                #     perturbed_dy = mouse_dy * multiplier_y
                # else:
                #     perturbed_dx = mouse_dx
                #     perturbed_dy = mouse_dy
                #---------------------------------------------------
                
            #     # Update frame position
            #     frame_x += perturbed_dx
            #     frame_y += perturbed_dy
                
            #     # Apply boundary constraints
            #     frame_x, frame_y = apply_boundary_constraints(
            #         frame_x, frame_y, FRAME_SIZE, screen_width, screen_height
            #     )
            
            # # Always update previous mouse position
            # prev_mouse_x = mouse_x
            # prev_mouse_y = mouse_y

        # ====================================================================
        # TRIAL STATE - CURSOR MOVEMENT WITH HYBRID PERTURBATION
        # needs a different noise scale!
        # ====================================================================
        # if state == 'trial':
        #     # Get current mouse position
        #     mouse_x, mouse_y = pygame.mouse.get_pos()
            
        #     # Calculate actual mouse displacement from previous frame
        #     mouse_dx = mouse_x - prev_mouse_x
        #     mouse_dy = mouse_y - prev_mouse_y
            
        #     # Only update if there's actual mouse movement
        #     if mouse_dx != 0 or mouse_dy != 0:
        #         # Get current trial's control level
        #         control = trials[current_trial]['control']
                
        #         # Advance noise time -- increasing would make noise evolve smoothly instead of jitter
        #         noise_t += 0.05
                
        #         # Generate Perlin noise values
        #         # Use different y-offsets to get independent noise for x and y (two independent 1D perlin streams instead of a single 2D why?)
        #         noise_val_x = perlin.perlin2(noise_t, 0)
        #         noise_val_y = perlin.perlin2(noise_t, 100)
                
        #         # Apply perturbation formula:

        #         add_x = (1 - control) * noise_val_x * NOISE_SCALE   # baseline drift
        #         mult_x = mouse_dx * (1 - control) * noise_val_x * NOISE_SCALE  # relative jitter
                
        #         add_y = (1 - control) * noise_val_y * NOISE_SCALE  # baseline drift
        #         mult_y = mouse_dy * (1 - control) * noise_val_y * NOISE_SCALE  # relative jitter
                
        #         perturbed_dx = mouse_dx + add_x + mult_x    
        #         perturbed_dy = mouse_dy + add_y + mult_y
                
        #         # Update frame position
        #         frame_x += perturbed_dx
        #         frame_y += perturbed_dy
                
        #         # Apply boundary constraints
        #         frame_x, frame_y = apply_boundary_constraints(
        #             frame_x, frame_y, FRAME_SIZE, screen_width, screen_height
        #         )
            
        #     # Always update previous mouse position
        #     prev_mouse_x = mouse_x
        #     prev_mouse_y = mouse_y
        # ====================================================================
        # RENDERING
        # ====================================================================
        screen.fill(BACKGROUND_COLOR)
        
        if state == 'fixation':
            # Draw fixation cross
            draw_fixation_cross(screen, FIXATION_COLOR, fixation_x, fixation_y)
            
            # Draw instruction
            if current_trial < len(trials):
                instruction = "Press SPACEBAR to start next trial"
            else:
                instruction = "Experiment complete!"
            draw_text(screen, instruction, font, TEXT_COLOR, screen_width // 2, screen_height - 100)
            
        elif state == 'trial':
            # Draw target square
            trial_color = trials[current_trial]['color']
            draw_square(screen, trial_color, target_x, target_y, SQUARE_SIZE)
            
            # Draw frame at perturbed cursor position
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
