import numpy as np
import pandas as pd
import os
# Check if running in an environment where display is available
try:
    # Try to set up for a GUI backend if possible (adjust path if needed for your venv)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'./.venv/Lib/site-packages/PyQt5/Qt5/plugins'
    import matplotlib
    matplotlib.use('qt5agg') # Use an interactive backend if available
    import matplotlib.pyplot as plt
    VISUALIZATION_ENABLED = True
    print("Matplotlib GUI backend loaded.")
except (ImportError, KeyError, RuntimeError, FileNotFoundError) as e:
    print(f"Warning: Matplotlib GUI backend ('qt5agg') could not be initialized ({e}). Using non-interactive 'Agg' backend. Plots will be saved or shown differently.")
    import matplotlib
    matplotlib.use('Agg') # Use non-interactive backend (saves figures instead of showing)
    import matplotlib.pyplot as plt
    VISUALIZATION_ENABLED = False # Visualization might still work by saving files

from PIL import Image
import math
import random

# --- Parameters ---
COORD_MIN = 0
COORD_MAX = 1000
NUM_ROUTES = 100  # N: Number of distinct routes to generate (for normal dataset)
NUM_SOURCES = 10 # M: Number of distinct signal sources (for normal dataset)

# --- New Route Generation Parameters (Normal Path Only) ---
# Requirement 2: Vary number of points and step size inversely
ROUTE_POINTS_MIN = 5
ROUTE_POINTS_MAX = 100
# Step range when points = ROUTE_POINTS_MIN (e.g., 5 points)
STEP_MIN_AT_MIN_POINTS = 100.0 # Example: 100
STEP_MAX_AT_MIN_POINTS = 200.0 # Example: 200
# Step range when points = ROUTE_POINTS_MAX (e.g., 100 points)
STEP_MIN_AT_MAX_POINTS = 20.0 # Example: 20
STEP_MAX_AT_MAX_POINTS = 40.0 # Example: 40

# --- Alternative Path Parameters (Unchanged Route Logic) ---
ROUTE_LEN_ALT = 10  # Fixed number of points for alternative path
STEP_MIN_ALT = 100.0 # Fixed step range for alternative path
STEP_MAX_ALT = 200.0 # Fixed step range for alternative path

# --- General Parameters ---
NOISE_MAP_FILE = 'noise_map/noise.png' # Path to the 1000x1000 noise map
RESOURCES_DIR = 'resources'
OUTPUT_CSV_FILE_NORMAL = os.path.join(RESOURCES_DIR,'dataset-v3_normal.csv')
OUTPUT_CSV_FILE_ALT = os.path.join(RESOURCES_DIR,'dataset-v3_alt.csv')
OUTPUT_CSV_FILE_VACUUM = os.path.join(RESOURCES_DIR,'dataset-v3_vacuum.csv') # Separate file for vacuum mode
VISUALIZE_SESSIONS = [0, 1, 2, 3] # Example session IDs to visualize (relative to the generated dataset)

# --- New Feature Flag ---
# Requirement 1: New RSSI logic controlled by this flag
# Set to True to use the simple logarithmic RSSI model (0 at 0m, -120 at 1000m)
# and disable noise map application.
# Set to False to use the original path loss model with noise map.
VACUUM_SET = True # <<< CHANGE THIS TO True/False AS NEEDED

# --- RSSI Parameters ---
# Original Model (Used if VACUUM_SET = False)
P0 = -30  # Reference power (RSSI at d0) in dBm
N_PATH_LOSS = 2.5 # Path loss exponent
D0 = 1.0 # Reference distance in meters
NOISE_SCALE_FACTOR = 15 # Max dBm reduction due to noise (only if VACUUM_SET = False)

# Vacuum Model Parameters (Used if VACUUM_SET = True)
# Requirement 1: RSSI=0 at distance=0, RSSI=-120 at distance=1000
RSSI_MAX_DIST = 1000.0 # Distance at which signal is considered lost
RSSI_AT_MAX_DIST = -120.0 # RSSI value at MAX_DIST
RSSI_AT_ZERO_DIST = 0.0   # RSSI value at zero distance
# Derived constant for the vacuum model: RSSI = RSSI(0) - K * log10(distance + 1)
# K = (RSSI(0) - RSSI(1000)) / log10(1000 + 1)
VACUUM_K = (RSSI_AT_ZERO_DIST - RSSI_AT_MAX_DIST) / math.log10(RSSI_MAX_DIST + 1.0) # Approx 120 / 3.0004 ~= 40

# --- Helper Functions ---

def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_noise_map(filepath):
    """Loads the noise map image and returns it as a NumPy array."""
    # Ensure directory exists
    ensure_dir(os.path.dirname(filepath))

    if not os.path.exists(filepath):
        print(f"Warning: Noise map file not found at '{filepath}'. Creating dummy (white) map.")
        dummy_noise = Image.new('L', (COORD_MAX, COORD_MAX), color=255) # White = no noise
        dummy_noise.save(filepath)

    try:
        img = Image.open(filepath).convert('L') # Convert to grayscale
        if img.size != (COORD_MAX, COORD_MAX):
            print(f"Warning: Noise map size {img.size} does not match coordinate range {COORD_MAX}x{COORD_MAX}. Resizing...")
            img = img.resize((COORD_MAX, COORD_MAX))
        # Flipud because image origin (0,0) is top-left, while plot origin is bottom-left
        noise_array = np.flipud(np.array(img))
        print(f"Noise map '{filepath}' loaded successfully.")
        return noise_array
    except Exception as e:
        print(f"Error loading noise map: {e}. Noise will not be applied.")
        return None

def calculate_rssi(distance):
    """
    Calculates RSSI based on the selected model (VACUUM_SET flag).
    Requirement 1: Implements both RSSI calculation methods.
    """
    if VACUUM_SET:
        # Simple Logarithmic Model: RSSI = 0 at d=0, -120 at d=1000
        # Formula: RSSI = RSSI_AT_ZERO_DIST - K * log10(distance + 1)
        # Ensure distance is non-negative
        distance = max(0.0, distance)
        # Add 1 before log10 to handle distance=0 correctly (log10(1)=0)
        rssi = RSSI_AT_ZERO_DIST - VACUUM_K * math.log10(distance + 1.0)
        # Clamp the minimum value to RSSI_AT_MAX_DIST
        rssi = max(RSSI_AT_MAX_DIST, rssi)
        return rssi
    else:
        # Original Log-Distance Path Loss Model
        # Avoid log(<=0) or unrealistic values very close
        eff_distance = max(D0, distance) # Use D0 as min distance for calculation
        # Original formula: P0 - 10 * n * log10(d / d0)
        # Handle potential math domain error if eff_distance / D0 is <= 0 (shouldn't happen with max(D0, dist))
        try:
            rssi = P0 - 10 * N_PATH_LOSS * math.log10(eff_distance / D0)
        except ValueError:
             rssi = P0 # Or some other default like -120 if distance is problematic
        return rssi

def apply_noise(rssi_in, x, y, noise_map):
    """Applies noise reduction based on the noise map, only if not in VACUUM_SET mode."""
    if noise_map is None or VACUUM_SET: # Do not apply noise in vacuum mode or if map failed
        return rssi_in

    try:
        # Get noise pixel value - clamp coordinates to be safe indices
        # Ensure integer indices within bounds [0, COORD_MAX-1]
        x_idx = min(max(int(round(x)), 0), COORD_MAX - 1)
        y_idx = min(max(int(round(y)), 0), COORD_MAX - 1)
        pixel_brightness = noise_map[y_idx, x_idx] # noise_map is already flipped vertically

        # Noise effect: Higher reduction for darker pixels (lower brightness)
        # Noise is value between 0 (no reduction) and NOISE_SCALE_FACTOR (max reduction)
        noise_reduction = NOISE_SCALE_FACTOR * (255.0 - pixel_brightness) / 255.0
        rssi_out = rssi_in - noise_reduction
        # Optional: Clamp RSSI to a minimum realistic value if needed after noise
        # rssi_out = max(rssi_out, -120.0)
        return rssi_out
    except IndexError:
        print(f"Warning: Index out of bounds accessing noise map at ({x:.2f}, {y:.2f}) -> [{x_idx}, {y_idx}]. Skipping noise for this point.")
        return rssi_in
    except Exception as e:
         print(f"Warning: Error applying noise at ({x:.2f}, {y:.2f}): {e}. Skipping noise for this point.")
         return rssi_in

# Route generation function adjusted to take exact num_steps
def generate_route(num_steps, min_step, max_step, x_min, x_max, y_min, y_max, persistence_alpha=0.75, turn_bias_90_deg=0.7):
    """
    Generates a single continuous route with a specific number of steps.
    """
    route = []
    # Start at a random position within bounds
    current_x = random.uniform(x_min, x_max)
    current_y = random.uniform(y_min, y_max)
    route.append((current_x, current_y))

    # Initial random direction
    current_angle = random.uniform(0, 2 * math.pi)

    # Need num_steps points, so generate num_steps-1 segments
    steps_generated = 0
    max_attempts = num_steps * 3 # Prevent infinite loop if stuck
    attempts = 0
    while steps_generated < num_steps - 1 and attempts < max_attempts:
        attempts += 1
        step_len = random.uniform(min_step, max_step)

        # --- Turn logic (same as before) ---
        if random.random() > persistence_alpha: # Turn
            if random.random() < turn_bias_90_deg: # Biased 90 deg turn
                angle_change = random.choice([math.pi / 2, -math.pi / 2])
                angle_change += random.uniform(-math.pi / 18, math.pi / 18) # Add jitter
            else: # More random turn
                angle_change = random.uniform(math.pi / 6, math.pi / 3) * random.choice([1, -1])
            current_angle += angle_change
        else: # Continue relatively straight
             current_angle += random.uniform(-math.pi / 36, math.pi / 36) # Slight drift

        current_angle = current_angle % (2 * math.pi) # Normalize angle
        # --- End Turn logic ---

        # Calculate next potential position
        next_x = current_x + step_len * math.cos(current_angle)
        next_y = current_y + step_len * math.sin(current_angle)

        # Boundary clamping
        next_x_clamped = max(x_min, min(x_max, next_x))
        next_y_clamped = max(y_min, min(y_max, next_y))

        # If the position was clamped, reflect angle to move away from edge
        if next_x != next_x_clamped or next_y != next_y_clamped:
             # Force a more significant turn next time
             current_angle += random.uniform(math.pi / 4, 3 * math.pi / 4) * random.choice([1,-1])

        next_x, next_y = next_x_clamped, next_y_clamped

        # Avoid getting stuck exactly at the same point after clamping
        if abs(next_x - current_x) < 1e-6 and abs(next_y - current_y) < 1e-6 :
             current_angle = random.uniform(0, 2 * math.pi) # Try a completely new direction
             continue # Skip adding this point, try generating next segment again

        # Add the valid point
        route.append((next_x, next_y))
        current_x, current_y = next_x, next_y
        steps_generated += 1

    if steps_generated < num_steps -1:
        print(f"Warning: Generated route only has {len(route)} points (requested {num_steps}) after {max_attempts} attempts.")

    return route


def visualize_session(df, session_id_to_plot, ax=None, title_prefix="", output_dir="visualizations"):
    """Visualizes a specific session from the dataframe."""
    if not VISUALIZATION_ENABLED: return # Skip if plotting disabled

    session_data = df[df['session_id'] == session_id_to_plot]
    if session_data.empty:
        print(f"Session {session_id_to_plot} not found in data for visualization.")
        if ax:
            ax.set_title(f"{title_prefix}Session {session_id_to_plot} (Not Found)")
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(COORD_MIN, COORD_MAX)
            ax.set_ylim(COORD_MIN, COORD_MAX)
            ax.grid(True)
        return

    standalone_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        standalone_plot = True

    meas_x = session_data['meas_x'].values
    meas_y = session_data['meas_y'].values
    # Ensure source coordinates are consistent within the session
    source_x = session_data['source_x'].iloc[0]
    source_y = session_data['source_y'].iloc[0]
    sin_a = session_data['sin_a'].values
    cos_a = session_data['cos_a'].values

    # Plot measurement points
    ax.plot(meas_x, meas_y, 'bo-', label='Measurement Path', markersize=4, linewidth=1)
    ax.plot(meas_x[0], meas_y[0], 'go', markersize=6, label='Start') # Mark start
    ax.plot(meas_x[-1], meas_y[-1], 'yo', markersize=6, label='End') # Mark end

    # Plot source location
    ax.plot(source_x, source_y, 'r*', markersize=15, label='Source')

    # Plot direction arrows (quiver)
    # Use a fixed scale for arrows for consistency, adjust 'arrow_display_scale' as needed
    arrow_display_scale = 50 # Lower value = longer arrows
    ax.quiver(meas_x, meas_y, cos_a, sin_a, color='gray', scale=arrow_display_scale,
              scale_units='xy', angles='xy', width=0.003, headwidth=3, headlength=4,
              label='Direction to Source')

    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")
    ax.set_title(f"{title_prefix}Session {session_id_to_plot} Visualization")
    ax.set_xlim(COORD_MIN, COORD_MAX)
    ax.set_ylim(COORD_MIN, COORD_MAX)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    if standalone_plot:
        ensure_dir(output_dir)
        filename = os.path.join(output_dir, f"{title_prefix.strip()}_Session_{session_id_to_plot}.png")
        plt.savefig(filename)
        print(f"Saved visualization: {filename}")
        plt.close(fig) # Close the figure after saving if running non-interactively


# --- Alternative Dataset Generation (Uses VACUUM_SET for RSSI, fixed route params) ---

def alternative_generate_dataset(output_file):
    """Generates the alternative dataset with cumulative measurements."""
    print("\n--- Starting Alternative Dataset Generation ---")
    print(f"VACUUM_SET Mode: {VACUUM_SET}")
    print(f"Route Length: {ROUTE_LEN_ALT}, Step Range: [{STEP_MIN_ALT}, {STEP_MAX_ALT}]")

    # 1. Load noise map (only used if VACUUM_SET = False)
    noise_map = load_noise_map(NOISE_MAP_FILE)

    # 2. Generate a single base route
    print("Generating base alternative route...")
    alt_route = generate_route(
        ROUTE_LEN_ALT, # num_steps
        STEP_MIN_ALT, STEP_MAX_ALT,
        COORD_MIN, COORD_MAX,
        COORD_MIN, COORD_MAX
    )
    print(f"Alternative route generated. Base length: {len(alt_route)} points.")
    if not alt_route:
        print("Error: Failed to generate alternative route. Skipping.")
        return None


    # 3. Generate source location
    print("Generating source...")
    source = (random.uniform(COORD_MIN, COORD_MAX), random.uniform(COORD_MIN, COORD_MAX))
    src_x, src_y = source

    # 4. Generate dataset entries (cumulatively)
    print("Generating alternative dataset entries (cumulative)...")
    dataset = []
    session_id = 0
    max_points_in_route = len(alt_route)

    # Generate sessions with 1 point, 2 points, ..., up to max_points_in_route
    for k in range(1, max_points_in_route + 1):
        current_session_data = []
        # Include the first k points from the generated route
        for i in range(k):
            meas_x, meas_y = alt_route[i]

            distance = math.hypot(meas_x - src_x, meas_y - src_y)
            rssi_raw = calculate_rssi(distance) # Uses VACUUM_SET logic internally
            # Apply noise (only if VACUUM_SET=False and noise_map is available)
            rssi_final = apply_noise(rssi_raw, meas_x, meas_y, noise_map)

            # Calculate direction angle and sin/cos
            angle = math.atan2(src_y - meas_y, src_x - meas_x)
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)

            current_session_data.append({
                'session_id': session_id, # All points in this cumulative set get the same ID
                'meas_x': meas_x,
                'meas_y': meas_y,
                'rssi': rssi_final,
                'source_x': src_x,
                'source_y': src_y,
                'sin_a': sin_a,
                'cos_a': cos_a,
                'distance': distance
            })
        # Only add non-empty sessions
        if current_session_data:
            dataset.extend(current_session_data)
            session_id += 1 # Increment session ID for the next cumulative set (k+1 points)

    if not dataset:
        print("Warning: No data generated for alternative dataset.")
        return None

    # 5. Create DataFrame and Save
    df_alt = pd.DataFrame(dataset)
    ensure_dir(os.path.dirname(output_file))
    df_alt.to_csv(output_file, index=False)
    print(f"Alternative dataset saved to '{output_file}' ({len(df_alt)} rows, {session_id} sessions)")

    # 6. Visualization (if enabled)
    viz_output_dir = "visualizations_alt" if not VACUUM_SET else "visualizations_vacuum_alt"
    if VISUALIZATION_ENABLED and VISUALIZE_SESSIONS:
        print(f"Visualizing alternative dataset sessions (saving to ./{viz_output_dir}/)...")
        num_viz = min(len(VISUALIZE_SESSIONS), session_id)
        plot_indices = [s for s in VISUALIZE_SESSIONS if s < session_id]

        if num_viz > 0 and plot_indices:
             # Create multiple figures for clarity if many sessions are visualized
             for sid in plot_indices:
                 visualize_session(df_alt, sid, ax=None, title_prefix="Alt ", output_dir=viz_output_dir) # Will save individually
        else:
            print("No valid sessions specified or found for visualization.")

    print("--- Alternative Dataset Generation Finished ---")
    return df_alt


# --- Main Dataset Generation (Uses VACUUM_SET for RSSI, variable route params) ---

def main_generate_dataset(output_file):
    """Generates the main dataset with multiple sources and routes."""
    print("\n--- Starting Main Dataset Generation ---")
    print(f"VACUUM_SET Mode: {VACUUM_SET}")
    print(f"Route Points Range: [{ROUTE_POINTS_MIN}, {ROUTE_POINTS_MAX}]")

    # 1. Load Noise Map (only used if VACUUM_SET = False)
    noise_map = load_noise_map(NOISE_MAP_FILE)

    # 2. Generate Routes (Requirement 2: variable points and steps)
    routes = []
    print(f"Generating {NUM_ROUTES} routes with variable parameters...")
    for r_idx in range(NUM_ROUTES):
        # Determine number of points for this route
        num_steps = random.randint(ROUTE_POINTS_MIN, ROUTE_POINTS_MAX)

        # Calculate interpolation factor (0 for min points, 1 for max points)
        if ROUTE_POINTS_MAX == ROUTE_POINTS_MIN:
            f = 0.5 # Avoid division by zero if range is just 1 point type
        else:
            # Linear interpolation factor based on number of points
            f = (num_steps - ROUTE_POINTS_MIN) / (ROUTE_POINTS_MAX - ROUTE_POINTS_MIN)

        # Interpolate step min/max based on num_steps (inversely)
        current_step_min = STEP_MIN_AT_MIN_POINTS * (1 - f) + STEP_MIN_AT_MAX_POINTS * f
        current_step_max = STEP_MAX_AT_MIN_POINTS * (1 - f) + STEP_MAX_AT_MAX_POINTS * f

        # Ensure min <= max, add small buffer if they are very close
        current_step_max = max(current_step_max, current_step_min + 1e-6)

        print(f"  Route {r_idx+1}: Requesting {num_steps} points, Step Range=[{current_step_min:.2f}, {current_step_max:.2f}]")

        # Generate the route with the calculated parameters
        route = generate_route(
            num_steps, # Pass exact number of steps requested
            current_step_min, current_step_max,
            COORD_MIN, COORD_MAX,
            COORD_MIN, COORD_MAX
        )
        if route: # Only add if route generation was successful
            routes.append(route)
            print(f"  Generated Route {r_idx+1}/{NUM_ROUTES} with actual {len(route)} points.")
        else:
            print(f"  Warning: Failed to generate Route {r_idx+1}.")

    if not routes:
        print("Error: No routes were generated. Exiting main dataset generation.")
        return None
    print("Routes generation finished.")


    # 3. Generate Source Locations
    print(f"Generating {NUM_SOURCES} source locations...")
    sources = [(random.uniform(COORD_MIN, COORD_MAX), random.uniform(COORD_MIN, COORD_MAX)) for _ in range(NUM_SOURCES)]
    print("Sources generated.")

    # 4. Generate Dataset Entries
    dataset = []
    session_id_counter = 0
    print("Generating main dataset entries...")

    for i, (src_x, src_y) in enumerate(sources):
        print(f"  Processing Source {i+1}/{NUM_SOURCES}...")
        for j, route in enumerate(routes):
            # Each combination of a source and a route forms a session
            current_session_data = []
            for meas_x, meas_y in route:
                distance = math.hypot(meas_x - src_x, meas_y - src_y)
                rssi_raw = calculate_rssi(distance) # Uses VACUUM_SET logic internally
                # Apply noise (only if VACUUM_SET=False and noise_map is available)
                rssi_final = apply_noise(rssi_raw, meas_x, meas_y, noise_map)

                # Calculate direction angle and sin/cos
                angle = math.atan2(src_y - meas_y, src_x - meas_x)
                sin_a = math.sin(angle)
                cos_a = math.cos(angle)

                current_session_data.append({
                    'session_id': session_id_counter, # Assign current session ID
                    'meas_x': meas_x,
                    'meas_y': meas_y,
                    'rssi': rssi_final,
                    'source_x': src_x,
                    'source_y': src_y,
                    'sin_a': sin_a,
                    'cos_a': cos_a,
                    'distance': distance
                })
            # Add the data for this session if any points were generated
            if current_session_data:
                dataset.extend(current_session_data)
                # Increment session ID for the next source-route pair
                session_id_counter += 1
    print(f"Main dataset entries generated ({len(dataset)} rows across {session_id_counter} sessions).")

    if not dataset:
        print("Warning: No data generated for main dataset.")
        return None

    # 5. Create DataFrame and Save
    df_main = pd.DataFrame(dataset)
    ensure_dir(os.path.dirname(output_file))
    df_main.to_csv(output_file, index=False)
    print(f"Main dataset saved to '{output_file}'")

    # 6. Visualization (if enabled)
    viz_output_dir = "visualizations_main" if not VACUUM_SET else "visualizations_vacuum_main"
    if VISUALIZATION_ENABLED and VISUALIZE_SESSIONS:
        print(f"Visualizing main dataset sessions (saving to ./{viz_output_dir}/)...")
        num_viz = min(len(VISUALIZE_SESSIONS), session_id_counter)
        plot_indices = [s for s in VISUALIZE_SESSIONS if s < session_id_counter]

        if num_viz > 0 and plot_indices:
            # Create multiple figures for clarity
            for sid in plot_indices:
                visualize_session(df_main, sid, ax=None, title_prefix="Main ", output_dir=viz_output_dir) # Will save individually
        else:
            print("No valid sessions specified or found for visualization.")

    print("--- Main Dataset Generation Finished ---")
    return df_main


# --- Script Execution ---
if __name__ == "__main__":

    # Determine output filenames based on VACUUM_SET flag
    if VACUUM_SET:
        main_output_file = OUTPUT_CSV_FILE_VACUUM # Overwrite vacuum file for main
        alt_output_file = OUTPUT_CSV_FILE_VACUUM.replace('.csv', '_alt.csv') # Make alt distinct
        print("Mode: VACUUM_SET = True. Using simple RSSI model, no noise.")
        print(f"Main output: {main_output_file}")
        print(f"Alt output: {alt_output_file}")
    else:
        main_output_file = OUTPUT_CSV_FILE_NORMAL
        alt_output_file = OUTPUT_CSV_FILE_ALT
        print("Mode: VACUUM_SET = False. Using path-loss RSSI model with noise.")
        print(f"Main output: {main_output_file}")
        print(f"Alt output: {alt_output_file}")


    # Generate the main dataset (using variable routes/steps)
    main_generate_dataset(main_output_file)

    # Generate the alternative dataset (using fixed routes/steps)
    # alternative_generate_dataset(alt_output_file)

    print("\nScript finished.")
    if not VISUALIZATION_ENABLED:
        print("Note: Visualizations were likely saved to files as interactive display was not available.")
    elif VISUALIZE_SESSIONS:
         print("Visualizations were generated (either shown interactively or saved to files).")