import pandas as pd
import numpy as np
import os, json5, shutil, stat, zipfile
from scipy import signal
from qsm import *
# Clean up experimental data
def clean_up_year_experimental_folder(year, useful_folders):
    # Path where the folders are located
    base_path = str(year) + os.sep
    # Read the folder names in 2023 folder
    all_tests_folders = os.listdir(base_path)    

    # Loop through all tests and delete not useful tests
    for test in all_tests_folders:
        folder_path = os.path.join(base_path, test)
        if test not in useful_folders: # Delete everything
            if os.path.isdir(folder_path):
                def handle_remove_readonly(func, path, _):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

                shutil.rmtree(folder_path, onerror=handle_remove_readonly)
                print(f"Deleted: {folder_path}")
            else:
                print(f"Folder not found: {folder_path}")
        else: # Clean inside the folder
            for root, dirs, files in os.walk(folder_path, topdown=False):
                # Delete .mp4 files
                for file in files:
                    if file.endswith('.mp4') or 'Telegram' in file:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
        

                # Delete directories with 'TelegramMedia' in the name
                for dir_name in dirs:
                    if 'Telegram' in dir_name:
                        dir_path = os.path.join(root, dir_name)
                        def handle_remove_readonly(func, path, _):
                            os.chmod(path, stat.S_IWRITE)
                            func(path)

                        shutil.rmtree(dir_path, onerror=handle_remove_readonly)
                        print(f"Deleted directory: {dir_path}")
                
            
            # Extract the only .zip file in the folder (non-recursively)
            zip_files = [f for f in os.listdir(folder_path) if f.endswith('ProtoLogger_lidar.csv.zip')]
            if len(zip_files) == 1:
                zip_path = os.path.join(folder_path, zip_files[0])
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(folder_path)
                print(f"Extracted zip file: {zip_path}")
            elif len(zip_files) > 1:
                print(f"Warning: More than one zip file found in {folder_path}")
            else:
                print(f"No zip file found in {folder_path}")


# Processing and loading experimental data
def load_process_experimental_data(data_path, test_name):
    def load_experimental_data(data_path, test_name):
        test_path = os.path.join(data_path, test_name)

        proto_logger_file = None
        for file in os.listdir(test_path):
            if file.endswith('_ProtoLogger_lidar.csv'):
                proto_logger_file = os.path.join(test_path, file)
                break

        if proto_logger_file:
            df = pd.read_csv(proto_logger_file, delimiter=' ', low_memory=False)
            print(f"Loaded file: {proto_logger_file}")
            
        return df

    def select_columns(df, columns_to_extract):
        missing_cols = [col for col in columns_to_extract if col not in df.columns]
        if missing_cols:
            print(f"Some of the requested columns are missing: {missing_cols}")

        # Filter rows where flight_phase_index is not NaN
        df_filtered = df[df['flight_phase_index'].notna()]

        # Select only relevant columns
        df_selected = df_filtered[columns_to_extract]

        return df_selected

    def group_cycles(df):
        def group_phases(index_vector):
            # Initialize
            group_ids = np.full(len(index_vector), -1)  # -1 = not part of any group
            current_group = 0
            in_group = False

            for i, val in enumerate(index_vector):
                if val == 1:
                    if not in_group:
                        in_group = True
                        current_group += 1
                    group_ids[i] = current_group
                else:
                    in_group = False  # End of current group
            return group_ids    
            
        def group_variable(var_vector, group_ids):
            grouped_var = []
            for gid in np.unique(group_ids):
                if gid > 0:
                    grouped_var.append(var_vector[group_ids == gid])
            return grouped_var

        results = [[], [], [], []]
        flight_idx = range(1, 5)
        for i in flight_idx:
            idx = df.flight_phase_index == i
            group_ids = group_phases(idx)

            grouped_var = group_variable(df.time.to_numpy(), group_ids)
            for n in range(0, len(grouped_var)): 
                results[i-1].append(pd.DataFrame(grouped_var[n], columns=[df.columns[0]]))    

            for var in df.columns[1:]:
                # group the variables
                grouped_var = group_variable(df[var].to_numpy(), group_ids)
                for n in range(0, len(grouped_var)):
                    # Append the grouped variable to the results
                    results[i-1][n][var] = pd.Series(grouped_var[n])

        cycle_dfs = []
        for n in range(len(results[0])): 
            cycle_dfs.append(pd.concat([results[0][n], results[1][n], results[2][n], results[3][n]], axis=0, ignore_index=True))
            cycle_dfs[n].reset_index(drop=True, inplace=True)
        
        return cycle_dfs
    # Check if the processed data file exists
    print('Reading from .csv file')
    # Read the experimental data
    exp_data = load_experimental_data(data_path, test_name)
    # Columns you want to extract
    columns_to_extract = ['time', 'date',     
        'kite_0_pitch', 'kite_velocity_abs',
        'ground_tether_reelout_speed', 'ground_tether_length', 'ground_tether_force',
        'airspeed_angle_of_attack', 'ground_mech_power',
        'kite_actual_depower',
        'kite_pos_east', 'kite_pos_north', 'kite_height', 
        'kite_elevation', 'kite_azimuth', 'kite_distance', 
        'airspeed_apparent_windspeed', 'kite_estimated_va', 'kite_measured_va',
        'kite_heading', 'kite_course',
        'lift_coeff', 'drag_coeff',
        '100m Wind Speed (m/s)',
        'flight_phase', 'flight_phase_index']

    exp_data = select_columns(exp_data, columns_to_extract)
    cycle_dfs = group_cycles(exp_data)

    return cycle_dfs

# Pattern analysis
def find_max_RO_tether_length(df):
    # Find the maximum ground tether length during the RO phase
    avg_ro_depwr = np.mean(df.kite_actual_depower[df.flight_phase_index==1])
    dep_idx = (df.kite_actual_depower -  avg_ro_depwr) > 8
    start_dep_idx = next((i for i, x in enumerate(dep_idx) if x != 0), -1)
    max_tether_length_RO = df.ground_tether_length[start_dep_idx]
    return max_tether_length_RO, start_dep_idx

def find_start_RO(df):
    dep_idx = df.ground_tether_reelout_speed >= 0.15
    start_RO_idx = next((i for i, x in enumerate(dep_idx) if x != 0), -1)
    return start_RO_idx

def find_RO_pattern_param(df_RO):
    def extract_complete_peaks(sig):    
        peaks, _ = signal.find_peaks(sig, distance = 5)
        valleys, _ = signal.find_peaks(-sig, distance = 5)

        complete_peaks = []

        # Loop through valleys to find enclosed peaks
        for i in range(len(valleys) - 1):
            start = valleys[i]
            end = valleys[i + 1]
            enclosed_peaks = [p for p in peaks if start < p < end]
            if enclosed_peaks:
                tallest = max(enclosed_peaks, key=lambda p: sig[p])
                complete_peaks.append(tallest)

        # Optionally check for a final complete cycle after the last valley
        if len(valleys) >= 1:
            last_valley = valleys[-1]
            trailing_peaks = [p for p in peaks if p > last_valley]
            if trailing_peaks:
                tallest = max(trailing_peaks, key=lambda p: sig[p])
                complete_peaks.append(tallest)

        return complete_peaks

    peaks_idx, _ = signal.find_peaks(np.abs(df_RO.kite_azimuth), prominence=0.1, distance=10)    
    max_az_trac = np.mean(np.abs(df_RO.kite_azimuth)[peaks_idx]) 

    peaks_idx_el = extract_complete_peaks(df_RO.kite_elevation)
    valleys_idx_el = extract_complete_peaks(-df_RO.kite_elevation)

    if len(peaks_idx_el) == 0 or len(valleys_idx_el) == 0:
        peaks_idx_el, _ = signal.find_peaks(df_RO.kite_elevation,  distance=10)
        valleys_idx_el, _ = signal.find_peaks(df_RO.kite_elevation, distance=10)
        raise(Exception)
    
    avg_el_peak = np.mean(df_RO.kite_elevation[peaks_idx_el])
    avg_el_valley = np.mean(df_RO.kite_elevation[valleys_idx_el])
    rel_el_angle = 0.5*(avg_el_peak - avg_el_valley) if not np.isnan(avg_el_peak) and not np.isnan(avg_el_valley) else np.nan
    avg_el_angle = 0.5*(avg_el_peak + avg_el_valley)

    return max_az_trac, rel_el_angle, avg_el_angle

def find_min_RI_tether_length(df):
    avg_riro_depwr = np.mean(df.kite_actual_depower[df.flight_phase_index == 4].iloc[-5:-1])
    dep_idx = (df.kite_actual_depower -  avg_riro_depwr) > 2
    dep_idx = dep_idx.iloc[::-1]
    start_pow_idx = len(dep_idx) - next((i for i, x in enumerate(dep_idx) if x != 0), -1)
    min_tether_length_RI = df.ground_tether_length[start_pow_idx]
    return min_tether_length_RI, start_pow_idx

# Identify actual flight phases
def find_actual_flight_phases(df):
    
    start_RO_idx = find_start_RO(df)
    df.drop(df.index[:start_RO_idx], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Change the flight_phase_index of elements of RORI to RO
    _, max_tether_length_RO_idx = find_max_RO_tether_length(df)
    df.loc[0:max_tether_length_RO_idx, 'flight_phase_index'] = 1
    
    # Change the flight_phase_index of elements of RIRO to RI
    _, min_tether_length_RO_idx = find_min_RI_tether_length(df)
    start_RIRO_idx = next((i for i, x in enumerate(df.flight_phase_index) if x == 4), -1)
    df.loc[start_RIRO_idx:min_tether_length_RO_idx, 'flight_phase_index'] = 3
    return df

# Running simulations
def run_simulation_from_exp_df(df, sys_props, speed_ctrl):
    df = find_actual_flight_phases(df)
    
    max_az_trac, rel_el_angle, avg_el_angle = find_RO_pattern_param(df[df.flight_phase_index == 1])

    avg_tether_force_RO_N = np.mean(df[df.flight_phase_index == 1].ground_tether_force)*9.806
    avg_tether_force_RI_N = np.mean(df[df.flight_phase_index == 3].ground_tether_force)*9.806
    
    avg_reeling_speed_RO = np.mean(df[df.flight_phase_index == 1].ground_tether_reelout_speed)
    avg_reeling_speed_RI = np.mean(df[df.flight_phase_index == 3].ground_tether_reelout_speed)
    avg_reeling_speed_RIRO = np.mean(df[df.flight_phase_index == 4].ground_tether_reelout_speed)

    if sys_props.__class__.__name__ == "SysPropsAeroCurves":
        def pitch_deg_from_depower(depower_percent):
            # https://kitepower.atlassian.net/wiki/spaces/FT/pages/32374796/Study+Active+depower+and+winch+combined+control
            m = -0.78 # From the link
            b = 23.25
            pitch_deg = m*depower_percent + b
            return pitch_deg
        
        depower_RO = np.mean(df[df.flight_phase_index == 1].kite_actual_depower)
        depower_RI = np.mean(df[df.flight_phase_index == 3].kite_actual_depower)
        sys_props.pitch_powered = np.deg2rad(pitch_deg_from_depower(depower_RO))
        sys_props.pitch_depowered = np.deg2rad(pitch_deg_from_depower(depower_RI))    
    
    max_tether_length_RO, _ = find_max_RO_tether_length(df)
    min_tether_length_RI, _ = find_min_RI_tether_length(df)

    env_state1 = LogProfile()
    env_state1.set_reference_wind_speed(np.mean(df[df.flight_phase_index == 1]['100m Wind Speed (m/s)']))
    env_state1.set_reference_height(100)
    env_state2 = LogProfile()
    env_state2.set_reference_wind_speed(np.mean(df[df.flight_phase_index == 2]['100m Wind Speed (m/s)']))
    env_state2.set_reference_height(100)
    env_state4 = LogProfile()
    env_state4.set_reference_wind_speed(np.mean(df[df.flight_phase_index == 4]['100m Wind Speed (m/s)']))
    env_state4.set_reference_height(100)

    if speed_ctrl:
        RO_settings = {'control': ('reeling_speed', avg_reeling_speed_RO),
                    'pattern': {'azimuth_angle': max_az_trac, 'rel_elevation_angle': rel_el_angle}, 'time_step': .01}
        
        cycle_settings = {'cycle': {
                            'traction_phase': TractionPhasePattern,
                            'include_transition_energy': True
                        },
                        'retraction': {
                            'control': ('reeling_speed', avg_reeling_speed_RI),
                            'time_step': .01

                        },
                        'transition': {
                            'control': ('reeling_speed', avg_reeling_speed_RIRO),
                        },
                        'traction': RO_settings
                    }
        
    else:    
        RO_settings = {'control': ('tether_force_ground', avg_tether_force_RO_N),
                'pattern': {'azimuth_angle': max_az_trac, 'rel_elevation_angle': rel_el_angle}}
        
        cycle_settings = {'cycle': {
                            'traction_phase': TractionPhasePattern,
                            'include_transition_energy': True
                        },
                        'retraction': {
                            'control': ('tether_force_ground', avg_tether_force_RI_N)

                        },
                        'transition': {
                            'control': ('reeling_speed', avg_reeling_speed_RIRO),
                        },
                        'traction': RO_settings
                    }

    cycle = Cycle(cycle_settings)
    cycle.elevation_angle_traction = avg_el_angle
    cycle.tether_length_start_retraction = max_tether_length_RO
    cycle.tether_length_start_traction = df.ground_tether_length[0]
    cycle.tether_length_end_retraction = min_tether_length_RI
    #env_retr, env_trans, env_trac
    
    cycle.run_simulation(sys_props, [env_state2, env_state4, env_state1], print_summary=False) 

    df_sim, cycle_results = pack_results_sim(cycle, env_state1)
    
    return df_sim, cycle_results

# Packing results 
def pack_results_sim(cycle, env_state):
    # --- Traction Phase ---
    time_trac = cycle.traction_phase.time
    time_trac = [x - cycle.traction_phase.time[0] for x in time_trac]
    reel_speeds = [s.reeling_speed for s in cycle.traction_phase.steady_states]
    RO_reel_speed = np.mean(reel_speeds)
    tether_forces = [s.tether_force_ground for s in cycle.traction_phase.steady_states]
    RO_force = np.mean(tether_forces)
    app_wind_speed = [s.apparent_wind_speed for s in cycle.traction_phase.steady_states]
    power_ground = [s.power_ground for s in cycle.traction_phase.steady_states]
    RO_power = np.mean(power_ground)
    x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp in cycle.traction_phase.kinematics])
    x_traj = list(x_traj)
    y_traj = list(y_traj)
    z_traj = list(z_traj)
    tether_length = [s.straight_tether_length for s in cycle.traction_phase.kinematics]
    flight_phase_index = 1*np.ones_like(tether_length)
    # Assuming each list is a column
    data = [time_trac, reel_speeds, tether_forces, tether_length, power_ground, x_traj, y_traj, z_traj, flight_phase_index]
    RO_sim_df = pd.DataFrame(list(zip(*data)), columns=['time', 'ground_tether_reelout_speed', 'ground_tether_force', 'ground_tether_length', 'ground_mech_power', 'x_pos', 'y_pos', 'z_pos', 'flight_phase_index'])

    # --- Retraction Phase ---
    time_retr = cycle.retraction_phase.time
    time_retr = [x - (cycle.retraction_phase.time[0] - time_trac[-1]) for x in time_retr]
    reel_speeds = [s.reeling_speed for s in cycle.retraction_phase.steady_states]
    RI_reel_speed = np.mean(reel_speeds)
    tether_forces = [s.tether_force_ground for s in cycle.retraction_phase.steady_states]
    RI_force = np.mean(tether_forces)    
    power_ground = [s.power_ground for s in cycle.retraction_phase.steady_states]
    RI_power = np.mean(power_ground)
    x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp in cycle.retraction_phase.kinematics])
    x_traj = list(x_traj)
    y_traj = list(y_traj)
    z_traj = list(z_traj)
    tether_length = [s.straight_tether_length for s in cycle.retraction_phase.kinematics]
    flight_phase_index = 3*np.ones_like(tether_length)
    # Assuming each list is a column
    data = [time_retr, reel_speeds, tether_forces, tether_length, power_ground, x_traj, y_traj, z_traj, flight_phase_index]
    RI_sim_df = pd.DataFrame(list(zip(*data)), columns=['time', 'ground_tether_reelout_speed', 'ground_tether_force', 'ground_tether_length', 'ground_mech_power','x_pos', 'y_pos', 'z_pos', 'flight_phase_index'])


    # --- Transition Phase ---
    time_RIRO = cycle.transition_phase.time
    time_RIRO = [x - (cycle.transition_phase.time[0] - time_retr[-1]) for x in time_RIRO]
    reel_speeds = [s.reeling_speed for s in cycle.transition_phase.steady_states]
    RIRO_reel_speed = np.mean(reel_speeds)
    tether_forces = [s.tether_force_ground for s in cycle.transition_phase.steady_states]
    RIRO_force = np.mean(tether_forces)
    power_ground = [s.power_ground for s in cycle.transition_phase.steady_states]
    RIRO_power = np.mean(power_ground)
    x_traj, y_traj, z_traj = zip(*[(kp.x, kp.y, kp.z) for kp in cycle.transition_phase.kinematics])
    x_traj = list(x_traj)
    y_traj = list(y_traj)
    z_traj = list(z_traj)
    tether_length = [s.straight_tether_length for s in cycle.transition_phase.kinematics]
    flight_phase_index = 4*np.ones_like(tether_length)
    # Assuming each list is a column
    data = [time_RIRO, reel_speeds, tether_forces, tether_length, power_ground, x_traj, y_traj, z_traj, flight_phase_index]
    RIRO_sim_df = pd.DataFrame(list(zip(*data)), columns=['time', 'ground_tether_reelout_speed', 'ground_tether_force', 'ground_tether_length', 'ground_mech_power', 'x_pos', 'y_pos', 'z_pos', 'flight_phase_index'])


    df_sim = pd.concat([RO_sim_df, RI_sim_df, RIRO_sim_df], axis=0, ignore_index=True)
    df_sim.reset_index(drop=True, inplace=True)
    cycle_results = ({'RI_reelout_speed_avg_mps': RI_reel_speed,
                      'RI_tether_force_avg_kgf': RI_force/9.8066,
                      'RO_reelout_speed_avg_mps': RO_reel_speed,
                      'RO_tether_force_avg_kgf': RO_force/9.8066,  
                      'RIRO_reelout_speed_avg_mps': RIRO_reel_speed,
                      'RIRO_tether_force_avg_kgf': RIRO_force/9.8066,
                       # Trajectory data and cycle settings
                      'RO_elevation_kite_avg_rad': cycle.traction_phase.elevation_angle,
                      'tether_length_reelout_min_m': cycle.tether_length_end_retraction,
                      'tether_length_reelout_max_m': cycle.tether_length_start_retraction,  
                      # Power
                      'RI_mech_power_avg_kW': cycle.retraction_phase.average_power,
                      'RO_mech_power_avg_kW': cycle.traction_phase.average_power,
                      'RIRO_mech_power_avg_kW': cycle.transition_phase.average_power,
                      'mech_power_cycle_avg_kW': cycle.average_power,
                      # Duration
                      'RO_duration_s': cycle.traction_phase.duration,
                      'RIRO_duration_s': cycle.transition_phase.duration,
                      'RI_duration_s': cycle.retraction_phase.duration,
                      'duration_cycle_s': cycle.duration,
                      'wind_speed_100m': env_state.calculate_wind(100)})
    
    return df_sim, cycle_results

def pack_cycle_exp_res(df): 
    cycle_res = {'RI_mech_power_avg_kW': np.mean(df[df.flight_phase_index == 3].ground_mech_power),
                 'RO_mech_power_avg_kW': np.mean(df[df.flight_phase_index == 1].ground_mech_power),
                 'RIRO_mech_power_avg_kW': np.mean(df[df.flight_phase_index == 4].ground_mech_power),
                 'mech_power_cycle_avg_kW': np.mean(df.ground_mech_power),
                 'wind_speed_100m': np.mean(df[df.flight_phase_index == 1]['100m Wind Speed (m/s)'])    
                 }   
    return cycle_res

# Init system properties
def init_sys_props(hardware_database_filename):
    """
    Build system properties for the kite system based on the hardware database and experimental data.
    Inputs:
        hardware_database_filename (str): Path to the hardware database JSON file.
        experimental_aero_coeff_filename (str): Path to the experimental aero coefficients pickle file.
    
    Returns:
        dict: A dictionary containing system properties.
    """
  
    # LOAD FROM hardware_database.json
    with open(hardware_database_filename) as f:
        hardware_data = json5.load(f)

    # Access kite values
    kite_data = hardware_data["Kite"]["KiteV9"]["values"]
    kcu_data = hardware_data["KCU"]["KCU2"]["values"]
    gs_data = hardware_data["GS"]["GS3"]["values"]
    tether_id = gs_data["tether"]['value']
    tether_data = hardware_data["Tether"][tether_id[0:-2]]["values"]

    sys_props = {
        'kite_projected_area': kite_data['kite_surface']['value'],  # [m^2]
        'kite_mass': kite_data['kite_mass']['value'] + kcu_data['kcu_mass']['value'],  # [kg]
        # CHECK PARAMETERSSSSS
        'tether_density': (tether_data["tether_density"]["value"]/
                    (np.pi/4*tether_data["tether_diameter"]["value"]**2)),  # [kg/m^3] - 0.85 GPa
        'tether_diameter': tether_data["tether_diameter"]["value"],  # [m]
        'tether_force_max_limit': 50000,  # ~ max_wing_loading*projected_area [N] 
        'tether_force_min_limit': 100,  # ~ min_wing_loading * projected_area [N]
        'kite_lift_coefficient_powered': None,  # [-] - in the range of .9 - 1.0
        'kite_drag_coefficient_powered': None,  # [-]
        'kite_lift_coefficient_depowered': None,  # [-]
        'kite_drag_coefficient_depowered': None,  # [-] - in the range of .1 - .2
        'reeling_speed_min_limit': 0.5,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
        'reeling_speed_max_limit': 10.5,  # [m/s] 
        'tether_drag_coefficient': tether_data["cd_tether"]["value"],  # [-]
    }
    return SystemProperties(sys_props)

def init_sys_props_aero(hardware_database_filename):
    """
    Build system properties for the kite system based on the hardware database and experimental data.
    Inputs:
        hardware_database_filename (str): Path to the hardware database JSON file.
        experimental_aero_coeff_filename (str): Path to the experimental aero coefficients pickle file.
    
    Returns:
        dict: A dictionary containing system properties.
    """

    with open(hardware_database_filename) as f:
        hardware_data = json5.load(f)

    kite_data = hardware_data["Kite"]["KiteV9"]["values"]
    kcu_data = hardware_data["KCU"]["KCU2"]["values"]
    gs_data = hardware_data["GS"]["GS3"]["values"]
    tether_id = gs_data["tether"]['value']
    tether_data = hardware_data["Tether"][tether_id[0:-2]]["values"]

    sys_props = {
        'kite_projected_area': kite_data['kite_surface']['value'],  # [m^2]
        'kite_mass': kite_data['kite_mass']['value'] + kcu_data['kcu_mass']['value'],  # [kg]
        'tether_density': (tether_data["tether_density"]["value"]/
                    (np.pi/4*tether_data["tether_diameter"]["value"]**2)),  # [kg/m^3] - 0.85 GPa
        'tether_diameter': tether_data["tether_diameter"]["value"],  # [m]
        'tether_force_max_limit': 50000,  #
        'tether_force_min_limit': 100,  # ~ min_wing_loading * projected_area [N]
        'reeling_speed_min_limit': 0.5,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
        'reeling_speed_max_limit': 10.5,  # [m/s] 
        'tether_drag_coefficient': tether_data["cd_tether"]["value"],  # [-]
    }
    return SysPropsAeroCurves(sys_props)

