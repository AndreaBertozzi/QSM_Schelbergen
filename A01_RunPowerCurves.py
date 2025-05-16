import os
import pandas as pd
import pprint as pp
from qsm import *
from tqdm import tqdm

# ADD PATH OF PROCESSED EXPERIMENTAL DATA
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/'
res_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/Results/'
experimental_fit_aero_coeff_filename = data_path + 'experimental_fit_aero_coeff_2025.csv'

def build_sys_props(hardware_database_filename, experimental_aero_coeff_filename):
    """
    Build system properties for the kite system based on the hardware database and experimental data.
    Inputs:
        hardware_database_filename (str): Path to the hardware database JSON file.
        experimental_aero_coeff_filename (str): Path to the experimental aero coefficients pickle file.
    
    Returns:
        dict: A dictionary containing system properties.
    """

    # Load kite data
    import json5
    import pickle, csv
    from math import pi
    import numpy

    # LOAD FROM hardware_database.json
    with open(hardware_database_filename) as f:
        hardware_data = json5.load(f)

    # Access V9.60 kite values
    kite_data = hardware_data["Kite"]["KiteV9"]["values"]
    kcu_data = hardware_data["KCU"]["KCU2"]["values"]
    gs_data = hardware_data["GS"]["GS3"]["values"]
    tether_id = gs_data["tether"]['value']
    tether_data = hardware_data["Tether"][tether_id[0:-2]]["values"]

    # LOAD FROM experimental_aero_coeff.csv
    experimental_aero_coeff = {}
    with open(experimental_aero_coeff_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            key, value = row
            experimental_aero_coeff[key] = float(value)  # or float(value) if needed


    sys_props_v9 = {
        'kite_projected_area': kite_data['kite_surface']['value'],  # [m^2] - 25 m^2 total flat area
        'kite_mass': kite_data['kite_mass']['value'] + kcu_data['kcu_mass']['value'],  # [kg] - 12 kg kite + 8 kg KCU
        # CHECK PARAMETERSSSSS
        'tether_density': (tether_data["tether_density"]["value"]/
                    (pi/4*tether_data["tether_diameter"]["value"]**2)),  # [kg/m^3] - 0.85 GPa
        'tether_diameter': tether_data["tether_diameter"]["value"],  # [m]
        'tether_force_max_limit': 50000,  # ~ max_wing_loading*projected_area [N] 
        'tether_force_min_limit': 100,  # ~ min_wing_loading * projected_area [N]
        'kite_lift_coefficient_powered': experimental_aero_coeff['c_l_ro'],  # [-] - in the range of .9 - 1.0
        'kite_drag_coefficient_powered': experimental_aero_coeff['c_d_ro'],  # [-]
        'kite_lift_coefficient_depowered': experimental_aero_coeff['c_l_ri'],  # [-]
        'kite_drag_coefficient_depowered': experimental_aero_coeff['c_d_ri'],  # [-] - in the range of .1 - .2
        'reeling_speed_min_limit': 0.5,  # [m/s] - ratio of 4 between lower and upper limit would reduce generator costs
        'reeling_speed_max_limit': 10.5,  # [m/s] 
        'tether_drag_coefficient': tether_data["cd_tether"]["value"],  # [-]
    }

    return sys_props_v9

def load_aero_coeff_fit_par(experimental_fit_aero_coeff_filename): 
    import csv
    import ast
    # LOAD FROM experimental_fit_aero_coeff_2025.csv
    experimental_aero_fit_coeff = {}
    with open(experimental_fit_aero_coeff_filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            key, value = row
            experimental_aero_fit_coeff[key] = ast.literal_eval(value)

    return experimental_aero_fit_coeff

# OPTIONS FOR SIMULATION
# Force or speed control (RO and RI)
force_control = False

if force_control: 
    speed_control = False
    control = 'force_control_'
else: 
    speed_control = True
    control = 'speed_control_'

# Constant or variable aerodynamic coefficients
aero = '_avg_aero_'  # '_avg_aero_', '_var_aero_', _fit_aero_

# Speed for transition phase [zero_speed, tran_speed, retr_speed]
transition_phase_speed = 'tran_speed'  # 'zero_speed', 'tran_speed', 'retr_speed'

sys_props_v9 = build_sys_props(data_path + 'hardware_database.json', data_path + 'experimental_aero_coeff.csv')
sys_props_v9 = SystemProperties(sys_props_v9)

sys_props_v9.kite_lift_coefficient_powered = 0.78  # [-] - in the range of .9 - 1.0
sys_props_v9.kite_drag_coefficient_powered = 0.14  # [-]
sys_props_v9.kite_lift_coefficient_depowered = 0.35  # [-]
sys_props_v9.kite_drag_coefficient_depowered =  0.08 # [-] - in the range of .1 - .2
# Load experimental data
data = []
for year in ['2024', '2025']:
    data.append(pd.read_csv(data_path + 'experimental_io_data_'+ year +'.csv'))

results = []


jj = 0  # 0 = 2024, 1 = 2025
year = ['2024', '2025'][jj]

filename = res_path + 'sim_'+ year + aero + control + transition_phase_speed + '.csv'

if os.path.exists(filename):
    print(filename)
    raise ValueError('Simulation results already exist.')

for ii in tqdm(range(len(data[jj])), desc="Running simulations"):
    env_state = LogProfile()
    env_state.set_reference_height(100)

    w_speed = data[jj]['lidar_wind_velocity_100m_avg_mps'][ii]
    if np.isnan(w_speed): print('NaN wind speed')
    env_state.set_reference_wind_speed(w_speed)


    if aero == '_var_aero_':  
        if np.isnan(data[jj]['c_l_o'][ii]): print('NaN c_l_o')
        if np.isnan(data[jj]['c_l_i'][ii]): print('NaN c_l_i')
        if np.isnan(data[jj]['c_d_o'][ii]): print('NaN c_d_o')
        if np.isnan(data[jj]['c_d_i'][ii]): print('NaN c_d_i')

        sys_props_v9.kite_lift_coefficient_powered = data[jj]['c_l_o'][ii]  # [-]
        sys_props_v9.kite_lift_coefficient_depowered = data[jj]['c_l_i'][ii]  # [-]
        sys_props_v9.kite_drag_coefficient_powered = data[jj]['c_d_o'][ii]  # [-]
        sys_props_v9.kite_drag_coefficient_depowered = data[jj]['c_d_i'][ii]  # [-]
    

    if aero == '_fit_aero_':
        fit_dict = load_aero_coeff_fit_par(experimental_fit_aero_coeff_filename)
        sys_props_v9.kite_lift_coefficient_powered = np.polyval(fit_dict['c_l_ro'], w_speed)  # [-]
        #sys_props_v9.kite_lift_coefficient_depowered = np.polyval(fit_dict['c_l_ri'], w_speed)  # [-]
        #sys_props_v9.kite_drag_coefficient_powered = np.polyval(fit_dict['c_d_ro'], w_speed)  # [-]
        #sys_props_v9.kite_drag_coefficient_depowered = np.polyval(fit_dict['c_d_ri'], w_speed)  # [-]
    
    
    # Force or speed control (retraction phase)

    retr_force = data[jj]['RI_tether_force_avg_kgf'][ii]*9.80665  # Convert to N
    retr_speed = data[jj]['RI_reelout_speed_avg_mps'][ii]
    if np.isnan(retr_force) or np.isnan(retr_speed): print('NaN retr_force or retr_speed')
    # Force or speed control (traction phase)
    trac_force = data[jj]['RO_tether_force_avg_kgf'][ii]*9.80665  # Convert to N
    trac_speed = data[jj]['RO_reelout_speed_avg_mps'][ii]
    if np.isnan(trac_force) or np.isnan(trac_speed): print('NaN trac_force or trac_speed')

    
    tran_speed = data[jj]['RIRO_reelout_speed_avg_mps'][ii] 
    if np.isnan(tran_speed): print('NaN tran_speed')

    # Pattern control (traction phase)
    max_az = data[jj]['phi_max'][ii]
    rel_el_angle = data[jj]['rel_theta'][ii]  # [rad]
    if np.isnan(max_az) or np.isnan(rel_el_angle): print('NaN max_az or rel_el_angle')


    settings = {
                    'cycle': {
                        'traction_phase': TractionPhasePattern,
                        'include_transition_energy': True
                    },
                    'retraction': {
                        'control': ('tether_force_ground', retr_force)

                    },
                    'transition': {
                        'control': ('reeling_speed', 0.0),
                    },
                    'traction': {
                        'control': ('tether_force_ground', trac_force),
                        'pattern': {'azimuth_angle': max_az,
                                    'rel_elevation_angle': rel_el_angle}
                    },
                }

    if transition_phase_speed == 'zero_speed':
        settings['transition']['control'] = ('reeling_speed', 0.0)
    elif transition_phase_speed == 'tran_speed':
        settings['transition']['control'] = ('reeling_speed', tran_speed)
    elif transition_phase_speed == 'retr_speed':        
        settings['transition']['control'] = ('reeling_speed', retr_speed) 
    else: 
        raise ValueError("Invalid retraction phase speed option. Choose 'zero_speed', 'tran_speed', or 'retr_speed'.")  
    
    
    if force_control:
        settings['retraction']['control'] = ('tether_force_ground', retr_force)
        settings['traction']['control'] = ('tether_force_ground', trac_force)
        
    if speed_control: 
        settings['retraction']['control'] = ('reeling_speed', retr_speed)
        settings['traction']['control'] = ('reeling_speed', trac_speed)
    
        
    cycle = Cycle(settings)

    #trac = TractionPhasePattern(settings['traction'])

    if np.isnan(data[jj]['tether_length_reelout_min_m'][ii]) or np.isnan(data[jj]['tether_length_reelout_max_m'][ii]): print('NaN stroke')
    cycle.tether_length_end_retraction = data[jj]['tether_length_reelout_min_m'][ii]
    cycle.tether_length_start_retraction = data[jj]['tether_length_reelout_max_m'][ii]
    if np.isnan(data[jj]['RO_elevation_kite_avg_rad'][ii]): print('NaN elevation angle')
    cycle.elevation_angle_traction = data[jj]['RO_elevation_kite_avg_rad'][ii]       

    try:
        probl_phase, _ = cycle.run_simulation(sys_props_v9, env_state, print_summary=False) 
        
        # --- Retraction Phase ---
        reel_speeds = [s.reeling_speed for s in cycle.retraction_phase.steady_states]
        retr_speed = sum(reel_speeds) / len(reel_speeds)

        tether_forces = [s.tether_force_ground for s in cycle.retraction_phase.steady_states]
        retr_force = sum(tether_forces) / len(tether_forces)

         # --- Traction Phase ---
        reel_speeds = [s.reeling_speed for s in cycle.traction_phase.steady_states]
        trac_speed = sum(reel_speeds) / len(reel_speeds)

        tether_forces = [s.tether_force_ground for s in cycle.traction_phase.steady_states]
        trac_force = sum(tether_forces) / len(tether_forces)

        app_wind_speed = [s.apparent_wind_speed for s in cycle.traction_phase.steady_states]
        app_wind_speed = sum(app_wind_speed) / len(app_wind_speed)

        # --- Transition Phase ---
        reel_speeds = [s.reeling_speed for s in cycle.transition_phase.steady_states]
        tran_speed = sum(reel_speeds) / len(reel_speeds)

        tether_forces = [s.tether_force_ground for s in cycle.transition_phase.steady_states]
        tran_force = sum(tether_forces) / len(tether_forces)
            
        results.append({'sim_success': True,   
                            # Kite data
                            'c_l_ro': sys_props_v9.kite_lift_coefficient_powered,
                            'c_d_ro': sys_props_v9.kite_drag_coefficient_powered,
                            'c_l_ri': sys_props_v9.kite_lift_coefficient_depowered,
                            'c_d_ri': sys_props_v9.kite_drag_coefficient_depowered,
                            'RI_reelout_speed_avg_mps': retr_speed,
                            'RI_tether_force_avg_kgf': retr_force/9.80665,
                            'RO_reelout_speed_avg_mps': trac_speed,
                            'RO_tether_force_avg_kgf': trac_force/9.80665,  
                            'RIRO_reelout_speed_avg_mps': tran_speed,
                            'RIRO_tether_force_avg_kgf': tran_force/9.80665,
                            # Trajectory data and cycle settings
                            'RO_elevation_kite_avg_rad': cycle.traction_phase.elevation_angle,
                            'tether_length_reelout_min_m': cycle.tether_length_end_retraction,
                            'tether_length_reelout_max_m': cycle.tether_length_start_retraction,  
                            # Power
                            'RI_mech_power_avg_kW': cycle.retraction_phase.average_power/1000.,
                            'RO_mech_power_avg_kW': cycle.traction_phase.average_power/1000.,
                            'RIRO_mech_power_avg_kW': cycle.transition_phase.average_power/1000.,
                            'mech_power_cycle_avg_kW': cycle.average_power/1000.,
                            # Duration
                            'RO_duration_s': cycle.traction_phase.duration,
                            'RIRO_duration_s': cycle.transition_phase.duration,
                            'RI_duration_s': cycle.retraction_phase.duration,
                            'duration_cycle_s': cycle.duration,
                            # Wind data
                            'windspeed_gndmast_mps': env_state.calculate_wind(6),
                            'lidar_wind_velocity_40m_avg_mps': env_state.calculate_wind(40),
                            'lidar_wind_velocity_100m_avg_mps': env_state.calculate_wind(100),
                            'lidar_wind_velocity_200m_avg_mps': env_state.calculate_wind(200),
                            'apparent_wind_traction_mps': app_wind_speed,
                            # Diagnosis
                            'error_code': None,
                            'error_message': None,
                            'probl_phase': probl_phase            
                    })
        
    except Exception as e:
        #print('Simulation failed')
        results.append({'sim_success': False,   
                        # Kite data
                        'c_l_ro': sys_props_v9.kite_lift_coefficient_powered,
                        'c_d_ro': sys_props_v9.kite_drag_coefficient_powered,
                        'c_l_ri': sys_props_v9.kite_lift_coefficient_depowered,
                        'c_d_ri': sys_props_v9.kite_drag_coefficient_depowered,
                        'RI_reelout_speed_avg_mps': data[jj]['RI_reelout_speed_avg_mps'][ii],
                        'RI_tether_force_avg_kgf': data[jj]['RI_tether_force_avg_kgf'][ii],
                        'RO_reelout_speed_avg_mps': data[jj]['RO_reelout_speed_avg_mps'][ii],
                        'RO_tether_force_avg_kgf': data[jj]['RO_tether_force_avg_kgf'][ii],  
                        'RIRO_reelout_speed_avg_mps': data[jj]['RIRO_reelout_speed_avg_mps'][ii],
                        'RIRO_tether_force_avg_kgf': data[jj]['RIRO_tether_force_avg_kgf'][ii],
                        # Trajectory data and cycle settings
                        'RO_elevation_kite_avg_rad': cycle.traction_phase.elevation_angle,
                        'tether_length_reelout_min_m': cycle.tether_length_end_retraction,
                        'tether_length_reelout_max_m': cycle.tether_length_start_retraction,  
                        # Power
                        'RI_mech_power_avg_kW': None,
                        'RO_mech_power_avg_kW': None,
                        'RIRO_mech_power_avg_kW': None,
                        'mech_power_cycle_avg_kW': None,
                         # Duration
                        'RO_duration_s': None,
                        'RIRO_duration_s': None,
                        'RI_duration_s': None,
                        'duration_cycle_s': None,
                        # Wind data
                        'windspeed_gndmast_mps': env_state.calculate_wind(6),
                        'lidar_wind_velocity_40m_avg_mps': env_state.calculate_wind(40),
                        'lidar_wind_velocity_100m_avg_mps': env_state.calculate_wind(100),
                        'lidar_wind_velocity_200m_avg_mps': env_state.calculate_wind(200),
                        'apparent_wind_traction_mps': None,
                        # Diagnosis
                        'error_code': int(e.code) if hasattr(e, 'code') else int(0),
                        'error_message': e.msg if hasattr(e, 'msg') else 'not_available',
                        'probl_phase': probl_phase if 'probl_phase' in locals() else 'other'
                })          
        
    continue
    
results = pd.DataFrame(results)

results.to_csv(filename, index=False)    
