import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle, json5
from qsm import *
from utils_exp_validation import *
import time

# --- Load experimental data ---
# 2024-01-30_10-55-14_GS3 
# 2024-02-15_12-38-12_GS3
# Define the directory and test name
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/2024/'
test_name = 'Test-2024-02-15_GS3/'
#test_name = 'Test-2024-02-16_GS3/'
#test_name = 'Test-2024-02-23_GS3/'
#test_name = 'Test-2024-02-27_GS3/'
#test_name = 'Test-2024-02-29_GS3/'
#test_name = 'Test-2024-03-01_GS3/'


cycle_dfs = load_process_experimental_data(data_path, test_name)

# --- Load kite system properties --- 
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/'
sys_props_v9 = init_sys_props(data_path + 'hardware_database.json') # Instance of SystemProperties class
# Update aerodynamic coefficients
sys_props_v9.kite_lift_coefficient_powered = 0.78  # [-]
sys_props_v9.kite_drag_coefficient_powered = 0.14  # [-]
sys_props_v9.kite_lift_coefficient_depowered = 0.35  # [-]
sys_props_v9.kite_drag_coefficient_depowered =  0.08 # [-] 

# This is supposed to calculate angle of attack in the qsm but does not work yet
sys_props_aero_v9 = init_sys_props_aero(data_path + 'hardware_database.json')
df_cl = pd.read_csv(data_path + 'lift_uri.csv', delimiter=";", decimal=",", header=None, names=["alpha", "c_l"])
df_cd = pd.read_csv(data_path + 'drag_uri.csv', delimiter=";", decimal=",", header=None, names=["alpha", "c_d"])
sys_props_aero_v9.angles_of_attack_lift = np.deg2rad(df_cl.alpha.to_numpy())
sys_props_aero_v9.angles_of_attack_drag = np.deg2rad(df_cd.alpha.to_numpy())
sys_props_aero_v9.kite_lift_coefficients_curve = df_cl.c_l.to_numpy()
sys_props_aero_v9.kite_drag_coefficients_curve = df_cd.c_d.to_numpy()
# -----------------------------------------------------------------------------------

all_sim_dfs = [] 
all_exp_dfs = []
cycle_results_sim = pd.DataFrame()  
cycle_results_exp = pd.DataFrame() 

for i, df in enumerate(cycle_dfs[7:8]):
    try:
        t1 = time.time()
        df_sim, cycle_res_sim = run_simulation_from_exp_df(df, sys_props_v9, True)
        print('Cycle time:', time.time()-t1)
        # Append experimental results
        all_exp_dfs.append(df)         
        cycle_res_exp = pack_cycle_exp_res(df)
        cycle_res_exp['cycle'] = i
        cycle_results_exp = pd.concat([cycle_results_exp, pd.DataFrame([cycle_res_exp])])    

        # Append simulation results
        all_sim_dfs.append(df_sim)        
        cycle_res_sim['cycle'] = i
        cycle_results_sim = pd.concat([cycle_results_sim, pd.DataFrame([cycle_res_sim])])     
        
    except Exception as e:
        print(f'Sim failed for cycle {i}: {e}')
        continue

print("Simulation complete.")

#with open('Results_' + test_name[:-1] +'.pkl', "wb") as file:
#    pickle.dump((all_sim_dfs, all_exp_dfs, cycle_results_sim, cycle_results_exp), file)


