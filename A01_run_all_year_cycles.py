import os, pickle
import pandas as pd
from qsm import *
from utils_exp_validation import *
from tqdm import tqdm
# ADD PATH OF PROCESSED EXPERIMENTAL DATA
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/'
res_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/Results/'

year = 2024
all_tests_list = os.listdir(data_path + str(year) + os.sep)

# --- Load kite system properties --- 
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/'
sys_props_v9 = init_sys_props(data_path + 'hardware_database.json') # Instance of SystemProperties class
# Update aerodynamic coefficients
sys_props_v9.kite_lift_coefficient_powered = 0.78  # [-]
sys_props_v9.kite_drag_coefficient_powered = 0.14  # [-]
sys_props_v9.kite_lift_coefficient_depowered = 0.35  # [-]
sys_props_v9.kite_drag_coefficient_depowered =  0.08 # [-] 


cycle_results_sim = pd.DataFrame()  
cycle_results_exp = pd.DataFrame() 

for i in tqdm(range(len(all_tests_list)), desc="Simulating all cycles:", unit="item"):
    test_name = all_tests_list[i]
    try:
        cycle_dfs = load_process_experimental_data(data_path + str(year) + os.sep, test_name)    
        for i, df in enumerate(cycle_dfs):
            try:
                _, cycle_res_sim = run_simulation_from_exp_df(df, sys_props_v9, True)

                # Append experimental results
                cycle_res_exp = pack_cycle_exp_res(df)
                cycle_res_exp['cycle'] = i
                cycle_results_exp = pd.concat([cycle_results_exp, pd.DataFrame([cycle_res_exp])])    

                # Append simulation results
                cycle_res_sim['cycle'] = i
                cycle_results_sim = pd.concat([cycle_results_sim, pd.DataFrame([cycle_res_sim])])     
            
            except Exception as e:
                print(f'Sim failed for cycle {i}: {e}')
                continue
    except Exception as e:
        print(f'No Lidar file in {test_name}: {e}')
        continue


with open('Results_all_cycles_' + str(year) + '.pkl', "wb") as file:
    pickle.dump((cycle_results_sim, cycle_results_exp), file)
print("Simulation complete.")
