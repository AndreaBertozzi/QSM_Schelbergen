import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from plotting_utils import *
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/2024/'
test_name = 'Test-2024-02-15_GS3/'
#test_name = 'Test-2024-02-23_GS3/'
#test_name = 'Test-2024-02-27_GS3/'
#test_name = 'Test-2024-03-01_GS3/'
#test_name = 'Test-2024-02-29_GS3/'

with open('Results_' + test_name[:-1] +'.pkl', "rb") as file:
    all_sim_dfs, all_exp_dfs, cycle_results_sim, cycle_results_exp = pickle.load(file)


#idx = 40
#cycle_to_cycle_plot(all_sim_dfs[idx], all_exp_dfs[idx], cycle_results_sim.iloc[idx], cycle_results_exp.iloc[idx])
#plt.show()

for idx in range(len(all_sim_dfs)):
    cycle_to_cycle_plot(all_sim_dfs[idx], all_exp_dfs[idx], cycle_results_sim.iloc[idx], cycle_results_exp.iloc[idx])
    plt.show()


#with open('Results_all_cycles_' + str(2024) + '.pkl', "rb") as file:
#    cycle_results_sim, cycle_results_exp = pickle.load(file)


print(cycle_results_exp.columns)

plt.scatter(cycle_results_exp.wind_speed_100m, cycle_results_exp.mech_power_cycle_avg_kW)
plt.scatter(cycle_results_sim.wind_speed_100m, cycle_results_sim.mech_power_cycle_avg_kW)

#plt.figure()
#plt.scatter(cycle_results_exp.wind_speed_100m, cycle_results_exp.duration_)
#plt.scatter(cycle_results_sim.wind_speed_100m, cycle_results_sim.mech_power_cycle_avg_kW)


plt.show()
