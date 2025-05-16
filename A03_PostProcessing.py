import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

exp_data = pickle.load(open('exp_data_processed.pkl', 'rb'))
data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/2024/'
#test_name = 'Test-2024-01-30_GS3/'
#test_name = 'Test-2024-02-15_GS3/'

colors = ["#FF6347",  # Tomato
          "#4682B4",  # SteelBlue
          "#32CD32",  # LimeGreen
          "#FFD700"]  # Gold

# Create the colormap
my_cmap = mcolors.ListedColormap(colors)


with open("all_results1.pkl", "rb") as file:
    all_sim_dfs, all_exp_dfs,  cycle_results_sim, cycle_results_exp = pickle.load(file)

plt.scatter(cycle_results_exp.wind_speed_100m, cycle_results_exp.mech_power_cycle_avg_kW/1000)
plt.scatter(cycle_results_sim.wind_speed_100m, cycle_results_sim.mech_power_cycle_avg_kW/1000)
plt.ylim([0, 30])

#with open("all_results1.pkl", "rb") as file:
#    all_sim_dfs, all_exp_dfs,  cycle_results_sim, cycle_results_exp = pickle.load(file)


#plt.scatter(cycle_results_exp.wind_speed_100m, cycle_results_exp.mech_power_cycle_avg_kW/1000)
#plt.scatter(cycle_results_sim.wind_speed_100m, cycle_results_sim.mech_power_cycle_avg_kW/1000)
#plt.ylim([0, 30])




plt.show()

"""
'RI_reelout_speed_avg_mps', 'RI_tether_force_avg_kgf',
'RO_reelout_speed_avg_mps', 'RO_tether_force_avg_kgf',
'RIRO_reelout_speed_avg_mps', 'RIRO_tether_force_avg_kgf',
'RO_elevation_kite_avg_rad', 'tether_length_reelout_min_m',
'tether_length_reelout_max_m', 'RI_mech_power_avg_kW',
'RO_mech_power_avg_kW', 'RIRO_mech_power_avg_kW',
'mech_power_cycle_avg_kW', 'RO_duration_s', 'RIRO_duration_s',
'RI_duration_s', 'duration_cycle_s', 'wind_speed_100m', 'cycle'

"""

#plt.scatter(cycle_results_exp.cycle, cycle_results_exp.mech_power_cycle_avg_kW)
#plt.scatter(cycle_results_sim.cycle, cycle_results_sim.mech_power_cycle_avg_kW)

#time', 'ground_tether_reelout_speed', 'ground_tether_force', 'ground_tether_length', 'ground_mech_power', 
# 'x_pos', 'y_pos', 'z_pos', 'flight_phase_index'
error = np.abs(cycle_results_exp.mech_power_cycle_avg_kW-cycle_results_sim.mech_power_cycle_avg_kW)
best_idx = np.argmin(error)
worst_idx = np.argmax(error)


def cycle_to_cycle_plot(df_sim, df_exp,cycle_sim, cycle_exp):
    colors = ["#FF6347",  # Tomato
            "#4682B4",  # SteelBlue
            "#32CD32",  # LimeGreen
            "#FFD700"]  # Gold

    # Create the colormap
    my_cmap = mcolors.ListedColormap(colors)


    ec = '#00395d'
    sc = '#00aeef'

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[0.8, 1], wspace=0.4)

    fig.suptitle('Cycle comparison: ' + str(int(cycle_exp.cycle)))
    # 3D Plot on the Left
    ax3d = fig.add_subplot(gs[0:3, 0], projection='3d')
    ax3d.view_init(elev=35, azim=40)
    for i in range(1,5):
        ax3d.plot(df_sim[df_sim.flight_phase_index==i].x_pos, 
                df_sim[df_sim.flight_phase_index==i].y_pos,
                df_sim[df_sim.flight_phase_index==i].z_pos,
                c = colors[i-1], linewidth = 2)

    x_pos = df_exp.ground_tether_length * np.cos(df_exp.kite_azimuth.to_numpy()) * np.cos(df_exp.kite_elevation.to_numpy())
    y_pos = df_exp.ground_tether_length * np.sin(df_exp.kite_azimuth.to_numpy()) * np.cos(df_exp.kite_elevation.to_numpy())
    z_pos = df_exp.ground_tether_length * np.sin(df_exp.kite_elevation.to_numpy())
        
    for i in range(1,5):
        ax3d.plot(x_pos[df_exp.flight_phase_index==i], 
                y_pos[df_exp.flight_phase_index==i],
                z_pos[df_exp.flight_phase_index==i],
                c = colors[i-1], linewidth = 2, linestyle = '--')
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_title("3D trajectories")
    ax3d.legend(['Reel-out', 'RORI', 'Reel-in', 'RIRO'])


    ax0 = fig.add_subplot(gs[3, 0])
    # Data extraction
    labels = ["RO", "RI", "RIRO", "Cycle"]
    sim_values = [
        cycle_sim["RO_mech_power_avg_kW"]/1000,
        cycle_sim["RI_mech_power_avg_kW"]/1000,
        cycle_sim["RIRO_mech_power_avg_kW"]/1000,
        cycle_sim["mech_power_cycle_avg_kW"]/1000
        
    ]
    exp_values = [
        cycle_exp["RO_mech_power_avg_kW"]/1000,
        cycle_exp["RI_mech_power_avg_kW"]/1000,
        cycle_exp["RIRO_mech_power_avg_kW"]/1000,
        cycle_exp["mech_power_cycle_avg_kW"]/1000
    ]

    # Plot settings
    x = np.arange(len(labels))
    width = 0.35  # width of the bars

    bar1 = ax0.bar(x - width / 2, sim_values, width, label="Simulation", color=sc)
    bar2 = ax0.bar(x + width / 2, exp_values, width, label="Experiment", color=ec)
    xlb, xub = plt.xlim()
    ax0.hlines(0, xlb, xub, colors=['black'], linewidth = 1)
    # Adding labels, title, and formatting
    ax0.set_xlabel("Power Metrics")
    ax0.set_ylabel("Average power (kW)")
    ax0.set_title("Comparison of average mechanical power")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)


    # Subplot 1 (Length)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(df_exp.time - df_exp.time[0], df_exp.ground_tether_length, label="Experiment", c=ec)
    ax1.plot(df_sim.time, df_sim.ground_tether_length, label="Simulation", c=sc)
    ax1.set_title("Tether length")
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_ylabel('Length [m]')


    # Subplot 2 (Reelout Speed)
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(df_exp.time - df_exp.time[0], df_exp.ground_tether_reelout_speed, label="Experiment", c=ec)
    ax2.plot(df_sim.time, df_sim.ground_tether_reelout_speed, label="Simulation", c=sc)
    ax2.set_title("Reelout speed")
    ax2.set_xticks([])
    ax2.set_ylabel('Speed [m/s]')


    # Subplot 3 (Force)
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(df_exp.time - df_exp.time[0], df_exp.ground_tether_force * 9.806 / 1000, label="Experiment", c=ec)
    ax3.plot(df_sim.time, df_sim.ground_tether_force / 1000, label="Simulation", c=sc)
    ax3.set_title("Ground tether force")
    ax3.set_ylabel('Force [kN]')
    ax3.set_xticks([])

    # Subplot 4 (Power)
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.plot(df_exp.time - df_exp.time[0], df_exp.ground_mech_power / 1000, label="Experiment", c=ec)
    ax4.plot(df_sim.time, df_sim.ground_mech_power / 1000, label="Simulation", c=sc)
    ax4.set_title("Mechanical power")
    ax4.set_ylabel('Power [kW]')
    ax4.set_xlabel('Time [s]')


#idx = 13

#cycle_to_cycle_plot(all_sim_dfs[idx], all_exp_dfs[idx], cycle_results_sim.iloc[idx], cycle_results_exp.iloc[idx])

#plt.show()
for idx in range(len(all_sim_dfs)):
    cycle_to_cycle_plot(all_sim_dfs[idx], all_exp_dfs[idx], cycle_results_sim.iloc[idx], cycle_results_exp.iloc[idx])
    plt.show()

