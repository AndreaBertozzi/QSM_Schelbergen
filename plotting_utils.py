import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from utils_exp_validation import *

colors = ["#FF6347",  # Tomato
          "#4682B4",  # SteelBlue
          "#32CD32",  # LimeGreen
          "#FFD700"]  # Gold

# Create the colormap
my_cmap = mcolors.ListedColormap(colors)

def plot_max_tether_length_RO(df):
    _, max_tether_length_RO_idx = find_max_RO_tether_length(df)

    plt.subplot(2,1,1)
    plt.plot(df.time - df.time[0], df.ground_tether_length, c='#274c77', linewidth=2)
    y_lb, y_ub = plt.ylim()
    plt.vlines(df.time[max_tether_length_RO_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RO')
    plt.scatter(df.time[max_tether_length_RO_idx] - df.time[0], df.ground_tether_length[max_tether_length_RO_idx], c='#d62828', label='Max tether length')
    plt.ylabel('Ground tether length [m]')
    plt.ylim([y_lb, y_ub])
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(df.time - df.time[0], df.kite_actual_depower, c='#274c77', linewidth=2)
    y_lb, y_ub = plt.ylim()
    plt.vlines(df.time[max_tether_length_RO_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RO')
    plt.xlabel('Time [s]')
    plt.scatter(df.time[max_tether_length_RO_idx] - df.time[0], df.kite_actual_depower[max_tether_length_RO_idx], c='#d62828', label='Start depowering')
    plt.ylabel('Kite depower [%]')
    plt.ylim([y_lb, y_ub])
    plt.legend()
    _, max_tether_length_RO_idx = find_max_RO_tether_length(df)

    plt.subplot(2,1,1)
    plt.plot(df.time - df.time[0], df.ground_tether_length, c='#274c77', linewidth=2)
    y_lb, y_ub = plt.ylim()
    plt.vlines(df.time[max_tether_length_RO_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RO')
    plt.scatter(df.time[max_tether_length_RO_idx] - df.time[0], df.ground_tether_length[max_tether_length_RO_idx], c='#d62828', label='Max tether length')
    plt.ylabel('Ground tether length [m]')
    plt.ylim([y_lb, y_ub])
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(df.time - df.time[0], df.kite_actual_depower, c='#274c77', linewidth=2)
    y_lb, y_ub = plt.ylim()
    plt.vlines(df.time[max_tether_length_RO_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RO')
    plt.xlabel('Time [s]')
    plt.scatter(df.time[max_tether_length_RO_idx] - df.time[0], df.kite_actual_depower[max_tether_length_RO_idx], c='#d62828', label='Start depowering')
    plt.ylabel('Kite depower [%]')
    plt.ylim([y_lb, y_ub])
    plt.legend()

def plot_pattern_param(df):
    x_pos = df.ground_tether_length * np.cos(df.kite_azimuth.to_numpy()) * np.cos(df.kite_elevation.to_numpy())
    y_pos = df.ground_tether_length * np.sin(df.kite_azimuth.to_numpy()) * np.cos(df.kite_elevation.to_numpy())
    z_pos = df.ground_tether_length * np.sin(df.kite_elevation.to_numpy())

    x_pos = x_pos[df.flight_phase_index == 1]
    y_pos = y_pos[df.flight_phase_index == 1]
    z_pos = z_pos[df.flight_phase_index == 1]
    L_min, _ = find_min_RI_tether_length(df)
    L_max, _ = find_max_RO_tether_length(df)
    
    max_az, rel_el_angle, avg_el_angle = find_RO_pattern_param(df[df.flight_phase_index == 1])
    plt.subplot(1,2,1) 
    plt.plot(np.sqrt(x_pos**2 + y_pos**2), z_pos, c =colors[0],linewidth=2)
    
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_el_angle)*x_plt                          
    plot_arc(0, avg_el_angle + rel_el_angle, L_min)
    plot_arc(0, avg_el_angle + rel_el_angle, L_max)
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, avg_el_angle, 50)
    plt.text(55, 10, r'$\theta_{avg}$', {'fontsize': 12})
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_el_angle+rel_el_angle)*x_plt
    plot_arc(avg_el_angle, avg_el_angle+rel_el_angle, 70)
    plot_arc(avg_el_angle, avg_el_angle+rel_el_angle, 75)
    plt.text(12, 50, r'$\theta_{rel}$', {'fontsize': 12})
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(avg_el_angle-rel_el_angle)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.axis('equal')
    plt.xlim([0, L_max + 20])
    plt.hlines(0, 0, L_max + 20, colors=['black'], linestyles=['-'], linewidth=1)
    plt.xticks([0, L_min, L_max], ['GS', r'L$_{min}$', 'L$_{max}$'])
    plt.yticks([], [])

    plt.subplot(1,2,2)
    plt.plot(x_pos, y_pos, c =colors[0],linewidth=2)   
    plt.hlines(0, 0, 250, colors=['black'], linestyles=['-'], linewidth=1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(max_az)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.plot(x_plt, -y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, max_az, 50)
    plt.text(55, 10, r'$\phi_{max}$', {'fontsize': 12})
    plt.axis('equal')
    plt.xlim([0, 250])
    plt.xticks([], [])
    plt.yticks([], [])

def plot_min_tether_length_RI(df):
    _, min_tether_length_RI_idx = find_min_RI_tether_length(df)

    plt.subplot(2,1,1)
    plt.plot(df.time - df.time[0], df.ground_tether_length, c='#274c77', linewidth=2)
    y_lb, y_ub = plt.ylim()
    plt.vlines(df.time[min_tether_length_RI_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RI')
    plt.scatter(df.time[min_tether_length_RI_idx] - df.time[0], df.ground_tether_length[min_tether_length_RI_idx], c='#d62828', label='Min tether length')
    plt.ylabel('Ground tether length [m]')
    plt.ylim([y_lb, y_ub])
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(df.time - df.time[0], df.kite_actual_depower, c='#274c77', linewidth=2)
    y_lb, y_ub = plt.ylim()
    plt.vlines(df.time[min_tether_length_RI_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RI')
    plt.xlabel('Time [s]')
    plt.scatter(df.time[min_tether_length_RI_idx] - df.time[0], df.kite_actual_depower[min_tether_length_RI_idx], c='#d62828', label='Kite powered')
    plt.ylabel('Kite depower [%]')
    plt.ylim([y_lb, y_ub])
    plt.legend()

def plot_reeling_speeds(df):
    phase_idxs = [1,3,4]
    titles = ['RO', 'RI', 'RIRO']
    for p, idx in enumerate(phase_idxs):
        plt.subplot(3,1,p+1)
        plt.plot(df[df.flight_phase_index == idx].time - df.time[0], df[df.flight_phase_index == idx].ground_tether_reelout_speed, c = colors[idx-1], linewidth = 2)
        t_lb, t_ub = plt.xlim()
        plt.title(titles[p])
        plt.ylabel(r'$s_{reel}$ [m/s]')
        plt.hlines(np.mean(df[df.flight_phase_index == idx].ground_tether_reelout_speed), t_lb, t_ub, colors='grey', linestyles = ':')
        if p+1 == 3: plt.xlabel('Time [s]')
        plt.tight_layout()

def cycle_to_cycle_plot(df_sim, df_exp,cycle_sim, cycle_exp):
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

def plot_arc(angle0, angle1, radius):
    a_range = np.linspace(angle0, angle1, 30)
    x_cor = radius*np.cos(a_range)
    y_cor = radius*np.sin(a_range)
    plt.plot(x_cor, y_cor, linewidth=1, color='black')
####################################################################