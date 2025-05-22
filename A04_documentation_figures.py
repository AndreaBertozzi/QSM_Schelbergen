import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd 
from qsm import *
from scipy import signal
from utils_exp_validation import *
from plotting_utils import *

data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/2024/'
test_name = 'Test-2024-02-15_GS3/'
#test_name = 'Test-2024-02-16_GS3/'
#test_name = 'Test-2024-02-23_GS3/'
#test_name = 'Test-2024-02-27_GS3/'
#test_name = 'Test-2024-02-29_GS3/'
#test_name = 'Test-2024-03-01_GS3/'

cycle_dfs = load_process_experimental_data(data_path, test_name)

with open("Results_Test-2024-02-15_GS3.pkl", "rb") as file:
    all_sim_dfs, all_exp_dfs,  cycle_results_sim, cycle_results_exp = pickle.load(file)


my_cmap = mcolors.ListedColormap(colors)

df_sim = all_sim_dfs[7]
df_exp = all_exp_dfs[7]


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.plot(df_sim.x_pos[df_sim.flight_phase_index == 1], df_sim.y_pos[df_sim.flight_phase_index == 1], df_sim.z_pos[df_sim.flight_phase_index == 1], c = colors[0], linewidth = 2)
#ax.plot(df_sim.x_pos[df_sim.flight_phase_index == 3], df_sim.y_pos[df_sim.flight_phase_index == 3], df_sim.z_pos[df_sim.flight_phase_index == 3], c = colors[2], linewidth = 2)
#ax.plot(df_sim.x_pos[df_sim.flight_phase_index == 4], df_sim.y_pos[df_sim.flight_phase_index == 4], df_sim.z_pos[df_sim.flight_phase_index == 4], c = colors[3], linewidth = 2)
#ax.legend(['Reel-out', 'Reel-in', 'RIRO'])
#plt.show()

def operational_par_traction(df_sim):
    av_el = 0.610945054945055 
    rel_el = 0.15416666666666665
    max_az = 0.6348644688644689
    L_min = df_sim.ground_tether_length[df_sim.flight_phase_index == 1][0]
    L_max = df_sim.ground_tether_length[df_sim.flight_phase_index == 1].to_numpy()[-1]
    x_plt = np.array([0, 250])
    y_plt = np.tan(av_el)*x_plt
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.sqrt(df_sim.x_pos[df_sim.flight_phase_index == 1]**2 + df_sim.y_pos[df_sim.flight_phase_index == 1]**2),                 
                    df_sim.z_pos[df_sim.flight_phase_index == 1], c = colors[0], linewidth = 2)

    plot_arc(0, av_el+rel_el, L_min)
    plot_arc(0, av_el+rel_el, L_max)
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, av_el, 50)
    plt.text(55, 10, r'$\theta_{avg}$', {'fontsize': 12})
    x_plt = np.array([0, 250])
    y_plt = np.tan(av_el+rel_el)*x_plt
    plot_arc(av_el, av_el+rel_el, 70)
    plot_arc(av_el, av_el+rel_el, 75)
    plt.text(12, 50, r'$\theta_{rel}$', {'fontsize': 12})
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(av_el-rel_el)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.axis('equal')
    plt.xlim([0, L_max + 20])
    plt.hlines(0, 0, L_max + 20, colors=['black'], linestyles=['-'], linewidth=1)
    plt.xticks([0, L_min, L_max], ['GS', r'L$_{min}$', 'L$_{max}$'])
    plt.yticks([], [])
    plt.subplot(1,2,2)
    plt.plot(df_sim.x_pos[df_sim.flight_phase_index == 1], df_sim.y_pos[df_sim.flight_phase_index == 1], c = colors[0], linewidth = 2)
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


df_exp = cycle_dfs[7]

#plt.scatter(df_exp.time - df_exp.time[0], df_exp.ground_tether_length, c=df_exp.flight_phase_index, cmap=my_cmap, s=4)
#plt.show()


#df_exp = actual_flight_phases(df_exp)


"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x_pos = df_exp.ground_tether_length * np.cos(df_exp.kite_azimuth.to_numpy()) * np.cos(df_exp.kite_elevation.to_numpy())
y_pos = df_exp.ground_tether_length * np.sin(df_exp.kite_azimuth.to_numpy()) * np.cos(df_exp.kite_elevation.to_numpy())
z_pos = df_exp.ground_tether_length * np.sin(df_exp.kite_elevation.to_numpy())

ax.plot(x_pos[df_exp.flight_phase_index == 1], y_pos[df_exp.flight_phase_index == 1], z_pos[df_exp.flight_phase_index == 1], c = colors[0], linewidth = 2)
ax.plot(x_pos[df_exp.flight_phase_index == 2], y_pos[df_exp.flight_phase_index == 2], z_pos[df_exp.flight_phase_index == 2], c = colors[1], linewidth = 2)
ax.plot(x_pos[df_exp.flight_phase_index == 3], y_pos[df_exp.flight_phase_index == 3], z_pos[df_exp.flight_phase_index == 3], c = colors[2], linewidth = 2)
ax.plot(x_pos[df_exp.flight_phase_index == 4], y_pos[df_exp.flight_phase_index == 4], z_pos[df_exp.flight_phase_index == 4], c = colors[3], linewidth = 2)
ax.legend(['Reel-out', 'RORI', 'Reel-in', 'RIRO'])
plt.show()
"""

"""
df = df_exp
_, max_tether_length_RO_idx = find_max_RO_tether_length(df)
plt.subplot(2,1,1)
for i in range(1,5):
    plt.plot(df[df_exp.flight_phase_index == i].time - df.time[0], df.ground_tether_length[df_exp.flight_phase_index == i], c=colors[i-1], linewidth=2,  zorder=1)
y_lb, y_ub = plt.ylim()
plt.vlines(df.time[max_tether_length_RO_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RO')
plt.scatter(df.time[max_tether_length_RO_idx] - df.time[0], df.ground_tether_length[max_tether_length_RO_idx], c='#d62828', label='Max tether length',zorder=2)
plt.ylabel('Tether length [m]')
plt.ylim([y_lb, y_ub])
plt.legend()
plt.subplot(2,1,2)
plt.scatter(df.time[max_tether_length_RO_idx] - df.time[0], df.kite_actual_depower[max_tether_length_RO_idx], c='#d62828', label='Start depowering', zorder=3)
for i in range(1,5):
    plt.plot(df[df_exp.flight_phase_index == i].time - df.time[0], df[df_exp.flight_phase_index == i].kite_actual_depower, c=colors[i-1], linewidth=2,  zorder=1)
y_lb, y_ub = plt.ylim()
plt.vlines(df.time[max_tether_length_RO_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RO')
plt.xlabel('Time [s]')
plt.hlines(df.kite_actual_depower[max_tether_length_RO_idx], 0, (df.time.to_numpy()[-1]-df.time[0]), color='#a9a29c', linestyle=':', label=r'2 % threshold')
plt.ylabel('Kite depower [%]')
plt.ylim([y_lb, y_ub])
plt.legend()

plt.figure()
df = df_exp
_, min_tether_length_RI_idx = find_min_RI_tether_length(df)
plt.subplot(2,1,1)
for i in range(1,5):
    plt.plot(df[df_exp.flight_phase_index == i].time - df.time[0], df.ground_tether_length[df_exp.flight_phase_index == i], c=colors[i-1], linewidth=2,  zorder=1)
y_lb, y_ub = plt.ylim()
plt.vlines(df.time[min_tether_length_RI_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RI')
plt.scatter(df.time[min_tether_length_RI_idx] - df.time[0], df.ground_tether_length[min_tether_length_RI_idx], c='#d62828', label='Min tether length',zorder=2)
plt.ylabel('Tether length [m]')
plt.ylim([y_lb, y_ub])
plt.legend()
plt.subplot(2,1,2)
plt.scatter(df.time[min_tether_length_RI_idx] - df.time[0], df.kite_actual_depower[min_tether_length_RI_idx], c='#d62828', label='End powering',  zorder=3)
for i in range(1,5):
    plt.plot(df[df_exp.flight_phase_index == i].time - df.time[0], df[df_exp.flight_phase_index == i].kite_actual_depower, c=colors[i-1], linewidth=2,  zorder=1)
y_lb, y_ub = plt.ylim()
plt.vlines(df.time[min_tether_length_RI_idx] - df.time[0], y_lb, y_ub, color='#a9a29c', linestyle='--', label='Actual end of RI')
plt.xlabel('Time [s]')
plt.hlines(df.kite_actual_depower[min_tether_length_RI_idx], 0, (df.time.to_numpy()[-1]-df.time[0]), color='#a9a29c', linestyle=':', label=r'2 % threshold')
plt.ylabel('Kite depower [%]')
plt.ylim([y_lb, y_ub])
plt.legend()

plt.show()
"""


"""
df_exp = cycle_dfs[7]
df = df_exp.copy()
df = actual_flight_phases(df)
df = df[df.flight_phase_index == 1]

plt.subplot(1,2,1)
plt.plot(df.time - df.time[0], np.rad2deg(np.abs(df.kite_azimuth)), c=colors[0], linewidth = 2, zorder = 1)
peaks_idx,_ = signal.find_peaks(np.abs(df.kite_azimuth.to_numpy()), distance = 10)
plt.scatter(df.time.to_numpy()[peaks_idx] - df.time[0], np.rad2deg(np.abs(df.kite_azimuth))[peaks_idx], c='grey', zorder = 2)
tlb, tup = plt.xlim()
plt.hlines(np.mean(np.rad2deg(np.abs(df.kite_azimuth))[peaks_idx]), tlb, tup, ['grey'], ['--'], label=r'$|\phi_{max}|$', linewidth=1)
plt.text(22, np.mean(np.rad2deg(np.abs(df.kite_azimuth))[peaks_idx])+0.4, r'$\phi_{max}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$|\phi|$ [deg]')

plt.subplot(1,2,2)
plt.plot(df.time - df.time[0], np.rad2deg((df.kite_elevation)), c=colors[0], linewidth = 2, zorder = 1)
peaks_idx = extract_complete_peaks(df.kite_elevation)
valleys_idx = extract_complete_peaks(-df.kite_elevation)
plt.scatter(df.time.to_numpy()[peaks_idx] - df.time[0], np.rad2deg(df.kite_elevation)[peaks_idx], c='grey', zorder = 2)
plt.scatter(df.time.to_numpy()[valleys_idx] - df.time[0], np.rad2deg(df.kite_elevation)[valleys_idx], c='grey', zorder = 2)

avg_el_peak = np.mean(df.kite_elevation[peaks_idx])
avg_el_valley = np.mean(df.kite_elevation[valleys_idx])
rel_el_angle = 0.5*(avg_el_peak - avg_el_valley) 
avg_el_angle = 0.5*(avg_el_peak + avg_el_valley)

plt.hlines([np.rad2deg(avg_el_angle), 
            np.rad2deg(avg_el_angle + rel_el_angle),
            np.rad2deg(avg_el_angle - rel_el_angle)], tlb, tup, ['grey', 'grey', 'grey'], ['--', ':', ':'], linewidth=1)

plt.text(0, np.rad2deg(avg_el_angle)+0.4, r'$\theta_{avg}$')
plt.text(20, np.rad2deg(avg_el_angle+rel_el_angle)+0.4, r'$\theta_{avg} + \theta_{rel}$')
plt.text(0, np.rad2deg(avg_el_angle-rel_el_angle)+0.4, r'$\theta_{avg} - \theta_{rel}$')
plt.xlabel('Time [s]')
plt.ylabel(r'$\theta$ [deg]')

plt.show()

"""


def plot_operational_par_traction(df_sim):
    av_el = 0.610945054945055 
    rel_el = 0.15416666666666665
    max_az = 0.6348644688644689
    L_min = df_sim.ground_tether_length[df_sim.flight_phase_index == 1][0]
    L_max = df_sim.ground_tether_length[df_sim.flight_phase_index == 1].to_numpy()[-1]
    x_plt = np.array([0, 250])
    y_plt = np.tan(av_el)*x_plt
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.plot(np.sqrt(df_sim.x_pos[df_sim.flight_phase_index == 1]**2 + df_sim.y_pos[df_sim.flight_phase_index == 1]**2),                 
                    df_sim.z_pos[df_sim.flight_phase_index == 1], c = colors[0], linewidth = 2)

    plot_arc(0, av_el+rel_el, L_min)
    plot_arc(0, av_el+rel_el, L_max)
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plot_arc(0, av_el, 50)
    plt.text(55, 10, r'$\theta_{avg}$', {'fontsize': 12})
    x_plt = np.array([0, 250])
    y_plt = np.tan(av_el+rel_el)*x_plt
    plot_arc(av_el, av_el+rel_el, 70)
    plot_arc(av_el, av_el+rel_el, 75)
    plt.text(12, 50, r'$\theta_{rel}$', {'fontsize': 12})
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    x_plt = np.array([0, 250])
    y_plt = np.tan(av_el-rel_el)*x_plt
    plt.plot(x_plt, y_plt, c='black', linestyle = '--', linewidth = 1)
    plt.axis('equal')
    plt.xlim([0, L_max + 20])
    plt.hlines(0, 0, L_max + 20, colors=['black'], linestyles=['-'], linewidth=1)
    plt.xticks([0, L_min, L_max], ['GS', r'L$_{min}$', 'L$_{max}$'])
    plt.yticks([], [])

    plt.subplot(1,2,2)
    plt.plot(df_sim.x_pos[df_sim.flight_phase_index == 1], df_sim.y_pos[df_sim.flight_phase_index == 1], c = colors[0], linewidth = 2)
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


plot_pattern_param(df_exp) 




plt.show()