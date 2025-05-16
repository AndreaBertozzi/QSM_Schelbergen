import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd 
from qsm import *
from scipy import signal
def find_max_RO_tether_length(df):
    # Find the maximum ground tether length during the RO phase
    avg_ro_depwr = np.mean(df.kite_actual_depower[df.flight_phase_index==1])
    dep_idx = (df.kite_actual_depower -  avg_ro_depwr) > 2
    start_dep_idx = next((i for i, x in enumerate(dep_idx) if x != 0), -1)
    max_tether_length_RO = df.ground_tether_length[start_dep_idx]
    return max_tether_length_RO, start_dep_idx

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

def find_RO_pattern_param(df_RO):

    peaks_idx, _ = signal.find_peaks(np.abs(df_RO.kite_azimuth-np.mean(df_RO.kite_azimuth)), prominence = 0.04)    
    max_az_trac = np.mean(np.abs(df_RO.kite_azimuth - np.mean(df_RO.kite_azimuth))[peaks_idx]) 

    peaks_idx_el, _ = signal.find_peaks(df_RO.kite_elevation, prominence = 0.04, distance=10)
    valleys_idx_el, _ = signal.find_peaks(-df_RO.kite_elevation, prominence = 0.04, distance=10)

    #plt.plot(df_RO.time, df_RO.kite_elevation)
    #t_lb, t_up = plt.xlim()
    #plt.scatter(df_RO.time[peaks_idx_el], df_RO.kite_elevation[peaks_idx_el])
    #plt.scatter(df_RO.time[valleys_idx_el], df_RO.kite_elevation[valleys_idx_el])
    #plt.hlines(np.mean(df_RO.kite_elevation), t_lb, t_up)
    #plt.show()

    avg_el_peak = np.mean(df_RO.kite_elevation[peaks_idx_el])
    avg_el_valley = np.mean(df_RO.kite_elevation[valleys_idx_el])
    rel_el_angle = 0.5*(avg_el_peak - avg_el_valley) if not np.isnan(avg_el_peak) and not np.isnan(avg_el_valley) else np.nan
    
    avg_el_angle = np.mean(df_RO.kite_elevation)

    return max_az_trac, rel_el_angle, avg_el_angle

def plot_pattern_param(df):
    x_pos = df.kite_distance * np.cos(df.kite_azimuth.to_numpy()) * np.cos(df.kite_elevation.to_numpy())
    y_pos = df.kite_distance * np.sin(df.kite_azimuth.to_numpy()) * np.cos(df.kite_elevation.to_numpy())
    z_pos = df.kite_distance * np.sin(df.kite_elevation.to_numpy())

    x_pos = x_pos[df.flight_phase_index == 1]
    y_pos = y_pos[df.flight_phase_index == 1]
    z_pos = z_pos[df.flight_phase_index == 1]


    max_az_trac, rel_el_angle, avg_el_angle = find_RO_pattern_param(df[df.flight_phase_index == 1])
    plt.subplot(1,2,1)                              
    plt.scatter(x_pos, y_pos, c = 'black', s = 4)
    x_plot = np.array([0, np.max(x_pos)])
    m_line = np.tan(max_az_trac)
    y_plot1 = m_line*x_plot
    y_plot2 = -m_line*x_plot
    plt.plot(x_plot, y_plot1, x_plot, y_plot2, c = 'grey', linestyle = ':')

    plt.subplot(1,2,2)
    plt.scatter(x_pos, z_pos, c = 'black', s = 4)
    x_plot = np.array([0, np.max(x_pos)])
    m_line = np.tan(avg_el_angle)
    y_plot1 = m_line*x_plot
    m_line = np.tan(avg_el_angle + rel_el_angle)
    y_plot2 = m_line*x_plot
    m_line = np.tan(avg_el_angle - rel_el_angle)
    y_plot3 = m_line*x_plot
    plt.plot(x_plot, y_plot1, x_plot, y_plot2, x_plot, y_plot3, c = 'grey', linestyle = ':')
    plt.ylim([0, np.max(x_pos)])

def find_min_RI_tether_length(df):
    avg_riro_depwr = np.mean(df.kite_actual_depower[df.flight_phase_index == 4].iloc[-5:-1])
    dep_idx = (df.kite_actual_depower -  avg_riro_depwr) > 2
    dep_idx = dep_idx.iloc[::-1]
    start_pow_idx = len(dep_idx) - next((i for i, x in enumerate(dep_idx) if x != 0), -1)
    min_tether_length_RI = df.ground_tether_length[start_pow_idx]
    return min_tether_length_RI, start_pow_idx

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

def actual_flight_phases(df):
    # Change the flight_phase_index of elements of RORI to RO
    max_tether_length_RO, max_tether_length_RO_idx = find_max_RO_tether_length(df)
    df.loc[0:max_tether_length_RO_idx, 'flight_phase_index'] = 1

    # Change the flight_phase_index of elements of RIRO to RI
    min_tether_length_RO, min_tether_length_RO_idx = find_min_RI_tether_length(df)
    start_RIRO_idx = next((i for i, x in enumerate(df.flight_phase_index) if x == 4), -1)
    df.loc[start_RIRO_idx:min_tether_length_RO_idx, 'flight_phase_index'] = 3
    return df

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

def group_cycles(df):
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



exp_data = pickle.load(open('exp_data_processed.pkl', 'rb'))
cycle_dfs = group_cycles(exp_data)



data_path = 'C:/Users/andre/OneDrive - Delft University of Technology/Andrea_Kitepower/Experimental_data/2024/'
test_name = 'Test-2024-01-30_GS3/'
test_name = 'Test-2024-02-15_GS3/'


with open("all_results.pkl", "rb") as file:
    all_sim_dfs, all_exp_dfs,  cycle_results_sim, cycle_results_exp = pickle.load(file)


def plot_arc(angle0, angle1, radius):
    a_range = np.linspace(angle0, angle1, 30)
    x_cor = radius*np.cos(a_range)
    y_cor = radius*np.sin(a_range)
    plt.plot(x_cor, y_cor, linewidth=1, color='black')
#plt.scatter(cycle_results_exp.wind_speed_100m, cycle_results_exp.mech_power_cycle_avg_kW)
#plt.scatter(cycle_results_sim.wind_speed_100m, cycle_results_sim.mech_power_cycle_avg_kW)

#plt.show()

colors = ["#FF6347",  # Tomato
          "#4682B4",  # SteelBlue
          "#32CD32",  # LimeGreen
          "#FFD700"]  # Gold

# Create the colormap
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



plot_pattern_param(df_exp) 




plt.show()