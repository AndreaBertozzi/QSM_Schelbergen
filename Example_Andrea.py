## EXAMPLE QUASI STEADY MODEL

# Import required packages.
from qsm import Environment, SystemProperties, Cycle
import matplotlib.pyplot as plt
import numpy as np

# Configure the default plotting settings
size=13
params = {'legend.fontsize': 'large',
          'figure.figsize': (8,5),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.85,
          'ytick.labelsize': size*0.85,
          'axes.titlepad': 25}
plt.rcParams.update(params)


# Create an object setting the wind conditions.
env_state = {
    'wind_speed': 5.5,  # [m/s]
    'air_density': 1.225,  # [kg/m^3]
}
env_state = Environment(**env_state)

sys_prop = SystemProperties({})

cycle = Cycle()

sim = cycle.run_simulation(sys_prop, env_state, print_summary=True)
cycle.time_plot(('tether_length', 'power_ground'), ('Tether length [m]', 'Power [W]'), fig_num=None)

#cycle.time_plot(('reeling_speed', 'power_ground', 'tether_force_ground'),
#                ('Reeling speed [m/s]', 'Power [W]', 'Tether force [N]'), fig_num=None)
plt.show()



