# -*- coding: utf-8 -*-
"""Utility functions."""

import matplotlib.pyplot as plt
from numpy import all, diff, array

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def flatten_dict(input_dict, parent_key='', sep='.'):
    """Recursive function to convert multi-level dictionary to flat dictionary.

    Args:
        input_dict (dict): Dictionary to be flattened.
        parent_key (str): Key under which `input_dict` is stored in the higher-level dictionary.
        sep (str): Separator used for joining together the keys pointing to the lower-level object.

    """
    items = []  # list for gathering resulting key, value pairs
    for k, v in input_dict.items():
        new_key = parent_key + sep + k.replace(" ", "") if parent_key else k.replace(" ", "")
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def zip_el(*args):
    """"Zip iterables, only if input lists have same length.

    Args:
        *args: Variable number of lists.

    Returns:
        list: Iterator that aggregates elements from each of the input lists.

    Raises:
        AssertError: If input lists do not have the same length.

    """
    lengths = [len(l) for l in [*args]]
    assert all(diff(lengths) == 0), "All the input lists should have the same length."
    return zip(*args)

def plot_traces(x, data_sources, source_labels, plot_parameters, y_labels=None, y_scaling=None, fig_num=None,
                plot_kwargs={}, plot_markers=None, x_label='Time [s]'):
    """Plot the time trace of a parameter from multiple sources.

    Args:
        x (tuple): Sequence of points along x.
        data_sources (tuple): Sequence of time traces of the different data sources.
        source_labels (tuple): Labels corresponding to the data sources.
        plot_parameters (tuple): Sequence of attributes/keys of the objects/dictionaries of the time traces.
        y_labels (tuple, optional): Y-axis labels corresponding to `plot_parameters`.
        y_scaling (tuple, optional): Scaling factors corresponding to `plot_parameters`.
        fig_num (int, optional): Number of figure used for the plot, if None a new figure is created.
        plot_kwargs (dict, optional): Line plot keyword arguments.

    """
    if y_labels is None:
        y_labels = plot_parameters
    if y_scaling is None:
        y_scaling = [None for _ in range(len(plot_parameters))]
    if fig_num:
        axes = plt.figure(fig_num).get_axes()
    else:
        axes = []
    if not axes:
        _, axes = plt.subplots(len(plot_parameters), 1, sharex=True, num=fig_num)
    if len(axes) == 1:
        axes = (axes,)

    for p, y_lbl, f, ax in zip_el(plot_parameters, y_labels, y_scaling, axes):
        for trace, s_lbl in zip_el(data_sources, source_labels):
            y = None
            #TODO: see if it is a better option to make this function a method and check if p is an attribute as condition
            if p == s_lbl:
                y = trace
            elif isinstance(trace[0], dict):
                if p in trace[0]:
                    y = [item[p] for item in trace]
            elif hasattr(trace[0], p):
                y = [getattr(item, p) for item in trace]
            if y:
                if f:
                    y = array(y)*f
                ax.plot(x, y, label=s_lbl, **plot_kwargs)
                if plot_markers:
                    marker_vals = [y[x.index(t)] for t in plot_markers]
                    ax.plot(plot_markers, marker_vals, 's', markerfacecolor='None')
        ax.set_ylabel(y_lbl)
        ax.grid(True)
        # ax.legend()
    axes[-1].set_xlabel(x_label)
    axes[-1].set_xlim([0, None])

def load_config(file_path):
    """Function to read config.yaml file.
    
    Args:
        file_path (string): The path of the .yaml configuration file.
    """

    
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    kite = config["kite"]
    tether = config["tether"]
    bounds = config["bounds"]

    params_dict = {
        # Kite
        "kite_mass": kite["mass"],
        "kite_projected_area": kite["projected_area"],
        "kite_drag_coefficient_powered": kite["drag_coefficient"]["powered"],
        "kite_drag_coefficient_depowered": kite["drag_coefficient"]["depowered"],
        "kite_lift_coefficient_powered": kite["lift_coefficient"]["powered"],
        "kite_lift_coefficient_depowered": kite["lift_coefficient"]["depowered"],

        # Tether
        "total_tether_length": tether["length"],
        "tether_diameter": tether["diameter"],
        "tether_density": tether["density"],
        "tether_drag_coefficient": tether["drag_coefficient"],

        # Bounds
        "avg_elevation_min_limit": bounds["avg_elevation"]["min"]*pi/180,
        "avg_elevation_max_limit": bounds["avg_elevation"]["max"]*pi/180,
        "max_azimuth_min_limit": bounds["max_azimuth"]["min"]*pi/180,
        "max_azimuth_max_limit": bounds["max_azimuth"]["max"]*pi/180,
        "rel_elevation_min_limit": bounds["relative_elevation"]["min"]*pi/180,
        "rel_elevation_max_limit": bounds["relative_elevation"]["max"]*pi/180,
        "reeling_speed_min_limit": bounds["speed_limits"]["min"],
        "reeling_speed_max_limit": bounds["speed_limits"]["max"],
        "tether_force_min_limit": bounds["force_limits"]["min"]*9.806,
        "tether_force_max_limit": bounds["force_limits"]["max"]*9.806,
        "tether_stroke_min_limit": bounds["tether_stroke"]["min"],
        "tether_stroke_max_limit": bounds["tether_stroke"]["max"],
        "min_tether_length_min_limit": bounds["minimum_tether_length"]["min"],
        "min_tether_length_max_limit": bounds["minimum_tether_length"]["max"],        
    }

    return params_dict

    