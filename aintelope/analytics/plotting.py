from typing import Optional

import dateutil.parser as dparser
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt

"""
Create and return plots for various analytics.
"""


def plot_history(events):
    """
    Plot the events from a history.
    args:
        events: pandas DataFrame
    return:
        plot: matplotlib.axes.Axes
    """
    plot = "NYI"

    return plot


def plot_groupby(all_events, group_keys, score_dimensions):
    keys = group_keys + ["Reward"] + score_dimensions
    data = pd.DataFrame(columns=keys)
    for events in all_events:
        if len(data) == 0:
            data = events[
                keys
            ].copy()  # needed to avoid Pandas complaining about empty dataframe
        else:
            data = pd.concat([data, events[keys]])

    data["Reward"] = data["Reward"].astype(float)
    data[score_dimensions] = data[score_dimensions].astype(float)
    data["Score"] = data[score_dimensions].sum(axis=1)

    plot_data = data.groupby(group_keys).mean()

    return plot_data


def plot_performance(all_events, score_dimensions, save_path: Optional[str]):
    """
    Plot performance between rewards and scores.
    Accepts a list of event records from which a boxplot is done.
    TODO: further consideration should be had on *what* to average over.
    """
    plot_data1 = (
        "Episode",
        plot_groupby(all_events, ["Run_id", "Episode", "Agent_id"], score_dimensions),
    )
    plot_data2 = (
        "Step",
        plot_groupby(all_events, ["Run_id", "Step", "Agent_id"], score_dimensions),
    )
    plot_datas = [plot_data1, plot_data2]

    # fig = plt.figure()
    fig, subplots = plt.subplots(2)

    for index, subplot in enumerate(subplots):
        (plot_label, plot_data) = plot_datas[index]

        subplot.plot(plot_data["Reward"].to_numpy(), label="Reward")
        subplot.plot(plot_data["Score"].to_numpy(), label="Score")
        for score_dimension in score_dimensions:
            subplot.plot(plot_data[score_dimension].to_numpy(), label=score_dimension)

        subplot.set_title("By " + plot_label)
        subplot.set(xlabel=plot_label, ylabel="Mean Reward")
        subplot.legend()

    if save_path:
        save_plot(fig, save_path)

    # enable this code if you want the plot to open automatically
    # plt.ion()
    # fig.show()
    # plt.draw()
    # plt.pause(0.1)
    # input("Press [enter] to continue.")

    return fig


def plot_heatmap(agent, env):
    """
    Plot how the agent sees the values in an environment.
    """
    plot = "NYI"
    return plot


def save_plot(plot, save_path):
    """
    Save plot to file. Will get deprecated if nothing else comes here.
    """
    plot.savefig(save_path)
