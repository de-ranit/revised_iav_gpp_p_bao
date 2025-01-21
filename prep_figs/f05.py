#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot scatter plot of observed vs simulated annual GPP before and
after using moisture stress in P Model. The experiment is
site year optimization using P Model.

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Thu Feb 01 2024 12:06:47 CET
"""

import os
from pathlib import Path
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib
from permetrics import RegressionMetric
import seaborn as sns

# import ipdb

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = (
    r"\renewcommand{\familydefault}{\sfdefault}"  # use sans-serif font
)
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# set the font to computer modern sans
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "cmss10"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def get_ann_gpp_ai(p_mod_res_path):
    """
    get the annual observed and simulated GPP (with and without
    using moisture stress) and aridity index for each site

    Parameters:
    -----------
    p_mod_res_path (Path object) : path to the serialized model results

    Returns:
    --------
    gpp_obs_yr_arr (np.array) : array of observed annual GPP from
    all the good years of all the sites
    gpp_sim_yr_arr (np.array) : array of simulated annual GPP from
    all the good years of all the sites
    gpp_sim_no_moisture_stress_yr_arr (np.array) : array of simulated annual GPP
    without using moisture stress from all the good years of all the sites
    ai_arr (np.array) : array of aridity index for each site
    """

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{p_mod_res_path}/*.npy")
    mod_res_file_list.sort()  # sort the files by site ID

    # filter out bad sites
    filtered_mod_res_file_list = [
        files
        for files in mod_res_file_list
        if not (
            "CG-Tch" in files
            or "MY-PSO" in files
            or "GH-Ank" in files
            or "US-LWW" in files
        )
    ]

    # empty lists to store the results
    gpp_obs_yr = []
    gpp_sim_yr = []
    gpp_sim_no_moisture_stress_yr = []
    ai_list = []

    # for each site
    for res_file in filtered_mod_res_file_list:
        # open the results file
        res_dict = np.load(res_file, allow_pickle=True).item()

        # store the results in lists
        gpp_obs_yr.append(res_dict["GPP_NT_yearly_filtered"])
        gpp_sim_yr.append(res_dict["GPP_sim_yearly_filtered"])

        good_yr_mask = np.where(res_dict["good_gpp_y_idx"] == 1.0, True, False)
        gpp_sim_no_moisture_stress_yr.append(
            res_dict["GPP_sim_no_moisture_yearly"][good_yr_mask]
        )

        no_of_good_yr = len(res_dict["GPP_NT_yearly_filtered"])
        ai_list.append(np.full(no_of_good_yr, res_dict["arid_ind"]))

    # concatenate the results of all the sites
    gpp_obs_yr_arr = np.concatenate(gpp_obs_yr)
    gpp_sim_yr_arr = np.concatenate(gpp_sim_yr)
    gpp_sim_no_moisture_stress_yr_arr = np.concatenate(gpp_sim_no_moisture_stress_yr)
    ai_arr = np.concatenate(ai_list)

    return gpp_obs_yr_arr, gpp_sim_yr_arr, gpp_sim_no_moisture_stress_yr_arr, ai_arr


def plot_axs(x, y, z, axs, lim, title):
    """
    plot each subplot

    Parameters:
    -----------
    x (np.array) : array of observed GPP
    y (np.array) : array of simulated GPP
    z (np.array) : array of aridity index
    axs (matplotlib.axes) : axes object
    lim (tuple) : tuple of limits for x and y axes
    title (str) : title of the subplot
    """

    # calculate NSE and R2 between observed and simulated GPP
    evaluator_sd = RegressionMetric(x, y, decimal=3)
    nse = evaluator_sd.NSE()
    r2 = (evaluator_sd.PCC()) ** 2

    # colors to use to construct the colormap
    colors = [
        "#CD7461",
        "#DC904F",
        "#EBBD35",
        "#FEE127",
        "#CFDC2E",
        "#71C047",
        "#58BB61",
        "#44AE9E",
        "#4388A9",
        "#4388A9",
    ]

    # create colormap from the list of colors with a smooth gradient
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors, N=100)

    # map aridity index to colors
    c_colors = np.digitize(z, np.linspace(0, 2, len(colors))) - 1

    # plot the scatter plot
    scatter = axs.scatter(x, y, c=c_colors, cmap=cmap)

    # customize the axes
    axs.set_xlim(lim)
    axs.set_ylim(lim)
    axs.tick_params(axis="both", which="major", labelsize=22.0)
    axs.set_title(f"{title}\nNSE: {nse}, R$^2$: {round(r2,3)}", fontsize=26)

    # add a 1:1 line and a fitted regression line
    axs.plot(np.array([lim[0], lim[1]]), np.array([lim[0], lim[1]]), color="black")

    coeff = np.polyfit(x, y, 1)
    fit_line = coeff[0] * x + coeff[1]

    axs.plot(x, fit_line, color="red", linestyle="--", dashes=(1.6, 4))
    if round(coeff[1],3) < 0:
        axs.text(1, 10.6, f"y = {round(coeff[0],3)}x - {abs(round(coeff[1],3))}", fontsize=22)
    else:
        axs.text(1, 10.6, f"y = {round(coeff[0],3)}x + {round(coeff[1],3)}", fontsize=22)

    sns.despine(ax=axs, top=True, right=True)

    return scatter


def plot_fig_main(p_mod_res_path):
    """
    plot the figure with the scatter plot of observed vs simulated GPP

    Parameters:
    -----------
    p_mod_res_path (Path object) : path to the serialized model results

    Returns:
    --------
    None
    """

    # get the annual observed and simulated GPP (with and without using moisture stress)
    (
        gpp_obs_yr_arr,
        gpp_sim_yr_arr,
        gpp_sim_no_moisture_stress_yr_arr,
        ai_arr,
    ) = get_ann_gpp_ai(p_mod_res_path)

    # get the max and min gpp to set equal limits for x and y axes
    max_gpp = max(gpp_obs_yr_arr.max(), gpp_sim_yr_arr.max())
    max_gpp_no_moisture = max(
        gpp_obs_yr_arr.max(), gpp_sim_no_moisture_stress_yr_arr.max()
    )
    min_gpp = min(gpp_obs_yr_arr.min(), gpp_sim_yr_arr.min())
    min_gpp_no_moisture = min(
        gpp_obs_yr_arr.min(), gpp_sim_no_moisture_stress_yr_arr.min()
    )

    fig_width = 15
    # fig_height = fig_width * (
    #     9 / 16
    # )  # Height in inches to maintain a 16:9 aspect ratio
    fig_height = 6

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=2, nrows=1, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    # add the scatter plots
    scatter = plot_axs(
        gpp_sim_no_moisture_stress_yr_arr,
        gpp_obs_yr_arr,
        ai_arr,
        axs[0],
        (min_gpp_no_moisture, max_gpp_no_moisture),
        r"(a) P$_{\text{hr}}$ model",
    )
    plot_axs(
        gpp_sim_yr_arr,
        gpp_obs_yr_arr,
        ai_arr,
        axs[1],
        (min_gpp, max_gpp),
        r"(b) P$^{\text{W}}_{\text{hr}}$ model",
    )

    # add the colorbar
    cbar = fig.colorbar(scatter, ax=axs, orientation="vertical")
    cbar.set_ticks(list(np.linspace(0, 9, 11)))
    cbar_tick_labs = [str(round(i, 1)) for i in list(np.linspace(0, 2, 11))]
    cbar_tick_labs[0] = f"{cbar_tick_labs[0]} (arid)"
    cbar_tick_labs[-1] = f"{cbar_tick_labs[-1]} (wet)"
    cbar.set_ticklabels(cbar_tick_labs)  # type: ignore
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.set_title("Aridity\nindex [-]", fontsize=24, pad=21)

    # add a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            lw=1.5,
            linestyle="-",
            color="black",
            label="1:1 line",
            markerfacecolor="black",
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            lw=2,
            linestyle="--",
            dashes=(1.6, 2),
            color="red",
            label="fitted regression line",
            markerfacecolor="red",
            markersize=14,
        ),
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=24,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=True,
        bbox_to_anchor=(0.03, -0.5),
    )

    fig.supxlabel(
        (
            r"Annual average $\text{GPP}_{\text{sim}}$"
            "\n"
            r"[$\mathrm{\mu} \text{mol CO}_2 \cdot \text{m}^{-2}\cdot \text{s}^{-1}$]" 
        ),
        y=-0.12,
        x=0.44,
        fontsize = 30
    )
    fig.supylabel(
        (
            r"Annual average $\text{GPP}_{\text{EC}}$"
            "\n"
            r"[$\mathrm{\mu} \text{mol CO}_2 \cdot \text{m}^{-2}\cdot \text{s}^{-1}$]" 
        ),
        x=0.03,
        fontsize = 30
    )

    # save the figure
    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        f"{fig_path}/f05.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{fig_path}/f05.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")
    # paths where results the modelling experiments can be found

    # plot the figure
    plot_fig_main(result_paths.per_site_yr_p_model_res_path)
