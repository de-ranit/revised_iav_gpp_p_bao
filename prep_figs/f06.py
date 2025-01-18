#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot scatter plot of observed vs simulated annual GPP when LUE model
is run with daily and sub-daily forcing. The experiment is
site year optimization using LUE model.

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Thu Feb 01 2024 14:28:49 CET
"""

import os
from pathlib import Path
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from permetrics import RegressionMetric
import seaborn as sns

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


def get_ann_gpp(lue_hr_data_exp_path, lue_dd_data_exp_path):
    """
    collect annual observed and simulated GPP from all the good years
    for the two experiments (daily and sub-daily forcing) of LUE model

    Parameters:
    -----------
    lue_hr_data_exp_path (Path object) : path to the serialized model results
    of LUE model with sub-daily forcing
    lue_dd_data_exp_path (Path object) : path to the serialized model results
    of LUE model with daily forcing

    Returns:
    --------
    hr_gpp_obs_yr_arr (np.array) : array of observed annual GPP from
    all the good years of all the sites for LUE model with sub-daily forcing
    hr_gpp_sim_yr_arr (np.array) : array of simulated annual GPP from
    all the good years of all the sites for LUE model with sub-daily forcing
    dd_gpp_obs_yr_arr (np.array) : array of observed annual GPP from
    all the good years of all the sites for LUE model with daily forcing
    dd_gpp_sim_yr_arr (np.array) : array of simulated annual GPP from
    all the good years of all the sites for LUE model with daily forcing
    """

    # find all the files with serialized model results
    mod_res_file_list_hr = glob.glob(f"{lue_hr_data_exp_path}/*.npy")
    mod_res_file_list_hr.sort()  # sort the files by site ID

    # filter out bad sites
    filtered_mod_res_file_list_hr = [
        files
        for files in mod_res_file_list_hr
        if not (
            "CG-Tch" in files
            or "MY-PSO" in files
            or "GH-Ank" in files
            or "US-LWW" in files
        )
    ]

    # empty lists to store the results
    hr_gpp_obs_yr = []
    hr_gpp_sim_yr = []
    dd_gpp_obs_yr = []
    dd_gpp_sim_yr = []

    # for each site
    for res_file in filtered_mod_res_file_list_hr:
        # open the results file with sub-daily forcing
        res_dict_hr = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_hr["SiteID"]

        # open the result file for corresponding site with daily forcing
        res_dict_dd = np.load(
            f"{lue_dd_data_exp_path}/{site_id}_result.npy", allow_pickle=True
        ).item()

        # collect the results
        hr_gpp_obs_yr.append(res_dict_hr["GPP_NT_yearly_filtered"])
        hr_gpp_sim_yr.append(res_dict_hr["GPP_sim_yearly_filtered"])

        dd_gpp_obs_yr.append(res_dict_dd["GPP_NT_yearly_filtered"])
        dd_gpp_sim_yr.append(res_dict_dd["GPP_sim_yearly_filtered"])

    # concatenate the results of all the sites
    hr_gpp_obs_yr_arr = np.concatenate(hr_gpp_obs_yr)
    hr_gpp_sim_yr_arr = np.concatenate(hr_gpp_sim_yr)
    dd_gpp_obs_yr_arr = np.concatenate(dd_gpp_obs_yr)
    dd_gpp_sim_yr_arr = np.concatenate(dd_gpp_sim_yr)

    return hr_gpp_obs_yr_arr, hr_gpp_sim_yr_arr, dd_gpp_obs_yr_arr, dd_gpp_sim_yr_arr


def plot_axs(x, y, axs, lim, title):
    """
    prepare the axes for the scatter plot of observed vs simulated annual GPP
    for LUE model with daily and sub-daily forcing

    Parameters:
    -----------
    x (np.array) : array of observed GPP
    y (np.array) : array of simulated GPP
    axs (matplotlib.axes) : axes object
    lim (tuple) : limits for the axes
    title (str) : title for the axes

    Returns:
    --------
    None
    """
    # calculate NSE and R2 between observed and simulated GPP
    evaluator_sd = RegressionMetric(x, y, decimal=3)
    nse = evaluator_sd.NSE()
    r2 = (evaluator_sd.PCC()) ** 2

    # plot the scatter plot
    axs.scatter(x, y, c="#52B4FF", alpha=0.7)

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

    if round(coeff[1],3) < 0.0:
        axs.text(1, 7, f"y = {round(coeff[0],3)}x - {abs(round(coeff[1],3))}", fontsize=18)
    else:
        axs.text(1, 7, f"y = {round(coeff[0],3)}x + {round(coeff[1],3)}", fontsize=18)

    sns.despine(ax=axs, top=True, right=True)


def plot_fig_main(lue_hr_data_exp_path, lue_dd_data_exp_path):
    """
    plot the figure with the scatter plot of observed vs simulated annual GPP

    Parameters:
    -----------
    lue_hr_data_exp_path (Path object) : path to the serialized model results
    of LUE model with sub-daily forcing
    lue_dd_data_exp_path (Path object) : path to the serialized model results
    of LUE model with daily forcing

    Returns:
    --------
    None
    """
    # get the annual observed and simulated GPP (using daily and sub-daily forcing)
    (
        hr_gpp_obs_yr_arr,
        hr_gpp_sim_yr_arr,
        dd_gpp_obs_yr_arr,
        dd_gpp_sim_yr_arr,
    ) = get_ann_gpp(lue_hr_data_exp_path, lue_dd_data_exp_path)

    # determine the limits for the axes
    max_gpp_hr = max(hr_gpp_obs_yr_arr.max(), hr_gpp_sim_yr_arr.max())
    max_gpp_dd = max(dd_gpp_obs_yr_arr.max(), dd_gpp_sim_yr_arr.max())
    min_gpp_hr = min(hr_gpp_obs_yr_arr.min(), hr_gpp_sim_yr_arr.min())
    min_gpp_dd = min(dd_gpp_obs_yr_arr.min(), dd_gpp_sim_yr_arr.min())

    # prepare the figure
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 6), sharex=True, sharey=True)

    # add the scatter plots
    plot_axs(
        hr_gpp_sim_yr_arr,
        hr_gpp_obs_yr_arr,
        axs[0],
        (min_gpp_hr, max_gpp_hr),
        r"(a) Bao$_{\text{hr}}$ model",
    )
    plot_axs(
        dd_gpp_sim_yr_arr,
        dd_gpp_obs_yr_arr,
        axs[1],
        (min_gpp_dd, max_gpp_dd),
        r"(b) Bao$_{\text{dd}}$ model",
    )

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
            color="red",
            dashes=(1.6, 2),
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
        bbox_to_anchor=(-0.1, -0.5),
    )

    fig.supxlabel(
        (
            r"Annual average $\text{GPP}_{\text{sim}}$"
            "\n"
            r"[$\mathrm{\mu} \text{mol CO}_2 \cdot \text{m}^{-2}\cdot \text{s}^{-1}$]"
        ),
        y=-0.12,
        x=0.52,
        fontsize=30,
    )
    fig.supylabel(
        (
            r"Annual average $\text{GPP}_{\text{EC}}$"
            "\n"
            r"[$\mathrm{\mu} \text{mol CO}_2 \cdot \text{m}^{-2}\cdot \text{s}^{-1}$]"
        ),
        x=0.03,
        fontsize=30,
    )

    # save the figure
    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        f"{fig_path}/f06.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{fig_path}/f06.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    # plot the figure
    plot_fig_main(
        result_paths.per_site_yr_lue_model_res_path,
        result_paths.per_site_yr_dd_lue_model_res_path,
    )
