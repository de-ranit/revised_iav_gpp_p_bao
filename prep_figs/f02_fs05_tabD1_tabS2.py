#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create f02 and fs05 - overall model performance across timescales

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Tue Jul 09 2024 16:15:23 CEST
"""
import os
from pathlib import Path
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = (
    r"\renewcommand{\familydefault}{\sfdefault}"  # use sans-serif font
)
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}"  # use amsmath package
)

# set the font to computer modern sans
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "cmss10"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def get_nse_arr(p_mod_res_path, lue_mod_res_path, lue_dd_mod_res_path):
    """
    get the hourly and yearly NSE for each site from site year optimization
    of P Model and LUE Model

    Parameters:
    -----------
    p_mod_res_path (Path object) : path to the serialized model results
    of site year optimization of P Model
    lue_mod_res_path (Path object) : path to the serialized model results
    of site year optimization of LUE Model

    Returns:
    --------
    nse_hr_p (np.ndarray) : array of hourly NSE of P Model
    nse_d_p (np.ndarray) : array of daily NSE of P Model
    nse_w_p (np.ndarray) : array of weekly NSE of P Model
    nse_m_p (np.ndarray) : array of monthly NSE of P Model
    nse_yy_p (np.ndarray) : array of yearly NSE of P Model

    nse_hr_p_no_sm (np.ndarray) : array of hourly NSE of P Model without moisture stress
    nse_d_p_no_sm (np.ndarray) : array of daily NSE of P Model without moisture stress
    nse_w_p_no_sm (np.ndarray) : array of weekly NSE of P Model without moisture stress
    nse_m_p_no_sm (np.ndarray) : array of monthly NSE of P Model without moisture stress
    nse_yy_p_no_sm (np.ndarray) : array of yearly NSE of P Model without moisture stress

    nse_hr_lue (np.ndarray) : array of hourly NSE of LUE Model
    nse_d_lue (np.ndarray) : array of daily NSE of LUE Model
    nse_w_lue (np.ndarray) : array of weekly NSE of LUE Model
    nse_m_lue (np.ndarray) : array of monthly NSE of LUE Model
    nse_yy_lue (np.ndarray) : array of yearly NSE of LUE Model

    nse_d_lue_dd (np.ndarray) : array of daily NSE of LUE Model optimized with daily data
    nse_w_lue_dd (np.ndarray) : array of weekly NSE of LUE Model optimized with daily data
    nse_m_lue_dd (np.ndarray) : array of monthly NSE of LUE Model optimized with daily data
    nse_y_lue_dd (np.ndarray) : array of yearly NSE of LUE Model optimized with daily data
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

    # arrays to store the results
    nse_hr_p = np.zeros(len(filtered_mod_res_file_list))
    nse_d_p = np.zeros(len(filtered_mod_res_file_list))
    nse_w_p = np.zeros(len(filtered_mod_res_file_list))
    nse_m_p = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_p = np.zeros(len(filtered_mod_res_file_list))

    nse_hr_p_no_sm = np.zeros(len(filtered_mod_res_file_list))
    nse_d_p_no_sm = np.zeros(len(filtered_mod_res_file_list))
    nse_w_p_no_sm = np.zeros(len(filtered_mod_res_file_list))
    nse_m_p_no_sm = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_p_no_sm = np.zeros(len(filtered_mod_res_file_list))

    nse_hr_lue = np.zeros(len(filtered_mod_res_file_list))
    nse_d_lue = np.zeros(len(filtered_mod_res_file_list))
    nse_w_lue = np.zeros(len(filtered_mod_res_file_list))
    nse_m_lue = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_lue = np.zeros(len(filtered_mod_res_file_list))

    nse_d_lue_dd = np.zeros(len(filtered_mod_res_file_list))
    nse_w_lue_dd = np.zeros(len(filtered_mod_res_file_list))
    nse_m_lue_dd = np.zeros(len(filtered_mod_res_file_list))
    nse_y_lue_dd = np.zeros(len(filtered_mod_res_file_list))

    # for each site
    for ix, res_file in enumerate(filtered_mod_res_file_list):
        # open the results file for P Model
        res_dict_p = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_p["SiteID"]

        # open the corresponding results file for LUE Model
        res_dict_lue = np.load(
            f"{lue_mod_res_path}/{site_id}_result.npy", allow_pickle=True
        ).item()

        res_dict_lue_dd = np.load(
            f"{lue_dd_mod_res_path}/{site_id}_result.npy", allow_pickle=True
        ).item()

        # fill the arrays with hourly, daily, weekly, monthly and yearly NSE
        nse_hr_p[ix] = res_dict_p["NSE"][f"NSE_{res_dict_p['Temp_res']}"]
        nse_d_p[ix] = res_dict_p["NSE"]["NSE_d"]
        nse_w_p[ix] = res_dict_p["NSE"]["NSE_w"]
        nse_m_p[ix] = res_dict_p["NSE"]["NSE_m"]
        nse_yy_p[ix] = res_dict_p["NSE"]["NSE_y"]

        nse_hr_p_no_sm[ix] = res_dict_p["NSE_no_moisture_Stress"][
            f"NSE_{res_dict_p['Temp_res']}"
        ]
        nse_d_p_no_sm[ix] = res_dict_p["NSE_no_moisture_Stress"]["NSE_d"]
        nse_w_p_no_sm[ix] = res_dict_p["NSE_no_moisture_Stress"]["NSE_w"]
        nse_m_p_no_sm[ix] = res_dict_p["NSE_no_moisture_Stress"]["NSE_m"]
        nse_yy_p_no_sm[ix] = res_dict_p["NSE_no_moisture_Stress"]["NSE_y"]

        nse_hr_lue[ix] = res_dict_lue["NSE"][f"NSE_{res_dict_lue['Temp_res']}"]
        nse_d_lue[ix] = res_dict_lue["NSE"]["NSE_d"]
        nse_w_lue[ix] = res_dict_lue["NSE"]["NSE_w"]
        nse_m_lue[ix] = res_dict_lue["NSE"]["NSE_m"]
        nse_yy_lue[ix] = res_dict_lue["NSE"]["NSE_y"]

        nse_d_lue_dd[ix] = res_dict_lue_dd["NSE"]["NSE_d"]
        nse_w_lue_dd[ix] = res_dict_lue_dd["NSE"]["NSE_w"]
        nse_m_lue_dd[ix] = res_dict_lue_dd["NSE"]["NSE_m"]
        nse_y_lue_dd[ix] = res_dict_lue_dd["NSE"]["NSE_y"]

    return (
        nse_hr_p,
        nse_d_p,
        nse_w_p,
        nse_m_p,
        nse_yy_p,
        nse_hr_p_no_sm,
        nse_d_p_no_sm,
        nse_w_p_no_sm,
        nse_m_p_no_sm,
        nse_yy_p_no_sm,
        nse_hr_lue,
        nse_d_lue,
        nse_w_lue,
        nse_m_lue,
        nse_yy_lue,
        nse_d_lue_dd,
        nse_w_lue_dd,
        nse_m_lue_dd,
        nse_y_lue_dd,
    )


def calc_nnse_rm_nan(nse_arr):
    """
    calculate the normalized NSE and remove nan values

    Parameters:
    -----------
    nse_arr (np.ndarray) : array of NSE values

    Returns:
    --------
    nnse_arr (np.ndarray) : array of normalized NSE values with nan values removed
    """

    nnse_arr = 1.0 / (2.0 - nse_arr)
    nnse_arr = nnse_arr[~np.isnan(nnse_arr)]

    return nnse_arr


def plot_axs(
    ax, nse_syr, nse_allyr, nse_allyr_iav, nse_pft, nse_glob, title, bw_adjust, cut
):
    """
    plot the KDE of NNSE for different optimization types

    Parameters:
    -----------
    ax (matplotlib axis) : axis to plot the histogram
    nse_syr (np.ndarray) : array of NSE for site year optimization
    nse_allyr (np.ndarray) : array of NSE for site optimization
    nse_allyr_iav (np.ndarray) : array of NSE for site optimization using IAV cost
    nse_pft (np.ndarray) : array of NSE for PFT optimization
    nse_glob (np.ndarray) : array of NSE for global optimization
    title (str) : title of the plot
    bw_adjust (float) : bandwidth adjustment for the KDE
    cut (float) : cut off value for the KDE

    Returns:
    --------
    dict : dictionary containing the median NNSE for each optimization type

    """

    # calculate NNSE and remove nan values
    nnse_syr = calc_nnse_rm_nan(nse_syr)
    nnse_allyr = calc_nnse_rm_nan(nse_allyr)
    nnse_allyr_iav = calc_nnse_rm_nan(nse_allyr_iav)
    nnse_pft = calc_nnse_rm_nan(nse_pft)
    nnse_glob = calc_nnse_rm_nan(nse_glob)

    # plot the histograms and KDE - but make histogram invisible and only show KDE
    sns.histplot(
        x=nnse_syr,
        stat="percent",
        kde=True,
        kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
        color="white",
        edgecolor="white",
        binrange=(0, 1.0),
        binwidth=0.1,
        ax=ax,
    )
    ax.lines[0].set_color("#56B4E9")

    sns.histplot(
        x=nnse_allyr,
        stat="percent",
        kde=True,
        kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
        binrange=(0, 1.0),
        binwidth=0.1,
        ax=ax,
        color="white",
        edgecolor="white",
    )
    ax.lines[1].set_color("#009E73")

    sns.histplot(
        x=nnse_pft,
        stat="percent",
        kde=True,
        kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
        binrange=(0, 1.0),
        binwidth=0.1,
        ax=ax,
        color="white",
        edgecolor="white",
    )
    ax.lines[2].set_color("#BBBBBB")

    sns.histplot(
        x=nnse_glob,
        stat="percent",
        kde=True,
        kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
        binrange=(0, 1.0),
        binwidth=0.1,
        ax=ax,
        color="white",
        edgecolor="white",
    )
    ax.lines[3].set_color("#CC79A7")

    sns.histplot(
        x=nnse_allyr_iav,
        stat="percent",
        kde=True,
        kde_kws={"bw_adjust": bw_adjust, "cut": cut, "clip": (0.0, 1.0)},
        binrange=(0, 1.0),
        binwidth=0.1,
        ax=ax,
        color="white",
        edgecolor="white",
    )
    ax.lines[4].set_color("#E6C300")

    # add vertical lines for the median values
    ax.axvline(x=np.median(nnse_syr), linestyle=":", color="#56B4E9")
    ax.axvline(x=np.median(nnse_allyr), linestyle=":", color="#009E73")
    ax.axvline(x=np.median(nnse_pft), linestyle=":", color="#BBBBBB")
    ax.axvline(x=np.median(nnse_glob), linestyle=":", color="#CC79A7")
    ax.axvline(x=np.median(nnse_allyr_iav), linestyle=":", color="#E6C300")

    # set the axis properties
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticklabels([round(x, 1) for x in np.linspace(0.0, 1.0, 6).tolist()])
    ax.tick_params(axis="both", which="major", labelsize=26.0)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("")
    ax.set_title(f"{title} ({len(nnse_syr)})", size=30)

    sns.despine(ax=ax, top=True, right=True)

    return {
        "syr_med": np.median(nnse_syr),
        "nnse_allyr_iav_med": np.median(nnse_allyr_iav),
        "allyr_med": np.median(nnse_allyr),
        "pft_med": np.median(nnse_pft),
        "glob_med": np.median(nnse_glob),
    }


def plot_fig_main(result_paths):
    """
    plot the main figure

    Parameters:
    -----------
    result_paths (namedtuple) : named tuple containing the paths to the
    serialized model results

    Returns:
    --------
    None
    """

    # get the arrays of NSE
    (
        nse_hr_p_syr,
        nse_d_p_syr,
        nse_w_p_syr,
        nse_m_p_syr,
        nse_yy_p_syr,
        nse_hr_p_no_sm_syr,
        nse_d_p_no_sm_syr,
        nse_w_p_no_sm_syr,
        nse_m_p_no_sm_syr,
        nse_yy_p_no_sm_syr,
        nse_hr_lue_syr,
        nse_d_lue_syr,
        nse_w_lue_syr,
        nse_m_lue_syr,
        nse_yy_lue_syr,
        nse_d_lue_dd_syr,
        nse_w_lue_dd_syr,
        nse_m_lue_dd_syr,
        nse_y_lue_dd_syr,
    ) = get_nse_arr(
        result_paths.per_site_yr_p_model_res_path,
        result_paths.per_site_yr_lue_model_res_path,
        result_paths.per_site_yr_dd_lue_model_res_path,
    )

    # get the arrays of NSE
    (
        nse_hr_p_allyr,
        nse_d_p_allyr,
        nse_w_p_allyr,
        nse_m_p_allyr,
        nse_yy_p_allyr,
        nse_hr_p_no_sm_allyr,
        nse_d_p_no_sm_allyr,
        nse_w_p_no_sm_allyr,
        nse_m_p_no_sm_allyr,
        nse_yy_p_no_sm_allyr,
        nse_hr_lue_allyr,
        nse_d_lue_allyr,
        nse_w_lue_allyr,
        nse_m_lue_allyr,
        nse_yy_lue_allyr,
        nse_d_lue_dd_allyr,
        nse_w_lue_dd_allyr,
        nse_m_lue_dd_allyr,
        nse_y_lue_dd_allyr,
    ) = get_nse_arr(
        result_paths.per_site_p_model_res_path,
        result_paths.per_site_lue_model_res_path,
        result_paths.per_site_dd_lue_model_res_path,
    )

    # get the arrays of NSE
    (
        nse_hr_p_allyr_iav,
        nse_d_p_allyr_iav,
        nse_w_p_allyr_iav,
        nse_m_p_allyr_iav,
        nse_yy_p_allyr_iav,
        nse_hr_p_no_sm_allyr_iav,
        nse_d_p_no_sm_allyr_iav,
        nse_w_p_no_sm_allyr_iav,
        nse_m_p_no_sm_allyr_iav,
        nse_yy_p_no_sm_allyr_iav,
        nse_hr_lue_allyr_iav,
        nse_d_lue_allyr_iav,
        nse_w_lue_allyr_iav,
        nse_m_lue_allyr_iav,
        nse_yy_lue_allyr_iav,
        nse_d_lue_dd_allyr_iav,
        nse_w_lue_dd_allyr_iav,
        nse_m_lue_dd_allyr_iav,
        nse_y_lue_dd_allyr_iav,
    ) = get_nse_arr(
        result_paths.per_site_p_model_res_path_iav,
        result_paths.per_site_lue_model_res_path_iav,
        result_paths.per_site_dd_lue_model_res_path_iav,
    )

    # get the arrays of NSE
    (
        nse_hr_p_pft,
        nse_d_p_pft,
        nse_w_p_pft,
        nse_m_p_pft,
        nse_yy_p_pft,
        nse_hr_p_no_sm_pft,
        nse_d_p_no_sm_pft,
        nse_w_p_no_sm_pft,
        nse_m_p_no_sm_pft,
        nse_yy_p_no_sm_pft,
        nse_hr_lue_pft,
        nse_d_lue_pft,
        nse_w_lue_pft,
        nse_m_lue_pft,
        nse_yy_lue_pft,
        nse_d_lue_dd_pft,
        nse_w_lue_dd_pft,
        nse_m_lue_dd_pft,
        nse_y_lue_dd_pft,
    ) = get_nse_arr(
        result_paths.per_pft_p_model_res_path,
        result_paths.per_pft_lue_model_res_path,
        result_paths.per_pft_dd_lue_model_res_path,
    )

    # get the arrays of NSE
    (
        nse_hr_p_glob,
        nse_d_p_glob,
        nse_w_p_glob,
        nse_m_p_glob,
        nse_yy_p_glob,
        nse_hr_p_no_sm_glob,
        nse_d_p_no_sm_glob,
        nse_w_p_no_sm_glob,
        nse_m_p_no_sm_glob,
        nse_yy_p_no_sm_glob,
        nse_hr_lue_glob,
        nse_d_lue_glob,
        nse_w_lue_glob,
        nse_m_lue_glob,
        nse_yy_lue_glob,
        nse_d_lue_dd_glob,
        nse_w_lue_dd_glob,
        nse_m_lue_dd_glob,
        nse_y_lue_dd_glob,
    ) = get_nse_arr(
        result_paths.glob_opti_p_model_res_path,
        result_paths.glob_opti_lue_model_res_path,
        result_paths.glob_opti_dd_lue_model_res_path,
    )

    #######################################################################
    # plot figure f02 showing overall model performance at hourly/daily
    # and yearly scale

    fig_width = 20
    fig_height = 10

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=4, nrows=2, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    hr_p_nnse_dict = plot_axs(
        axs[0, 0],
        nse_hr_p_syr,
        nse_hr_p_allyr,
        nse_hr_p_allyr_iav,
        nse_hr_p_pft,
        nse_hr_p_glob,
        r"(a) P$^{\text{W}}_{\text{hr}}$ model",
        1.0,
        3,
    )
    hr_p_no_sm_nnse_dict = plot_axs(
        axs[0, 1],
        nse_hr_p_no_sm_syr,
        nse_hr_p_no_sm_allyr,
        nse_hr_p_no_sm_allyr_iav,
        nse_hr_p_no_sm_pft,
        nse_hr_p_no_sm_glob,
        r"(b) P$_{\text{hr}}$ model",
        1.0,
        6,
    )
    hr_lue_hr_nnse_dict = plot_axs(
        axs[0, 2],
        nse_hr_lue_syr,
        nse_hr_lue_allyr,
        nse_hr_lue_allyr_iav,
        nse_hr_lue_pft,
        nse_hr_lue_glob,
        r"(c) Bao$_{\text{hr}}$ model",
        1.0,
        3,
    )
    dd_lue_dd_nnse_dict = plot_axs(
        axs[0, 3],
        nse_d_lue_dd_syr,
        nse_d_lue_dd_allyr,
        nse_d_lue_dd_allyr_iav,
        nse_d_lue_dd_pft,
        nse_d_lue_dd_glob,
        r"(d) Bao$_{\text{dd}}$ model",
        1.0,
        3,
    )

    yy_p_nnse_dict = plot_axs(
        axs[1, 0],
        nse_yy_p_syr,
        nse_yy_p_allyr,
        nse_yy_p_allyr_iav,
        nse_yy_p_pft,
        nse_yy_p_glob,
        r"(e) P$^{\text{W}}_{\text{hr}}$ model",
        0.7,
        0.4,
    )
    yy_p_no_sm_nnse_dict = plot_axs(
        axs[1, 1],
        nse_yy_p_no_sm_syr,
        nse_yy_p_no_sm_allyr,
        nse_yy_p_no_sm_allyr_iav,
        nse_yy_p_no_sm_pft,
        nse_yy_p_no_sm_glob,
        r"(f) P$_{\text{hr}}$ model",
        0.7,
        0.4,
    )
    yy_lue_hr_nnse_dict = plot_axs(
        axs[1, 2],
        nse_yy_lue_syr,
        nse_yy_lue_allyr,
        nse_yy_lue_allyr_iav,
        nse_yy_lue_pft,
        nse_yy_lue_glob,
        r"(g) Bao$_{\text{hr}}$ model",
        0.7,
        0.4,
    )
    yy_lue_dd_nnse_dict = plot_axs(
        axs[1, 3],
        nse_y_lue_dd_syr,
        nse_y_lue_dd_allyr,
        nse_y_lue_dd_allyr_iav,
        nse_y_lue_dd_pft,
        nse_y_lue_dd_glob,
        r"(h) Bao$_{\text{dd}}$ model",
        0.7,
        0.4,
    )

    fig.text(
        0.12,
        0.95,
        r"\textbf{Hourly/ daily (only for subplot d) scale}",
        ha="left",
        fontsize=30,
    )
    fig.text(0.12, 0.5, r"\textbf{Annual aggregation}", ha="left", fontsize=30)

    fig.supxlabel("NNSE [-]", y=-0.01, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.07, fontsize=36)

    fig.subplots_adjust(hspace=0.4)

    # Adding legend manually
    colors = ["#56B4E9", "#E6C300", "#009E73", "#BBBBBB", "#CC79A7"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=25,
        )
        for i, opti_type in enumerate(
            [
                r"per site--year parameterization",
                r"per site parameterization using $Cost^{IAV}$",
                "per site parameterization",
                "per PFT parameterization",
                "global parameterization",
            ]
        )
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(-1.3, -0.9),
    )

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f02.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f02.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    ##########################################################################
    # plot figure fs05 showing overall model performance at daily, weekly and
    # monthly scale

    fig_width = 20
    fig_height = 15

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=4, nrows=3, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    dd_p_nnse_dict = plot_axs(
        axs[0, 0],
        nse_d_p_syr,
        nse_d_p_allyr,
        nse_d_p_allyr_iav,
        nse_d_p_pft,
        nse_d_p_glob,
        r"(a) P$^{\text{W}}_{\text{hr}}$ model",
        1.0,
        3,
    )
    dd_p_no_sm_nnse_dict = plot_axs(
        axs[0, 1],
        nse_d_p_no_sm_syr,
        nse_d_p_no_sm_allyr,
        nse_d_p_no_sm_allyr_iav,
        nse_d_p_no_sm_pft,
        nse_d_p_no_sm_glob,
        r"(b) P$_{\text{hr}}$ model",
        1.0,
        6,
    )
    dd_lue_hr_nnse_dict = plot_axs(
        axs[0, 2],
        nse_d_lue_syr,
        nse_d_lue_allyr,
        nse_d_lue_allyr_iav,
        nse_d_lue_pft,
        nse_d_lue_glob,
        r"(c) Bao$_{\text{hr}}$ model",
        1.0,
        3,
    )
    dd_lue_dd_nnse_dict = plot_axs(
        axs[0, 3],
        nse_d_lue_dd_syr,
        nse_d_lue_dd_allyr,
        nse_d_lue_dd_allyr_iav,
        nse_d_lue_dd_pft,
        nse_d_lue_dd_glob,
        r"(d) Bao$_{\text{dd}}$ model",
        1.0,
        3,
    )

    ww_p_nnse_dict = plot_axs(
        axs[1, 0],
        nse_w_p_syr,
        nse_w_p_allyr,
        nse_w_p_allyr_iav,
        nse_w_p_pft,
        nse_w_p_glob,
        r"(e) P$^{\text{W}}_{\text{hr}}$ model",
        1.0,
        3,
    )
    ww_p_no_sm_nnse_dict = plot_axs(
        axs[1, 1],
        nse_w_p_no_sm_syr,
        nse_w_p_no_sm_allyr,
        nse_w_p_no_sm_allyr_iav,
        nse_w_p_no_sm_pft,
        nse_w_p_no_sm_glob,
        r"(f) P$_{\text{hr}}$ model",
        1.0,
        6,
    )
    ww_lue_hr_nnse_dict = plot_axs(
        axs[1, 2],
        nse_w_lue_syr,
        nse_w_lue_allyr,
        nse_w_lue_allyr_iav,
        nse_w_lue_pft,
        nse_w_lue_glob,
        r"(g) Bao$_{\text{hr}}$ model",
        1.0,
        3,
    )
    ww_lue_dd_nnse_dict = plot_axs(
        axs[1, 3],
        nse_w_lue_dd_syr,
        nse_w_lue_dd_allyr,
        nse_w_lue_dd_allyr_iav,
        nse_w_lue_dd_pft,
        nse_w_lue_dd_glob,
        r"(h) Bao$_{\text{dd}}$ model",
        1.0,
        3,
    )

    mm_p_nnse_dict = plot_axs(
        axs[2, 0],
        nse_m_p_syr,
        nse_m_p_allyr,
        nse_m_p_allyr_iav,
        nse_m_p_pft,
        nse_m_p_glob,
        r"(i) P$^{\text{W}}_{\text{hr}}$ model",
        1.0,
        3,
    )
    mm_p_no_sm_nnse_dict = plot_axs(
        axs[2, 1],
        nse_m_p_no_sm_syr,
        nse_m_p_no_sm_allyr,
        nse_m_p_no_sm_allyr_iav,
        nse_m_p_no_sm_pft,
        nse_m_p_no_sm_glob,
        r"(j) P$_{\text{hr}}$ model",
        1.0,
        6,
    )
    mm_lue_hr_nnse_dict = plot_axs(
        axs[2, 2],
        nse_m_lue_syr,
        nse_m_lue_allyr,
        nse_m_lue_allyr_iav,
        nse_m_lue_pft,
        nse_m_lue_glob,
        r"(k) Bao$_{\text{hr}}$ model",
        1.0,
        3,
    )
    mm_lue_dd_nnse_dict = plot_axs(
        axs[2, 3],
        nse_m_lue_dd_syr,
        nse_m_lue_dd_allyr,
        nse_m_lue_dd_allyr_iav,
        nse_m_lue_dd_pft,
        nse_m_lue_dd_glob,
        r"(l) Bao$_{\text{dd}}$ model",
        1.0,
        3,
    )

    fig.text(0.12, 0.93, r"\textbf{Daily aggregation}", ha="left", fontsize=30)
    fig.text(0.12, 0.64, r"\textbf{Weekly aggregation}", ha="left", fontsize=30)
    fig.text(0.12, 0.35, r"\textbf{Monthly aggregation}", ha="left", fontsize=30)

    fig.subplots_adjust(hspace=0.5)
    fig.supxlabel("NNSE [-]", y=0.03, fontsize=36)
    fig.supylabel("Fraction of" + r" sites [\%]", x=0.07, fontsize=36)

    # Adding legend manually
    colors = ["#56B4E9", "#E6C300", "#009E73", "#BBBBBB", "#CC79A7"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=25,
        )
        for i, opti_type in enumerate(
            [
                r"per site--year parameterization",
                r"per site parameterization using $Cost^{IAV}$",
                "per site parameterization",
                "per PFT parameterization",
                "global parameterization",
            ]
        )
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-1.3, -1.2),
    )

    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./supplement_figs/fs05.png", dpi=300, bbox_inches="tight")
    plt.savefig("./supplement_figs/fs05.pdf", dpi=300, bbox_inches="tight")

    #################################################################################
    # print the median NNSE values for each optimization type (Table D1 and Table S2)

    df_hr = pd.DataFrame(
        [hr_p_nnse_dict, hr_p_no_sm_nnse_dict, hr_lue_hr_nnse_dict, dd_lue_dd_nnse_dict]
    )
    df_yr = pd.DataFrame(
        [yy_p_nnse_dict, yy_p_no_sm_nnse_dict, yy_lue_hr_nnse_dict, yy_lue_dd_nnse_dict]
    )

    df_dd = pd.DataFrame(
        [dd_p_nnse_dict, dd_p_no_sm_nnse_dict, dd_lue_hr_nnse_dict, dd_lue_dd_nnse_dict]
    )
    df_ww = pd.DataFrame(
        [ww_p_nnse_dict, ww_p_no_sm_nnse_dict, ww_lue_hr_nnse_dict, ww_lue_dd_nnse_dict]
    )
    df_mm = pd.DataFrame(
        [mm_p_nnse_dict, mm_p_no_sm_nnse_dict, mm_lue_hr_nnse_dict, mm_lue_dd_nnse_dict]
    )

    print("hr/daily")
    print(df_hr.to_latex(index=False, float_format="%.3f"))
    print("yearly")
    print(df_yr.to_latex(index=False, float_format="%.3f"))
    print("daily")
    print(df_dd.to_latex(index=False, float_format="%.3f"))
    print("weekly")
    print(df_ww.to_latex(index=False, float_format="%.3f"))
    print("monthly")
    print(df_mm.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    # get the result paths collection module
    result_paths_coll = importlib.import_module("result_path_coll")

    # plot the figure
    plot_fig_main(result_paths_coll)
