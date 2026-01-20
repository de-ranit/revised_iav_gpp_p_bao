#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot overall model performance results

author: rde
first created: Sun Nov 16 2025 16:02:58 CET
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
    r"\usepackage{amsmath}"  # use amsmath package
)
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def get_nse_arr(p_mod_res_path, lue_mod_res_path):
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
    nse_yy_p = np.zeros(len(filtered_mod_res_file_list))

    nse_hr_lue = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_lue = np.zeros(len(filtered_mod_res_file_list))

    # for each site
    for ix, res_file in enumerate(filtered_mod_res_file_list):
        # open the results file for P Model
        res_dict_p = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_p["SiteID"]

        # open the corresponding results file for LUE Model
        res_dict_lue = np.load(
            f"{lue_mod_res_path}/{site_id}_result.npy", allow_pickle=True
        ).item()

        # fill the arrays with hourly, daily, weekly, monthly and yearly NSE
        nse_hr_p[ix] = res_dict_p["NSE"][f"NSE_{res_dict_p['Temp_res']}"]
        nse_yy_p[ix] = res_dict_p["NSE"]["NSE_y"]

        nse_hr_lue[ix] = res_dict_lue["NSE"][f"NSE_{res_dict_lue['Temp_res']}"]
        nse_yy_lue[ix] = res_dict_lue["NSE"]["NSE_y"]

    return (
        nse_hr_p,
        nse_yy_p,
        nse_hr_lue,
        nse_yy_lue,
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
        nse_yy_p_syr,
        nse_hr_lue_syr,
        nse_yy_lue_syr,
    ) = get_nse_arr(
        result_paths.per_site_yr_p_model_res_path,
        result_paths.per_site_yr_lue_model_res_path,
    )

    # get the arrays of NSE
    (
        nse_hr_p_allyr,
        nse_yy_p_allyr,
        nse_hr_lue_allyr,
        nse_yy_lue_allyr,
    ) = get_nse_arr(
        result_paths.per_site_p_model_res_path,
        result_paths.per_site_lue_model_res_path,
    )

    # get the arrays of NSE
    (
        nse_hr_p_allyr_iav,
        nse_yy_p_allyr_iav,
        nse_hr_lue_allyr_iav,
        nse_yy_lue_allyr_iav,
    ) = get_nse_arr(
        result_paths.per_site_p_model_res_path_iav,
        result_paths.per_site_lue_model_res_path_iav,
    )

    # get the arrays of NSE
    (
        nse_hr_p_pft,
        nse_yy_p_pft,
        nse_hr_lue_pft,
        nse_yy_lue_pft,
    ) = get_nse_arr(
        result_paths.per_pft_p_model_res_path,
        result_paths.per_pft_lue_model_res_path,
    )

    # get the arrays of NSE
    (
        nse_hr_p_glob,
        nse_yy_p_glob,
        nse_hr_lue_glob,
        nse_yy_lue_glob,
    ) = get_nse_arr(
        result_paths.glob_opti_p_model_res_path,
        result_paths.glob_opti_lue_model_res_path,
    )

    #######################################################################
    # plot figure f02 showing overall model performance at hourly/daily
    # and yearly scale

    fig_width = 20
    fig_height = 12

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=2, nrows=2, figsize=(fig_width, fig_height), sharex=True, sharey=True
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

    hr_lue_hr_nnse_dict = plot_axs(
        axs[0, 1],
        nse_hr_lue_syr,
        nse_hr_lue_allyr,
        nse_hr_lue_allyr_iav,
        nse_hr_lue_pft,
        nse_hr_lue_glob,
        r"(b) Bao$_{\text{hr}}$ model",
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
        r"(c) P$^{\text{W}}_{\text{hr}}$ model",
        0.7,
        0.4,
    )

    yy_lue_hr_nnse_dict = plot_axs(
        axs[1, 1],
        nse_yy_lue_syr,
        nse_yy_lue_allyr,
        nse_yy_lue_allyr_iav,
        nse_yy_lue_pft,
        nse_yy_lue_glob,
        r"(d) Bao$_{\text{hr}}$ model",
        0.7,
        0.4,
    )

    fig.text(
        0.12,
        0.95,
        r"\textbf{Hourly scale}",
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
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-0.2, -1.0),
    )

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f05_best_ever_opti_overall_model_performance.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f05_best_ever_opti_overall_model_performance.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    #################################################################################
    # print the median NNSE values for each optimization type (Table D1 and Table S2)

    df_hr = pd.DataFrame(
        [hr_p_nnse_dict, hr_lue_hr_nnse_dict]  # , dd_lue_dd_nnse_dict]
    )
    df_yr = pd.DataFrame(
        [yy_p_nnse_dict, yy_lue_hr_nnse_dict]  # , yy_lue_dd_nnse_dict]
    )

    print("hr/daily")
    print(df_hr.to_latex(index=False, float_format="%.3f"))
    print("yearly")
    print(df_yr.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    # get the result paths collection module
    result_paths_coll = importlib.import_module("result_path_coll")

    # plot the figure
    plot_fig_main(result_paths_coll)
