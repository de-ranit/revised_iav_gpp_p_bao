#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot model performance between
1. ET vs ET_CORR
2. Nominal vs strict data filtering

author: rde
first created: Fri Dec 12 2025 15:36:19 CET
"""
import os
from pathlib import Path
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


def get_nse_arr(
    p_mod_res_path_default,
    p_mod_res_path_changed,
    lue_mod_res_path_default,
    lue_mod_res_path_changed,
):

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{p_mod_res_path_default}/*.npy")
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
    nse_hr_p_changed = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_p = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_p_changed = np.zeros(len(filtered_mod_res_file_list))

    nse_hr_lue = np.zeros(len(filtered_mod_res_file_list))
    nse_hr_lue_changed = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_lue = np.zeros(len(filtered_mod_res_file_list))
    nse_yy_lue_changed = np.zeros(len(filtered_mod_res_file_list))

    # for each site
    for ix, res_file in enumerate(filtered_mod_res_file_list):
        # open the results file for P Model
        res_dict_p = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_p["SiteID"]

        try:
            res_dict_p_changed = np.load(
                f"{p_mod_res_path_changed}/{site_id}_result.npy", allow_pickle=True
            ).item()
            # fill the arrays with hourly, daily, weekly, monthly and yearly NSE
            nse_hr_p[ix] = res_dict_p["NSE"][f"NSE_{res_dict_p['Temp_res']}"]
            nse_yy_p[ix] = res_dict_p["NSE"]["NSE_y"]

            nse_hr_p_changed[ix] = res_dict_p_changed["NSE"][
                f"NSE_{res_dict_p_changed['Temp_res']}"
            ]
            if np.isnan(nse_hr_p_changed[ix]):
                nse_hr_p[ix] = np.nan

            nse_yy_p_changed[ix] = res_dict_p_changed["NSE"]["NSE_y"]
            if np.isnan(nse_yy_p_changed[ix]):
                nse_yy_p[ix] = np.nan

        except FileNotFoundError:
            nse_hr_p[ix] = np.nan
            nse_yy_p[ix] = np.nan

            nse_hr_p_changed[ix] = np.nan
            nse_yy_p_changed[ix] = np.nan

        # open the corresponding results file for LUE Model
        res_dict_lue = np.load(
            f"{lue_mod_res_path_default}/{site_id}_result.npy", allow_pickle=True
        ).item()

        try:
            res_dict_lue_changed = np.load(
                f"{lue_mod_res_path_changed}/{site_id}_result.npy", allow_pickle=True
            ).item()

            nse_hr_lue[ix] = res_dict_lue["NSE"][f"NSE_{res_dict_lue['Temp_res']}"]
            nse_yy_lue[ix] = res_dict_lue["NSE"]["NSE_y"]

            nse_hr_lue_changed[ix] = res_dict_lue_changed["NSE"][
                f"NSE_{res_dict_lue_changed['Temp_res']}"
            ]
            if np.isnan(nse_hr_lue_changed[ix]):
                nse_hr_lue[ix] = np.nan

            nse_yy_lue_changed[ix] = res_dict_lue_changed["NSE"]["NSE_y"]
            if np.isnan(nse_yy_lue_changed[ix]):
                nse_yy_lue[ix] = np.nan

        except FileNotFoundError:
            nse_hr_lue[ix] = np.nan
            nse_yy_lue[ix] = np.nan

            nse_hr_lue_changed[ix] = np.nan
            nse_yy_lue_changed[ix] = np.nan

    return (
        calc_nnse_rm_nan(nse_hr_p),
        calc_nnse_rm_nan(nse_hr_p_changed),
        calc_nnse_rm_nan(nse_yy_p),
        calc_nnse_rm_nan(nse_yy_p_changed),
        calc_nnse_rm_nan(nse_hr_lue),
        calc_nnse_rm_nan(nse_hr_lue_changed),
        calc_nnse_rm_nan(nse_yy_lue),
        calc_nnse_rm_nan(nse_yy_lue_changed),
    )


def plot_axs(ax, nnse_syr, nnse_syr_changed, title, bw_adjust, cut):

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
        x=nnse_syr_changed,
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

    # add vertical lines for the median values
    ax.axvline(x=np.median(nnse_syr), linestyle=":", color="#56B4E9")
    ax.axvline(x=np.median(nnse_syr_changed), linestyle=":", color="#009E73")

    ax.text(0.1, 18.0, f"{round(np.median(nnse_syr),3)}", fontsize=18, color="#56B4E9")
    ax.text(
        0.1,
        14.0,
        f"{round(np.median(nnse_syr_changed),3)}",
        fontsize=18,
        color="#009E73",
    )

    # set the axis properties
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticklabels([round(x, 1) for x in np.linspace(0.0, 1.0, 6).tolist()])
    ax.tick_params(axis="both", which="major", labelsize=26.0)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("")
    ax.set_title(f"{title} ({len(nnse_syr)})", size=30)

    sns.despine(ax=ax, top=True, right=True)


def plot_fig(
    p_default, p_changed, bao_default, bao_changed, changed_legend, fig_file_name
):

    (
        nnse_hr_p_syr,
        nnse_hr_p_syr_changed,
        nnse_yy_p_syr,
        nnse_yy_p_syr_changed,
        nnse_hr_lue_syr,
        nnse_hr_lue_syr_changed,
        nnse_yy_lue_syr,
        nnse_yy_lue_syr_changed,
    ) = get_nse_arr(
        p_default,
        p_changed,
        bao_default,
        bao_changed,
    )

    fig_width = 20
    fig_height = 12

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=2, nrows=2, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    plot_axs(
        axs[0, 0],
        nnse_hr_p_syr,
        nnse_hr_p_syr_changed,
        r"(a) P$^{\text{W}}_{\text{hr}}$ model",
        1.0,
        3,
    )

    plot_axs(
        axs[0, 1],
        nnse_hr_lue_syr,
        nnse_hr_lue_syr_changed,
        r"(b) Bao$_{\text{hr}}$ model",
        1.0,
        3,
    )

    plot_axs(
        axs[1, 0],
        nnse_yy_p_syr,
        nnse_yy_p_syr_changed,
        r"(c) P$^{\text{W}}_{\text{hr}}$ model",
        0.7,
        0.4,
    )

    plot_axs(
        axs[1, 1],
        nnse_yy_lue_syr,
        nnse_yy_lue_syr_changed,
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
    colors = ["#56B4E9", "#009E73"]
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
        for i, opti_type in enumerate(changed_legend)
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=28,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-0.2, -0.67),
    )

    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    
    plt.savefig(f"./supplement_figs/{fig_file_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"./supplement_figs/{fig_file_name}.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    p_model_syr_default_pop_cmaes = Path(
        "../model_results/P_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_default_pop_cmaes/serialized_model_results/"
    )
    p_model_syr_default_pop_cmaes_et_corr = Path(
        "../model_results/P_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_et_corr_default_pop_cmaes/serialized_model_results/"
    )
    p_model_syr_deafult_pop_cmaes_strict_data_filter = Path(
        "../model_results/P_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_strict_cost_nnse_unc_default_pop_cmaes/serialized_model_results/"
    )

    bao_model_syr_default_pop_cmaes = Path(
        "../model_results/LUE_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_default_pop_cmaes/serialized_model_results/"
    )
    bao_model_syr_default_pop_cmaes_et_corr = Path(
        "../model_results/LUE_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_et_corr_default_pop_cmaes/serialized_model_results/"
    )
    bao_model_syr_deafult_pop_cmaes_strict_data_filter = Path(
        "../model_results/LUE_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_strict_cost_lue_default_pop_cmaes/serialized_model_results/"
    )

    plot_fig(
        p_model_syr_default_pop_cmaes,
        p_model_syr_default_pop_cmaes_et_corr,
        bao_model_syr_default_pop_cmaes,
        bao_model_syr_default_pop_cmaes_et_corr,
        [r"using $\mathit{ET_{LE}}$", r"using $\mathit{ET_{LE_{CORR}}}$"],
        "fs12_overall_mod_perform_et_corr",
    )

    plot_fig(
        p_model_syr_default_pop_cmaes,
        p_model_syr_deafult_pop_cmaes_strict_data_filter,
        bao_model_syr_default_pop_cmaes,
        bao_model_syr_deafult_pop_cmaes_strict_data_filter,
        ["nominal data filtering", "strict data filtering"],
        "fs13_overall_mod_perform_strict",
    )
