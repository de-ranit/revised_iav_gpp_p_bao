#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare f03 - uncertainty analysis in annual GPP

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Tue Jan 21 2025 16:57:23 CET
"""
import os
import sys
from pathlib import Path
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd


# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

# import the parameter module to get intial parameter values
from src.common.get_data import read_nc_data  # pylint: disable=C0413
from src.common.get_data import df_to_dict  # pylint: disable=C0413
from src.postprocess.prep_results import filter_data_up_dd  # pylint: disable=C0413

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


def prep_data(exp_path, ip_data_path, is_p_model=False):
    """
    Prepare data for the uncertainty analysis. Aggregate several variables
    of EC derived GPP to annual scale, and quality filter the data. Calculate range
    of annual GPP obtained from various GPP variables and calculate how much of simulated
    annual GPP is within the range

    Parameters:
    -----------
    exp_path (str) : path to the directory containing the model results
    ip_data_path (str) : path to the directory containing the input data/ forcing data
    is_p_model (bool) : whether the model is a P model or not

    Returns:
    --------
    return_dict (dict) : dictionary containing the 
    fraction of simulated GPP within the min-max range
    """

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{exp_path}/*.npy")
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

    # collect the fraction of simulated GPP within the min-max range
    frac_gpp_sim_within_min_max_list = []
    frac_gpp_sim_no_moisture_within_min_max_list = []

    for (
        file
    ) in (
        filtered_mod_res_file_list
    ):  # only for sites where there is more than three good years

        res_file = np.load(file, allow_pickle=True).item()

        if ~np.isnan(res_file["NSE"]["NSE_y"]):
            forcing_data_filename = glob.glob(
                os.path.join(ip_data_path, f"{res_file['SiteID']}.*.nc")
            )[0]
            ip_df_dict = read_nc_data(
                forcing_data_filename, {"fPAR_var": "FPAR_FLUXNET_EO"}
            )

            gpp_y_coll_dict = {}
            for parti_method in ["NT", "DT"]:
                for qntl in [5, 16, 25, 50, 75, 84, 95]:
                    drop_gpp_data_indices = res_file[
                        f"GPP_drop_idx_{ip_df_dict['Temp_res']}"
                    ].astype(bool)

                    # create a df to resmaple the GPP values to annual scale
                    gpp_resample_df = pd.DataFrame(
                        {
                            "Time": ip_df_dict["Time"],
                            "GPP_obs": ip_df_dict[f"GPP_{parti_method}_{qntl}"],
                            "GPP_sim": res_file[f"GPP_sim_{ip_df_dict['Temp_res']}"],
                            "NEE_QC": ip_df_dict[f"NEE_QC_{parti_method}_{qntl}"],
                            "drop_gpp_idx": ~drop_gpp_data_indices,
                        }
                    )

                    # resample to annual
                    gpp_df_y = gpp_resample_df.resample("Y", on="Time").mean()
                    gpp_df_y = gpp_df_y.reset_index()
                    gpp_y_dict = df_to_dict(gpp_df_y)
                    gpp_obs_y_filtered, _, time_y_filtered, _ = filter_data_up_dd(
                        gpp_y_dict
                    )  # exclude years with mostly bad data

                    # index of common years between annual GPP used for
                    # model eval and other GPP variables
                    common_idx = np.where(
                        time_y_filtered == res_file["Time_yearly_filtered"]
                    )[0]
                    common_gpp_sim_y = gpp_obs_y_filtered[common_idx]

                    gpp_y_coll_dict[f"GPP_{parti_method}_{qntl}_y"] = common_gpp_sim_y

            gpp_y_coll_dict["GPP_NT_yearly_filtered"] = res_file[
                "GPP_NT_yearly_filtered"
            ]

            stacked_gpp_obs_arrays = np.vstack(list(gpp_y_coll_dict.values()))

            # get the range (min-max) of GPP from different variables
            min_array = np.min(stacked_gpp_obs_arrays, axis=0)
            max_array = np.max(stacked_gpp_obs_arrays, axis=0)

            # calculate fraction of site years in a site within the range
            mask_gpp_sim_within_min_max = (
                res_file["GPP_sim_yearly_filtered"] >= min_array
            ) & (res_file["GPP_sim_yearly_filtered"] <= max_array)
            frac_gpp_sim_within_min_max = np.sum(mask_gpp_sim_within_min_max) / len(
                mask_gpp_sim_within_min_max
            )

            frac_gpp_sim_within_min_max_list.append(frac_gpp_sim_within_min_max)

            # if P-model; calculate the same for GPP simulated with no moisture stress
            if is_p_model:
                good_yr_mask = res_file["good_gpp_y_idx"].astype(bool)
                mask_gpp_sim_within_min_max = (
                    res_file["GPP_sim_no_moisture_yearly"][good_yr_mask] >= min_array
                ) & (res_file["GPP_sim_no_moisture_yearly"][good_yr_mask] <= max_array)
                frac_gpp_sim_within_min_max = np.sum(mask_gpp_sim_within_min_max) / len(
                    mask_gpp_sim_within_min_max
                )

                frac_gpp_sim_no_moisture_within_min_max_list.append(
                    frac_gpp_sim_within_min_max
                )

    # prepare the return dictionary
    if is_p_model:
        return_dict = {
            "frac_within_min_max": np.array(frac_gpp_sim_within_min_max_list),
            "frac_no_moisture_within_min_max": np.array(
                frac_gpp_sim_no_moisture_within_min_max_list
            ),
        }
    else:
        return_dict = {
            "frac_within_min_max": np.array(frac_gpp_sim_within_min_max_list)
        }

    return return_dict


def plot_axs(
    ax, frac_syr, frac_allyr, frac_allyr_iav, frac_pft, frac_glob, title, bw_adjust, cut
):
    """
    plot the KDE of fraction of site years within range of GPP for different experiments
    Parameters:
    -----------
    ax (matplotlib axis) : axis to plot the histogram
    frac_syr (np.array) : fraction of site years within range of
    GPP for per site--year parameterization
    frac_allyr (np.array) : fraction of site years within range of GPP for per site parameterization
    frac_allyr_iav (np.array) : fraction of site years within range
    of GPP for per site parameterization using IAV
    frac_pft (np.array) : fraction of site years within range of GPP for per PFT parameterization
    frac_glob (np.array) : fraction of site years within range of GPP for global parameterization
    title (str) : title of the plot
    bw_adjust (float) : bandwidth adjustment for the KDE
    cut (float) : cut off value for the KDE

    Returns:
    --------
    dict : dictionary containing the mean fraction for each optimization type

    """

    # plot the histograms and KDE - but make histogram invisible and only show KDE
    sns.histplot(
        x=frac_syr,
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
        x=frac_allyr,
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
        x=frac_pft,
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
        x=frac_glob,
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
        x=frac_allyr_iav,
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

    ax.axvline(x=np.mean(frac_syr), linestyle=":", color="#56B4E9")
    ax.axvline(x=np.mean(frac_allyr), linestyle=":", color="#009E73")
    ax.axvline(x=np.mean(frac_pft), linestyle=":", color="#BBBBBB")
    ax.axvline(x=np.mean(frac_glob), linestyle=":", color="#CC79A7")
    ax.axvline(x=np.mean(frac_allyr_iav), linestyle=":", color="#E6C300")

    # set the axis properties
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticklabels([round(x, 1) for x in np.linspace(0.0, 1.0, 6).tolist()])
    ax.tick_params(axis="both", which="major", labelsize=26.0)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("")
    ax.set_title(f"{title} ({len(frac_syr)})", size=30)

    sns.despine(ax=ax, top=True, right=True)

    return {
        "syr_med": np.mean(frac_syr),
        "nnse_allyr_iav_med": np.mean(frac_allyr_iav),
        "allyr_med": np.mean(frac_allyr),
        "pft_med": np.mean(frac_pft),
        "glob_med": np.mean(frac_glob),
    }


def plot_fig_main(
    hr_p_model_res_path_coll,
    hr_bao_model_res_path_coll,
    dd_bao_model_res_path_coll,
    ip_data_path,
    ip_data_path_dd,
):
    """
    prepare the data and plot the figure

    Parameters:
    -----------
    hr_p_model_res_path_coll (dict) : dictionary containing the paths to the results of P model
    hr_bao_model_res_path_coll (dict) : dictionary containing the paths to the results of Bao model
    calibrated using hourly data
    dd_bao_model_res_path_coll (dict) : dictionary containing the paths to the results of Bao model
    calibrated using daily data
    ip_data_path (str) : path to the directory containing the input data/ forcing data at hourly
    resolution
    ip_data_path_dd (str) : path to the directory containing the input data/ forcing data at daily
    resolution

    Returns:
    --------
    None

    """

    # prepare the data
    hr_p_model_dict = {}
    for exp in hr_p_model_res_path_coll.keys():
        hr_p_model_dict[exp] = prep_data(
            hr_p_model_res_path_coll[exp], ip_data_path, True
        )

    hr_bao_model_dict = {}
    for exp in hr_bao_model_res_path_coll.keys():
        hr_bao_model_dict[exp] = prep_data(
            hr_bao_model_res_path_coll[exp], ip_data_path
        )

    dd_bao_model_dict = {}
    for exp in dd_bao_model_res_path_coll.keys():
        dd_bao_model_dict[exp] = prep_data(
            dd_bao_model_res_path_coll[exp], ip_data_path_dd
        )

    # prepare the figure
    fig_width = 12
    fig_height = 9

    fig, axs = plt.subplots(
        ncols=2, nrows=2, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    hr_p_frac_dict = plot_axs(
        axs[0, 0],
        hr_p_model_dict["per_site_yr"]["frac_within_min_max"],
        hr_p_model_dict["per_site"]["frac_within_min_max"],
        hr_p_model_dict["per_site_iav"]["frac_within_min_max"],
        hr_p_model_dict["per_pft"]["frac_within_min_max"],
        hr_p_model_dict["glob_opti"]["frac_within_min_max"],
        r"(a) P$^{\text{W}}_{\text{hr}}$ model",
        1.0,
        3,
    )

    hr_p_no_sm_frac_dict = plot_axs(
        axs[0, 1],
        hr_p_model_dict["per_site_yr"]["frac_no_moisture_within_min_max"],
        hr_p_model_dict["per_site"]["frac_no_moisture_within_min_max"],
        hr_p_model_dict["per_site_iav"]["frac_no_moisture_within_min_max"],
        hr_p_model_dict["per_pft"]["frac_no_moisture_within_min_max"],
        hr_p_model_dict["glob_opti"]["frac_no_moisture_within_min_max"],
        r"(b) P$_{\text{hr}}$ model",
        1.0,
        3,
    )

    hr_bao_frac_dict = plot_axs(
        axs[1, 0],
        hr_bao_model_dict["per_site_yr"]["frac_within_min_max"],
        hr_bao_model_dict["per_site"]["frac_within_min_max"],
        hr_bao_model_dict["per_site_iav"]["frac_within_min_max"],
        hr_bao_model_dict["per_pft"]["frac_within_min_max"],
        hr_bao_model_dict["glob_opti"]["frac_within_min_max"],
        r"(c) Bao$_{\text{hr}}$ model",
        1.0,
        3,
    )

    dd_bao_frac_dict = plot_axs(
        axs[1, 1],
        dd_bao_model_dict["per_site_yr"]["frac_within_min_max"],
        dd_bao_model_dict["per_site"]["frac_within_min_max"],
        dd_bao_model_dict["per_site_iav"]["frac_within_min_max"],
        dd_bao_model_dict["per_pft"]["frac_within_min_max"],
        dd_bao_model_dict["glob_opti"]["frac_within_min_max"],
        r"(d) Bao$_{\text{dd}}$ model",
        1.0,
        3,
    )

    fig.supxlabel(
        r"Fraction of site--year per site within range of $\text{GPP}_{\text{EC}}$ [-]",
        y=-0.03,
        fontsize=34,
    )
    fig.supylabel("Fraction of" + r" sites [\%]", fontsize=34)

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
        fontsize=24,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(-0.1, -1.0),
    )

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f03_new.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f03_new.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")

    # get the mean fraction of simulated GPP within the range for each optimization type
    unc_df = pd.DataFrame(
        [hr_p_frac_dict, hr_p_no_sm_frac_dict, hr_bao_frac_dict, dd_bao_frac_dict]
    )
    print(unc_df.to_latex(index=False, float_format="%.3f"))


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    hr_p_model_res_path_coll_dict = {
        "per_site_yr": result_paths.per_site_yr_p_model_res_path,
        "per_site": result_paths.per_site_p_model_res_path,
        "per_site_iav": result_paths.per_site_p_model_res_path_iav,
        "per_pft": result_paths.per_pft_p_model_res_path,
        "glob_opti": result_paths.glob_opti_p_model_res_path,
    }
    # store all the paths in a dict (for lue model)
    hr_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_lue_model_res_path,
        "per_site": result_paths.per_site_lue_model_res_path,
        "per_site_iav": result_paths.per_site_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_lue_model_res_path,
    }
    # store all the paths in a dict (for lue model in daily resolution)
    dd_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_dd_lue_model_res_path,
        "per_site": result_paths.per_site_dd_lue_model_res_path,
        "per_site_iav": result_paths.per_site_dd_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_dd_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_dd_lue_model_res_path,
    }

    hr_ip_data_path = result_paths.hr_ip_data_path
    dd_ip_data_path = result_paths.dd_ip_data_path

    plot_fig_main(
        hr_p_model_res_path_coll_dict,
        hr_lue_model_res_path_coll,
        dd_lue_model_res_path_coll,
        hr_ip_data_path,
        dd_ip_data_path,
    )
