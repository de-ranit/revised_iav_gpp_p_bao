#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot boxplots of pairwise differences in NNSE (at different temporal resolution)
between different model experiments

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Tue Jan 30 2024 12:35:02 CET
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


def calc_pairwise_nse_diff(exp_1_path, exp_2_path):
    """
    calculate pairwise difference in NSE between two experiments

    parameters:
    -----------
    exp_1_path (str): path to the serialized model results of experiment 1
    exp_2_path (str): path to the serialized model results of experiment 2

    returns:
    --------
    nse_diff_dict (dict): dictionary with absolute difference in NSE
    (at different temporal resolution) between the two experiments
    """

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{exp_1_path}/*.npy")
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

    # initialize arrays to store the difference in NSE
    # at different temporal resolution
    nse_diff_arr = np.zeros(len(filtered_mod_res_file_list))
    nse_diff_d_arr = np.zeros(len(filtered_mod_res_file_list))
    nse_diff_w_arr = np.zeros(len(filtered_mod_res_file_list))
    nse_diff_m_arr = np.zeros(len(filtered_mod_res_file_list))
    nse_diff_y_arr = np.zeros(len(filtered_mod_res_file_list))

    # for each site
    for ix, res_file in enumerate(filtered_mod_res_file_list):
        # load the serialized model results from experiment 1
        exp_1_mod_res = np.load(res_file, allow_pickle=True).item()
        site_id = exp_1_mod_res["SiteID"]

        # load the serialized model results from experiment 2
        exp_2_mod_res = np.load(
            f"{exp_2_path}/{site_id}_result.npy", allow_pickle=True
        ).item()

        # get the NSE from each experiment at different temporal resolutions
        exp_1_nse = exp_1_mod_res["NSE"][f"NSE_{exp_1_mod_res['Temp_res']}"]
        exp_2_nse = exp_2_mod_res["NSE"][f"NSE_{exp_2_mod_res['Temp_res']}"]

        exp_1_nse_d = exp_1_mod_res["NSE"]["NSE_d"]
        exp_2_nse_d = exp_2_mod_res["NSE"]["NSE_d"]

        exp_1_nse_w = exp_1_mod_res["NSE"]["NSE_w"]
        exp_2_nse_w = exp_2_mod_res["NSE"]["NSE_w"]

        exp_1_nse_m = exp_1_mod_res["NSE"]["NSE_m"]
        exp_2_nse_m = exp_2_mod_res["NSE"]["NSE_m"]

        exp_1_nse_y = exp_1_mod_res["NSE"]["NSE_y"]
        exp_2_nse_y = exp_2_mod_res["NSE"]["NSE_y"]

        # normalize NSE
        exp_1_nse = 1.0 / (2.0 - exp_1_nse)
        exp_2_nse = 1.0 / (2.0 - exp_2_nse)

        exp_1_nse_d = 1.0 / (2.0 - exp_1_nse_d)
        exp_2_nse_d = 1.0 / (2.0 - exp_2_nse_d)

        exp_1_nse_w = 1.0 / (2.0 - exp_1_nse_w)
        exp_2_nse_w = 1.0 / (2.0 - exp_2_nse_w)

        exp_1_nse_m = 1.0 / (2.0 - exp_1_nse_m)
        exp_2_nse_m = 1.0 / (2.0 - exp_2_nse_m)

        exp_1_nse_y = 1.0 / (2.0 - exp_1_nse_y)
        exp_2_nse_y = 1.0 / (2.0 - exp_2_nse_y)

        # calculate the difference in normalized NSE between the two experiments
        # at different temporal resolutions
        diff = exp_1_nse - exp_2_nse
        diff_d = exp_1_nse_d - exp_2_nse_d
        diff_w = exp_1_nse_w - exp_2_nse_w
        diff_m = exp_1_nse_m - exp_2_nse_m
        diff_y = exp_1_nse_y - exp_2_nse_y

        # collect the difference in NSE between the two experiments
        nse_diff_arr[ix] = diff
        nse_diff_d_arr[ix] = diff_d
        nse_diff_w_arr[ix] = diff_w
        nse_diff_m_arr[ix] = diff_m
        nse_diff_y_arr[ix] = diff_y

    # store the difference in NSE between the two experiments
    # at different temporal resolutions in a dictionary
    nse_diff_dict = {
        "nse_diff": nse_diff_arr,
        "nse_diff_d": nse_diff_d_arr,
        "nse_diff_w": nse_diff_w_arr,
        "nse_diff_m": nse_diff_m_arr,
        "nse_diff_y": nse_diff_y_arr,
    }

    return nse_diff_dict


def plot_axs(axs, in_dict_p, in_dict_lue, var):
    """
    plot the boxplots of pairwise difference in NNSE

    parameters:
    -----------
    axs (axes): axes object to plot the boxplots
    in_dict (dict): dictionary with absolute difference in NNSE between
    different optimization experiments
    var (str): variable to plot
    title (str): title of the plot

    returns:
    --------
    None
    """

    # collect all the arrays in a list
    data_p = []
    for val in in_dict_p.values():
        data_p.append(val[var])

    # remove NaN from all the arrays
    for idx, arr in enumerate(data_p):
        data_p[idx] = arr[~np.isnan(arr)]

    data_lue = []
    for val in in_dict_lue.values():
        data_lue.append(val[var])

    # remove NaN from all the arrays
    for idx, arr in enumerate(data_lue):
        data_lue[idx] = arr[~np.isnan(arr)]

    data_dict = {
        "site year - site_iav": {"p_model": data_p[0], "lue_model": data_lue[0]},
        "site year - site": {"p_model": data_p[1], "lue_model": data_lue[1]},
        "site year - PFT": {"p_model": data_p[2], "lue_model": data_lue[2]},
        "site year - global": {"p_model": data_p[3], "lue_model": data_lue[3]},
        "site_iav - site": {"p_model": data_p[4], "lue_model": data_lue[4]},
        "site_iav - PFT": {"p_model": data_p[5], "lue_model": data_lue[5]},
        "site_iav - global": {"p_model": data_p[6], "lue_model": data_lue[6]},
        "site - PFT": {"p_model": data_p[7], "lue_model": data_lue[7]},
        "site - global": {"p_model": data_p[8], "lue_model": data_lue[8]},
        "PFT - global": {"p_model": data_p[9], "lue_model": data_lue[9]},
    }

    # x axis labels
    xlabs = [
        r"site--year - site ($Cost^{IAV}$)",
        r"site--year - site",
        r"site--year - PFT",
        r"site--year - global",
        r"site ($Cost^{IAV}$) - site",
        r"site ($Cost^{IAV}$) - PFT",
        r"site ($Cost^{IAV}$) - global",
        "site - PFT",
        "site - global",
        "PFT - global",
    ]

    colors = ["#FEB337", "#015296"]
    # plot the boxplots
    for ix, opti_name in enumerate(data_dict.keys()):
        for jx, model_name in enumerate(["p_model", "lue_model"]):
            # position for the boxplot
            position = 2 * ix + jx + 0.5
            # create the boxplot with specific color
            axs.boxplot(  # bx =
                np.array(data_dict[opti_name][model_name]),
                widths=0.6,
                positions=[position],
                vert=True,
                patch_artist=True,
                showfliers=True,
                # flierprops=dict(marker="o", color="red", markersize=5),
                boxprops=dict(facecolor=colors[jx]),
                medianprops=dict(color="black"),
            )
            # axs.text(
            #     position,
            #     bx["medians"][0].get_ydata()[0],
            #     f"Median: {bx['medians'][0].get_ydata()[0]:.2f}",
            #     verticalalignment="bottom",
            # )

    axs.set_xticks(np.arange(1.0, 21.0, 2.0), labels=xlabs, rotation=45, ha="right")
    axs.tick_params(axis="both", which="major", labelsize=35)
    axs.set_xlabel(
        "Pairs of parameterization strategies",
        fontdict={"style": "italic", "size": 38},
    )
    axs.set_ylabel(
        "NNSE differences [-]",
        fontdict={"style": "italic", "size": 38},
    )

    grid_x = np.arange(2.0, 19.0, 2.0)
    for i in grid_x:
        axs.axvline(i, color="gray", linestyle=(0, (5, 10)))

    sns.despine(ax=axs, top=True, right=True)  # remove the top and right spines


def plot_fig_between_opti(in_dict_p, in_dict_lue, var, op_filename, op_folder):
    """
    plot the figure with the boxplots of pairwise difference in NNSE

    parameters:
    -----------
    in_dict_p (dict): dictionary with absolute difference in NNSE between
    different optimization experiments in P model
    in_dict_lue (dict): dictionary with absolute difference in NNSE between
    different optimization experiments in LUE model
    var (str): variable to plot
    op_filename (str): output filename
    op_folder (str): output folder

    returns:
    --------
    None

    """

    # dimension of figure
    fig_width = 16
    fig_height = 9

    # create the figure
    _, axs = plt.subplots(
        ncols=1, nrows=1, figsize=(fig_width, fig_height), sharex=True
    )

    plot_axs(axs, in_dict_p, in_dict_lue, var)

    # Adding legend manually
    colors = ["#FEB337", "#015296"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=40,
        )
        for i, opti_type in enumerate(
            [
                r"P$^{\text{W}}_{\text{hr}}$ model",
                r"Bao$_{\text{hr}}$ model",
            ]
        )
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=40,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=True,
        bbox_to_anchor=(0.5, -0.85),
    )

    # save the figure
    fig_path = Path(op_folder)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(f"./{op_folder}/{op_filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"./{op_folder}/{op_filename}.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")


def plot_fig_main(p_model_res_path, lue_model_res_path):
    """
    make the figure with the boxplots of pairwise difference

    parameters:
    -----------
    p_model_res_path (dict): dictionary with paths to serialized model results
    from different P model experiments
    lue_model_res_path (dict): dictionary with paths to serialized model results
    from different LUE model experiments

    returns:
    --------
    None
    """
    # ############################################################################
    # calculate the pairwise difference in NSE between different optimization
    # experiments in P Model
    nse_diff_p_per_site_yr_per_site_iav = calc_pairwise_nse_diff(
        p_model_res_path["per_site_yr"], p_model_res_path["per_site_iav"]
    )
    nse_diff_p_per_site_yr_per_site = calc_pairwise_nse_diff(
        p_model_res_path["per_site_yr"], p_model_res_path["per_site"]
    )
    nse_diff_p_per_site_yr_per_pft = calc_pairwise_nse_diff(
        p_model_res_path["per_site_yr"], p_model_res_path["per_pft"]
    )
    nse_diff_p_per_site_yr_glob_opti = calc_pairwise_nse_diff(
        p_model_res_path["per_site_yr"], p_model_res_path["glob_opti"]
    )
    nse_diff_p_per_site_iav_per_site = calc_pairwise_nse_diff(
        p_model_res_path["per_site_iav"], p_model_res_path["per_site"]
    )
    nse_diff_p_per_site_iav_per_pft = calc_pairwise_nse_diff(
        p_model_res_path["per_site_iav"], p_model_res_path["per_pft"]
    )
    nse_diff_p_per_site_iav_glob_opti = calc_pairwise_nse_diff(
        p_model_res_path["per_site_iav"], p_model_res_path["glob_opti"]
    )
    nse_diff_p_per_site_per_pft = calc_pairwise_nse_diff(
        p_model_res_path["per_site"], p_model_res_path["per_pft"]
    )
    nse_diff_p_per_site_glob_opti = calc_pairwise_nse_diff(
        p_model_res_path["per_site"], p_model_res_path["glob_opti"]
    )
    nse_diff_p_per_pft_glob_opti = calc_pairwise_nse_diff(
        p_model_res_path["per_pft"], p_model_res_path["glob_opti"]
    )

    ############################################################################
    # calculate the pairwise difference in NSE between different optimization
    # experiments in LUE Model
    nse_diff_lue_per_site_yr_per_site_iav = calc_pairwise_nse_diff(
        lue_model_res_path["per_site_yr"], lue_model_res_path["per_site_iav"]
    )
    nse_diff_lue_per_site_yr_per_site = calc_pairwise_nse_diff(
        lue_model_res_path["per_site_yr"], lue_model_res_path["per_site"]
    )
    nse_diff_lue_per_site_yr_per_pft = calc_pairwise_nse_diff(
        lue_model_res_path["per_site_yr"], lue_model_res_path["per_pft"]
    )
    nse_diff_lue_per_site_yr_glob_opti = calc_pairwise_nse_diff(
        lue_model_res_path["per_site_yr"], lue_model_res_path["glob_opti"]
    )
    nse_diff_lue_per_site_iav_per_site = calc_pairwise_nse_diff(
        lue_model_res_path["per_site_iav"], lue_model_res_path["per_site"]
    )
    nse_diff_lue_per_site_iav_per_pft = calc_pairwise_nse_diff(
        lue_model_res_path["per_site_iav"], lue_model_res_path["per_pft"]
    )
    nse_diff_lue_per_site_iav_glob_opti = calc_pairwise_nse_diff(
        lue_model_res_path["per_site_iav"], lue_model_res_path["glob_opti"]
    )
    nse_diff_lue_per_site_per_pft = calc_pairwise_nse_diff(
        lue_model_res_path["per_site"], lue_model_res_path["per_pft"]
    )
    nse_diff_lue_per_site_glob_opti = calc_pairwise_nse_diff(
        lue_model_res_path["per_site"], lue_model_res_path["glob_opti"]
    )
    nse_diff_lue_per_pft_glob_opti = calc_pairwise_nse_diff(
        lue_model_res_path["per_pft"], lue_model_res_path["glob_opti"]
    )

    ############################################################################
    diff_dict_p = {
        "0": nse_diff_p_per_site_yr_per_site_iav,
        "1": nse_diff_p_per_site_yr_per_site,
        "2": nse_diff_p_per_site_yr_per_pft,
        "3": nse_diff_p_per_site_yr_glob_opti,
        "4": nse_diff_p_per_site_iav_per_site,
        "5": nse_diff_p_per_site_iav_per_pft,
        "6": nse_diff_p_per_site_iav_glob_opti,
        "7": nse_diff_p_per_site_per_pft,
        "8": nse_diff_p_per_site_glob_opti,
        "9": nse_diff_p_per_pft_glob_opti,
    }

    diff_dict_lue = {
        "0": nse_diff_lue_per_site_yr_per_site_iav,
        "1": nse_diff_lue_per_site_yr_per_site,
        "2": nse_diff_lue_per_site_yr_per_pft,
        "3": nse_diff_lue_per_site_yr_glob_opti,
        "4": nse_diff_lue_per_site_iav_per_site,
        "5": nse_diff_lue_per_site_iav_per_pft,
        "6": nse_diff_lue_per_site_iav_glob_opti,
        "7": nse_diff_lue_per_site_per_pft,
        "8": nse_diff_lue_per_site_glob_opti,
        "9": nse_diff_lue_per_pft_glob_opti,
    }

    plot_fig_between_opti(
        diff_dict_p,
        diff_dict_lue,
        "nse_diff_y",
        "f09",
        "figures",
    )

    plot_fig_between_opti(
        diff_dict_p,
        diff_dict_lue,
        "nse_diff",
        "fs08",
        "supplement_figs",
    )


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    # store all the paths in a dict (for p model)
    hr_p_model_res_path_coll = {
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

    # plot the figure
    plot_fig_main(hr_p_model_res_path_coll, hr_lue_model_res_path_coll)
