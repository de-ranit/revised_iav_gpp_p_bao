#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Determine peak (percentile 90 or any other value) of observed and simulated GPP per
site year and calculate the ratio of the two (and plot its histogram) to get an
idea about in which experiments the peaks are overestimated or underestimated

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Fri Jun 14 2024 15:58:34 CEST
"""

import os
from pathlib import Path
import importlib
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sns

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams["font.family"] = 'STIXGeneral'
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2  # make the axes edge linewidth thicker


def get_gpp_peak_bias(res_path, percentile):
    """
    Determine peak (percentile 90 or any other value) of observed and simulated GPP per
    site year and calculate the ratio of the two

    Parameters:
    -----------
    res_path (str): path where serialized model results are stored from an experiment
    percentile (int): percentile value to determine peak GPP

    Returns:
    --------
    peak_bias_arr (np array): array of peak GPP bias values from all the good quality site years
    """

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{res_path}/*.npy")
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

    # list to collect arrays of peak GPP values from each site
    gpp_sim_obs_peak_val_ratio_list = []

    for res_file in filtered_mod_res_file_list:
        # load serialized model results
        result_dict = np.load(res_file, allow_pickle=True).item()

        gpp_filter_mask = np.where(
            result_dict[f"GPP_drop_idx_{result_dict['Temp_res']}"] == 1, False, True
        )  # construct mask to filter out bad GPP values
        gpp_sim_filtered = result_dict[f"GPP_sim_{result_dict['Temp_res']}"][
            gpp_filter_mask
        ]  # filter out bad GPP values
        gpp_obs_filtered = result_dict[f"GPP_NT_{result_dict['Temp_res']}"][
            gpp_filter_mask
        ]  # filter out bad GPP values
        # timestatmps of filtered GPP values
        time_filtered = result_dict[f"Time_{result_dict['Temp_res']}"][gpp_filter_mask]
        # years of filtered GPP values
        time_filtered_yrs = time_filtered.astype("datetime64[Y]").astype(int) + 1970
        # unique years of filtered GPP values
        filtered_yr_yrs = np.unique(time_filtered_yrs)

        # construct arrays of peak GPP values
        gpp_obs_peak_val_arr = np.zeros(len(filtered_yr_yrs))
        gpp_sim_peak_val_arr = np.zeros(len(filtered_yr_yrs))

        # for each good year in a site, calculate and store peak GPP (P90) values
        for ix, yr in enumerate(filtered_yr_yrs):
            yr_mask = np.where(time_filtered_yrs == yr, True, False)
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:  # some sites may not have any GPP values for a given year
                    gpp_obs_peak_val_arr[ix] = np.nanpercentile(
                        gpp_obs_filtered[yr_mask], percentile
                    )  # calculate peak GPP obs (P90) value
                except Warning as e:
                    print(
                        (
                            f"{e} occured for {result_dict['SiteID']}_{yr}"
                            " while calculating GPP obs peak"
                        )
                    )
                    gpp_obs_peak_val_arr[ix] = np.nan
                try:
                    gpp_sim_peak_val_arr[ix] = np.nanpercentile(
                        gpp_sim_filtered[yr_mask], percentile
                    )  # calculate peak GPP sim (P90) value
                except Warning as e:
                    print(
                        (
                            f"{e} occured for {result_dict['SiteID']}_{yr}"
                            " while calculating GPP sim peak"
                        )
                    )
                    gpp_sim_peak_val_arr[ix] = np.nan

        # place nan where peak of GPPobs is zero. There
        # seems to be almost no GPP in the whole year
        # happens in some bad site year (IT-Ren 2001, IT-Ren 2004)
        # where some points have good data quality flags.
        # place 1 where both peaks of GPPobs and GPPsim are zero
        gpp_sim_obs_peak_val_ratio_list.append(
            np.where(
                (gpp_sim_peak_val_arr == 0) & (gpp_obs_peak_val_arr == 0),
                1,
                np.divide(
                    gpp_sim_peak_val_arr,
                    gpp_obs_peak_val_arr,
                    out=np.full_like(gpp_sim_peak_val_arr, np.nan),
                    where=gpp_obs_peak_val_arr != 0,
                ),
            )
        )

    # concatenate all the peak GPP bias values from all the good quality site years
    peak_bias_arr = np.concatenate(gpp_sim_obs_peak_val_ratio_list)

    return peak_bias_arr


def plot_axs(
    axs,
    site_yr_arr_p,
    site_arr_p,
    site_iav_arr_p,
    pft_arr_p,
    glob_arr_p,
    site_yr_arr_lue,
    site_arr_lue,
    site_iav_arr_lue,
    pft_arr_lue,
    glob_arr_lue,
):
    """
    plot the boxplots of peak GPP bias values from each experiment

    Parameters:
    -----------
    axs (matplotlib axes): axes object to plot the boxplots
    arr (np array): array of peak GPP bias values from all the good quality site years
    title (str): title of the subplot

    Returns:
    --------
    None
    """

    # collect all the arrays in a list
    data = [
        site_yr_arr_p,
        site_arr_p,
        site_iav_arr_p,
        pft_arr_p,
        glob_arr_p,
        site_yr_arr_lue,
        site_arr_lue,
        site_iav_arr_lue,
        pft_arr_lue,
        glob_arr_lue,
    ]
    # remove NaN from all the arrays
    for idx, arr in enumerate(data):
        data[idx] = arr[~np.isnan(arr)]

    data_dict = {
        "Site year": {"p_model": data[0], "lue_model": data[5]},
        "Site": {"p_model": data[1], "lue_model": data[6]},
        "Site_iav": {"p_model": data[2], "lue_model": data[7]},
        "PFT": {"p_model": data[3], "lue_model": data[8]},
        "Global": {"p_model": data[4], "lue_model": data[9]},
    }

    opti_name_list = ["Site year", "Site_iav", "Site", "PFT", "Global"]
    colors = ["#FEB337", "#015296"]

    # plot the boxplots
    for ix, opti_name in enumerate(opti_name_list):
        for jx, model_name in enumerate(["p_model", "lue_model"]):
            # position for the boxplot
            position = 3 * ix + jx + 1
            # create the boxplot with specific color
            axs.boxplot(  # box_dict =
                np.array(data_dict[opti_name][model_name]),
                widths=0.5,
                positions=[position],
                vert=True,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", color="red", markersize=5),
                boxprops=dict(facecolor=colors[jx]),
                medianprops=dict(color="black"),
            )

            # for line in box_dict["medians"]:
            #     # Get the median value
            #     median_value = line.get_ydata()[0]

            #     # Get the x position for the annotation
            #     x_position = line.get_xdata()[1]

            #     # Get the y position for the annotation
            #     y_position = median_value

            #     # Place the text annotation on the plot
            #     axs.text(
            #         x_position,
            #         y_position,
            #         f"{median_value:.2f}",
            #         ha="center",
            #         va="bottom",
            #         fontsize=8,
            #         color="blue",
            #     )

            # IQR = (
            #     box_dict["whiskers"][1].get_ydata()[0]
            #     - box_dict["whiskers"][0].get_ydata()[0]
            # )
            # axs.text(
            #     x_position,
            #     y_position + 0.1,
            #     f"IQR: {IQR:.2f}",
            #     verticalalignment="bottom",
            #     color="black",
            #     fontsize=8,
            # )

    positions_x_ticks = [1.5, 4.5, 7.5, 10.5, 13.5]
    xlabs = [
        "per site year",
        "Per site \n" + r"($Cost^{IAV}$)",
        "per site",
        "per PFT",
        "Global",
    ]
    axs.set_xticks(positions_x_ticks, labels=xlabs, rotation=45, ha="right")
    # axs.set_xticklabels(xlabs)

    # set the limits and ticks of the axes
    axs.set_ylim(-0.2, 2.4)
    axs.set_xlim(-0.05, 14.4)
    axs.set_yticks(np.arange(-0.2, 2.1, 0.2))
    axs.set_yticklabels([f"{tick:.1f}" for tick in np.arange(-0.2, 2, 0.2)] + [r"$>$2"])

    # add the extreme fliers to the plot and annotate
    # total count of fliers
    data_p_model = [data_dict[opti_name]["p_model"] for opti_name in opti_name_list]
    data_lue_model = [data_dict[opti_name]["lue_model"] for opti_name in opti_name_list]

    for i, arr in enumerate(data_p_model):
        fliers = arr[arr > 2.2]
        flier_count = len(fliers)
        if flier_count > 0:
            y_position = 2.3  # Fixed position for the annotation
            axs.plot(
                [3 * i + 0 + 1] * flier_count,
                np.linspace(2.2, 2.4, flier_count),
                "o",
                color="red",
            )  # Plot the adjusted fliers
            axs.text(
                positions_x_ticks[i] - 1.4,
                y_position,
                f"N={flier_count}",
                va="center",
                color="black",
                fontsize=22,
            )  # Annotate with the count of fliers

    for i, arr in enumerate(data_lue_model):
        fliers = arr[arr > 2.2]
        flier_count = len(fliers)
        if flier_count > 0:
            y_position = 2.3  # Fixed position for the annotation
            axs.plot(
                [3 * i + 1 + 1] * flier_count,
                np.linspace(2.2, 2.4, flier_count),
                "o",
                color="red",
            )  # Plot the adjusted fliers
            axs.text(
                positions_x_ticks[i] + 0.6,
                y_position,
                f"N={flier_count}",
                va="center",
                color="black",
                fontsize=22,
            )  # Annotate with the count of fliers

    axs.tick_params(axis="both", which="major", labelsize=38)
    axs.set_xlabel(
        "Parameterization strategies",
        fontdict={"style": "italic", "size": 40},
    )
    axs.set_ylabel(
        r"Ratio of peaks of $P90_{GPP_{sim}}$" + "\nto " r"$P90_{GPP_{EC}}$ [-]",
        fontdict={"style": "italic", "size": 40},
    )

    sns.despine(ax=axs, top=True, right=True)  # remove the top and right spines


def plot_peak_bias_hist(p_model_res_path, lue_model_res_path, percentile):
    """
    plot the boxplot distribution of peak GPP bias values from each experiment

    Parameters:
    -----------
    p_model_res_path (dict): dict of paths where serialized model results are stored from
    different experiments of P model
    lue_model_res_path (dict): dict of paths where serialized model results are stored from
    different experiments of LUE model
    percentile (int): percentile value to determine peak GPP

    Returns:
    --------
    None
    """

    # get the array of peak GPP bias values from each experiment
    site_yr_p_model_peak_bias = get_gpp_peak_bias(
        p_model_res_path["per_site_yr"], percentile
    )
    per_site_p_model_peak_bias = get_gpp_peak_bias(
        p_model_res_path["per_site"], percentile
    )
    per_site_iav_p_model_peak_bias = get_gpp_peak_bias(
        p_model_res_path["per_site_iav"], percentile
    )
    per_pft_p_model_peak_bias = get_gpp_peak_bias(
        p_model_res_path["per_pft"], percentile
    )
    glob_opti_p_model_peak_bias = get_gpp_peak_bias(
        p_model_res_path["glob_opti"], percentile
    )

    site_yr_lue_model_peak_bias = get_gpp_peak_bias(
        lue_model_res_path["per_site_yr"], percentile
    )
    per_site_lue_model_peak_bias = get_gpp_peak_bias(
        lue_model_res_path["per_site"], percentile
    )
    per_site_iav_lue_model_peak_bias = get_gpp_peak_bias(
        lue_model_res_path["per_site_iav"], percentile
    )
    per_pft_lue_model_peak_bias = get_gpp_peak_bias(
        lue_model_res_path["per_pft"], percentile
    )
    glob_opti_lue_model_peak_bias = get_gpp_peak_bias(
        lue_model_res_path["glob_opti"], percentile
    )

    # dimension of figure
    fig_width = 16
    fig_height = 9

    # plot the boxplot of peak GPP bias values from each experiment
    _, axs = plt.subplots(
        ncols=1, nrows=1, figsize=(fig_width, fig_height), sharex=True
    )

    plot_axs(
        axs,
        site_yr_p_model_peak_bias,
        per_site_p_model_peak_bias,
        per_site_iav_p_model_peak_bias,
        per_pft_p_model_peak_bias,
        glob_opti_p_model_peak_bias,
        site_yr_lue_model_peak_bias,
        per_site_lue_model_peak_bias,
        per_site_iav_lue_model_peak_bias,
        per_pft_lue_model_peak_bias,
        glob_opti_lue_model_peak_bias,
    )

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
        # title="Legend",
        # title_fontsize=32,
        fontsize=40,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=True,
        bbox_to_anchor=(0.5, -0.68),
    )

    # save the figure
    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./supplement_figs/fs10.png", dpi=300, bbox_inches="tight")
    plt.savefig("./supplement_figs/fs10.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")


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

    # percentile value to determine peak GPP
    PERCENTILE_THRESHOLD_PEAK_GPP = 90

    # plot the histogram of peak GPP bias values from each experiment
    plot_peak_bias_hist(
        hr_p_model_res_path_coll,
        hr_lue_model_res_path_coll,
        PERCENTILE_THRESHOLD_PEAK_GPP,
    )
