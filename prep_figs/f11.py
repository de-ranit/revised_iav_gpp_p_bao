#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot correlation between differences in model performance in simulating peak GPP and annual GPP

author: rde
first created: Fri Jul 19 2024 14:48:41 CEST
"""
import os
from pathlib import Path
import importlib
import glob
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import seaborn as sns
from permetrics import RegressionMetric

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams["font.family"] = 'STIXGeneral'
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2  # make the axes edge linewidth thicker


def get_gpp_peak(res_path, percentile):
    """
    collect model performance in simulating peak GPP and annual GPP

    Parameters:
    -----------
    res_path (str): path where serialized model results are stored
    percentile (int): percentile value to determine peak GPP

    Returns:
    --------
    nnse_p90_per_site_list (np.array): array of model performance in simulating peak GPP
    nnse_annual_gpp_per_site_list (np.array): array of model performance in simulating annual GPP
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
    nnse_p90_per_site_list = []
    nnse_annual_gpp_per_site_list = []

    for res_file in filtered_mod_res_file_list:
        # load serialized model results
        result_dict = np.load(res_file, allow_pickle=True).item()

        if not np.isnan(result_dict["NSE"]["NSE_y"]):
            nnse_annual_gpp_per_site_list.append(
                1.0 / (2.0 - result_dict["NSE"]["NSE_y"])
            )

            good_yr_idx = result_dict["good_gpp_y_idx"].astype(bool)
            filtered_yearly_timestep = result_dict["Time_yearly"][good_yr_idx]

            good_years_in_yearly_ts = (
                filtered_yearly_timestep.astype("datetime64[Y]").astype(int) + 1970
            )

            hourly_data_mask = result_dict["GPP_drop_idx_Hourly"].astype(bool)
            filtered_hourly_timestep = result_dict["Time_Hourly"][~hourly_data_mask]

            gpp_obs_filtered = result_dict["GPP_NT_Hourly"][~hourly_data_mask]
            gpp_sim_filtered = result_dict["GPP_sim_Hourly"][~hourly_data_mask]

            coll_p90_obs = np.zeros(len(good_years_in_yearly_ts))
            coll_p90_sim = np.zeros(len(good_years_in_yearly_ts))

            for ix, years in enumerate(good_years_in_yearly_ts):
                year_mask = np.where(
                    filtered_hourly_timestep.astype("datetime64[Y]").astype(int) + 1970
                    == years,
                    True,
                    False,
                )
                gpp_obs_peak_val_arr = np.nanpercentile(
                    gpp_obs_filtered[year_mask], percentile
                )
                gpp_sim_peak_val_arr = np.nanpercentile(
                    gpp_sim_filtered[year_mask], percentile
                )

                coll_p90_obs[ix] = gpp_obs_peak_val_arr
                coll_p90_sim[ix] = gpp_sim_peak_val_arr

            nnse_evaluator_p90 = RegressionMetric(coll_p90_obs, coll_p90_sim, decimal=5)
            nnse_p90_per_site_list.append(1.0 / (2.0 - nnse_evaluator_p90.NSE()))

    return np.array(nnse_p90_per_site_list), np.array(nnse_annual_gpp_per_site_list)


def plot_axs_diff(
    ax,
    exp_1_p_model_nnse_p90,
    exp_1_lue_model_nnse_p90,
    exp_2_p_model_nnse_p90,
    exp_2_lue_model_nnse_p90,
    exp_1_p_model_annual_nnse,
    exp_1_lue_model_annual_nnse,
    exp_2_p_model_annual_nnse,
    exp_2_lue_model_annual_nnse,
    title,
):
    """
    plot the correlation between differences model performance in simulating peak GPP and annual GPP
    from two different optimization experiments

    Parameters:
    -----------
    ax (matplotlib.axes.Axes): axes object to plot the correlation
    exp_1_p_model_nnse_p90 (np.array): array of model performance in
        simulating (by P model) peak GPP from experiment 1
    exp_1_lue_model_nnse_p90 (np.array): array of model performance
        in simulating (by lue model) peak GPP from experiment 1
    exp_2_p_model_nnse_p90 (np.array): array of model performance in
        simulating (by P model) peak GPP from experiment 2
    exp_2_lue_model_nnse_p90 (np.array): array of model performance in
        simulatin (by lue model) peak GPP from experiment 2

    exp_1_p_model_annual_nnse (np.array): array of model performance in
        simulating (by P model) annual GPP from experiment 1
    exp_1_lue_model_annual_nnse (np.array): array of model performance
        in simulating (by lue model) annual GPP from experiment 1
    exp_2_p_model_annual_nnse (np.array): array of model performance in
        simulating (by P model) annual GPP from experiment 2
    exp_2_lue_model_annual_nnse (np.array): array of model performance in
        simulating (by lue model) annual GPP from experiment 2
    title (str): title of the plot

    Returns:
    --------
    None
    """

    ax.scatter(
        exp_1_p_model_nnse_p90 - exp_2_p_model_nnse_p90,
        exp_1_p_model_annual_nnse - exp_2_p_model_annual_nnse,
        color="#FEB337",
        alpha=0.7,
    )
    ax.scatter(
        exp_1_lue_model_nnse_p90 - exp_2_lue_model_nnse_p90,
        exp_1_lue_model_annual_nnse - exp_2_lue_model_annual_nnse,
        color="#015296",
        alpha=0.6,
        marker="^",
    )

    corr_p = ma.corrcoef(
        ma.masked_invalid(exp_1_p_model_annual_nnse - exp_2_p_model_annual_nnse),
        ma.masked_invalid(exp_1_p_model_nnse_p90 - exp_2_p_model_nnse_p90),
    )[0, 1]
    corr_lue = ma.corrcoef(
        ma.masked_invalid(exp_1_lue_model_annual_nnse - exp_2_lue_model_annual_nnse),
        ma.masked_invalid(exp_1_lue_model_nnse_p90 - exp_2_lue_model_nnse_p90),
    )[0, 1]

    ax.set_title(
        title
        + "\n"
        + r"P$^{\text{W}}_{\text{hr}}$ model:"
        + f" {corr_p:.2f}, "
        + r"Bao$_{\text{hr}}$ model:"
        + f" {corr_lue:.2f}",
        fontsize=42,
        pad=29,
    )

    ax.set_xticks(np.arange(-1, 1.5, 0.5))
    ax.set_yticks(np.arange(-1, 1.5, 0.5))
    ax.set_xticklabels(np.arange(-1, 1.5, 0.5))
    ax.set_yticklabels(np.arange(-1, 1.5, 0.5))
    ax.tick_params(axis="both", which="major", labelsize=38)

    # plot 1:1 line
    ax.plot(np.array([-1.0, 1.0]), np.array([-1.0, 1.0]), color="black")

    # set the x and y axis limits
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    # set aspect ratio equal
    ax.set_aspect("equal")

    sns.despine(ax=ax, top=True, right=True)

    # get the values which are reported in results section
    # if (title == r"(d) $j1$ = site--year, $j2$ = global") or (
    #     title == r"(e) $j1$ = site ($Cost^{IAV}$), $j2$ = site"
    # ):
    #     diff_nnse_p90_p_model = exp_1_p_model_nnse_p90 - exp_2_p_model_nnse_p90
    #     diff_nnse_annual_p_model = exp_1_p_model_annual_nnse - exp_2_p_model_annual_nnse

    #     diff_nnse_p90_lue_model = exp_1_lue_model_nnse_p90 - exp_2_lue_model_nnse_p90
    #     diff_nnse_annual_lue_model = (
    #         exp_1_lue_model_annual_nnse - exp_2_lue_model_annual_nnse
    #     )

    #     positive_mask_p_model = (diff_nnse_p90_p_model > 0) & (
    #         diff_nnse_annual_p_model > 0
    #     )
    #     positive_mask_lue_model = (diff_nnse_p90_lue_model > 0) & (
    #         diff_nnse_annual_lue_model > 0
    #     )

    #     perc_positive_val_p_model = round(
    #         (np.sum(positive_mask_p_model) / len(positive_mask_p_model)) * 100.0, 0
    #     )
    #     perc_positive_val_lue_model = round(
    #         (np.sum(positive_mask_lue_model) / len(positive_mask_lue_model)) * 100.0, 0
    #     )

    #     print(
    #         (
    #             "Percentage of positive values for P model:"
    #             f"{perc_positive_val_p_model} when {title}"
    #         )
    #     )
    #     print(
    #         (
    #             "Percentage of positive values for LUE model:"
    #             f"{perc_positive_val_lue_model} when {title}"
    #         )
    #     )


def set_ticks_for_selected_subplots(axs, selected_indices):
    """
    hide subplot x axis ticks and only enable for given subplots
    """

    # Hide x-axis ticks for all subplots
    for row in axs:
        for ax in row:
            ax.tick_params(axis="x", which="both", top=False, labelbottom=False)

    # Enable x-axis ticks for selected subplots
    for i, j in selected_indices:
        axs[i][j].tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=True
        )


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
    site_yr_p_model_nnse_p90, site_yr_p_model_annual_nnse = get_gpp_peak(
        p_model_res_path["per_site_yr"], percentile
    )
    per_site_p_model_nnse_p90, per_site_p_model_annual_nnse = get_gpp_peak(
        p_model_res_path["per_site"], percentile
    )
    per_site_iav_p_model_nnse_p90, per_site_iav_p_model_annual_nnse = get_gpp_peak(
        p_model_res_path["per_site_iav"], percentile
    )
    per_pft_p_model_nnse_p90, per_pft_p_model_annual_nnse = get_gpp_peak(
        p_model_res_path["per_pft"], percentile
    )
    glob_opti_p_model_nnse_p90, glob_opti_p_model_annual_nnse = get_gpp_peak(
        p_model_res_path["glob_opti"], percentile
    )

    site_yr_lue_model_nnse_p90, site_yr_lue_model_annual_nnse = get_gpp_peak(
        lue_model_res_path["per_site_yr"], percentile
    )
    per_site_lue_model_nnse_p90, per_site_lue_model_annual_nnse = get_gpp_peak(
        lue_model_res_path["per_site"], percentile
    )
    per_site_iav_lue_model_nnse_p90, per_site_iav_lue_model_annual_nnse = get_gpp_peak(
        lue_model_res_path["per_site_iav"], percentile
    )
    per_pft_lue_model_nnse_p90, per_pft_lue_model_annual_nnse = get_gpp_peak(
        lue_model_res_path["per_pft"], percentile
    )
    glob_opti_lue_model_nnse_p90, glob_opti_lue_model_annual_nnse = get_gpp_peak(
        lue_model_res_path["glob_opti"], percentile
    )

    fig_width = 30
    fig_height = 36

    fig, axs = plt.subplots(
        ncols=3, nrows=4, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )

    plot_axs_diff(
        axs[0, 0],
        site_yr_p_model_nnse_p90,
        site_yr_lue_model_nnse_p90,
        per_site_iav_p_model_nnse_p90,
        per_site_iav_lue_model_nnse_p90,
        site_yr_p_model_annual_nnse,
        site_yr_lue_model_annual_nnse,
        per_site_iav_p_model_annual_nnse,
        per_site_iav_lue_model_annual_nnse,
        r"(a) $j1$ = site--year, $j2$ = site ($Cost^{IAV}$)",
    )

    plot_axs_diff(
        axs[0, 1],
        site_yr_p_model_nnse_p90,
        site_yr_lue_model_nnse_p90,
        per_site_p_model_nnse_p90,
        per_site_lue_model_nnse_p90,
        site_yr_p_model_annual_nnse,
        site_yr_lue_model_annual_nnse,
        per_site_p_model_annual_nnse,
        per_site_lue_model_annual_nnse,
        r"(b) $j1$ = site--year, $j2$ = site",
    )

    plot_axs_diff(
        axs[0, 2],
        site_yr_p_model_nnse_p90,
        site_yr_lue_model_nnse_p90,
        per_pft_p_model_nnse_p90,
        per_pft_lue_model_nnse_p90,
        site_yr_p_model_annual_nnse,
        site_yr_lue_model_annual_nnse,
        per_pft_p_model_annual_nnse,
        per_pft_lue_model_annual_nnse,
        r"(c) $j1$ = site--year, $j2$ = PFT",
    )

    plot_axs_diff(
        axs[1, 0],
        site_yr_p_model_nnse_p90,
        site_yr_lue_model_nnse_p90,
        glob_opti_p_model_nnse_p90,
        glob_opti_lue_model_nnse_p90,
        site_yr_p_model_annual_nnse,
        site_yr_lue_model_annual_nnse,
        glob_opti_p_model_annual_nnse,
        glob_opti_lue_model_annual_nnse,
        r"(d) $j1$ = site--year, $j2$ = global",
    )

    plot_axs_diff(
        axs[1, 1],
        per_site_iav_p_model_nnse_p90,
        per_site_iav_lue_model_nnse_p90,
        per_site_p_model_nnse_p90,
        per_site_lue_model_nnse_p90,
        per_site_iav_p_model_annual_nnse,
        per_site_iav_lue_model_annual_nnse,
        per_site_p_model_annual_nnse,
        per_site_lue_model_annual_nnse,
        r"(e) $j1$ = site ($Cost^{IAV}$), $j2$ = site",
    )

    plot_axs_diff(
        axs[1, 2],
        per_site_iav_p_model_nnse_p90,
        per_site_iav_lue_model_nnse_p90,
        per_pft_p_model_nnse_p90,
        per_pft_lue_model_nnse_p90,
        per_site_iav_p_model_annual_nnse,
        per_site_iav_lue_model_annual_nnse,
        per_pft_p_model_annual_nnse,
        per_pft_lue_model_annual_nnse,
        r"(f) $j1$ = site ($Cost^{IAV}$), $j2$ = PFT",
    )

    plot_axs_diff(
        axs[2, 0],
        per_site_iav_p_model_nnse_p90,
        per_site_iav_lue_model_nnse_p90,
        glob_opti_p_model_nnse_p90,
        glob_opti_lue_model_nnse_p90,
        per_site_iav_p_model_annual_nnse,
        per_site_iav_lue_model_annual_nnse,
        glob_opti_p_model_annual_nnse,
        glob_opti_lue_model_annual_nnse,
        r"(g) $j1$ = site ($Cost^{IAV}$), $j2$ = global",
    )

    plot_axs_diff(
        axs[2, 1],
        per_site_p_model_nnse_p90,
        per_site_lue_model_nnse_p90,
        per_pft_p_model_nnse_p90,
        per_pft_lue_model_nnse_p90,
        per_site_p_model_annual_nnse,
        per_site_lue_model_annual_nnse,
        per_pft_p_model_annual_nnse,
        per_pft_lue_model_annual_nnse,
        r"(h) $j1$ = site, $j2$ = PFT",
    )

    plot_axs_diff(
        axs[2, 2],
        per_site_p_model_nnse_p90,
        per_site_lue_model_nnse_p90,
        glob_opti_p_model_nnse_p90,
        glob_opti_lue_model_nnse_p90,
        per_site_p_model_annual_nnse,
        per_site_lue_model_annual_nnse,
        glob_opti_p_model_annual_nnse,
        glob_opti_lue_model_annual_nnse,
        r"(i) $j1$ = site, $j2$ = global",
    )

    plot_axs_diff(
        axs[3, 0],
        per_pft_p_model_nnse_p90,
        per_pft_lue_model_nnse_p90,
        glob_opti_p_model_nnse_p90,
        glob_opti_lue_model_nnse_p90,
        per_pft_p_model_annual_nnse,
        per_pft_lue_model_annual_nnse,
        glob_opti_p_model_annual_nnse,
        glob_opti_lue_model_annual_nnse,
        r"(j) $j1$ = PFT, $j2$ = global",
    )

    set_ticks_for_selected_subplots(axs, [(2, 1), (2, 2), (3, 0)])

    fig.delaxes(axs[3, 1])
    fig.delaxes(axs[3, 2])
    fig.subplots_adjust(hspace=0.6, wspace=0.5)

    # Adding legend manually
    colors = ["#FEB337", "#015296"]
    markers = ["o", "^"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=markers[i],
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=30,
        )
        for i, opti_type in enumerate(
            [
                r"P$^{\text{W}}_{\text{hr}}$ model",
                r"Bao$_{\text{hr}}$ model",
            ]
        )
    ]

    legend_elements.append(
        Line2D(
            [0],
            [0],
            lw=4,
            linestyle="-",
            color="black",
            label="1:1 line",
            markerfacecolor="black",
            markersize=40,
        ),
    )

    plt.legend(
        handles=legend_elements,
        fontsize=40,
        loc="lower right",
        frameon=True,
        bbox_to_anchor=(3.0, 0.3),
    )

    fig.supxlabel(
        r"$\Delta NNSE_{P90}$ [-]",
        y=0.07,
        fontsize=55,
    )
    fig.supylabel(
        r"$\Delta NNSE_{y}$ [-]",
        x=0.04,
        fontsize=55,
    )

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f11.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f11.pdf", dpi=300, bbox_inches="tight")
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
