#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot boxplots of NSE from different optimization experiments of P Model
and LUE Model per bioclimatic class

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Thu Feb 01 2024 18:38:15 CET
"""

import os
from pathlib import Path
import importlib
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import seaborn as sns
from scipy import stats

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = (
    r"\renewcommand{\familydefault}{\sfdefault}"  # use sans-serif font
)
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"  # use amsmath font

# set the font to computer modern sans
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "cmss10"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def determine_bioclim(pft, kg):
    """
    Determine the bioclimatic class of the site
    based on the PFT and KG of the site

    Parameters:
    -----------
    pft (str) : PFT of the site
    kg (str) : KG of the site

    Returns:
    --------
    bioclim (str) : bioclimatic class of the site
    """
    if pft in ["EBF", "ENF", "DBF", "DNF", "MF", "WSA", "OSH", "CSH"]:
        bio = "forest"
    elif pft in ["GRA", "SAV", "CRO", "WET"]:
        bio = "grass"
    elif pft == "SNO":
        bio = "Polar"
    else:
        raise ValueError("PFT not recognized")

    if (kg[0] == "A") & (bio == "forest"):
        bioclim = "TropicalF"
    elif (kg[0] == "A") & (bio == "grass"):
        bioclim = "TropicalG"
    elif (kg[0] == "B") & (bio == "forest"):
        bioclim = "AridF"
    elif (kg[0] == "B") & (bio == "grass"):
        bioclim = "AridG"
    elif (kg[0] == "C") & (bio == "forest"):
        bioclim = "TemperateF"
    elif (kg[0] == "C") & (bio == "grass"):
        bioclim = "TemperateG"
    elif (kg[0] == "D") & (bio == "forest"):
        bioclim = "BorealF"
    elif (kg[0] == "D") & (bio == "grass"):
        bioclim = "BorealG"
    elif kg[0] == "E":
        bioclim = "Polar"
    else:
        raise ValueError(f"Bioclim could not be determined for {pft} and {kg}")

    return bioclim


def prep_data(res_path_dict):
    """
    prepare data to plot boxplots of NSE from different optimization
    experiments of P Model and LUE Model per bioclimate class

    Parameters:
    -----------
    res_path_dict (dict) : dict containing paths to the serialized model
    results from different optimization experiments of a certain model

    Returns:
    --------
    bioclim_mod_perform_dict (dict) : dict containing NSE from different
    optimization experiments of a certain Model per bioclimatic class
    """

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{res_path_dict['per_site_yr']}/*.npy")
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

    # create a dict to store the results from each bioclimatic class and each optimization
    # experiment
    bioclim_mod_perform_dict = {
        "TropicalF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "TropicalG": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "AridF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "AridG": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "TemperateF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "TemperateG": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "BorealF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "BorealG": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "Polar": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
    }

    # for each site
    for res_file in filtered_mod_res_file_list:
        # open the results file from per site year optimization
        res_dict_site_yr = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_site_yr["SiteID"]

        # open the results file from other experiments
        res_dict_site = np.load(
            f"{res_path_dict['per_site']}/{site_id}_result.npy", allow_pickle=True
        ).item()
        res_dict_pft = np.load(
            f"{res_path_dict['per_pft']}/{site_id}_result.npy", allow_pickle=True
        ).item()
        res_dict_glob = np.load(
            f"{res_path_dict['glob_opti']}/{site_id}_result.npy", allow_pickle=True
        ).item()

        pft = res_dict_site_yr["PFT"]  # get the site PFT
        kg = res_dict_site_yr["KG"]  # get the site KG
        bioclim = determine_bioclim(
            pft, kg
        )  # determine the bioclimatic class of the site

        # store NSE from different optimization experiments of a certain
        # Model based on the bioclimatic class of the site
        bioclim_mod_perform_dict[bioclim]["per_site_yr"].append(
            res_dict_site_yr["NSE"][f"NSE_{res_dict_site_yr['Temp_res']}"]
        )
        bioclim_mod_perform_dict[bioclim]["per_site"].append(
            res_dict_site["NSE"][f"NSE_{res_dict_site['Temp_res']}"]
        )
        bioclim_mod_perform_dict[bioclim]["per_pft"].append(
            res_dict_pft["NSE"][f"NSE_{res_dict_pft['Temp_res']}"]
        )
        bioclim_mod_perform_dict[bioclim]["glob"].append(
            res_dict_glob["NSE"][f"NSE_{res_dict_glob['Temp_res']}"]
        )

    return bioclim_mod_perform_dict


def plot_stat_significance(in_dict):
    """
    after https://rowannicholls.github.io/python/graphs/ax_based/boxplots_significance.html

    test if model performance distribution for a given PFT from
    two different parameterization methods are significantly different
    using non-parametric Kolmogorov-Smirnov test

    Parameters:
    -----------
    in_dict (dict) : dict containing NSE from different parameterization
    methods of a certain Model per PFT

    Returns:
    --------
    stat_pft_dict (dict) : dict containing significant combinations
    of NSE from different parameterization methods of a certain Model
    per PFT and corresponding p-values
    """

    pft_dict = {}

    for pft, val in in_dict.items():
        nnse_list_arr = []
        for _, nse in val.items():
            nnse_list_arr.append(1.0 / (2.0 - np.array(nse)))
        pft_dict[pft] = nnse_list_arr

    stat_pft_dict = {}
    for pft, val in pft_dict.items():
        significant_combinations = []
        ls = list(range(1, len(val) + 1))
        combinations = [
            (ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))
        ]
        for c in combinations:
            data1 = val[c[0] - 1]
            data2 = val[c[1] - 1]
            _, p_val = stats.ks_2samp(data1, data2)
            if p_val < 0.05:
                significant_combinations.append([c, p_val])

        stat_pft_dict[pft] = significant_combinations

    return stat_pft_dict


def plot_axs(ax, data_dict, title):
    """
    prepare the subplots for the figure

    Parameters:
    -----------
    ax (matplotlib.axes._subplots.AxesSubplot) : axes to plot the boxplots
    data_dict (dict) : dict containing NSE from different optimization
    experiments of a certain Model per bioclimatic class
    title (str) : title of the subplot

    Returns:
    --------
    None
    """

    # list of bioclimatic classes
    bioclim_list = list(data_dict.keys())

    # colors for the boxplots (of each optimization experiment)
    colors = ["#56B4E9", "#009E73", "#BBBBBB", "#CC79A7"]

    stat_dict = plot_stat_significance(data_dict)

    # create boxplots
    for ix, bioclim_name in enumerate(bioclim_list):
        for jx, opti_type in enumerate(["per_site_yr", "per_site", "per_pft", "glob"]):
            # position for the boxplot
            position = 4 * ix + jx + 1
            # create the boxplot with specific color
            ax.boxplot(  # box_dict =
                # np.where(
                #     np.array(data_dict[bioclim_name][opti_type]) < -1.0,
                #     -1.2,
                #     np.array(data_dict[bioclim_name][opti_type]),
                # ),
                1.0 / (2.0 - np.array(data_dict[bioclim_name][opti_type])),
                widths=0.5,
                positions=[position],
                patch_artist=True,
                boxprops=dict(facecolor=colors[jx]),
            )
            # for line in box_dict['medians']:
            #     # Get the median value
            #     median_value = line.get_ydata()[0]

            #     # Get the x position for the annotation
            #     x_position = line.get_xdata()[1]

            #     # Get the y position for the annotation
            #     y_position = median_value

            #     # Place the text annotation on the plot
            #     ax.text(x_position, y_position, f'{median_value:.2f}',
            #             ha='center', va='bottom', fontsize=8, color='blue')

    # set x-axis labels as bioclimatic classes
    # (with number of sites in each bioclimatic classes in brackets)
    ax.set_xticks(range(2, 4 * len(bioclim_list) + 1, 4))
    xticklabs = [
        f"{bioclim_name} ({len(data_dict[bioclim_name]['per_site_yr'])})"
        for bioclim_name in bioclim_list
    ]
    ax.set_xticklabels(xticklabs)

    grid_x = np.arange(4.5, 33, 4)
    for i in grid_x:
        ax.axvline(i, color="gray", linestyle=(0, (5, 10)))

    # add significance stars
    # after https://rowannicholls.github.io/python/graphs/ax_based/boxplots_significance.html
    bottom = 0.0
    top = 0.93
    yrange = top - bottom

    for ix, pft_name in enumerate(bioclim_list):
        sig_combi = stat_dict[pft_name]
        for i, significant_combination in enumerate(sig_combi):
            x1 = significant_combination[0][0] - 1
            x2 = significant_combination[0][1] - 1
            level = len(sig_combi) - i

            position_x1 = 4 * ix + x1 + 1
            position_x2 = 4 * ix + x2 + 1

            bar_height = (yrange * 0.08 * level) + top
            bar_tips = bar_height - (yrange * 0.02)

            ax.plot(
                [position_x1, position_x1, position_x2, position_x2],
                [bar_tips, bar_height, bar_height, bar_tips],
                lw=1,
                c="k",
            )

            p = significant_combination[1]
            if p < 0.001:
                sig_symbol = "***"
            elif p < 0.01:
                sig_symbol = "**"
            elif p < 0.05:
                sig_symbol = "*"
            elif p >= 0.05:
                sig_symbol = "n.s."
            text_height = bar_height + (yrange * 0.01)
            ax.text(
                (position_x1 + position_x2) * 0.5,
                text_height,
                sig_symbol,
                ha="center",
                c="k",
                fontsize=26,
            )

    ax.tick_params(axis="both", which="major", labelsize=38.0)
    ax.tick_params(axis="x", labelrotation=45)

    ax.set_title(
        title,
        fontsize=54,
    )

    sns.despine(ax=ax, top=True, right=True)


def assign_symbol(p_val):
    """
    assign symbol of statistical significance based on p-value
    """
    if p_val < 0.001:
        sig_symbol = "***"
    elif p_val < 0.01:
        sig_symbol = "**"
    elif p_val < 0.05:
        sig_symbol = "*"
    elif p_val >= 0.05:
        sig_symbol = "n.s."
    return f"{p_val}$^{{{sig_symbol}}}$"


def plot_fig(p_mod_res_path_dict, lue_mod_res_path_dict):
    """
    plot the figure

    Parameters:
    -----------
    p_mod_res_path_dict (dict) : dict containing paths to the serialized model
    results from different optimization experiments of P Model
    lue_mod_res_path_dict (dict) : dict containing paths to the serialized model
    results from different optimization experiments of LUE Model

    Returns:
    --------
    None
    """
    # prepare the data for each model
    p_mod_bioclim_mod_perform_dict = prep_data(p_mod_res_path_dict)
    lue_mod_bioclim_mod_perform_dict = prep_data(lue_mod_res_path_dict)

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=1, nrows=2, figsize=(40, 24), sharex=True, sharey=True
    )

    plot_axs(
        axs[0], p_mod_bioclim_mod_perform_dict, r"(a) P$^{\text{W}}_{\text{hr}}$ model"
    )
    plot_axs(axs[1], lue_mod_bioclim_mod_perform_dict, r"(b) Bao$_{\text{hr}}$ model")

    fig.subplots_adjust(hspace=0.2)

    # Adding legend manually
    colors = ["#56B4E9", "#009E73", "#BBBBBB", "#CC79A7"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=30,
        )
        for i, opti_type in enumerate(
            [
                r"per site--year parameterization",
                "per site parameterization",
                "per PFT parameterization",
                "global parameterization",
            ]
        )
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=40,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=True,
        bbox_to_anchor=(0.5, -0.6),
    )

    fig.supxlabel(r"Climate--vegetation types", y=-0.04, fontsize=54)
    fig.supylabel("NNSE [-]", x=0.08, fontsize=54)

    # save the figure
    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        "./supplement_figs/fs09.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        "./supplement_figs/fs09.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    # test if model performance between P-model and LUE model
    # is significantly different for a PFT and a parameterization method
    # using non-parametric Kolmogorov-Smirnov test
    sig_test_among_model_dict = {
        "PFT": [],
        "per_site_yr": [],
        "per_site": [],
        "per_pft": [],
        "glob": [],
    }
    for pft, mod_perform_dict in p_mod_bioclim_mod_perform_dict.items():
        sig_test_among_model_dict["PFT"].append(pft)

        for opti, nse_vals in mod_perform_dict.items():
            data1 = 1.0 / (2.0 - np.array(nse_vals))
            data2 = 1.0 / (2.0 - np.array(lue_mod_bioclim_mod_perform_dict[pft][opti]))

            _, p_val = stats.ks_2samp(data1, data2)
            sig_test_among_model_dict[opti].append(round(p_val, 3))

    sig_test_among_model_df = pd.DataFrame(sig_test_among_model_dict)

    sig_test_among_model_df.iloc[:, 1:5] = sig_test_among_model_df.iloc[
        :, 1:5
    ].applymap(assign_symbol)
    print(sig_test_among_model_df.to_latex(index=False, escape=False))


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    # store all the paths in a dict (for p model)
    hr_p_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_p_model_res_path,
        "per_site": result_paths.per_site_p_model_res_path,
        "per_pft": result_paths.per_pft_p_model_res_path,
        "glob_opti": result_paths.glob_opti_p_model_res_path,
    }

    # store all the paths in a dict (for lue model)
    hr_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_lue_model_res_path,
        "per_site": result_paths.per_site_lue_model_res_path,
        "per_pft": result_paths.per_pft_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_lue_model_res_path,
    }

    # plot the figure
    plot_fig(hr_p_model_res_path_coll, hr_lue_model_res_path_coll)
