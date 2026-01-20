#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calculate spatial and temporal variation of model parameters in terms of
mean absolute deviation (MAD)

author: rde
first created: Fri Mar 21 2025 17:46:37 CET
"""
import sys
import os
from pathlib import Path
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.lines as lines
import seaborn as sns
import pandas as pd
import xarray as xr

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)
from src.common.get_params import get_params


def collect_mad_params(res_path_syr, res_path_per_site, forcing_data_path, model_name):
    """
    collect parameter values from the serialized model results
    for both LUE and P model from across site and per site year optimization

    Parameters:
    -----------
    res_path (Path object) : path to the serialized model results
    model_name (str) : name of the model (LUE_model or P_model)
    opti_type (str) : type of optimization (per_site or per_site_year)

    Returns:
    --------
    param_dict (dict) : dictionary containing parameter values from across site
    or per site year optimization
    """

    # get the list of files containing the results from per site optimization
    mod_res_file_list = glob.glob(f"{res_path_per_site}/*.npy")
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

    # load a sample result file to get parameter names
    # and construct the dictionary to store parameter values
    if model_name == "LUE_model":
        sample_res_dict = np.load(
            filtered_mod_res_file_list[2], allow_pickle=True
        ).item()
        param_list = list(sample_res_dict["Opti_par_val"].keys())
        param_list.append("alpha")
    elif model_name == "P_model":
        sample_res_dict = np.load(
            filtered_mod_res_file_list[0], allow_pickle=True
        ).item()
        param_list = list(sample_res_dict["Opti_par_val"].keys())
    else:
        raise ValueError("model_name should be either 'LUE_model' or 'P_model'")

    param_range_dict = get_params({"KG": "C", "elev": 100})
    param_range_dict["alpha"] = {"ini": 0.9899452, "ub": 1.0, "lb": 0.0}

    param_val_per_site_dict = {}
    for param in param_list:
        param_val_per_site_dict[param] = []

    mad_temporal = {}
    for param in param_list:
        mad_temporal[param] = []

    # for each site, get the parameter values from different optimization experiments
    for res_file in filtered_mod_res_file_list:
        # open the results file from per site year optimization
        res_dict_site = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_site["SiteID"]

        site_data_path = f"{forcing_data_path}/{site_id}.*.hourly_for_PModel.nc"
        site_data_file  = glob.glob(f"{site_data_path}")[0]
        ds_site = xr.open_dataset(site_data_file)
        ta_gf_ts = ds_site["TA_GF"].values.ravel()
        yr_arr = ds_site["year"].values.ravel()
        neg_temp_mask = ta_gf_ts < 0.0 
        if neg_temp_mask.sum() > 0:
            snowfall_site = True
        else:
            snowfall_site = False

        # open the results file from other experiments
        res_dict_site_yr = np.load(
            f"{res_path_syr}/{site_id}_result.npy", allow_pickle=True
        ).item()  # site year optimization

        # get the indices of good years so that only parameter values
        # from good years are plotted in case of site year optimization
        good_gpp_y_idx = res_dict_site_yr["good_gpp_y_idx"]
        yr_keys = list(res_dict_site_yr["Opti_par_val"].keys())
        good_yr_keys = [
            yr_keys[i] for i, val in enumerate(good_gpp_y_idx) if val == 1.0
        ]

        for param in param_list:
            try:
                if param in ["sn_a", "meltRate_temp", "meltRate_netrad"] and (not snowfall_site):
                    pass

                # elif (res_dict_site["Opti_par_val"][param] == param_range_dict[param]["ub"]) or (res_dict_site["Opti_par_val"][param] == param_range_dict[param]["lb"]):
                #     print(f"Skipping site {site_id} for parameter {param} as it is at the bound.")

                else:
                    param_val_per_site_dict[param].append(
                        res_dict_site["Opti_par_val"][param]
                    )
                
                param_val_list = []
                if len(good_yr_keys) >= 3:
                    for yr in good_yr_keys:

                        yr_mask = yr_arr == int(yr)
                        ta_gf_ts_yr = ta_gf_ts[yr_mask]
                        neg_temp_mask_yr = ta_gf_ts_yr < 0.0

                        if param in ["sn_a", "meltRate_temp", "meltRate_netrad"] and (neg_temp_mask_yr.sum() == 0):
                            pass 

                        # elif (res_dict_site_yr["Opti_par_val"][yr][param] == param_range_dict[param]["ub"]) or (res_dict_site_yr["Opti_par_val"][yr][param] == param_range_dict[param]["lb"]):
                        #     print(f"Skipping site {site_id} - {yr} for parameter {param} as it is at the bound.")

                        else:
                            param_val_list.append(
                                res_dict_site_yr["Opti_par_val"][yr][param]
                            )

                    if len(param_val_list) > 0:
                        # using MAD
                        mad_temporal_val_per_site = np.sum(
                            np.absolute(
                                np.array(param_val_list)
                                - np.median(np.array(param_val_list))
                            )
                        ) / len(param_val_list)
                        mad_temporal[param].append(mad_temporal_val_per_site)

            except KeyError:
                pass

    mad_sp_site = {}
    for param, val in param_val_per_site_dict.items():
        param_site = np.array(val)

        # using MAD
        param_site_median = np.median(param_site)
        mad_sp = np.sum(np.absolute(param_site - param_site_median)) / len(param_site)
        mad_sp_site[param] = mad_sp

    for param, list_arr in mad_temporal.items():
        mad_temporal[param] = np.array(list_arr)

    return mad_sp_site, mad_temporal


def subplot_indices(nrows, ncols, idx):
    """
    calculate the row and column index of a subplot in a grid

    Parameters:
    -----------
    nrows (int): number of rows in the subplot grid
    ncols (int): number of columns in the subplot grid
    idx (int): index of the subplot in the grid as a single number
    in sequential order (1, 2, 3, ...)

    Returns:
    --------
    row_idx (int): row index of the subplot in the grid
    col_idx (int): column index of the subplot in the grid
    """
    # check if the passed arguments are valid
    if idx < 1 or idx > nrows * ncols:
        raise ValueError("Index out of range for the subplot grid.")
    # calculate row and column index
    row_idx = (idx - 1) // ncols
    col_idx = (idx - 1) % ncols

    return (row_idx, col_idx)


def format_number(value, base_str):
    """
    format the values either in scientific notation or in decimal notation

    Parameters:
    -----------
    value (float): value to be formatted
    base_str (str): string to be prepended to the formatted value

    Returns:
    --------
    ret_string (str): formatted value
    """

    # Convert the number to a string
    value_str = f"{value:.10f}"

    # Check the number of leading zeros after the decimal point
    leading_zeros = len(value_str.split(".")[1]) - len(
        value_str.split(".")[1].lstrip("0")
    )

    if leading_zeros <= 1:
        # Round to two decimal places and return as string
        ret_string = base_str + f"{value:.2f}"

    else:
        # Format in scientific notation
        mantissa, exponent = f"{value:.2e}".split("e")
        mantissa = float(mantissa)
        exponent = int(exponent)
        ret_string = (
            base_str
            + str(mantissa)
            + r"$\boldsymbol{{\times 10^{{{}}}}}$".format(  # pylint: disable=C0209
                str(exponent)
            )
        )

    return ret_string


def percentage_difference(a, b):
    """
    calculate relative difference between two values
    in percentage

    Parameters:
    -----------
    a (float): first value
    b (float): second value

    Returns:
    --------
    percentage_diff (float): relative difference between the two values
    in percentage
    """
    if a == 0:
        raise ValueError(
            "The value of 'a' should not be zero to avoid division by zero."
        )

    difference = b - a
    percentage_diff = (difference / a) * 100

    return percentage_diff


def plot_axs_param_dens(
    axis,
    mad_sp_site_val,
    mad_temporal,
    param_name,
    axtitle,
    model_name,
    var,
):
    """
    prepare the subplots with the density plot of the difference in parameter values
    from the mean parameter value for per site and per site year optimization

    Parameters:
    -----------
    axis (matplotlib.axes) : axes object
    per_site_arr (np.array) : array of the difference in parameter values from the mean
    parameter value for per site optimization
    per_site_yr_arr (np.array) : array of the difference in parameter values from the mean
    parameter value for per site year optimization
    param_name (str) : name of the parameter
    axtitle (str) : title for the axes
    model_name (str) : name of the model (LUE_model or P_model)

    Returns:
    --------
    None
    """
    # dictionary to store the parameter names and their corresponding units
    xlabel_dict = {
        "acclim_window": r"$A_t$ [days]",
        "LUE_max": r"$\varepsilon_{max}$"
        + "\n"
        + r"[$\mathrm{\mu} \text{mol CO}_2 \cdot \mathrm{\mu} \text{mol}\ \text{photons}^{-1}$]",
        "T_opt": r"$T_{opt}$ [$^{\circ}\text{C}$]",
        "K_T": r"$k_T$ [$^{\circ}\text{C}^{-1}$]",
        "Kappa_VPD": r"$\kappa$ [$\text{Pa}^{-1}$]",
        "Ca_0": r"$C_{a0}$ [ppm]",
        "C_Kappa": r"$C_{\kappa}$ [-]",
        "c_m": r"$C_m$ [ppm]",
        "gamma_fL_TAL": r"$\gamma$"
        + "\n"
        + r"[$\mathrm{\mu} \text{mol}\ \text{photons}^{-1} \cdot \text{m}^2 \cdot \text{s}$]",
        "mu_fCI": r"$\mu$ [-]",
        "W_I": r"$W_I$ [$\text{mm} \cdot \text{mm}^{-1}$]",
        "K_W": r"$K_W$ [-]",
        "AWC": r"AWC [mm]",
        "theta": r"$\theta$ [$\text{mm} \cdot \text{h}^{-1}$]",
        "alpha": r"$\alpha$ [-]",
        "alphaPT": r"$PET_{scalar}$ [-]",
        "meltRate_temp": (
            r"$MR_{tair}$" + "\n"
            r"[$\text{mm} \cdot ^{\circ}\text{C} \cdot \text{h}^{-1}$]"
        ),
        "meltRate_netrad": r"$MR_{netrad}$"
        + "\n"
        + r"[$\text{mm} \cdot \text{MJ}^{-1} \cdot \text{h}^{-1}$]",
        "sn_a": r"$sn_a$ [-]",
        "alpha_fT_Horn": r"$\alpha_{fT}$ [-]",
    }

    # set the color palette
    # source: highcontrast (https://packages.tesselle.org/khroma/articles/tol.html)
    cols = ['#004488', '#DDAA33', '#BB5566']

    sns.histplot(
        x=mad_temporal,
        stat="percent",
        kde=True,
        ax=axis,
        color="white",
        edgecolor="white",
    )
    axis.lines[0].set_color(cols[2])
    axis.lines[0].set_linewidth(2)

    # add the axis labels
    axis.set_xlabel(xlabel_dict[param_name], fontdict={"size": 36})
    axis.set_ylabel("", fontdict={"size": 36})
    axis.tick_params(axis="both", which="major", labelsize=34)
    axis.tick_params(axis="x", pad=10)  # , rotation=45)

    if (model_name == "LUE_model") and (var == "clim") and axtitle == "(g)":
        axis.set_xticks(np.array([0.00025, 0.00075]))

    axis.set_title(
        r"\textbf{{{}}}".format(f"{axtitle}"), fontsize=36  # pylint: disable=C0209
    )

    axis.axvline(
        x=mad_sp_site_val, linestyle=":", color=cols[0], linewidth=2, dashes=(4, 2)
    )
    axis.axvline(
        x=np.median(mad_temporal),
        linestyle=":",
        color=cols[2],
        linewidth=2,
        dashes=(4, 2),
    )

    if (model_name == "LUE_model") and (var == "clim"):
        if axtitle == "(c)" or axtitle == "(i)":
            mad_sp_site_text_x_pos = 1.02
            mad_temporal_text_x_pos = 1.05
        elif axtitle == "(d)":
            mad_sp_site_text_x_pos = 1.12
            mad_temporal_text_x_pos = 1.15
        elif axtitle == "(e)":
            mad_sp_site_text_x_pos = 1.01
            mad_temporal_text_x_pos = 1.00
        elif axtitle == "(h)":
            mad_sp_site_text_x_pos = 1.10
            mad_temporal_text_x_pos = 1.15
        elif axtitle == "(i)":
            mad_sp_site_text_x_pos = 1.10
            mad_temporal_text_x_pos = 1.02
        elif axtitle == "(f)" or axtitle == "(j)":
            mad_sp_site_text_x_pos = 1.14
            mad_temporal_text_x_pos = 1.16
        else:
            mad_sp_site_text_x_pos = 0.98
            mad_temporal_text_x_pos = 1.00

    if (model_name == "LUE_model") and (var == "hydro"):
        if axtitle == "(i)":
            mad_sp_site_text_x_pos = 1.07
            mad_temporal_text_x_pos = 1.09
        elif axtitle == "(c)":
            mad_sp_site_text_x_pos = 1.41
            mad_temporal_text_x_pos = 1.43
        elif axtitle == "(d)":
            mad_sp_site_text_x_pos = 1.28
            mad_temporal_text_x_pos = 1.30
        elif axtitle == "(e)":
            mad_sp_site_text_x_pos = 1.20
            mad_temporal_text_x_pos = 1.21
        elif axtitle == "(g)":
            mad_sp_site_text_x_pos = 1.13
            mad_temporal_text_x_pos = 1.15
        else:
            mad_sp_site_text_x_pos = 1.03
            mad_temporal_text_x_pos = 1.05

    if model_name == "P_model":
        if axtitle == "(a)":
            mad_sp_site_text_x_pos = 1.31
            mad_temporal_text_x_pos = 1.33
        elif axtitle == "(b)":
            mad_sp_site_text_x_pos = 1.11
            mad_temporal_text_x_pos = 1.13
        elif axtitle == "(e)" or axtitle == "(h)" or axtitle == "(i)":
            mad_sp_site_text_x_pos = 1.28
            mad_temporal_text_x_pos = 1.30
        else:
            mad_sp_site_text_x_pos = 1.00
            mad_temporal_text_x_pos = 1.01

    if model_name == "P_model":
        mad_sp_site_text_y_pos = 0.9
        mad_temporal_text_y_pos = 0.77
    else:
        mad_sp_site_text_y_pos = 0.9
        mad_temporal_text_y_pos = 0.77

    axis.text(
        mad_sp_site_text_x_pos,
        mad_sp_site_text_y_pos,
        r"\textbf{{{}}}".format(
            format_number(
                mad_sp_site_val, r"$\boldsymbol{\mathit{MAD_{SP}} =\ }$"
            )
        ),
        fontsize=24,
        ha="center",
        va="center",
        color=cols[0],
        transform=axis.transAxes,
    )
    axis.text(
        mad_temporal_text_x_pos,
        mad_temporal_text_y_pos,
        r"\textbf{{{}}}".format(
            format_number(
                np.median(mad_temporal),
                r"$\boldsymbol{\mathit{\overline{MAD_{TP}}} =\ }$",
            )
        ),
        fontsize=24,
        ha="center",
        va="center",
        color=cols[2],
        transform=axis.transAxes,
    )

    sns.despine(ax=axis, top=True, right=True)

    perc_diff_mad_tp_mad_sp_site = percentage_difference(
        np.median(mad_temporal), mad_sp_site_val
    )

    return perc_diff_mad_tp_mad_sp_site


def plot_param_variance(per_site_yr_res_path, per_site_res_path, forcing_data_path, model_name):
    """
    prepare the main figure for LUE model and P model with the difference in parameter values
    from the mean parameter value for per site and per site year optimization

    Parameters:
    -----------
    per_site_yr_res_path (Path object) : path to the serialized model results
    from per site year optimization
    per_site_res_path (Path object) : path to the serialized model results
    from per site optimization
    model_name (str) : name of the model (LUE_model or P_model)

    Returns:
    --------
    """

    # collect the parameter values from the serialized model results
    mad_sp_site_dict, mad_temporal_dict = collect_mad_params(
        per_site_yr_res_path, per_site_res_path, forcing_data_path, model_name
    )

    # source: highcontrast https://packages.tesselle.org/khroma/articles/tol.html#high-contrast
    cols = ['#004488', '#DDAA33', '#BB5566']

    # create a legend for the figure
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=r"$\mathit{MAD_{SP}}$",
            markerfacecolor=cols[0],
            markersize=28,
            # alpha=0.5,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=r"$\mathit{MAD_{TP}}$",
            markerfacecolor=cols[2],
            markersize=28,
            # alpha=0.5,
        ),
    ]

    if model_name == "LUE_model":
        # list of parameters for LUE model
        param_list_lue = [
            "LUE_max",
            "T_opt",
            "K_T",
            "alpha_fT_Horn",
            "Kappa_VPD",
            "Ca_0",
            "C_Kappa",
            "c_m",
            "gamma_fL_TAL",
            "mu_fCI",
            "W_I",
            "K_W",
            "alpha",
            "AWC",
            "theta",
            "alphaPT",
            "meltRate_temp",
            "meltRate_netrad",
            "sn_a",
        ]

        param_list_lue_clim = param_list_lue[0:10]

        # list of figure titles ((a), (b), (c),...) for LUE model
        fig_titles_lue_clim = [f"({chr(i)})" for i in range(ord("a"), ord("a") + 10)]

        # dictionary to map the parameter index to the subplot index
        splot_ix_dict_clim = {
            0: 1,
            8: 2,
            9: 3,
            1: 5,
            2: 6,
            3: 7,
            4: 9,
            5: 10,
            6: 11,
            7: 12,
        }

        ##################################################################
        # dictionary to map the subplot index to the figure title index
        title_ix_dict_clim = {key: i for i, key in enumerate(splot_ix_dict_clim.keys())}

        # create the figure for LUE model
        fig_lue_clim, axs = plt.subplots(nrows=3, ncols=4, figsize=(32, 25))

        perc_diff_dict_lue_clim = {
            "param": [],
            "perc_diff_tp_sp_site": [],
        }

        # add the density plots to the subplots
        for plot_ix, p_to_plot in enumerate(param_list_lue_clim):
            ax_x, ax_y = subplot_indices(3, 4, splot_ix_dict_clim[plot_ix])
            perc_diff_tp_sp_site = plot_axs_param_dens(
                axs[ax_x, ax_y],
                mad_sp_site_dict[p_to_plot],
                mad_temporal_dict[p_to_plot],
                p_to_plot,
                fig_titles_lue_clim[title_ix_dict_clim[plot_ix]],
                model_name,
                "clim",
            )
            perc_diff_dict_lue_clim["param"].append(p_to_plot)
            perc_diff_dict_lue_clim["perc_diff_tp_sp_site"].append(perc_diff_tp_sp_site)

        # remove the empty subplots
        for splot_ix in [4, 8]:
            fig_lue_clim.delaxes(axs[subplot_indices(3, 4, splot_ix)])

        # # adjust the spacing between the subplots
        fig_lue_clim.subplots_adjust(hspace=0.75, wspace=0.6)

        fig_lue_clim.add_artist(
            lines.Line2D(
                [0.1, 0.9], [0.35, 0.35], color="gray", linestyle="--", linewidth=2
            )
        )
        fig_lue_clim.add_artist(
            lines.Line2D(
                [0.1, 0.9], [0.63, 0.63], color="gray", linestyle="--", linewidth=2
            )
        )

        fig_lue_clim.supylabel(r"Fraction of sites [\%]", x=0.07, fontsize=46)

        fig_lue_clim.text(
            0.12,
            0.91,
            r"\textbf{Radiation parameters}",
            ha="left",
            fontsize=38,
        )

        fig_lue_clim.text(
            0.12,
            0.61,
            r"\textbf{Temperature parameters}",
            ha="left",
            fontsize=38,
        )

        fig_lue_clim.text(
            0.12,
            0.33,
            r"\textbf{VPD and CO\textsubscript{2} parameters}",
            ha="left",
            fontsize=38,
        )

        # add the legend to the figure
        plt.legend(
            handles=legend_elements,
            fontsize=40,
            loc="lower left",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(-3.0, -0.64),
        )

        # save the figure
        fig_path = Path("figures")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(
            "./figures/f06_param_clim_variance_lue.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            "./figures/f06_param_clim_variance_lue.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close("all")

        print("lue_clim")
        df_perc_diff_lue_clim = pd.DataFrame(perc_diff_dict_lue_clim)
        print(df_perc_diff_lue_clim.to_latex(index=False, float_format="%.2f"))
        print("##############")

        ##################################################################
        param_list_lue_hydro = param_list_lue[10:]
        fig_titles_lue_hydro = [f"({chr(i)})" for i in range(ord("a"), ord("a") + 9)]
        splot_ix_dict_hydro = {
            0: 1,
            1: 2,
            2: 3,
            3: 5,
            4: 6,
            5: 7,
            6: 8,
            7: 9,
            8: 10,
        }

        # dictionary to map the subplot index to the figure title index
        title_ix_dict_hydro = {
            key: i for i, key in enumerate(splot_ix_dict_hydro.keys())
        }

        # create the figure for LUE model
        fig_lue_hydro, axs = plt.subplots(nrows=3, ncols=4, figsize=(29, 25))

        perc_diff_dict_lue_hydro = {
            "param": [],
            "perc_diff_tp_sp_site": [],
        }

        # add the density plots to the subplots
        for plot_ix, p_to_plot in enumerate(param_list_lue_hydro):
            ax_x, ax_y = subplot_indices(3, 4, splot_ix_dict_hydro[plot_ix])
            perc_diff_tp_sp_site = plot_axs_param_dens(
                axs[ax_x, ax_y],
                mad_sp_site_dict[p_to_plot],
                mad_temporal_dict[p_to_plot],
                p_to_plot,
                fig_titles_lue_hydro[title_ix_dict_hydro[plot_ix]],
                model_name,
                "hydro",
            )
            perc_diff_dict_lue_hydro["param"].append(p_to_plot)
            perc_diff_dict_lue_hydro["perc_diff_tp_sp_site"].append(
                perc_diff_tp_sp_site
            )

        # remove the empty subplots
        for splot_ix in [4, 11, 12]:
            fig_lue_hydro.delaxes(axs[subplot_indices(3, 4, splot_ix)])

        # # adjust the spacing between the subplots
        fig_lue_hydro.subplots_adjust(hspace=0.6, wspace=0.93)

        fig_lue_hydro.add_artist(
            lines.Line2D(
                [0.1, 0.9], [0.64, 0.64], color="gray", linestyle="--", linewidth=2
            )
        )

        fig_lue_hydro.supylabel(r"Fraction of sites [\%]", x=0.06, fontsize=46)

        fig_lue_hydro.text(
            0.12,
            0.91,
            r"\textbf{Drought stress parameters}",
            ha="left",
            fontsize=38,
        )

        fig_lue_hydro.text(
            0.12,
            0.62,
            r"\textbf{Hydrological model parameters}",
            ha="left",
            fontsize=38,
        )

        # add the legend to the figure
        plt.legend(
            handles=legend_elements,
            fontsize=40,
            loc="lower left",
            frameon=True,
            bbox_to_anchor=(1.6, 0.2),
        )

        # save the figure
        fig_path = Path("figures")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(
            "./figures/f07_param_hydro_variance_lue.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            "./figures/f07_param_hydro_variance_lue.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close("all")

        print("lue_hydro")
        df_perc_diff_lue_hydro = pd.DataFrame(perc_diff_dict_lue_hydro)
        print(df_perc_diff_lue_hydro.to_latex(index=False, float_format="%.2f"))
        print("##############")

    ##############################################################################
    # prepare the figure for P model
    elif model_name == "P_model":
        # list of parameters for P model
        param_list_p = list(mad_sp_site_dict.keys())

        # list of figure titles ((a), (b), (c),...) for P model
        fig_titles_p = [
            f"({chr(i)})" for i in range(ord("a"), ord("a") + len(param_list_p))
        ]

        # dictionary to map the parameter index to the subplot index
        splot_ix_dict = {0: 1, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 11, 9: 12}
        # dictionary to map the subplot index to the figure title index
        title_ix_dict = {key: i for i, key in enumerate(splot_ix_dict.keys())}

        # create the figure for P model
        fig_p, axs = plt.subplots(nrows=4, ncols=3, figsize=(20, 26))

        perc_diff_dict_p_all = {
            "param": [],
            "perc_diff_tp_sp_site": [],
        }

        # add the density plots to the subplots
        for plot_ix, p_to_plot in enumerate(param_list_p):
            ax_x, ax_y = subplot_indices(4, 3, splot_ix_dict[plot_ix])
            perc_diff_tp_sp_site = plot_axs_param_dens(
                axs[ax_x, ax_y],
                mad_sp_site_dict[p_to_plot],
                mad_temporal_dict[p_to_plot],
                p_to_plot,
                fig_titles_p[title_ix_dict[plot_ix]],
                model_name,
                "all",
            )
            perc_diff_dict_p_all["param"].append(p_to_plot)
            perc_diff_dict_p_all["perc_diff_tp_sp_site"].append(perc_diff_tp_sp_site)

        # remove the empty subplots
        for splot_ix in [2, 3]:
            fig_p.delaxes(axs[subplot_indices(4, 3, splot_ix)])

        # adjust the spacing between the subplots
        fig_p.subplots_adjust(hspace=0.9, wspace=1.01)
        fig_p.supylabel(r"Fraction of sites [\%]", x=0.05, fontsize=50)

        fig_p.add_artist(
            lines.Line2D(
                [0.1, 0.9], [0.71, 0.71], color="gray", linestyle="--", linewidth=2
            )
        )
        fig_p.add_artist(
            lines.Line2D(
                [0.1, 0.9], [0.49, 0.49], color="gray", linestyle="--", linewidth=2
            )
        )

        fig_p.text(
            0.12,
            0.91,
            r"\textbf{Acclimation time}",
            ha="left",
            fontsize=38,
        )

        fig_p.text(
            0.12,
            0.69,
            r"\textbf{Drought stress parameters}",
            ha="left",
            fontsize=38,
        )

        fig_p.text(
            0.12,
            0.47,
            r"\textbf{Hydrological model parameters}",
            ha="left",
            fontsize=38,
        )

        # add the legend to the figure
        plt.legend(
            handles=legend_elements,
            fontsize=40,
            loc="upper right",
            frameon=True,
            bbox_to_anchor=(0.8, 6.4),
        )

        # save the figure
        fig_path = Path("supplement_figs")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(
            "./supplement_figs/fs14_param_variance_p.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            "./supplement_figs/fs14_param_variance_p.pdf",
            dpi=300,
            bbox_inches="tight",
        )

        print("p_all")
        df_perc_diff_p_all = pd.DataFrame(perc_diff_dict_p_all)
        print(df_perc_diff_p_all.to_latex(index=False, float_format="%.2f"))
        print("##############")

    else:
        raise ValueError("model_name must be either 'LUE_model' or 'P_model'")


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    # plot the difference in parameter values from the mean parameter value
    # from across site and within site year optimization for both LUE and P model
    plot_param_variance(
        result_paths.per_site_yr_lue_model_res_path,
        result_paths.per_site_lue_model_res_path,
        result_paths.hr_ip_data_path,
        "LUE_model",
    )
    plot_param_variance(
        result_paths.per_site_yr_p_model_res_path,
        result_paths.per_site_p_model_res_path,
        result_paths.hr_ip_data_path,
        "P_model",
    )
