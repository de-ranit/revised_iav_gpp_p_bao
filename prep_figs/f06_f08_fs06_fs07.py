#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot timeseries of aannual average GPP_EC vs GPP_sim for selected sites

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Thu Jun 13 2024 14:58:41 CEST
"""

import os
from pathlib import Path
import importlib
import numpy as np
import bottleneck as bn
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns

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


def get_qc_mod_perform(result_dict, model_name):
    """
    get the model performance metrics and min, max observed and simulated GPP values
    from different modelling experiment

    Parameters:
    -----------
    result_dict (dict): dictionary containing the results from the modelling experiment
    model_name (str): name of the model (P_model or LUE_model)

    Returns:
    --------
    qc_mod_perform_dict (dict): dictionary containing the model performance metrics
    and min and max GPP values from the modelling experiment
    """
    # get the model performance metrics
    nse = round(result_dict["NSE"]["NSE_y"], 3)
    r2 = round(result_dict["R2"]["R2_y"], 3)
    rmse = round(result_dict["RMSE"]["RMSE_y"], 3)
    corr = round(result_dict["corr_coeff"]["corr_coeff_y"], 3)
    var = round(result_dict["variability_coeff"]["variability_coeff_y"], 3)
    bias = round(result_dict["bias_coeff"]["bias_coeff_y"], 3)

    nse_gpp_no_stress = np.nan
    r2_gpp_no_stress = np.nan
    rmse_gpp_no_stress = np.nan
    corr_gpp_no_stress = np.nan
    var_gpp_no_stress = np.nan
    bias_gpp_no_stress = np.nan

    if model_name == "P_model":
        # find the range of GPP values
        max_gpp_val = max(
            bn.nanmax(
                result_dict["GPP_NT_yearly"][result_dict["good_gpp_y_idx"].astype(bool)]
            ),
            bn.nanmax(
                result_dict["GPP_sim_yearly"][
                    result_dict["good_gpp_y_idx"].astype(bool)
                ]
            ),
            bn.nanmax(
                result_dict["GPP_sim_no_moisture_yearly"][
                    result_dict["good_gpp_y_idx"].astype(bool)
                ]
            ),
        )
        min_gpp_val = min(
            bn.nanmin(
                result_dict["GPP_NT_yearly"][result_dict["good_gpp_y_idx"].astype(bool)]
            ),
            bn.nanmin(
                result_dict["GPP_sim_yearly"][
                    result_dict["good_gpp_y_idx"].astype(bool)
                ]
            ),
            bn.nanmin(
                result_dict["GPP_sim_no_moisture_yearly"][
                    result_dict["good_gpp_y_idx"].astype(bool)
                ]
            ),
        )

        # get the model performance metrics without moisture stress (in case of P model)
        nse_gpp_no_stress = round(result_dict["NSE_no_moisture_Stress"]["NSE_y"], 3)
        r2_gpp_no_stress = round(result_dict["R2_no_moisture_Stress"]["R2_y"], 3)
        rmse_gpp_no_stress = round(result_dict["RMSE_no_moisture_Stress"]["RMSE_y"], 3)
        corr_gpp_no_stress = round(
            result_dict["corr_coeff_no_moisture_Stress"]["corr_coeff_y"], 3
        )
        var_gpp_no_stress = round(
            result_dict["variability_coeff_no_moisture_Stress"]["variability_coeff_y"],
            3,
        )
        bias_gpp_no_stress = round(
            result_dict["bias_coeff_no_moisture_Stress"]["bias_coeff_y"], 3
        )

    elif model_name == "LUE_model":
        # find the range of GPP values
        max_gpp_val = max(
            bn.nanmax(
                result_dict["GPP_NT_yearly"][result_dict["good_gpp_y_idx"].astype(bool)]
            ),
            bn.nanmax(
                result_dict["GPP_sim_yearly"][
                    result_dict["good_gpp_y_idx"].astype(bool)
                ]
            ),
        )
        min_gpp_val = min(
            bn.nanmin(
                result_dict["GPP_NT_yearly"][result_dict["good_gpp_y_idx"].astype(bool)]
            ),
            bn.nanmin(
                result_dict["GPP_sim_yearly"][
                    result_dict["good_gpp_y_idx"].astype(bool)
                ]
            ),
        )
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {model_name}"
            "is not implemented"
        )

    return {
        "nse": nse,
        "r2": r2,
        "rmse": rmse,
        "corr": corr,
        "var": var,
        "bias": bias,
        "nse_gpp_no_stress": nse_gpp_no_stress,
        "r2_gpp_no_stress": r2_gpp_no_stress,
        "rmse_gpp_no_stress": rmse_gpp_no_stress,
        "corr_gpp_no_stress": corr_gpp_no_stress,
        "var_gpp_no_stress": var_gpp_no_stress,
        "bias_gpp_no_stress": bias_gpp_no_stress,
        "max_gpp_val": max_gpp_val,
        "min_gpp_val": min_gpp_val,
    }

def set_ticks_for_selected_subplots(axs, selected_indices):
    """
    hide subplot x axis ticks and only enable for given subplots
    """

    # Hide x-axis ticks for all subplots
    for row in axs:
        for ax in row:
            ax.tick_params(
                axis="x", which="both", top=False, labelbottom=False  # bottom=False
            )

    # Enable x-axis ticks for selected subplots
    for i, j in selected_indices:
        axs[i][j].tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=True
        )

def create_ax_gpp_ts(axs, result_dict, qc_mod_perform_dict, model_name, subplot_seq):
    """
    create the subplots of GPP timeseries from different modelling experiments

    Parameters:
    -----------
    axs (matplotlib axes object): axes object to plot the timeseries
    result_dict (dict): dictionary containing the results from the modelling experiment
    qc_mod_perform_dict (dict): dictionary containing the model performance metrics/ data
    to plot the bar indicating the good quality data
    model_name (str): name of the model (P_model or LUE_model)
    subplot_seq (str): subplot sequence (e.g. (a), (b), (c), (d))

    Returns:
    --------
    None
    """

    marker_size = 70
    line_w = 2

    # plot GPP
    axs.scatter(
        result_dict["Time_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        result_dict["GPP_NT_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        c="#E3BD6B",
        marker="o",
        s=marker_size,
    )  # observed GPP
    axs.plot(
        result_dict["Time_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        result_dict["GPP_NT_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        c="#E3BD6B",
        linewidth=line_w,
    )  # observed GPP
    if model_name == "P_model":
        axs.scatter(
            result_dict["Time_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
            result_dict["GPP_sim_no_moisture_yearly"][
                result_dict["good_gpp_y_idx"].astype(bool)
            ],
            c="#025196",
            marker="o",
            s=marker_size,
        )  # simulated GPP without moisture stress (in case of P model)
        axs.plot(
            result_dict["Time_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
            result_dict["GPP_sim_no_moisture_yearly"][
                result_dict["good_gpp_y_idx"].astype(bool)
            ],
            c="#025196",
            linewidth=line_w,
        )  # simulated GPP without moisture stress (in case of P model)
    axs.scatter(
        result_dict["Time_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        result_dict["GPP_sim_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        c="#295E11",
        marker="o",
        s=marker_size,
    )  # simulated GPP
    axs.plot(
        result_dict["Time_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        result_dict["GPP_sim_yearly"][result_dict["good_gpp_y_idx"].astype(bool)],
        c="#295E11",
        linewidth=line_w,
    )  # simulated GPP

    # set the subplot title
    if model_name == "P_model":
        axs.set_title(
            (
                f"{subplot_seq} \n"
                r"P$^{\text{W}}_{\text{hr}}$: (NSE: "
                + f"{qc_mod_perform_dict['nse']}, "
                + r"$r$: "
                + f"{qc_mod_perform_dict['corr']}, "
                + r"$\alpha_{NSE}$: "
                + f"{qc_mod_perform_dict['var']}, "
                + r"$\beta_{n}$: "
                + f"{qc_mod_perform_dict['bias']})\n"
                r"P$_{\text{hr}}$: (NSE: "
                + f"{qc_mod_perform_dict['nse_gpp_no_stress']}, "
                + r"$r$: "
                + f"{qc_mod_perform_dict['corr_gpp_no_stress']}, "
                + r"$\alpha_{NSE}$: "
                + f"{qc_mod_perform_dict['var_gpp_no_stress']}, "
                + r"$\beta_{n}$: "
                + f"{qc_mod_perform_dict['bias_gpp_no_stress']})"
            ),
            size=26,
        )
    else:
        axs.set_title(
            (
                f"{subplot_seq} \n"
                f"NSE: {qc_mod_perform_dict['nse']}, "
                + r"$r$: "
                + f"{qc_mod_perform_dict['corr']}, "
                + r"$\alpha_{NSE}$: "
                + f"{qc_mod_perform_dict['var']}, "
                + r"$\beta_{n}$: "
                + f"{qc_mod_perform_dict['bias']}"
            ),
            size=26,
        )

    axs.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )  # set the x-axis label as year
    axs.xaxis.set_major_locator(
        mdates.YearLocator(base=1)
    )  # set the x-axis ticks as every 2 years
    axs.tick_params(
        axis="x", which="major", labelsize=32.0, labelrotation=45.0
    )  # set the x-axis tick label font size and rotation
    axs.tick_params(
        axis="y", which="major", labelsize=32.0
    )  # set the y-axis tick label font size

    sns.despine(ax=axs, top=True, right=True)  # remove the top and right spines


def plot_fig(path_coll, site_id, model_name, op_folder, filename_order, full_site_name):
    """
    plot observed vs simulated GPP timeseries from different modelling experiments

    Parameters:
    -----------
    path_coll (dict): dictionary containing the paths to the results from different
    modelling experiments
    site_id (str): Site ID
    model_name (str): name of the model (P_model or LUE_model)

    Returns:
    --------
    None
    """

    # load the results from different modelling experiments
    per_site_yr_res_dict = np.load(
        f"{path_coll['per_site_yr']}/{site_id}_result.npy", allow_pickle=True
    ).item()
    per_site_res_dict = np.load(
        f"{path_coll['per_site']}/{site_id}_result.npy", allow_pickle=True
    ).item()
    per_site_iav_res_dict = np.load(
        f"{path_coll['per_site_iav']}/{site_id}_result.npy", allow_pickle=True
    ).item()
    per_pft_res_dict = np.load(
        f"{path_coll['per_pft']}/{site_id}_result.npy", allow_pickle=True
    ).item()
    glob_opti_res_dict = np.load(
        f"{path_coll['glob_opti']}/{site_id}_result.npy", allow_pickle=True
    ).item()

    # get the model performance metrics and data to plot the bar indicating the
    per_site_yr_qc_mod_perform_dict = get_qc_mod_perform(
        per_site_yr_res_dict, model_name
    )
    per_site_qc_mod_perform_dict = get_qc_mod_perform(per_site_res_dict, model_name)
    per_site_iav_qc_mod_perform_dict = get_qc_mod_perform(
        per_site_iav_res_dict, model_name
    )
    per_pft_qc_mod_perform_dict = get_qc_mod_perform(per_pft_res_dict, model_name)
    glob_opti_qc_mod_perform_dict = get_qc_mod_perform(glob_opti_res_dict, model_name)

    fig_width = 28  # 4.7244
    fig_height = fig_width * (
        9 / 16
    )  # Height in inches to maintain a 16:9 aspect ratio

    # create the figure
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )  # 22, 15

    # set title with site info
    if model_name == "P_model":
        y_pos_title = 1.01
    else:
        y_pos_title = 0.98

    fig.suptitle(
        r"\textbf{{{}}}".format(  # pylint: disable=C0209
            (
                f"{full_site_name}"
                f" (Site ID: {per_site_yr_res_dict['SiteID']}"
                f" PFT: {per_site_yr_res_dict['PFT']}"
                f" KG: {per_site_yr_res_dict['KG']})"
            )
        ),
        x=0.5,
        y=y_pos_title,
        fontsize=36,
        weight="bold",
    )

    # create the subplots
    create_ax_gpp_ts(
        axs[0,0],
        per_site_yr_res_dict,
        per_site_yr_qc_mod_perform_dict,
        model_name,
        r"\textbf{(a) Per site--year parameterization}",
    )
    # create the subplots
    create_ax_gpp_ts(
        axs[0,1],
        per_site_iav_res_dict,
        per_site_iav_qc_mod_perform_dict,
        model_name,
        r"\textbf{(b) Per site parameterization using} $\mathbf{Cost^{IAV}}$",
    )
    create_ax_gpp_ts(
        axs[0,2],
        per_site_res_dict,
        per_site_qc_mod_perform_dict,
        model_name,
        r"\textbf{(c) Per site parameterization}",
    )
    create_ax_gpp_ts(
        axs[1,0],
        per_pft_res_dict,
        per_pft_qc_mod_perform_dict,
        model_name,
        r"\textbf{(d) Per PFT parameterization}",
    )
    create_ax_gpp_ts(
        axs[1,1],
        glob_opti_res_dict,
        glob_opti_qc_mod_perform_dict,
        model_name,
        r"\textbf{(e) Global parameterization}",
    )

    set_ticks_for_selected_subplots(axs, [(0, 2), (1, 0), (1, 1)])

    fig.delaxes(axs[1,2])  # remove the last subplot

    # create a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=r"$GPP_{EC}$",
            markerfacecolor="#E3BD6B",
            markersize=20,
        ),
    ]

    if model_name == "P_model":
        legend_elements.insert(
            1,
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=r"$GPP_{sim}$ by P$_{\text{hr}}$ model",
                markerfacecolor="#025196",
                markersize=20,
            ),
        )
        legend_elements.insert(
            2,
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=r"$GPP_{sim}$ by P$^{\text{W}}_{\text{hr}}$ model",
                markerfacecolor="#295E11",
                markersize=20,
            ),
        )
    elif model_name == "LUE_model":
        legend_elements.insert(
            1,
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=r"$GPP_{sim}$",
                markerfacecolor="#295E11",
                markersize=20,
            ),
        )

    if model_name == "P_model":
        fig.subplots_adjust(
            hspace=0.45, wspace=0.5
        )  # adjust the space between subplots
        leg_x_pos = 2.5
    elif model_name == "LUE_model":
        fig.subplots_adjust(hspace=0.3)
        leg_x_pos = 1.8

    # add the legend
    plt.legend(
        handles=legend_elements,
        fontsize=30,
        loc="lower right",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(leg_x_pos, 0.3),
    )

    if site_id == "AU-ASM":
        x_pos = 0.04
    elif site_id == "US-Ne1":
        x_pos = 0.03
    elif site_id == "DE-Hai":
        x_pos = 0.05

    fig.supxlabel(
        "Years",
        fontsize=40,
    )  # set the x-axis label
    fig.supylabel(
        "Annual average GPP"
        + "\n"
        + r"[$\mathrm{\mu} \text{mol CO}_2 \cdot \text{m}^{-2}\cdot \text{s}^{-1}$]",
        fontsize=40,
        x=x_pos,
    )  # set the y-axis label

    # save the figure
    fig_path = Path(op_folder)
    os.makedirs(fig_path, exist_ok=True)
    fig_filename = os.path.join(fig_path, f"{filename_order}.png")
    fig_filename_vector = os.path.join(fig_path, f"{filename_order}.pdf")
    plt.savefig(fig_filename, dpi=200, bbox_inches="tight")
    plt.savefig(fig_filename_vector, dpi=200, bbox_inches="tight")
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

    # plot the timeseries of GPP from AU-ASM (using P Model)
    plot_fig(
        hr_p_model_res_path_coll, "AU-ASM", "P_model", "figures", "f06", "Alice Springs"
    )

    # plot the timeseries of GPP from US-Ne1 (using P Model)
    plot_fig(
        hr_p_model_res_path_coll,
        "US-Ne1",
        "P_model",
        "supplement_figs",
        "fs06",
        "Mead - irrigated continuous maize site",
    )

    # store all the paths in a dict (for lue model)
    hr_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_lue_model_res_path,
        "per_site": result_paths.per_site_lue_model_res_path,
        "per_site_iav": result_paths.per_site_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_lue_model_res_path,
    }

    plot_fig(
        hr_lue_model_res_path_coll, "DE-Hai", "LUE_model", "figures", "f08", "Hainich"
    )

    # store all the paths in a dict (for lue model in daily resolution)
    dd_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_dd_lue_model_res_path,
        "per_site": result_paths.per_site_dd_lue_model_res_path,
        "per_site_iav": result_paths.per_site_dd_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_dd_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_dd_lue_model_res_path,
    }

    # plot the timeseries of GPP from DE-Hai (using LUE Model in daily resolution)
    plot_fig(
        dd_lue_model_res_path_coll,
        "DE-Hai",
        "LUE_model",
        "supplement_figs",
        "fs07",
        "Hainich",
    )
