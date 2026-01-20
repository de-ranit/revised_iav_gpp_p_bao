#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot the initial and calibrated parameter values for each PFT
obatined from different parameterization experiments using P Model
and LUE Model

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Wed Mar 19 2025 14:52:24 CET
"""
import os
import sys
import copy
from pathlib import Path
import importlib
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import seaborn as sns

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

# import the parameter module to get intial parameter values
from src.common.get_params import get_params  # pylint: disable=C0413


def get_param_vals(res_path_dict, model_name):
    """
    collect parameter values from different optimization experiments
    for each PFT and store them in a dictionary

    Parameters:
    -----------
    res_path_dict (dict): dictionary containing paths to the results
    of different optimization experiments for a certain model
    model_name (str): name of the model (either 'LUE_model' or 'P_model')

    Returns:
    --------
    param_coll_per_pft (dict): dictionary containing parameter values for each PFT
    from different optimization experiments
    glob_param_vals (dict): dictionary containing parameter values from the global
    optimization experiment
    """
    # get the list of files containing the results from per site optimization
    mod_res_file_list = glob.glob(f"{res_path_dict['per_site']}/*.npy")
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

    # list of PFT
    pft_list = [
        "CRO",
        "CSH",
        "DBF",
        "DNF",
        "EBF",
        "ENF",
        "GRA",
        "MF",
        "OSH",
        "SAV",
        "SNO",
        "WET",
        "WSA",
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

    # create a dictionary to store parameter values
    # from different optimization experiments
    val_per_pft = {}
    for param in param_list:
        val_per_pft[param] = {
            "per_site_yr": [],
            "per_site": [],
            "per_site_iav": [],
            "per_pft": [],
        }

    # create a dictionary to store parameter values
    # from different optimization experiments for each PFT
    param_coll_per_pft = {}
    for pft in pft_list:
        param_coll_per_pft[pft] = copy.deepcopy(
            val_per_pft
        )  # create a deeo copy of the dictionary to avoid dictionary mutation

    # load the global optimization results for a certain site
    glob_res_dict = np.load(
        f"{res_path_dict['glob_opti']}/AR-SLu_result.npy", allow_pickle=True
    ).item()
    opti_par_dict = glob_res_dict["Opti_par_val"]
    param_name_arr = np.array(list(opti_par_dict.keys()))
    param_val_arr = np.array(list(opti_par_dict.values()))

    # in case of LUE model, get the alpha_fT_Horn value from another site
    # since it was not optimized for AR-SLu
    if model_name == "LUE_model":
        glob_res_dict_2 = np.load(
            f"{res_path_dict['glob_opti']}/AT-Neu_result.npy", allow_pickle=True
        ).item()
        param_name_arr = np.insert(param_name_arr, 3, "alpha_fT_Horn")
        param_val_arr = np.insert(
            param_val_arr, 3, glob_res_dict_2["Opti_par_val"]["alpha_fT_Horn"]
        )

    # create a dictionary to store parameter values from the global optimization experiment
    glob_param_vals = {}
    for ix, p_name in enumerate(param_name_arr):
        glob_param_vals[p_name] = param_val_arr[ix]

    # for each site, get the parameter values from different optimization experiments
    for res_file in filtered_mod_res_file_list:
        # open the results file from per site year optimization
        res_dict_site = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_site["SiteID"]

        site_data_path = Path(f"{res_path_dict['ip_data_path']}/{site_id}.*.hourly_for_PModel.nc")
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
            f"{res_path_dict['per_site_yr']}/{site_id}_result.npy", allow_pickle=True
        ).item()  # site year optimization
        res_dict_pft = np.load(
            f"{res_path_dict['per_pft']}/{site_id}_result.npy", allow_pickle=True
        ).item()  # per pft optimization
        res_dict_site_iav = np.load(
            f"{res_path_dict['per_Site_iav']}/{site_id}_result.npy", allow_pickle=True
        ).item()  # per site iav optimization

        # get the PFT of the site
        site_pft = res_dict_site_yr["PFT"]

        # get the indices of good years so that only parameter values
        # from good years are plotted in case of site year optimization
        good_gpp_y_idx = res_dict_site_yr["good_gpp_y_idx"]
        yr_keys = list(res_dict_site_yr["Opti_par_val"].keys())
        good_yr_keys = [
            yr_keys[i] for i, val in enumerate(good_gpp_y_idx) if val == 1.0
        ]

        # for each parameter, get the parameter values from different optimization experiments
        for param in param_list:
            try:
                if param in ["sn_a", "meltRate_temp", "meltRate_netrad"] and (not snowfall_site):
                    pass
                else:
                    param_coll_per_pft[site_pft][param]["per_site"].append(
                        res_dict_site["Opti_par_val"][param]
                    )  # complete the dictionary for each PFT and per site optimization
                    param_coll_per_pft[site_pft][param]["per_site_iav"].append(
                        res_dict_site_iav["Opti_par_val"][param]
                    )  # complete the dictionary for each PFT and per site optimization

                for yr in good_yr_keys:
                    yr_mask = yr_arr == int(yr)
                    ta_gf_ts_yr = ta_gf_ts[yr_mask]
                    neg_temp_mask_yr = ta_gf_ts_yr < 0.0

                    if param in ["sn_a", "meltRate_temp", "meltRate_netrad"] and (neg_temp_mask_yr.sum() == 0):
                        pass 
                    else:
                        param_coll_per_pft[site_pft][param]["per_site_yr"].append(
                            res_dict_site_yr["Opti_par_val"][yr][param]
                        )  # complete the dictionary for each PFT and per site year optimization

            except KeyError:  # alpha and alpha_fT_Horn are not optimized for all sites
                pass

            # get the parameter values from per PFT optimization
            # if it's already not collected for a certain PFT
            if len(param_coll_per_pft[site_pft][param]["per_pft"]) == 0:
                try:
                    param_coll_per_pft[site_pft][param]["per_pft"].append(
                        res_dict_pft["Opti_par_val"][param]
                    )
                except KeyError:
                    pass

    return param_coll_per_pft, glob_param_vals


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


def plot_axs(ax, data_dict, glob_param_vals, ini_param_dict, param_name, title):
    """
    prepare the subplots for each parameter

    Parameters:
    -----------
    ax (matplotlib.axes._subplots.AxesSubplot): axis object
    data_dict (dict): dictionary containing parameter values for each PFT
    from different optimization experiments
    glob_param_vals (dict): dictionary containing parameter values from the global
    optimization experiment
    ini_param_dict (dict): dictionary containing initial parameter values
    param_name (str): name of the parameter
    title (str): order of the subplot in title ((a), (b), (c), ...)

    Returns:
    --------
    None
    """

    # dictionary containing the names of the parameters and their units
    ylabel_dict = {
        "acclim_window": "$A_t$ [days]",
        "LUE_max": r"$\varepsilon_{max}$"
        + "\n"
        + r"[$\mathrm{\mu} \text{mol CO}_2 \cdot$"
        + "\n"
        + r"$\mathrm{\mu} \text{mol}\ \text{photons}^{-1}$]",
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
        "W_I": r"$W_I$" + "\n" + r"[$\text{mm} \cdot \text{mm}^{-1}$]",
        "K_W": r"$K_W$ [-]",
        "AWC": r"$AWC$ [mm]",
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

    # dictionary containing the full names of the parameters (to be used in the title)
    title_dict = {
        "acclim_window": "Acclimation time",
        "LUE_max": "Maximum light use efficiency",
        "T_opt": "Optimal temperature",
        "K_T": "Sensitivity to temperature changes",
        "Kappa_VPD": "Sensitivity to VPD changes",
        "Ca_0": "Minimum optimal atmospheric CO$_2$ concentration",
        "C_Kappa": "Sensitivity to atmospheric\nCO$_2$ concentration changes",
        "c_m": "CO$_2$ fertilization intensity indicator",
        "gamma_fL_TAL": "Light saturation curve indicator",
        "mu_fCI": "Sensitivity to cloudiness index changes",
        "W_I": "Optimal soil moisture",
        "K_W": "Sensitivity to soil moisture changes",
        "AWC": "Available water capacity",
        "theta": "Rate of evapotranspiration",
        "alpha": "Lag parameter for soil moisture effect",
        "alphaPT": "Scalar factor for potential evapotranspiration",
        "meltRate_temp": "Snow melt rate for temperature",
        "meltRate_netrad": "Snow melt rate for net radiation",
        "sn_a": "Sublimation resistance",
        "alpha_fT_Horn": "Lag parameter for temperature effect",
    }

    # list of PFT
    pft_list = list(data_dict.keys())

    # get the parameter values for each PFT
    # from per PFT optimization experiment
    param_per_pft_val = np.zeros(len(pft_list))
    for ix, pft_name in enumerate(pft_list):
        try:
            param_per_pft_val[ix] = data_dict[pft_name][param_name]["per_pft"][0]
        except IndexError:
            param_per_pft_val[ix] = np.nan

    # colors for the boxplots (of each optimization experiment)
    colors = ["#56B4E9", "#E6C300", "#009E73"]

    box_width = 0.4  # width of the boxplots to place them correctly

    median_dict = {}
    for pft_name in pft_list:
        median_dict[pft_name] = {}

    iqr_dict = {}
    for pft_name in pft_list:
        iqr_dict[pft_name] = {}

    # create boxplots
    for ix, pft_name in enumerate(pft_list):
        for jx, opti_type in enumerate(["per_site_yr", "per_site_iav", "per_site"]):
            if jx == 0:  # calculate the position of the boxplot
                position = ix - box_width / 2 - 0.1
            elif jx == 1:
                position = ix
            else:
                position = ix + box_width / 2 + 0.1

            # if there are less than 3 points in the distribution,
            # plot them as scatter points
            if len(data_dict[pft_name][param_name][opti_type]) < 6:
                ax.scatter(
                    [position] * len(data_dict[pft_name][param_name][opti_type]),
                    data_dict[pft_name][param_name][opti_type],
                    color=colors[jx],
                    s=40,
                )
            else:
                ax.boxplot(  # box_dict =
                    np.array(data_dict[pft_name][param_name][opti_type]),
                    positions=[position],
                    patch_artist=True,
                    boxprops=dict(facecolor=colors[jx]),
                )

                # # Get the median values
                # for line in box_dict["medians"]:
                #     median_value = line.get_ydata()[0]
                #     median_dict[pft_name][opti_type] = f"{median_value:.2f}"

                # # Get the IQR values
                # iqr = (
                #     box_dict["whiskers"][1].get_ydata()[0]
                #     - box_dict["whiskers"][0].get_ydata()[0]
                # )
                # iqr_dict[pft_name][opti_type] = f"{iqr:.2f}"

    # ######### get various data which are discussed in Results section ########
    # if param_name == "AWC":
    #     print(param_name)
    #     collect_median = []
    #     collect_pft = []
    #     for k, v in median_dict.items():
    #         try:
    #             collect_median.append(float(v["per_site"]))
    #             collect_pft.append(k)
    #             # print(f"{v['per_site']} mm ({k}),")
    #         except:
    #             pass
    #     print(
    #         (
    #             f"max awc: {np.array(collect_median).max()} mm"
    #             f"(PFT: {collect_pft[np.array(collect_median).argmax()]})"
    #         )
    #     )
    #     print(
    #         (
    #             f"min awc: {np.array(collect_median).min()} mm"
    #             f"(PFT: {collect_pft[np.array(collect_median).argmin()]})"
    #         )
    #     )

    # if param_name == "Ca_0":
    #     print(param_name)
    #     collect_iqr = []
    #     collect_pft = []
    #     for k, v in iqr_dict.items():
    #         try:
    #             # print(f"{v['per_site']} ppm ({k}),")
    #             collect_iqr.append(float(v["per_site"]))
    #             collect_pft.append(k)
    #         except:
    #             pass
    #     print(
    #         (
    #             f"max iqr: {np.array(collect_iqr).max()} ppm"
    #             f"(PFT: {collect_pft[np.array(collect_iqr).argmax()]})"
    #         )
    #     )
    #     print(
    #         (
    #             f"min iqr: {np.array(collect_iqr).min()} ppm"
    #             f"(PFT: {collect_pft[np.array(collect_iqr).argmin()]})"
    #         )
    #     )

    # if param_name == "acclim_window":
    #     print(param_name)
    #     collect_iqr = []
    #     collect_pft = []
    #     for k, v in iqr_dict.items():
    #         try:
    #             # print(f"{int(np.floor(float(v['per_site'])))} days ({k}),")
    #             collect_iqr.append(int(np.floor(float(v["per_site"]))))
    #             collect_pft.append(k)
    #         except:
    #             pass
    #     print(
    #         (
    #             f"max iqr: {np.array(collect_iqr).max()} days"
    #             f"(PFT: {collect_pft[np.array(collect_iqr).argmax()]})"
    #         )
    #     )
    #     print(
    #         (
    #             f"min iqr: {np.array(collect_iqr).min()} days"
    #             f"(PFT: {collect_pft[np.array(collect_iqr).argmin()]})"
    #         )
    #     )

    # if param_name == "acclim_window":
    #     print(param_name)
    #     collect_median = []
    #     collect_pft = []
    #     for k, v in median_dict.items():
    #         try:
    #             print(f"{int(np.floor(float(v['per_site'])))} days ({k}),")
    #             collect_median.append(int(np.floor(float(v["per_site"]))))
    #             collect_pft.append(k)
    #         except:
    #             pass
    #     print(
    #         (
    #             f"max median: {np.array(collect_median).max()} days"
    #             f"(PFT: {collect_pft[np.array(collect_median).argmax()]})"
    #         )
    #     )
    #     print(
    #         (
    #             f"min median: {np.array(collect_median).min()} days"
    #             f"(PFT: {collect_pft[np.array(collect_median).argmin()]})"
    #         )
    #     )
    # ##########################################

    # scatter plot of parameter values from per PFT optimization
    for kx, pft_par_val in enumerate(param_per_pft_val):
        ax.scatter(
            [kx],
            [pft_par_val],
            marker="*",
            s=250,
            c="#D55E00",
            alpha=0.6,
            zorder=3.5,
        )

    # add initial parameter values and parameter values from global optimization as
    # horizontal lines
    ax.axhline(
        y=glob_param_vals[param_name],
        color="#CC79A7",
        linestyle="-.",  # dashes=(1.6, 4)
    )  # linewidth=1)
    ax.axhline(
        y=ini_param_dict[param_name], color="black", linestyle="--", dashes=(1.6, 4)
    )  # linewidth=1)

    # add dotted lines separating the PFTs
    grid_x = np.arange(0.5, 12.5, 1)
    for i in grid_x:
        ax.axvline(i, color="gray", linestyle=(0, (5, 10)))

    # set x-axis labels as PFT (with number of sites in each PFT in brackets)
    xticklabs = [
        f"{pft_name}\n({len(data_dict[pft_name][param_name]['per_site'])})"
        for pft_name in pft_list
    ]
    ax.set_xticks(range(len(pft_list)), labels=xticklabs, ha="center")

    # fine tune other axis properties
    ax.tick_params(axis="both", which="major", labelsize=24.0)
    # ax.tick_params(axis="x", labelrotation=45)

    # ax.set_xlabel("PFT", fontdict={"size": 31})
    ax.set_ylabel(ylabel_dict[param_name], fontdict={"size": 32})

    ax.set_title(f"{title} {title_dict[param_name]}", weight="bold", fontsize=32)

    sns.despine(ax=ax, top=True, right=True)  # remove the top and right spines


def plot_fig(res_path_dict, model_name):
    """
    prepare the figure containing subplots for each parameter

    Parameters:
    -----------
    res_path_dict (dict): dictionary containing paths to the results
    of different optimization experiments for a certain model
    model_name (str): name of the model (either 'LUE_model' or 'P_model')

    Returns:
    --------
    None
    """
    # get the parameter values from different optimization experiments
    param_coll_per_pft, glob_param_vals = get_param_vals(res_path_dict, model_name)

    # get the initial parameter values
    ini_param_dict = get_params({"KG": ["c"], "elev": 0.0})
    ini_param_dict_2 = get_params({"KG": ["B"], "elev": 0.0})
    ini_param_dict["alpha"] = ini_param_dict_2["alpha"]

    for param, val in ini_param_dict.items():
        try:
            ini_param_dict[param] = val["ini"]
            # make initials of Kappa_VPD and K_W negative
            # as they were made positive during optimization
            if param in ["Kappa_VPD", "K_W"]:
                ini_param_dict[param] = -val["ini"]
        except TypeError:
            pass

    # create a common legend to be shared by all figures
    colors = ["#56B4E9", "#E6C300", "#009E73"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=20,
        )
        for i, opti_type in enumerate(
            [
                r"per site--year parameterization",
                r"per site parameterization using $Cost^{IAV}$",
                "per site parameterization",
            ]
        )
    ]
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="*",
            alpha=0.6,
            color="w",
            label="per PFT parameterization",
            markerfacecolor="#D55E00",
            markersize=25,
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="",
            label="global parameterization",
            linestyle="-.",
            linewidth=4,
            color="#CC79A7",
        )
    )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="",
            label="initial parameter value",
            linestyle="--",
            linewidth=4,
            color="black",
        )
    )

    if model_name == "P_model":
        # # list of selected parameters from P Model which will be shown in a figure
        param_to_plot_main_p = ["acclim_window", "AWC", "W_I", "theta"]

        # generate a list of titles for the subplots (a), (b), (c), ...
        fig_titles = [
            f"({chr(i)})" for i in range(ord("a"), ord("a") + len(param_to_plot_main_p))
        ]

        # create the main figure
        main_fig_p, axs = plt.subplots(ncols=1, nrows=4, figsize=(18, 20), sharex=True)

        # add subplots for each parameter
        for plot_ix, p_to_plot in enumerate(param_to_plot_main_p):
            plot_axs(
                axs[plot_ix],
                param_coll_per_pft,
                glob_param_vals,
                ini_param_dict,
                p_to_plot,
                fig_titles[plot_ix],
            )

        # adjust the space between subplots
        main_fig_p.supxlabel("PFT", y=0.03, fontsize=42)

        # add the legend
        plt.legend(
            handles=legend_elements,
            fontsize=32,
            loc="lower center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.46, -1.2),
        )

        # save the figure
        fig_path = Path("figures")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig("./figures/f08_main_p_param.png", dpi=300, bbox_inches="tight")
        plt.savefig("./figures/f08_main_p_param.pdf", dpi=300, bbox_inches="tight")
        plt.close("all")

        ##############################
        # get the list of remaining parameters from P Model which will be shown
        # in the supplementary figure
        param_to_plot_suppli_p = [
            i
            for i in list(param_coll_per_pft["CRO"].keys())
            if i not in param_to_plot_main_p
        ]

        # generate a list of titles for the subplots (a), (b), (c), ...
        fig_titles = [
            f"({chr(i)})"
            for i in range(ord("a"), ord("a") + len(param_to_plot_suppli_p))
        ]

        # create the supplementary figure
        suppli_fig_p, axs = plt.subplots(
            ncols=2, nrows=3, figsize=(30, 20), sharex=True
        )

        # add subplots for each parameter
        for plot_ix, p_to_plot in enumerate(param_to_plot_suppli_p):
            ax_x, ax_y = subplot_indices(7, 2, plot_ix + 1)
            plot_axs(
                axs[ax_x, ax_y],
                param_coll_per_pft,
                glob_param_vals,
                ini_param_dict,
                p_to_plot,
                fig_titles[plot_ix],
            )

        # adjust the space between subplots
        suppli_fig_p.subplots_adjust(hspace=0.22)
        suppli_fig_p.supxlabel("PFT", y=0.03, fontsize=42)

        # add the legend
        plt.legend(
            handles=legend_elements,
            fontsize=32,
            loc="lower center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(-0.1, -0.9),
        )

        # save the figure
        fig_path = Path("supplement_figs")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(
            "./supplement_figs/fs15_suppli_p_param.png", dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            "./supplement_figs/fs15_suppli_p_param.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close("all")

    elif model_name == "LUE_model":
        # ##############################
        # # list of selected parameters from LUE Model which will be shown in a figure
        # # in the main paper
        param_to_plot_main_lue = ["LUE_max", "T_opt", "Kappa_VPD", "Ca_0", "AWC", "W_I"]

        # generate a list of titles for the subplots (a), (b), (c), ...
        fig_titles = [
            f"({chr(i)})"
            for i in range(ord("a"), ord("a") + len(param_to_plot_main_lue))
        ]

        # create the main figure
        main_fig_lue, axs = plt.subplots(
            ncols=1, nrows=6, figsize=(18, 22), sharex=True
        )

        # add subplots for each parameter
        for plot_ix, p_to_plot in enumerate(param_to_plot_main_lue):
            # ax_x, ax_y = subplot_indices(3, 2, plot_ix + 1)
            plot_axs(
                axs[plot_ix],
                param_coll_per_pft,
                glob_param_vals,
                ini_param_dict,
                p_to_plot,
                fig_titles[plot_ix],
            )

        # adjust the space between subplots
        main_fig_lue.subplots_adjust(hspace=0.38)
        main_fig_lue.supxlabel("PFT", y=0.04, fontsize=38)

        # add the legend
        plt.legend(
            handles=legend_elements,
            fontsize=32,
            loc="lower center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.46, -1.8),
        )

        # save the figure
        fig_path = Path("figures")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig("./figures/f09_main_lue_param.png", dpi=300, bbox_inches="tight")
        plt.savefig("./figures/f09_main_lue_param.pdf", dpi=300, bbox_inches="tight")
        plt.close("all")

        ##############################
        # get the list of remaining parameters which will be shown
        # in the supplementary figure
        param_to_plot_suppli_lue = [
            i
            for i in list(param_coll_per_pft["CRO"].keys())
            if i not in param_to_plot_main_lue
        ]
        # adjust the order of the parameters to be shown in the figure
        param_to_plot_suppli_lue.insert(7, param_to_plot_suppli_lue.pop(-1))
        param_to_plot_suppli_lue.insert(1, param_to_plot_suppli_lue.pop(-1))

        # generate a list of titles for the subplots (a), (b), (c), ...
        fig_titles = [
            f"({chr(i)})"
            for i in range(ord("a"), ord("a") + len(param_to_plot_suppli_lue))
        ]

        # create the supplementary figure
        suppli_fig_lue, axs = plt.subplots(
            ncols=2, nrows=7, figsize=(30, 34), sharex=True
        )

        # add subplots for each parameter
        for plot_ix, p_to_plot in enumerate(param_to_plot_suppli_lue):
            ax_x, ax_y = subplot_indices(7, 2, plot_ix + 1)
            plot_axs(
                axs[ax_x, ax_y],
                param_coll_per_pft,
                glob_param_vals,
                ini_param_dict,
                p_to_plot,
                fig_titles[plot_ix],
            )

        # adjust the space between subplots
        suppli_fig_lue.subplots_adjust(hspace=0.47)
        # delete the last subplot (which is empty)
        suppli_fig_lue.delaxes(axs[6, 1])
        axs[5, 1].tick_params(
            axis="x", which="both", bottom=True, top=False, labelbottom=True
        )
        suppli_fig_lue.supxlabel("PFT", y=0.07, fontsize=42)

        # add the legend
        plt.legend(
            handles=legend_elements,
            fontsize=28,
            loc="lower center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(1.68, 0.03),
        )

        # save the figure
        fig_path = Path("supplement_figs")
        os.makedirs(fig_path, exist_ok=True)
        plt.savefig(
            "./supplement_figs/fs16_suppli_lue_param.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            "./supplement_figs/fs16_suppli_lue_param.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close("all")


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    # store all the paths in a dict (for p model)
    hr_p_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_p_model_res_path,
        "per_site": result_paths.per_site_p_model_res_path,
        "per_Site_iav": result_paths.per_site_p_model_res_path_iav,
        "per_pft": result_paths.per_pft_p_model_res_path,
        "glob_opti": result_paths.glob_opti_p_model_res_path,
        "ip_data_path": result_paths.hr_ip_data_path
    }

    # store all the paths in a dict (for lue model)
    hr_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_lue_model_res_path,
        "per_site": result_paths.per_site_lue_model_res_path,
        "per_Site_iav": result_paths.per_site_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_lue_model_res_path,
        "ip_data_path": result_paths.hr_ip_data_path
    }

    # plot the figures
    plot_fig(hr_p_model_res_path_coll, "P_model")
    plot_fig(hr_lue_model_res_path_coll, "LUE_model")
