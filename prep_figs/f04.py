#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make a dot plot with percentage contributions of each factor to the sum of squares
in ANOVA for NNSE (annual) and NNSE (hourly)

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Tue Aug 27 2024 16:33:43 CEST
"""

import os
from pathlib import Path
import importlib
import glob
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import levene

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


def determine_bioclim(pft, kg):
    """
    determine the bioclimatic zone based on the PFT and KG

    Parameters
    ----------
    pft (str) : plant functional type
    kg (str) : Koppen-Geiger climate zone

    Returns
    -------
    bioclim (str) : bioclimatic zone
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


def collect_data(
    filtered_mod_res_file_list,
    nse_list,
    model_name_list,
    opti_type_list,
    input_timescale_list,
    output_timescale_list,
    site_id_list,
    no_of_good_years_list,
    pft_list,
    kg_list,
    bioclim_list,
    lat_list,
    lon_list,
    model_name,
    opti_name,
    site_data_dict,
    nse_var_name,
):
    """
    collect the data for ANOVA analysis for NNSE

    Parameters
    ----------
    filtered_mod_res_file_list (list) : list of files with serialized model results
    nse_list (list) : list to store NSE values
    model_name_list (list) : list to store model names
    opti_type_list (list) : list to store optimization strategies
    input_timescale_list (list) : list to store input timescales
    output_timescale_list (list) : list to store output timescales
    site_id_list (list) : list to store site IDs
    no_of_good_years_list (list) : list to store number of good years
    pft_list (list) : list to store plant functional types
    kg_list (list) : list to store Koppen-Geiger climate zones
    bioclim_list (list) : list to store bioclimatic zones
    lat_list (list) : list to store latitudes
    lon_list (list) : list to store longitudes
    model_name (str) : model name
    opti_name (str) : optimization strategy
    site_data_dict (dict) : dictionary with site information
    nse_var_name (str) : variable name for NSE

    Returns
    -------
    nse_list (list) : list with NSE values
    model_name_list (list) : list with model names
    opti_type_list (list) : list with optimization strategies
    input_timescale_list (list) : list with input timescales
    output_timescale_list (list) : list with output timescales
    site_id_list (list) : list with site IDs
    no_of_good_years_list (list) : list with number of good years
    pft_list (list) : list with plant functional types
    kg_list (list) : list with Koppen-Geiger climate zones
    bioclim_list (list) : list with bioclimatic zones
    lat_list (list) : list with latitudes
    lon_list (list) : list with longitudes

    """

    for res_file in filtered_mod_res_file_list:
        result_dict = np.load(res_file, allow_pickle=True).item()
        try:
            nse_list.append(result_dict[nse_var_name]["NSE_Hourly"])
            model_name_list.append(model_name)
            opti_type_list.append(opti_name)
            input_timescale_list.append(result_dict["Temp_res"])
            output_timescale_list.append("Hourly")
            site_id_list.append(result_dict["SiteID"])
            no_of_good_years_list.append(len(result_dict["Time_yearly_filtered"]))
            pft_list.append(result_dict["PFT"])
            kg_list.append(result_dict["KG"])
            bioclim_list.append(
                determine_bioclim(result_dict["PFT"], result_dict["KG"])
            )
            lat_list.append(site_data_dict[result_dict["SiteID"]]["Lat"])
            lon_list.append(site_data_dict[result_dict["SiteID"]]["Lon"])
        except KeyError:
            nse_list.append(np.nan)
            model_name_list.append(model_name)
            opti_type_list.append(opti_name)
            input_timescale_list.append(result_dict["Temp_res"])
            output_timescale_list.append("Hourly")
            site_id_list.append(result_dict["SiteID"])
            no_of_good_years_list.append(len(result_dict["Time_yearly_filtered"]))
            pft_list.append(result_dict["PFT"])
            kg_list.append(result_dict["KG"])
            bioclim_list.append(
                determine_bioclim(result_dict["PFT"], result_dict["KG"])
            )
            lat_list.append(site_data_dict[result_dict["SiteID"]]["Lat"])
            lon_list.append(site_data_dict[result_dict["SiteID"]]["Lon"])

        for op_timescale in ["d", "w", "m", "y"]:
            nse_list.append(result_dict[nse_var_name][f"NSE_{op_timescale}"])
            model_name_list.append(model_name)
            opti_type_list.append(opti_name)
            input_timescale_list.append(result_dict["Temp_res"])
            if op_timescale == "d":
                output_timescale_list.append("Daily")
            elif op_timescale == "w":
                output_timescale_list.append("Weekly")
            elif op_timescale == "m":
                output_timescale_list.append("Monthly")
            elif op_timescale == "y":
                output_timescale_list.append("Yearly")
            site_id_list.append(result_dict["SiteID"])
            no_of_good_years_list.append(len(result_dict["Time_yearly_filtered"]))
            pft_list.append(result_dict["PFT"])
            kg_list.append(result_dict["KG"])
            bioclim_list.append(
                determine_bioclim(result_dict["PFT"], result_dict["KG"])
            )
            lat_list.append(site_data_dict[result_dict["SiteID"]]["Lat"])
            lon_list.append(site_data_dict[result_dict["SiteID"]]["Lon"])

    return (
        nse_list,
        model_name_list,
        opti_type_list,
        input_timescale_list,
        output_timescale_list,
        site_id_list,
        no_of_good_years_list,
        pft_list,
        kg_list,
        bioclim_list,
        lat_list,
        lon_list,
    )


def prep_nse_data(exp_path, site_data_dict, model_name, opti_name, consider_p_hr=None):
    """
    prepare the data for ANOVA analysis for NNSE

    Parameters
    ----------
    exp_path (str) : path to the serialized model results
    site_data_dict (dict) : dictionary with site information
    model_name (str) : model name
    opti_name (str) : optimization strategy

    Returns
    -------
    return_df (pd.DataFrame) : dataframe with the data for ANOVA analysis

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

    nse_list = []
    model_name_list = []
    opti_type_list = []
    input_timescale_list = []
    output_timescale_list = []
    site_id_list = []
    no_of_good_years_list = []
    pft_list = []
    kg_list = []
    bioclim_list = []
    lat_list = []
    lon_list = []

    if model_name == "p":
        (
            nse_list,
            model_name_list,
            opti_type_list,
            input_timescale_list,
            output_timescale_list,
            site_id_list,
            no_of_good_years_list,
            pft_list,
            kg_list,
            bioclim_list,
            lat_list,
            lon_list,
        ) = collect_data(
            filtered_mod_res_file_list,
            nse_list,
            model_name_list,
            opti_type_list,
            input_timescale_list,
            output_timescale_list,
            site_id_list,
            no_of_good_years_list,
            pft_list,
            kg_list,
            bioclim_list,
            lat_list,
            lon_list,
            model_name,
            opti_name,
            site_data_dict,
            "NSE",
        )

        if consider_p_hr:
            (
                nse_list,
                model_name_list,
                opti_type_list,
                input_timescale_list,
                output_timescale_list,
                site_id_list,
                no_of_good_years_list,
                pft_list,
                kg_list,
                bioclim_list,
                lat_list,
                lon_list,
            ) = collect_data(
                filtered_mod_res_file_list,
                nse_list,
                model_name_list,
                opti_type_list,
                input_timescale_list,
                output_timescale_list,
                site_id_list,
                no_of_good_years_list,
                pft_list,
                kg_list,
                bioclim_list,
                lat_list,
                lon_list,
                "P_no_sm",
                opti_name,
                site_data_dict,
                "NSE_no_moisture_Stress",
            )
    else:
        (
            nse_list,
            model_name_list,
            opti_type_list,
            input_timescale_list,
            output_timescale_list,
            site_id_list,
            no_of_good_years_list,
            pft_list,
            kg_list,
            bioclim_list,
            lat_list,
            lon_list,
        ) = collect_data(
            filtered_mod_res_file_list,
            nse_list,
            model_name_list,
            opti_type_list,
            input_timescale_list,
            output_timescale_list,
            site_id_list,
            no_of_good_years_list,
            pft_list,
            kg_list,
            bioclim_list,
            lat_list,
            lon_list,
            model_name,
            opti_name,
            site_data_dict,
            "NSE",
        )

    # create a dataframe with all the data
    return_df = pd.DataFrame(
        [
            nse_list,
            model_name_list,
            opti_type_list,
            input_timescale_list,
            output_timescale_list,
            site_id_list,
            no_of_good_years_list,
            pft_list,
            kg_list,
            bioclim_list,
            lat_list,
            lon_list,
        ]
    ).T

    col_names = [
        "NSE",
        "Model_name",
        "Opti_type",
        "Input_timescale",
        "Output_timescale",
        "SiteID",
        "No_of_good_years",
        "PFT",
        "KG",
        "Bioclim",
        "Lat",
        "Lon",
    ]
    return_df.columns = col_names

    return return_df


def homoscedasticity_levene_test(df, args_list):
    """
    Do Levene test to check for homoscedasticity

    Parameters
    ----------
    df (pd.DataFrame) : dataframe with the data for ANOVA analysis
    args_list (list) : list of factors to group the data by

    Returns
    -------
    None
    """
    grouped_nse = df.groupby(args_list)

    groups_nse = [group["NSE"] for name, group in grouped_nse]
    _, p_nse_y = levene(*groups_nse)

    if p_nse_y > 0.05:
        print(f"The variances are equal across groups (p value: {p_nse_y}).")
    else:
        print(f"The variances are not equal across groups (p value: {p_nse_y}).")


def n_way_anova(data, target, anova_type, *args):
    """
    perform N-way ANOVA

    Parameters
    ----------
    data (pd.DataFrame) : dataframe with the data for ANOVA analysis
    target (str) : target variable
    anova_type (str) : type of ANOVA to perform
    args (list) : list of factors to group the data by

    Returns
    -------
    anova_results (pd.DataFrame) : ANOVA results
    """

    anova_string_list = []
    for idx, arg in enumerate(args):
        if idx == 0:
            anova_string_list.append(f"C({arg})")
        else:
            if anova_type == "interaction":
                anova_string_list.append(f"* C({arg})")
            else:
                anova_string_list.append(f"+ C({arg})")

    anova_string = " ".join(anova_string_list)
    model = ols(f"{target} ~ {anova_string}", data=data).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)

    return anova_results


def percentage_ss_contrib(anova_results_df, factors_list):
    """
    calculate the percentage contribution of each factor to the sum of squares

    Parameters
    ----------
    anova_results_df (pd.DataFrame) : ANOVA results
    factors_list (list) : list of factors to calculate the percentage contribution for

    Returns
    -------
    perc_factor_dict (dict) : percentage contribution of each factor to NNSE
    """
    ss_factor_dict = {}
    for factors in factors_list:
        if factors == "Residual":
            ss_factor_dict[factors] = anova_results_df.loc["Residual", "sum_sq"]
        else:
            ss_factor_dict[factors] = anova_results_df.loc[f"C({factors})", "sum_sq"]

    # Calculate total sum of squares (excluding residuals)
    total_ss = []
    for ss in ss_factor_dict.values():
        total_ss.append(ss)
    total_ss = sum(total_ss)

    perc_factor_dict = {}
    for factor_name, factor_ss in ss_factor_dict.items():
        perc_factor_dict[factor_name] = (factor_ss / total_ss) * 100

    return perc_factor_dict


def main_prep_nse_anova_tab(
    hr_p_path, hr_lue_path, dd_lue_path, site_data_dict, consider_p_hr
):
    """
    prepare the data for ANOVA analysis for NNSE and perform the ANOVA

    Parameters
    ----------
    hr_p_path (dict) : dictionary with paths to the serialized model results for p model
    hr_lue_path (dict) : dictionary with paths to the serialized model results for lue model
    dd_lue_path (dict) : dictionary with paths to the serialized model results for lue model
    when optimized at daily resolution
    site_data_dict (dict) : dictionary with site information
    consider_p_hr (bool) : whether to consider the p model with no sm at hourly resolution

    Returns
    -------
    perc_factor_nse_y_dict (dict) : percentage contribution of each factor to NNSE (annual)
    perc_factor_nse_hr_dict (dict) : percentage contribution of each factor to NNSE (hourly)

    """

    p_model_hr_coll = {}
    for exp, exp_path in hr_p_path.items():
        p_model_hr_coll[exp] = prep_nse_data(
            exp_path, site_data_dict, "p", exp, consider_p_hr
        )

    lue_model_hr_coll = {}
    for exp, exp_path in hr_lue_path.items():
        lue_model_hr_coll[exp] = prep_nse_data(exp_path, site_data_dict, "lue", exp)

    lue_model_dd_coll = {}
    for exp, exp_path in dd_lue_path.items():
        lue_model_dd_coll[exp] = prep_nse_data(
            exp_path, site_data_dict, "lue", exp, consider_p_hr
        )

    concat_df_list = []
    for nse_dict in [p_model_hr_coll, lue_model_hr_coll, lue_model_dd_coll]:
        for exp_df in nse_dict.values():
            concat_df_list.append(exp_df)

    main_nse_anova_df = pd.concat(concat_df_list).reset_index(drop=True)

    # subset the data for ANOVA analysis to find contribution of each factor
    # to annual NNSE
    nse_y_anova_df = main_nse_anova_df[
        (main_nse_anova_df["Input_timescale"] == "Hourly")
        & (main_nse_anova_df["Output_timescale"] == "Yearly")
        & (main_nse_anova_df["NSE"].notnull())
    ].copy()
    nse_y_anova_df["NSE"] = nse_y_anova_df["NSE"].astype(float)
    nse_y_anova_df["Lat"] = nse_y_anova_df["Lat"].astype(float)
    nse_y_anova_df["Lon"] = nse_y_anova_df["Lon"].astype(float)

    print("anova_results_nse_y")
    print("________________________")
    nse_y_anova_df["NSE"] = 1.0 / (2.0 - nse_y_anova_df["NSE"])
    # do a Levene test to check for homoscedasticity
    homoscedasticity_levene_test(
        nse_y_anova_df, ["Model_name", "Opti_type", "PFT", "KG", "Bioclim"]
    )
    # perform the N-way ANOVA
    anova_results_nse_y = n_way_anova(
        nse_y_anova_df,
        "NSE",
        # "interaction",
        "no_interaction",
        "Model_name",
        "Opti_type",
        # "SiteID",
        "No_of_good_years",
        "PFT",
        "KG",
        "Bioclim",
        # "Lat",
        # "Lon",
    )

    # print the ANOVA results
    print(anova_results_nse_y)

    # calculate the percentage contribution of each factor to the sum of squares
    percentage_factor_nse_y_dict = percentage_ss_contrib(
        anova_results_nse_y,
        ["Model_name", "Opti_type", "No_of_good_years", "PFT", "KG", "Bioclim"],
    )  # , 'Residual'])

    ########################################################################################
    # subset the data for ANOVA analysis to find contribution of each factor
    # to hourly NNSE
    nse_hr_anova_df = main_nse_anova_df[
        (main_nse_anova_df["Input_timescale"] == "Hourly")
        & (main_nse_anova_df["Output_timescale"] == "Hourly")
        & (main_nse_anova_df["NSE"].notnull())
    ].copy()
    nse_hr_anova_df["NSE"] = nse_hr_anova_df["NSE"].astype(float)
    nse_hr_anova_df["Lat"] = nse_hr_anova_df["Lat"].astype(float)
    nse_hr_anova_df["Lon"] = nse_hr_anova_df["Lon"].astype(float)

    print("anova_results_nse_hr")
    print("________________________")
    nse_hr_anova_df["NSE"] = 1.0 / (2.0 - nse_hr_anova_df["NSE"])

    # do a Levene test to check for homoscedasticity
    homoscedasticity_levene_test(
        nse_hr_anova_df, ["Model_name", "Opti_type", "PFT", "KG", "Bioclim"]
    )

    # perform the N-way ANOVA for hourly NNSE
    anova_results_nse_hr = n_way_anova(
        nse_hr_anova_df,
        "NSE",
        # "interaction",
        "no_interaction",
        "Model_name",
        "Opti_type",
        # "SiteID",
        "No_of_good_years",
        "PFT",
        "KG",
        "Bioclim",
        # "Lat",
        # "Lon",
    )

    # print the ANOVA results
    print(anova_results_nse_hr)

    # calculate the percentage contribution of each factor to the sum of squares
    percentage_factor_nse_hr_dict = percentage_ss_contrib(
        anova_results_nse_hr,
        ["Model_name", "Opti_type", "No_of_good_years", "PFT", "KG", "Bioclim"],
    )

    return percentage_factor_nse_y_dict, percentage_factor_nse_hr_dict


def make_fig(
    ss_perc_nse_y_dict_incl_no_sm_p,
    ss_perc_nse_hr_dict_incl_no_sm_p,
    ss_perc_nse_y_dict,
    ss_perc_nse_hr_dict,
):
    """
    make a dot plot showing the percentage contributions of each factor
    to the sum of squares in ANOVA for NNSE (annual) and NNSE (hourly)

    Parameters
    ----------
    ss_perc_nse_y_dict_incl_no_sm_p (dict) : percentage contribution of each factor to NNSE (annual)
    including P-model with no moisture
    ss_perc_nse_hr_dict_incl_no_sm_p (dict) : percentage contribution of each factor to NNSE (hourly)
    including P-model with no moisture
    ss_perc_nse_y_dict (dict) : percentage contribution of each factor to NNSE (annual)
    ss_perc_nse_hr_dict (dict) : percentage contribution of each factor to NNSE (hourly)

    Returns
    -------
    None
    """

    data_dicts_list = [
        ss_perc_nse_y_dict,
        ss_perc_nse_hr_dict,
        ss_perc_nse_y_dict_incl_no_sm_p,
        ss_perc_nse_hr_dict_incl_no_sm_p,
    ]
    dict_name_list = [
        "Annual NNSE\n" + r"(excluding P$_{\text{hr}}$ model)",
        "Hourly NNSE\n" + r"(excluding P$_{\text{hr}}$ model)",
        "Annual NNSE\n" + r"(including P$_{\text{hr}}$ model)",
        "Hourly NNSE\n" + r"(including P$_{\text{hr}}$ model)",
    ]
    dict_keys = list(ss_perc_nse_y_dict_incl_no_sm_p.keys())
    colors = ["#3498db", "#BBBBBB", "#2ecc71", "#f1c40f", "#9b59b6", "#34495e"]

    # set the figure size
    fig_width = 8  # Width in inches
    # fig_height = fig_width * (3 / 4)
    fig_height = 5

    _, axs = plt.subplots(figsize=(fig_width, fig_height))

    for i, d in enumerate(data_dicts_list):
        for j, key in enumerate(dict_keys):
            axs.plot(d[key], i, "o", color=colors[j], markersize=8, alpha=0.8)

    # Set y-axis labels
    axs.set_yticks([0, 1, 2, 3])
    axs.set_yticklabels(dict_name_list)

    # Set labels
    axs.set_xlabel(r"Percentage contribution to sum of squares [\%]", fontsize=22)
    axs.set_ylabel("Model performances [-]", fontsize=22)

    axs.tick_params(axis="both", which="major", labelsize=18)

    # create a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Model",
            markerfacecolor="#3498db",
            markersize=10,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Parameterization strategy",
            markerfacecolor="#BBBBBB",
            markersize=10,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Number of good site years",
            markerfacecolor="#2ecc71",
            markersize=10,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Plant--functional types (PFT)",
            markerfacecolor="#f1c40f",
            markersize=10,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Köppen--Geiger climate classes (KG)",
            markerfacecolor="#9b59b6",
            markersize=10,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=r"Climate--vegetation type",
            markerfacecolor="#34495e",
            markersize=10,
            alpha=0.8,
        ),
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=18,
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.4, -0.49),
        labelspacing=0.8,
        handlelength=0.2,
        handletextpad=0.8,
        borderpad=0.5,
    )

    sns.despine(ax=axs, top=True, right=True)

    # save the figure
    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f04.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f04.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
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
    # store all the paths in a dict (for lue model in daily resolution)
    dd_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_dd_lue_model_res_path,
        "per_site": result_paths.per_site_dd_lue_model_res_path,
        "per_site_iav": result_paths.per_site_dd_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_dd_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_dd_lue_model_res_path,
    }

    # read the csv file with ancillary site info and collect in a dictionary
    site_info = pd.read_csv("../site_info/SiteInfo_BRKsite_list.csv", low_memory=False)
    site_aux_data_dict = {}
    site_list = []
    for row in site_info.itertuples():
        site_aux_data_dict[row.SiteID] = {
            "Lat": row.Lat,
            "Lon": row.Lon,
            "KG": row.KG,
            "PFT": row.PFT,
            "elev": row.elev,
        }
        site_list.append(row.SiteID)

    # percentage contributions of each of the factors in sum of squares
    # in ANOVA for NNSE (annual) and NNSE (hourly)
    perc_factor_nse_y_dict_incl_p_no_sm, perc_factor_nse_hr_dict_incl_p_no_sm = (
        main_prep_nse_anova_tab(
            hr_p_model_res_path_coll,
            hr_lue_model_res_path_coll,
            dd_lue_model_res_path_coll,
            site_aux_data_dict,
            consider_p_hr=True,
        )
    )

    perc_factor_nse_y_dict, perc_factor_nse_hr_dict = main_prep_nse_anova_tab(
        hr_p_model_res_path_coll,
        hr_lue_model_res_path_coll,
        dd_lue_model_res_path_coll,
        site_aux_data_dict,
        consider_p_hr=False,
    )

    # plot the pie chart with the percentage contributions
    make_fig(
        perc_factor_nse_y_dict_incl_p_no_sm,
        perc_factor_nse_hr_dict_incl_p_no_sm,
        perc_factor_nse_y_dict,
        perc_factor_nse_hr_dict,
    )
