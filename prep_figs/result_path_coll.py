#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
contains path to all the model experiment results
This module will be imported as module in other scripts to access the
paths to the model results.

author: rde
first created: Mon Feb 05 2024 15:19:02 CET
"""

from pathlib import Path


per_site_yr_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_p_model_res_path_iav = Path(
    (
        "../model_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_pft_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)
glob_opti_p_model_res_path = Path(
    (
        "../model_results/P_model/"
        "global_opti_BRK15_FPAR_FLUXNET_EO_CO2_"
        "MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever"
        "/serialized_model_results/"
    )
)

per_site_yr_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_site_lue_model_res_path_iav = Path(
    (
        "../model_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever"
        "/serialized_model_results/"
    )
)
per_pft_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever/"
        "serialized_model_results/"
    )
)
glob_opti_lue_model_res_path = Path(
    (
        "../model_results/LUE_model/"
        "global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA"
        "_nominal_cost_lue_best_opti_ever/serialized_model_results/"
    )
)


hr_ip_data_path = Path(
    "/path/to/hourly/forcing/data/in/nc/format"
)
dd_ip_data_path = Path(
    "/path/to/daily/forcing/data/in/nc/format"
)
