#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
contains path to all the model experiment results obtained for
CMAES optimization with big population size.
This module will be imported as module in other scripts to access the
paths to the model results.

author: rde
first created: Mon Feb 05 2024 15:19:02 CET
"""

from pathlib import Path


per_site_yr_p_model_res_path = Path(
    (
        "../opti_results/P_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes"
        "/opti_dicts/"
    )
)

per_site_p_model_res_path = Path(
    (
        "../opti_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes"
        "/opti_dicts/"
    )
)
per_site_p_model_res_path_iav = Path(
    (
        "../opti_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes"
        "/opti_dicts/"
    )
)

#################################################################################
per_site_yr_lue_model_res_path = Path(
    (
        "../opti_results/LUE_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_cmaes"
        "/opti_dicts/"
    )
)

per_site_lue_model_res_path = Path(
    (
        "../opti_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_cmaes"
        "/opti_dicts/"
    )
)

per_site_lue_model_res_path_iav = Path(
    (
        "../opti_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_cost_iav_big_pop_cmaes"
        "/opti_dicts/"
    )
)
#################################################################################