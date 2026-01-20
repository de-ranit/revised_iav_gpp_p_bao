#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this module contains the cost function for the LUE model

author: rde
first created: 2023-12-26
"""

import sys
import logging
import warnings
import numpy as np
import pandas as pd

from src.lue_model.run_lue_model import run_lue_model
from src.common.get_params import get_params
from src.lue_model.partial_sensitivity_funcs import (
    f_temp_horn,
    f_vpd_co2_preles,
    f_water_horn,
    f_light_scalar_tal,
)
from src.p_model.p_model_cost_function import cost_nnse
from src.p_model.p_model_cost_function import cost_nrmse
from src.p_model.p_model_cost_function import get_weights_from_normalize_uncertainty

logger = logging.getLogger(__name__)


# ToDo: when RANDUNC is not used in cost function, remove it from
# one of the data filtering criteria in filter_data function
def filter_lue_data(
    lue_model_op,
    ip_df_dict,
    data_filtering,
    co2_var_name,
    nstepsday=None,
    site_year=None,
    et_var_name="ET",
):
    """
    filter bad quailty data, data gaps before calculating cost function or
    calculating model performance metrices

    parameters:
    lue_model_op (dict): output dictionary from running LUE model
    ip_df_dict (dict): input dictionary of forcing variables
    data_filtering (str): data filtering criteria, such as "nominal" or "strict"
    co2_var_name (str): name of the CO2 variable used in the model
    nstepsday (float): temporal resolution or number of timesteps per day)
    site_year (float): year of data to be used for optimization
                       (only used when optimizing each year separately)

    returns:
    gpp_obs (array): filtered observed GPP
    gpp_sim (array): filtered simulated GPP
    nee_unc (array): filtered uncertainty in observed GPP
    et_obs (array): filtered observed ET
    et_sim (array): filtered simulated ET
    et_unc (array): filtered uncertainty in observed ET
    drop_gpp_data_indices (array): indices of points which
                                   were dropped from calculating GPP component of cost function
    drop_et_data_indices (array): indices of points which were dropped
                                  from calculating ET component of cost function

    """

    if data_filtering == "nominal":  # filter data based on nominal strictness
        # (use timesteps where forcing data were filled using downscaled ERA5 data)
        # remove nan GPP_obs, GPP_sim, NEE_RANDUNC,
        # as cost function can't be calculated for these points
        gpp_nan_indices = (
            np.isnan(ip_df_dict["NEE_RANDUNC"])
            | np.isnan(ip_df_dict["GPP_NT"])
            | np.isnan(lue_model_op["gpp_lue"])
        )

        if ip_df_dict["Temp_res"] == "Daily":  # for daily data
            # NEE_QC:  <0.8 = bad; >=0.8 = good
            bad_gpp_obs_indices = ip_df_dict["NEE_QC"] < 0.8
        else:
            # NEE_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
            bad_gpp_obs_indices = (ip_df_dict["NEE_QC"] == 0.0) | (
                ip_df_dict["NEE_QC"] == 0.5
            )

        # replace negative GPP obs values at night
        # (when SW_IN_GF <= 0.0) with 0.0 and use them in cost function
        ip_df_dict["GPP_NT"] = np.where(
            (ip_df_dict["SW_IN_GF"] <= 0.0) & (ip_df_dict["GPP_NT"] < 0.0),
            0.0,
            ip_df_dict["GPP_NT"],
        )

        negative_obs_gpp_indices = (
            ip_df_dict["GPP_NT"] < 0.0
        )  # remove noisy gpp_obs which are negative during daytime

        if co2_var_name == "CO2":  # when using site co2 measurements
            if (
                ip_df_dict["Temp_res"] != "Daily"
            ):  # Daily doesn't have CO2_QC for site co2
                # bad co2 when CO2_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
                remove_co2_indices = (ip_df_dict["CO2_QC"] == 0.0) | (
                    ip_df_dict["CO2_QC"] == 0.5
                )
            else:  # for daily data, don't do any filtering (all remove_indices set to False)
                remove_co2_indices = np.full(ip_df_dict[co2_var_name].shape, False)
                warnings.warn(
                    "Daily data doesn't have CO2_QC for site CO2, so no filtering is done"
                )
        else:  # when using MLO-NOAA CO2 or site CO2 gapfilled with MLO-NOAA CO2,
            # don't do any filtering (all remove_indices set to False)
            remove_co2_indices = np.full(ip_df_dict[co2_var_name].shape, False)

        # indice of points which will be dropped from calculating GPP component of cost function
        drop_gpp_data_indices = (
            gpp_nan_indices
            | bad_gpp_obs_indices
            | negative_obs_gpp_indices
            | remove_co2_indices
        )

        # remove nan ET_obs, ET_sim, ET_RANDUNC,
        # as cost function can't be calculated for these points
        et_nan_indices = (
            np.isnan(ip_df_dict[et_var_name])
            | np.isnan(ip_df_dict["ET_RANDUNC"])
            | np.isnan(lue_model_op["wai_results"]["etsno"])
        )

        if ip_df_dict["Temp_res"] == "Daily":  # for daily data
            # bad quality ETobs when LE_QC < 0.8 = bad; >=0.8 = good
            bad_et_obs_indices = ip_df_dict["LE_QC"] < 0.8
        else:
            # bad quality ETobs when LE_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
            bad_et_obs_indices = (ip_df_dict["LE_QC"] == 0.0) | (
                ip_df_dict["LE_QC"] == 0.5
            )

        # indice of points which will be dropped from calculating ET component of cost function
        drop_et_data_indices = et_nan_indices | bad_et_obs_indices

    # data_filtering == "strict" only use good quality observed data, drop anything else
    # strict data filtering only works for hourly data
    elif data_filtering == "strict":
        if ip_df_dict["Temp_res"] == "Daily":  # for daily data
            sys.exit("strict data filtering only works for hourly data, exiting")

        # remove nan GPP_obs, GPP_sim, NEE_RANDUNC,
        # as cost function can't be calculated for these points
        gpp_nan_indices = (
            np.isnan(ip_df_dict["NEE_RANDUNC"])
            | np.isnan(ip_df_dict["GPP_NT"])
            | np.isnan(lue_model_op["gpp_lue"])
        )
        # bad gpp when NEE_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
        bad_gpp_obs_indices = (ip_df_dict["NEE_QC"] == 0.0) | (
            ip_df_dict["NEE_QC"] == 0.5
        )
        negative_obs_gpp_indices = (
            ip_df_dict["GPP_NT"] < 0.0
        )  # remove noisy gpp_obs which are negative (even during nighttime)
        ppfd_fill_indices = (
            ip_df_dict["PPFD_IN_FILL_FLAG"] == 1.0
        )  # remove points where PPFD was filled (flag == 1.0)
        ta_fill_indices = (
            ip_df_dict["TA_FILL_FLAG"] == 1.0
        )  # remove points where TA was filled (flag == 1.0)
        vpd_fill_indices = (
            ip_df_dict["VPD_FILL_FLAG"] == 1.0
        )  # remove points where VPD was filled (flag == 1.0)

        if co2_var_name == "CO2":  # when using site co2 measurements
            # bad co2 when CO2_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
            remove_co2_indices = (ip_df_dict["CO2_QC"] == 0.0) | (
                ip_df_dict["CO2_QC"] == 0.5
            )
        elif (
            co2_var_name == "CO2_GF"
        ):  # when using CO2 where gaps and bad quality site co2 are filled with MLO NOAA CO2
            remove_co2_indices = (
                ip_df_dict["CO2_FILL_FLAG"] == 1.0
            )  # remove points where CO2 was filled (flag == 1.0)
        else:  # when using MLO-NOAA CO2, don't do any filtering
            # (all remove_indices set to False)
            remove_co2_indices = np.full(ip_df_dict[co2_var_name].shape, False)

        # indice of points which will be dropped from calculating GPP component of cost function
        drop_gpp_data_indices = (
            gpp_nan_indices
            | bad_gpp_obs_indices
            | negative_obs_gpp_indices
            | ppfd_fill_indices
            | ta_fill_indices
            | vpd_fill_indices
            | remove_co2_indices
        )

        # remove nan ET_obs, ET_sim, ET_RANDUNC,
        # as cost function can't be calculated for these points
        et_nan_indices = (
            np.isnan(ip_df_dict[et_var_name])
            | np.isnan(ip_df_dict["ET_RANDUNC"])
            # | np.isnan(lue_model_op["wai_results"]["et"])
            | np.isnan(lue_model_op["wai_results"]["etsno"])
        )
        # bad quality ETobs when LE_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
        bad_et_obs_indices = (ip_df_dict["LE_QC"] == 0.0) | (ip_df_dict["LE_QC"] == 0.5)

        netrad_fill_indices = (ip_df_dict["NETRAD_FILL_FLAG"] == 1.0) | (
            ip_df_dict["NETRAD_FILL_FLAG"] == 2.0
        )  # remove points where netrad was filled with NETRAD_ERA5 (flag == 1.0)
        # or based on regression with SW_IN_GF (flag == 2.0)
        ta_fill_indices = (
            ip_df_dict["TA_FILL_FLAG"] == 1.0
        )  # remove points where TA was filled (flag == 1.0) since TA is used in ET calculation
        p_fill_indices = (
            ip_df_dict["P_FILL_FLAG"] == 1.0
        )  # remove points where P was filled (flag == 1.0) since P is used in ET calculation

        # indice of points which will be dropped from calculating ET component of cost function
        drop_et_data_indices = (
            et_nan_indices
            | bad_et_obs_indices
            | netrad_fill_indices
            | ta_fill_indices
            | p_fill_indices
        )
    else:
        sys.exit("data_filtering must be one of: nominal, strict")

    if site_year is None:  # all year optimization
        # filter data based on drop_indices and get the arrays to be used in cost function
        gpp_obs = ip_df_dict["GPP_NT"][~drop_gpp_data_indices]
        gpp_sim = lue_model_op["gpp_lue"][~drop_gpp_data_indices]
        nee_unc = ip_df_dict["NEE_RANDUNC"][~drop_gpp_data_indices]

        et_obs = ip_df_dict[et_var_name][~drop_et_data_indices]
        # et_sim = lue_model_op["wai_results"]["et"][~drop_et_data_indices]
        et_sim = lue_model_op["wai_results"]["etsno"][~drop_et_data_indices]
        et_unc = ip_df_dict["ET_RANDUNC"][~drop_et_data_indices]

        # log if less than 3 months of data left after filtering
        if nstepsday is not None:
            if (gpp_obs.size < (3.0 * 30.0 * nstepsday)) or (
                et_obs.size < (3.0 * 30.0 * nstepsday)
            ):
                logger.warning(
                    "%s has %d good quality data points, which is less than 3 months of data",
                    ip_df_dict["SiteID"],
                    gpp_obs.size,
                )
    else:  # site year optimization
        # filter data based on drop_indices and
        # get the arrays of the year to be used in cost function
        gpp_obs = ip_df_dict["GPP_NT"][
            ~drop_gpp_data_indices & (ip_df_dict["year"] == site_year)
        ]
        gpp_sim = lue_model_op["gpp_lue"][
            ~drop_gpp_data_indices & (ip_df_dict["year"] == site_year)
        ]
        nee_unc = ip_df_dict["NEE_RANDUNC"][
            ~drop_gpp_data_indices & (ip_df_dict["year"] == site_year)
        ]

        et_obs = ip_df_dict[et_var_name][
            ~drop_et_data_indices & (ip_df_dict["year"] == site_year)
        ]
        et_sim = lue_model_op["wai_results"]["etsno"][
            ~drop_et_data_indices & (ip_df_dict["year"] == site_year)
        ]
        et_unc = ip_df_dict["ET_RANDUNC"][
            ~drop_et_data_indices & (ip_df_dict["year"] == site_year)
        ]

        # log if less than 3 months of data left after filtering
        if nstepsday is not None:
            if (gpp_obs.size < (3.0 * 30.0 * nstepsday)) or (
                et_obs.size < (3.0 * 30.0 * nstepsday)
            ):
                logger.warning(
                    (
                        "%s (%s) has %d good quality GPP and %d ET data points,"
                        "which is less than 3 months of data"
                    ),
                    ip_df_dict["SiteID"],
                    str(int(site_year)),
                    gpp_obs.size,
                    et_obs.size,
                )

    return (
        gpp_obs,
        gpp_sim,
        nee_unc,
        et_obs,
        et_sim,
        et_unc,
        drop_gpp_data_indices,
        drop_et_data_indices,
    )


def calc_cum(df, var_name):
    # df["cumulative_sum"] = df.groupby("year")[var_name].cumsum()

    # df["cumulative_count"] = df.groupby("year").cumcount() + 1

    # # Calculate the cumulative average
    # df["cumulative_average"] = df["cumulative_sum"] / df["cumulative_count"]

    # # Get the cumulative average as a numpy array
    # cumulative_average = df["cumulative_average"].to_numpy()

    return df.groupby("year")[var_name].cumsum().to_numpy()  # cumulative_average


def calc_yearwise_cum(
    gpp_obs,
    gpp_sim,
    nee_unc,
    et_obs,
    et_sim,
    et_unc,
    drop_gpp_data_indices,
    drop_et_data_indices,
    year_arr,
):

    gpp_year_arr = year_arr[~drop_gpp_data_indices]
    et_year_arr = year_arr[~drop_et_data_indices]

    gpp_df = pd.DataFrame(
        {
            "year": gpp_year_arr,
            "gpp_obs": gpp_obs,
            "gpp_sim": gpp_sim,
            "nee_unc": nee_unc,
        }
    )

    et_df = pd.DataFrame(
        {
            "year": et_year_arr,
            "et_obs": et_obs,
            "et_sim": et_sim,
            "et_unc": et_unc,
        }
    )

    gpp_obs_yr_cum_avg = calc_cum(gpp_df, "gpp_obs")
    gpp_sim_yr_cum_avg = calc_cum(gpp_df, "gpp_sim")
    nee_unc_yr_cum_avg = calc_cum(gpp_df, "nee_unc")

    et_obs_yr_cum_avg = calc_cum(et_df, "et_obs")
    et_sim_yr_cum_avg = calc_cum(et_df, "et_sim")
    et_unc_yr_cum_avg = calc_cum(et_df, "et_unc")

    return (
        gpp_obs_yr_cum_avg,
        gpp_sim_yr_cum_avg,
        nee_unc_yr_cum_avg,
        et_obs_yr_cum_avg,
        et_sim_yr_cum_avg,
        et_unc_yr_cum_avg,
        gpp_year_arr,
        et_year_arr,
    )

def calc_mean_yearwise_cost(obs_arr, sim_arr, unc_arr, yr_arr):
    split_indices = np.where(np.diff(yr_arr))[0] + 1

    obs_subsets = np.split(obs_arr, split_indices)
    sim_subsets = np.split(sim_arr, split_indices)
    unc_subsets = np.split(unc_arr, split_indices)

    year_cost_arr = np.zeros(len(np.unique(yr_arr)))
    for ix, val in enumerate(obs_subsets):
        year_cost_arr[ix] = 1.0 - cost_nnse(
            val,
            sim_subsets[ix],
            get_weights_from_normalize_uncertainty(unc_subsets[ix]),
        )

    return year_cost_arr.mean()

def lue_model_cost_function(
    p_values_scalar,
    param_names,
    ip_df_dict,
    ip_df_daily_wai,
    wai_output,
    nstepsday,
    fpar_var_name,
    co2_var_name,
    data_filtering,
    cost_func,
    synthetic_data,
    site_year=None,
    consider_yearly_cost=False,
    et_var_name="ET",
):
    """
    Calculate cost function value for LUE model with good quality observed and simulated data

    parameters:
    p_values_scalar (array): scalar of parameter values given by optimizer
    param_names (list): list of parameter names
    ip_df_dict (dict): input dictionary of forcing variables
    ip_df_daily_wai (dict): input dictionary of daily variables needed to calculate WAI
    wai_output (dict): output dictionary to store variables from the bucket model
    nstepsday (float): number of timesteps per day (24 for hourly, 48 for half-hourly)
    fpar_var_name (str): name of the fpar variable used in the model
    co2_var_name (str): name of the CO2 variable used in the model
    data_filtering (str): data filtering criteria, such as "nominal" or "strict"
    cost_func (str): cost function to be used, such as "cost_nnse_unc", "cost_nrmse_unc", "cost_lue"
    synthetic_data (dict): dictionary of synthetic forcing data
                           to calculate 3rd and 4th part of "cost_lue"
    site_year (float): year of data to be used for optimization in case of site year optimization
    consider_yearly_cost (bool): whether to consider an additional constraint on IAV of GPP

    returns:
    total_cost (float): total cost function value
    """

    # run LUE model with parameters given by optimizer
    lue_model_op = run_lue_model(
        p_values_scalar,
        param_names,
        ip_df_dict,
        ip_df_daily_wai,
        wai_output,
        nstepsday,
        fpar_var_name,
        co2_var_name,
    )

    # get filtered data for calculating cost function
    if site_year is None:  # all year optimization
        (
            gpp_obs,
            gpp_sim,
            nee_unc,
            et_obs,
            et_sim,
            et_unc,
            drop_gpp_data_indices,
            drop_et_data_indices,
        ) = filter_lue_data(
            lue_model_op,
            ip_df_dict,
            data_filtering,
            co2_var_name,
            nstepsday,
            et_var_name=et_var_name,
        )
    else:  # site year optimization
        gpp_obs, gpp_sim, nee_unc, et_obs, et_sim, et_unc, *_ = filter_lue_data(
            lue_model_op,
            ip_df_dict,
            data_filtering,
            co2_var_name,
            nstepsday,
            site_year,
            et_var_name=et_var_name,
        )

    if (gpp_obs.size == 0) or (
        et_obs.size == 0
    ):  # no data left after filtering to calculate cost function
        gpp_cost = np.nan
        et_cost = np.nan
        total_cost = gpp_cost + et_cost
    else:  # continue with calculating cost function when we have some data left after filtering
        if cost_func == "cost_nnse_unc":
            gpp_nnse = cost_nnse(
                gpp_obs, gpp_sim, get_weights_from_normalize_uncertainty(nee_unc)
            )
            et_nnse = cost_nnse(
                et_obs, et_sim, get_weights_from_normalize_uncertainty(et_unc)
            )

            # return 1 - nnse, so that when the cost is minimized,
            # the nnse is maximized (better model performance)
            gpp_cost = 1.0 - gpp_nnse
            et_cost = 1.0 - et_nnse

            total_cost = gpp_cost + et_cost

        elif cost_func == "cost_nrmse_unc":
            gpp_cost = cost_nrmse(gpp_obs, gpp_sim, normalize_model=2, unc=nee_unc)
            et_cost = cost_nrmse(et_obs, et_sim, normalize_model=2, unc=et_unc)

            total_cost = gpp_cost + et_cost

        elif cost_func == "cost_nrmse":
            gpp_cost = cost_nrmse(gpp_obs, gpp_sim, normalize_model=2)
            et_cost = cost_nrmse(et_obs, et_sim, normalize_model=2)

            total_cost = gpp_cost + et_cost

        elif cost_func == "cost_lue":
            #############################################
            # cost func part 1 and part2 (GPP and ET)
            #############################################
            gpp_nnse = cost_nnse(
                gpp_obs, gpp_sim, get_weights_from_normalize_uncertainty(nee_unc)
            )
            et_nnse = cost_nnse(
                et_obs, et_sim, get_weights_from_normalize_uncertainty(et_unc)
            )

            # cost is 1 - nnse, so that when the cost is minimized,
            # the nnse is maximized (better model performance)
            gpp_cost = 1.0 - gpp_nnse
            et_cost = 1.0 - et_nnse

            #############################################################
            # cost func part 3 and part4 (ideal and non-ideal conditions)
            #############################################################
            # recalculate parameter values based on the scalar values given by optimizer
            updated_params = get_params(ip_df_dict, param_names, p_values_scalar)

            # calculate sensitivity function for temperature
            # with synthetic data and parameters given by optimizer
            f_tair_synthetic = f_temp_horn(
                synthetic_data["TA"],
                updated_params["T_opt"],
                updated_params["K_T"],
                updated_params["alpha_fT_Horn"],
            )

            # calculate sensitivity function for VPD
            # with synthetic data and parameters given by optimizer
            _, f_vpd_part_synthetic, _ = f_vpd_co2_preles(
                synthetic_data["VPD"],
                synthetic_data["CO2"],
                updated_params["Kappa_VPD"],
                updated_params["Ca_0"],
                updated_params["C_Kappa"],
                updated_params["c_m"],
            )

            # calculate sensitivity function for soil moisture
            # with synthetic data and parameters given by optimizer
            f_water_synthetic = f_water_horn(
                synthetic_data["wai_nor"],
                updated_params["W_I"],
                updated_params["K_W"],
                updated_params["alpha"],
            )

            # calculate sensitivity function for light scalar
            # with synthetic data and parameters given by optimizer
            f_light_synthetic = f_light_scalar_tal(
                synthetic_data["fPAR"],
                synthetic_data["PPFD"],
                updated_params["gamma_fL_TAL"],
            )

            # in ideal conditions (the ranges of environmental forcings in synthetic data),
            # all sensitivity functions should be as close to 1.0
            cost_ideal_cond = (
                abs(1.0 - f_tair_synthetic.max())
                + abs(1.0 - f_vpd_part_synthetic.max())
                + abs(1.0 - f_water_synthetic.max())
                + abs(1.0 - f_light_synthetic.max())
            ) * 1e3  # multiply by a scalar to make the cost
            # function value comparable to other parts

            # in non ideal conditions, the sensitivity functions
            # should be lower than certain threshold

            # when temperature is below 0 degC (non-ideal), fT should be lower than 0.2.
            # in case fT is above 0.2, then try to minimize the difference
            f_tair_synthetic_non_ideal = f_tair_synthetic[synthetic_data["TA"] < 0.0]
            f_tair_synthetic_non_ideal = np.where(
                f_tair_synthetic_non_ideal > 0.2, f_tair_synthetic_non_ideal - 0.2, 0.0
            )

            # when VPD is above 2000 Pa (non-ideal), fVPD (only VPD part) should be lower than 0.9.
            # in case fVPD (only VPD part) is above 0.9, then try to minimize the difference
            f_vpd_part_synthetic_non_ideal = f_vpd_part_synthetic[
                synthetic_data["VPD"] > 2000.0
            ]
            f_vpd_part_synthetic_non_ideal = np.where(
                f_vpd_part_synthetic_non_ideal > 0.9,
                f_vpd_part_synthetic_non_ideal - 0.9,
                0.0,
            )

            # when normalized wai is below 0.01 (non-ideal), fW should be lower than 0.2.
            # in case fW is above 0.2, then try to minimize the difference
            f_water_synthetic_non_ideal = f_water_synthetic[
                synthetic_data["wai_nor"] < 0.01
            ]
            f_water_synthetic_non_ideal = np.where(
                f_water_synthetic_non_ideal > 0.2,
                f_water_synthetic_non_ideal - 0.2,
                0.0,
            )

            # sum up the difference of fX and thresholds under non ideal conditions
            cost_non_ideal_cond = (
                f_tair_synthetic_non_ideal.sum()
                + f_vpd_part_synthetic_non_ideal.sum()
                + f_water_synthetic_non_ideal.sum()
            )

            # calculate total costs
            total_cost = gpp_cost + et_cost + cost_ideal_cond + cost_non_ideal_cond

        else:
            raise ValueError(
                "for LUE model, cost_func must be one of: cost_nnse_unc,"
                "cost_nrmse_unc, cost_nrmse, cost_lue"
            )

        # add an additional constraint on IAV of GPP
        if (consider_yearly_cost) and (site_year is None):
            (
                gpp_obs_yr_cum_avg,
                gpp_sim_yr_cum_avg,
                nee_unc_yr_cum_avg,
                et_obs_yr_cum_avg,
                et_sim_yr_cum_avg,
                et_unc_yr_cum_avg,
                gpp_yr_arr,
                et_yr_arr,
            ) = calc_yearwise_cum(
                gpp_obs,
                gpp_sim,
                nee_unc,
                et_obs,
                et_sim,
                et_unc,
                drop_gpp_data_indices,
                drop_et_data_indices,
                ip_df_dict["year"],
            )

            # gpp_cost_yr = calc_mean_yearwise_cost(
            #     gpp_obs_yr_cum_avg, gpp_sim_yr_cum_avg, nee_unc_yr_cum_avg, gpp_yr_arr
            # )
            # et_cost_yr = calc_mean_yearwise_cost(
            #     et_obs_yr_cum_avg, et_sim_yr_cum_avg, et_unc_yr_cum_avg, et_yr_arr
            # )

            gpp_yr_nnse = cost_nnse(
                gpp_obs_yr_cum_avg,
                gpp_sim_yr_cum_avg,
                get_weights_from_normalize_uncertainty(nee_unc_yr_cum_avg),
            )
            et_yr_nnse = cost_nnse(
                et_obs_yr_cum_avg,
                et_sim_yr_cum_avg,
                get_weights_from_normalize_uncertainty(et_unc_yr_cum_avg),
            )
    
            # cost is 1 - nnse, so that when the cost is minimized,
            # the nnse is maximized (better model performance)
            gpp_cost_yr = 1.0 - gpp_yr_nnse
            et_cost_yr = 1.0 - et_yr_nnse

            total_cost = total_cost + gpp_cost_yr + et_cost_yr

        elif (consider_yearly_cost) and (site_year is not None):
            raise ValueError(
                "for site year optimization, consider_yearly_cost must be False"
            )

    return total_cost
