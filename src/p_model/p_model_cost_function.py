#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this module calculates the cost function for P model

author: rde
first created: 2023-11-07

"""
import sys
import warnings
import logging
import numpy as np
import pandas as pd

from src.p_model.run_gpp_p_acclim import run_p_model

logger = logging.getLogger(__name__)

# ToDo: when RANDUNC is not used in cost function, remove it from
# one of the data filtering criteria in filter_data function
def filter_data(
    p_model_acclim_fw_op,
    ip_df_dict,
    data_filtering,
    co2_var_name,
    time_info=None,
    site_year=None,
    et_var_name="ET",
):
    """
    filter bad quailty data, data gaps before calculating cost function or
    calculating model performance metrices

    parameters:
    p_model_acclim_fw_op (dict): output dictionary from running PModel with acclimation
    ip_df_dict (dict): input dictionary of forcing variables
    data_filtering (str): data filtering criteria, such as "nominal" or "strict"
    co2_var_name (str): name of the CO2 variable used in the model
    time_info (dict): dictionary with time related information
                      (e.g., temporal resolution or number of timesteps per day)
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
            | np.isnan(p_model_acclim_fw_op["GPPp_opt_fW"])
        )
        # bad gpp when NEE_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
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
            # bad co2 when CO2_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
            remove_co2_indices = (ip_df_dict["CO2_QC"] == 0.0) | (
                ip_df_dict["CO2_QC"] == 0.5
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
            # | np.isnan(p_model_acclim_fw_op["wai_results"]["et"])
            | np.isnan(p_model_acclim_fw_op["wai_results"]["etsno"])
        )
        # bad quality ETobs when LE_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
        bad_et_obs_indices = (ip_df_dict["LE_QC"] == 0.0) | (ip_df_dict["LE_QC"] == 0.5)
        # negative_obs_et_indices = (ip_df_dict[et_var_name] < 0.0) | (
        #     p_model_acclim_fw_op["wai_results"]["et"] < 0.0
        # )  # remove negative et_obs and et_sim

        # indice of points which will be dropped from calculating ET component of cost function
        drop_et_data_indices = (
            et_nan_indices | bad_et_obs_indices
        )  # | negative_obs_et_indices

    elif data_filtering == "strict":  # data_filtering == "strict" only use good quality observed data, drop anything else
        # remove nan GPP_obs, GPP_sim, NEE_RANDUNC,
        # as cost function can't be calculated for these points
        gpp_nan_indices = (
            np.isnan(ip_df_dict["NEE_RANDUNC"])
            | np.isnan(ip_df_dict["GPP_NT"])
            | np.isnan(p_model_acclim_fw_op["GPPp_opt_fW"])
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
        else:  # when using MLO-NOAA CO2, don't do any filtering (all remove_indices set to False)
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
            # | np.isnan(p_model_acclim_fw_op["wai_results"]["et"])
            | np.isnan(p_model_acclim_fw_op["wai_results"]["etsno"])
        )
        # bad quality ETobs when LE_QC:  0.0 = bad; 0.5 = medium; 1.0 = good
        bad_et_obs_indices = (ip_df_dict["LE_QC"] == 0.0) | (ip_df_dict["LE_QC"] == 0.5)
        # negative_obs_et_indices = (ip_df_dict[et_var_name] < 0.0) | (
        #     p_model_acclim_fw_op["wai_results"]["et"] < 0.0
        # )  # remove negative et_obs and et_sim
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
        )  # | negative_obs_et_indices
    else:
        sys.exit("data_filtering must be one of: nominal, strict")

    if site_year is None:
        # filter data based on drop_indices and get the arrays to be used in cost function
        gpp_obs = ip_df_dict["GPP_NT"][~drop_gpp_data_indices]
        gpp_sim = p_model_acclim_fw_op["GPPp_opt_fW"][~drop_gpp_data_indices]
        nee_unc = ip_df_dict["NEE_RANDUNC"][~drop_gpp_data_indices]

        et_obs = ip_df_dict[et_var_name][~drop_et_data_indices]
        # et_sim = p_model_acclim_fw_op["wai_results"]["et"][~drop_et_data_indices]
        et_sim = p_model_acclim_fw_op["wai_results"]["etsno"][~drop_et_data_indices]
        et_unc = ip_df_dict["ET_RANDUNC"][~drop_et_data_indices]

        # log if less than 3 months of data left after filtering
        if time_info is not None:
            if (gpp_obs.size < (3.0 * 30.0 * time_info["nstepsday"])) or (
                et_obs.size < (3.0 * 30.0 * time_info["nstepsday"])
            ):
                logger.warning(
                    "%s has %d good quality data points, which is less than 3 months of data",
                    ip_df_dict["SiteID"],
                    gpp_obs.size,
                )
    else:
        # filter data based on drop_indices and
        # get the arrays of the year to be used in cost function
        gpp_obs = ip_df_dict["GPP_NT"][
            ~drop_gpp_data_indices & (ip_df_dict["year"] == site_year)
        ]
        gpp_sim = p_model_acclim_fw_op["GPPp_opt_fW"][
            ~drop_gpp_data_indices & (ip_df_dict["year"] == site_year)
        ]
        nee_unc = ip_df_dict["NEE_RANDUNC"][
            ~drop_gpp_data_indices & (ip_df_dict["year"] == site_year)
        ]

        et_obs = ip_df_dict[et_var_name][
            ~drop_et_data_indices & (ip_df_dict["year"] == site_year)
        ]
        # et_sim = p_model_acclim_fw_op["wai_results"]["et"][
        #     ~drop_et_data_indices & (ip_df_dict["year"] == site_year)
        # ]
        et_sim = p_model_acclim_fw_op["wai_results"]["etsno"][
            ~drop_et_data_indices & (ip_df_dict["year"] == site_year)
        ]
        et_unc = ip_df_dict["ET_RANDUNC"][
            ~drop_et_data_indices & (ip_df_dict["year"] == site_year)
        ]

        # log if less than 3 months of data left after filtering
        if time_info is not None:
            if (gpp_obs.size < (3.0 * 30.0 * time_info["nstepsday"])) or (
                et_obs.size < (3.0 * 30.0 * time_info["nstepsday"])
            ):
                logger.warning(
                    (
                        "%s (%s) has %d good quality GPP and %d ET data points,",
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


def get_weights_from_normalize_uncertainty(uncertainty):
    """
    normalize uncertainty between 0 and 1,
    then return (1 - normalized uncertainty) as weights to be used in cost function

    Parameters:
    uncertainty (array): uncertainty in observed data

    Returns:
    weights (array): weights to be used in cost function
    """
    min_uncertainty = np.min(uncertainty)
    max_uncertainty = np.max(uncertainty)
    # a runtime warning will be produced when max_uncertainty == min_uncertainty
    # for e.g., SJ-Adv_2012 (in daily resolution) has only 2 good quality data points
    # left with the same uncertainty values. This will lead to nan weights and nan cost values
    # and optimization will be aborted (doesn't matter as we don't have enough data anyway)
    normalized_uncertainty = (uncertainty - min_uncertainty) / (
        max_uncertainty - min_uncertainty
    )

    weights = 1.0 - normalized_uncertainty
    return weights


def cost_nse(obs, sim, unc=None):
    """
    calculate Nash-Sutcliffe efficiency (range -inf to 1) between observed and simulated data
    ref: weighted NSE: https://doi.org/10.1029/2011WR010527

    Parameters:
    obs (array): observed data
    sim (array): simulated data
    unc (array): uncertainty in observed data

    Returns:
    nse (float): Nash-Sutcliffe efficiency between observed and simulated data
    """
    if unc is None:  # if no uncertainty is given, use unweighted NSE
        nse = 1.0 - np.sum((obs - sim) ** 2.0) / np.sum((obs - np.mean(obs)) ** 2.0)
    else:  # if uncertainty is given, use weighted NSE
        # nse = 1.0 - np.sum((obs - sim) ** 2.0 / unc**2.0) / np.sum(
        #     (obs - np.mean(obs)) ** 2.0 / unc**2.0
        # )
        nse = 1.0 - np.sum(unc * (obs - sim) ** 2.0) / np.sum(
            unc * (obs - np.mean(obs)) ** 2.0
        )

    return nse


def cost_nnse(obs, sim, unc=None):
    """
    calculate normalized Nash-Sutcliffe efficiency (between 0 to 1) from NSE

    Parameters:
    obs (array): observed data
    sim (array): simulated data
    unc (array): uncertainty in observed data

    Returns:
    nnse (float): normalized Nash-Sutcliffe efficiency between observed and simulated data
    """
    if unc is None:
        nnse = 1.0 / (2.0 - cost_nse(obs, sim))
    else:
        nnse = 1.0 / (2.0 - cost_nse(obs, sim, unc))

    return nnse

# ToDo: unc is not correctly implemented in cost_nrmse, like cost_nnse
def cost_nrmse(obs, sim, normalize_model=2, unc=None):
    """
    calculate normalized (by different methods) root mean square error

    Parameters:
    obs (array): observed data
    sim (array): simulated data
    normalize_model (int): method to normalize rmse values
                           (1 = mean, 2 = range, 3 = log, 4=standard deviation)
    unc (array): uncertainty in observed data

    Returns:
    nrmse (float): normalized root mean square error between observed and simulated data

    """
    if unc is None:
        rmse = np.sqrt(np.mean((sim - obs) ** 2.0))
        if normalize_model == 1:
            nrmse = rmse / np.mean(sim)
        elif normalize_model == 2:
            nrmse = rmse / (np.max(obs) - np.min(obs))
        elif normalize_model == 3:
            nrmse = np.sqrt(np.sum(np.log((sim + 1.0) / (obs + 1.0)) ** 2.0) / len(obs))
        else:
            nrmse = rmse / sim.std()
    else:
        warnings.warn(
                (
                    "cost_nrmse_unc is not correctly implemented yet"
                    "with weights calculated from normalized uncertainty"
                )
            )
        rmse = np.sqrt(np.mean((sim - obs) ** 2.0 / unc**2.0))
        if normalize_model == 1:
            nrmse = rmse / np.mean(sim / unc)
        elif normalize_model == 2:
            nrmse = rmse / (np.max(obs / unc) - np.min(obs / unc))
        elif normalize_model == 3:
            nrmse = np.sqrt(
                np.sum(np.log((sim / unc + 1.0) / (obs / unc + 1.0)) ** 2.0) / len(obs)
            )
        else:
            nrmse = rmse / (sim / unc).std()

    return nrmse

def calc_cum(df, var_name):
    # df["cumulative_sum"] = df.groupby("year")[var_name].cumsum()

    # df["cumulative_count"] = df.groupby("year").cumcount() + 1

    # # Calculate the cumulative average
    # df["cumulative_average"] = df["cumulative_sum"] / df["cumulative_count"]

    # # Get the cumulative average as a numpy array
    # cumulative_average = df["cumulative_average"].to_numpy()

    return df.groupby("year")[var_name].cumsum().to_numpy() #cumulative_average


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


def p_model_cost_function(
    p_values_scalar,
    p_names,
    ip_df_dict,
    model_op_no_acclim_sd,
    ip_df_daily_wai,
    wai_output,
    time_info,
    fpar_var_name,
    co2_var_name,
    data_filtering,
    cost_func,
    site_year=None,
    consider_yearly_cost=False,
    et_var_name="ET",
):
    """
    calculate cost function for GPP and ET components

    Parameters:
    p_values_scalar (array): scalar of parameter values given by optimizer
    p_names (list): list of parameter names which were optimized
    ip_df_dict (dict): input dictionary of forcing variables
    model_op_no_acclim_sd (dict): output dictionary from running PModel without acclimation
    ip_df_daily_wai (dict): input dictionary to calculate
                            hourly avergae of WAI per day during spinup
    wai_output (dict): output dictionary to store WAI calculation
                       results (contains arrays of zeros per variable)
    time_info (dict): dictionary with time related information
                      (e.g., temporal resolution or number of timesteps per day)
    fpar_var_name (str): name of the fPAR variable used in the model
    co2_var_name (str): name of the CO2 variable used in the model
    data_filtering (str): data filtering criteria, such as "nominal" or "strict"
    cost_func (str): cost function to be used for optimization
    site_year (float): year of data to be used for optimization
                       (only used when optimizing each year separately)
    consider_yearly_cost (bool): whether to consider an additional constraint on IAV of GPP

    Returns:
    cost (float): cost function value

    """

    # run PModel to get simulated GPP and ET
    p_model_acclim_fw_op = run_p_model(
        p_values_scalar,
        p_names,
        ip_df_dict,
        model_op_no_acclim_sd,
        ip_df_daily_wai,
        wai_output,
        time_info,
        fpar_var_name,
        co2_var_name,
    )
    # get filtered data for calculating cost function
    if site_year is None:  # all year optimization
        gpp_obs, gpp_sim, nee_unc, et_obs, et_sim, et_unc, drop_gpp_data_indices, drop_et_data_indices = filter_data(
            p_model_acclim_fw_op,
            ip_df_dict,
            data_filtering,
            co2_var_name,
            time_info,
            et_var_name=et_var_name,
        )
    else:  # site year optimization
        gpp_obs, gpp_sim, nee_unc, et_obs, et_sim, et_unc, *_ = filter_data(
            p_model_acclim_fw_op,
            ip_df_dict,
            data_filtering,
            co2_var_name,
            time_info,
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

        elif cost_func == "cost_nnse":
            gpp_nnse = cost_nnse(gpp_obs, gpp_sim)
            et_nnse = cost_nnse(et_obs, et_sim)

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

        else:
            raise ValueError(f"cost function {cost_func} is not implemented for P model")
        
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
                gpp_obs_yr_cum_avg, gpp_sim_yr_cum_avg, get_weights_from_normalize_uncertainty(nee_unc_yr_cum_avg)
            )
            et_yr_nnse = cost_nnse(
                et_obs_yr_cum_avg, et_sim_yr_cum_avg, get_weights_from_normalize_uncertainty(et_unc_yr_cum_avg)
            )

            # cost is 1 - nnse, so that when the cost is minimized,
            # the nnse is maximized (better model performance)
            gpp_cost_yr = 1.0 - gpp_yr_nnse
            et_cost_yr = 1.0 - et_yr_nnse

            total_cost = gpp_cost + et_cost + gpp_cost_yr + et_cost_yr

    return total_cost
