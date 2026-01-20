#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare the results of model opti/eval, collect obs and sim data at different temporal resolutions,
calculate model performance metrices

author: rde
first created: 2023-11-10
"""

import logging
import warnings
import numpy as np
import pandas as pd
from permetrics import RegressionMetric

from src.p_model.p_model_cost_function import filter_data
from src.lue_model.lue_model_cost_function import filter_lue_data
from src.common.get_data import df_to_dict
from src.common.get_params import get_params

logger = logging.getLogger(__name__)


def calc_variability_metrics(obs, sim):
    """
    calculate variability metrics for observed and simulated data
    ref: https://doi.org/10.1016/j.jhydrol.2009.08.003

    parameters:
    obs (array): observed data
    sim (array): simulated data

    returns:
    coefficient of variation
    """

    return sim.std() / obs.std()


def calc_bias_metrics(obs, sim):
    """
    calculate bias metrics for observed and simulated data
    ref: https://doi.org/10.1016/j.jhydrol.2009.08.003

    parameters:
    obs (array): observed data
    sim (array): simulated data

    returns:
    bias
    """

    return (sim.mean() - obs.mean()) / obs.std()


def filter_data_up_dd(in_dict):
    """
    filter out nan and bad data (temporal resolution daily and above) from input dictionary

    parameters:
    in_dict (dict): dictionary with input data

    returns:
    gpp_obs_filtered (array): filtered gpp_obs data
    gpp_sim_filtered (array): filtered gpp_sim data
    """
    # when more than 50% of data in a day/week/month/year were dropped
    # during optimization, then that day/week/month/year is bad
    filter_idx = in_dict["drop_gpp_idx"] < 0.5

    # get good data to calculate model perfromance metrices
    good_data_mask = ~(filter_idx)
    gpp_obs_filtered = in_dict["GPP_obs"][good_data_mask]
    gpp_sim_filtered = in_dict["GPP_sim"][good_data_mask]
    time_filtered = in_dict["Time"][good_data_mask]

    return gpp_obs_filtered, gpp_sim_filtered, time_filtered, good_data_mask


def calc_gpp_model_perform(
    ip_df_dict,
    model_op,
    filtered_gpp_obs,
    filtered_gpp_sim,
    drop_gpp_data_indices,
    sim_gpp_var_name,
    # no_of_param,
):
    """
    calculate model performance metrices for GPP at different temporal resolutions

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    model_op (dict): dictionary with PModel output with acclimation
    filtered_gpp_obs (array): filtered gpp_obs data
    filtered_gpp_sim (array): filtered gpp_sim data
    sim_gpp_var_name (str): name of gpp_sim variable to
                            be used to calculate model performance metrices
    drop_gpp_data_indices (bool): indices at which bad quality subdaily data were
                                  removed during optimization/ evaluation
    no_of_param (int): number of parameters in the model

    returns:
    model_perform_metric_dict (dict): dictionary with model performance
                                      metrices at different temporal resolutions
    gpp_d_dict (dict): dictionary with resampled gpp_obs and gpp_sim data at daily scale
    gpp_w_dict (dict): dictionary with resampled gpp_obs and gpp_sim data at weekly scale
    gpp_m_dict (dict): dictionary with resampled gpp_obs and gpp_sim data at monthly scale
    gpp_y_dict (dict): dictionary with resampled gpp_obs and gpp_sim data at yearly scale
    """
    try:  # when using site CO2 as a variable in subdaily model run
        co2_qc_arr = ip_df_dict["CO2_QC"]
    except KeyError:  # CO2_QC is not available in the daily input data
        co2_qc_arr = np.ones(ip_df_dict["Time"].size)

    # create a dataframe with subdaily gpp_obs, gpp_sim, NEE_QC and CO2_QC
    gpp_resample_df = pd.DataFrame(
        {
            "Time": ip_df_dict["Time"],
            "GPP_obs": ip_df_dict["GPP_NT"],
            "GPP_sim": model_op[sim_gpp_var_name],
            "NEE_QC": ip_df_dict["NEE_QC"],
            "CO2_QC": co2_qc_arr,
            "drop_gpp_idx": ~drop_gpp_data_indices,
        }
    )

    # resample the dataframe to different temporal resolutions
    gpp_df_d = gpp_resample_df.resample("D", on="Time").mean()
    gpp_df_w = gpp_resample_df.resample("W", on="Time").mean()
    gpp_df_m = gpp_resample_df.resample("M", on="Time").mean()
    gpp_df_y = gpp_resample_df.resample("Y", on="Time").mean()

    gpp_df_d = gpp_df_d.reset_index()
    gpp_df_w = gpp_df_w.reset_index()
    gpp_df_m = gpp_df_m.reset_index()
    gpp_df_y = gpp_df_y.reset_index()

    # create dictionaries from the resampled dataframes
    gpp_d_dict = df_to_dict(gpp_df_d)
    gpp_w_dict = df_to_dict(gpp_df_w)
    gpp_m_dict = df_to_dict(gpp_df_m)
    gpp_y_dict = df_to_dict(gpp_df_y)

    # filter good quality data from the dictionaries to calculate model performance metrices
    gpp_obs_d_filtered, gpp_sim_d_filtered, _, good_gpp_d_idx = filter_data_up_dd(
        gpp_d_dict
    )
    gpp_obs_w_filtered, gpp_sim_w_filtered, _, good_gpp_w_idx = filter_data_up_dd(
        gpp_w_dict
    )
    gpp_obs_m_filtered, gpp_sim_m_filtered, _, good_gpp_m_idx = filter_data_up_dd(
        gpp_m_dict
    )
    gpp_obs_y_filtered, gpp_sim_y_filtered, time_y_filtered, good_gpp_y_idx = (
        filter_data_up_dd(gpp_y_dict)
    )

    gpp_y_filtered_dict = {
        "Time": time_y_filtered,
        "GPP_obs": gpp_obs_y_filtered,
        "GPP_sim": gpp_sim_y_filtered,
    }

    # calculate model performance metrices at different temporal resolutions
    evaluator_sd = RegressionMetric(
        filtered_gpp_obs, filtered_gpp_sim, decimal=5
    )  # not used in case of daily model run
    if ip_df_dict["Temp_res"] == "Daily":
        # in case of daily data, calculate metrices directly
        # from the filtered data as used in cost function
        evaluator_d = RegressionMetric(filtered_gpp_obs, filtered_gpp_sim, decimal=5)
        good_gpp_d_idx = ~drop_gpp_data_indices
    else:
        evaluator_d = RegressionMetric(
            gpp_obs_d_filtered, gpp_sim_d_filtered, decimal=5
        )
    evaluator_w = RegressionMetric(gpp_obs_w_filtered, gpp_sim_w_filtered, decimal=5)
    evaluator_m = RegressionMetric(gpp_obs_m_filtered, gpp_sim_m_filtered, decimal=5)
    evaluator_y = RegressionMetric(gpp_obs_y_filtered, gpp_sim_y_filtered, decimal=5)

    # initialize a dictionary to store model performance metrices
    model_perform_metric_dict = {
        "nse_sd": np.nan,
        "nse_d": np.nan,
        "nse_w": np.nan,
        "nse_m": np.nan,
        "nse_y": np.nan,
        "cod_sd": np.nan,
        "cod_d": np.nan,
        "cod_w": np.nan,
        "cod_m": np.nan,
        "cod_y": np.nan,
        "r2_sd": np.nan,
        "r2_d": np.nan,
        "r2_w": np.nan,
        "r2_m": np.nan,
        "r2_y": np.nan,
        "kge_sd": np.nan,
        "kge_d": np.nan,
        "kge_w": np.nan,
        "kge_m": np.nan,
        "kge_y": np.nan,
        "rmse_sd": np.nan,
        "rmse_d": np.nan,
        "rmse_w": np.nan,
        "rmse_m": np.nan,
        "rmse_y": np.nan,
        "corr_coeff_sd": np.nan,
        "corr_coeff_d": np.nan,
        "corr_coeff_w": np.nan,
        "corr_coeff_m": np.nan,
        "corr_coeff_y": np.nan,
        "variability_coeff_sd": np.nan,
        "variability_coeff_d": np.nan,
        "variability_coeff_w": np.nan,
        "variability_coeff_m": np.nan,
        "variability_coeff_y": np.nan,
        "bias_coeff_sd": np.nan,
        "bias_coeff_d": np.nan,
        "bias_coeff_w": np.nan,
        "bias_coeff_m": np.nan,
        "bias_coeff_y": np.nan,
    }

    # Some runtime warnings (Zerodivision) may be produced, when the GPPsim are all zero.
    # This happens when fPAR FLUXNET EO is somehow zero (maybe no good quality MODIS data present,
    # and NDVI/fPAR couldn't be calculated) for all timesteps (e.g., CG-Tch, GH-Ank, MY-PSO, US-LWW)
    runtime_error_list = []  # list to collect the runtime warnings
    # list of functions to calculate model performance metrices at different temporal resolutions
    calc_metric_func_list = [
        # sub-daily metrices are same as daily, when daily data is used
        lambda: model_perform_metric_dict.update({"nse_sd": evaluator_sd.NSE()}),
        lambda: model_perform_metric_dict.update({"cod_sd": evaluator_sd.COD()}),
        lambda: model_perform_metric_dict.update({"r2_sd": (evaluator_sd.PCC()) ** 2}),
        lambda: model_perform_metric_dict.update({"kge_sd": evaluator_sd.KGE()}),
        lambda: model_perform_metric_dict.update({"rmse_sd": evaluator_sd.RMSE()}),
        lambda: model_perform_metric_dict.update({"nse_d": evaluator_d.NSE()}),
        lambda: model_perform_metric_dict.update({"cod_d": evaluator_d.COD()}),
        lambda: model_perform_metric_dict.update({"r2_d": (evaluator_d.PCC()) ** 2}),
        lambda: model_perform_metric_dict.update({"kge_d": evaluator_d.KGE()}),
        lambda: model_perform_metric_dict.update({"rmse_d": evaluator_d.RMSE()}),
        lambda: model_perform_metric_dict.update({"nse_w": evaluator_w.NSE()}),
        lambda: model_perform_metric_dict.update({"cod_w": evaluator_w.COD()}),
        lambda: model_perform_metric_dict.update({"r2_w": (evaluator_w.PCC()) ** 2}),
        lambda: model_perform_metric_dict.update({"kge_w": evaluator_w.KGE()}),
        lambda: model_perform_metric_dict.update({"rmse_w": evaluator_w.RMSE()}),
        lambda: model_perform_metric_dict.update({"nse_m": evaluator_m.NSE()}),
        lambda: model_perform_metric_dict.update({"cod_m": evaluator_m.COD()}),
        lambda: model_perform_metric_dict.update({"r2_m": (evaluator_m.PCC()) ** 2}),
        lambda: model_perform_metric_dict.update({"kge_m": evaluator_m.KGE()}),
        lambda: model_perform_metric_dict.update({"rmse_m": evaluator_m.RMSE()}),
        lambda: model_perform_metric_dict.update({"nse_y": evaluator_y.NSE()}),
        lambda: model_perform_metric_dict.update({"cod_y": evaluator_y.COD()}),
        lambda: model_perform_metric_dict.update({"r2_y": (evaluator_y.PCC()) ** 2}),
        lambda: model_perform_metric_dict.update({"kge_y": evaluator_y.KGE()}),
        lambda: model_perform_metric_dict.update({"rmse_y": evaluator_y.RMSE()}),
        # calculate correlation coefficient
        lambda: model_perform_metric_dict.update(
            {"corr_coeff_sd": np.corrcoef(filtered_gpp_obs, filtered_gpp_sim)[0, 1]}
        ),
        lambda: model_perform_metric_dict.update(
            {"corr_coeff_d": np.corrcoef(gpp_obs_d_filtered, gpp_sim_d_filtered)[0, 1]}
        ),
        lambda: model_perform_metric_dict.update(
            {"corr_coeff_w": np.corrcoef(gpp_obs_w_filtered, gpp_sim_w_filtered)[0, 1]}
        ),
        lambda: model_perform_metric_dict.update(
            {"corr_coeff_m": np.corrcoef(gpp_obs_m_filtered, gpp_sim_m_filtered)[0, 1]}
        ),
        lambda: model_perform_metric_dict.update(
            {"corr_coeff_y": np.corrcoef(gpp_obs_y_filtered, gpp_sim_y_filtered)[0, 1]}
        ),
        # calculate variability coefficient
        lambda: model_perform_metric_dict.update(
            {
                "variability_coeff_sd": calc_variability_metrics(
                    filtered_gpp_obs, filtered_gpp_sim
                )
            }
        ),
        lambda: model_perform_metric_dict.update(
            {
                "variability_coeff_d": calc_variability_metrics(
                    gpp_obs_d_filtered, gpp_sim_d_filtered
                )
            }
        ),
        lambda: model_perform_metric_dict.update(
            {
                "variability_coeff_w": calc_variability_metrics(
                    gpp_obs_w_filtered, gpp_sim_w_filtered
                )
            }
        ),
        lambda: model_perform_metric_dict.update(
            {
                "variability_coeff_m": calc_variability_metrics(
                    gpp_obs_m_filtered, gpp_sim_m_filtered
                )
            }
        ),
        lambda: model_perform_metric_dict.update(
            {
                "variability_coeff_y": calc_variability_metrics(
                    gpp_obs_y_filtered, gpp_sim_y_filtered
                )
            }
        ),
        # calculate bias coefficient
        lambda: model_perform_metric_dict.update(
            {"bias_coeff_sd": calc_bias_metrics(filtered_gpp_obs, filtered_gpp_sim)}
        ),
        lambda: model_perform_metric_dict.update(
            {"bias_coeff_d": calc_bias_metrics(gpp_obs_d_filtered, gpp_sim_d_filtered)}
        ),
        lambda: model_perform_metric_dict.update(
            {"bias_coeff_w": calc_bias_metrics(gpp_obs_w_filtered, gpp_sim_w_filtered)}
        ),
        lambda: model_perform_metric_dict.update(
            {"bias_coeff_m": calc_bias_metrics(gpp_obs_m_filtered, gpp_sim_m_filtered)}
        ),
        lambda: model_perform_metric_dict.update(
            {"bias_coeff_y": calc_bias_metrics(gpp_obs_y_filtered, gpp_sim_y_filtered)}
        ),
    ]

    with warnings.catch_warnings():
        warnings.filterwarnings("error")  # turn warnings into errors
        # calculate model performance metrices
        for func_ix, metric_func in enumerate(calc_metric_func_list):
            if (gpp_obs_w_filtered.size < 3) and (
                func_ix in [10, 11, 12, 13, 14, 27, 32, 37]  # , 42]
            ):
                # if there are less than 3 weekly good quality data points,
                # then skip the weekly metrices
                logger.warning(
                    "%s : only %d points left to calculate weekly metrices, skipping",
                    ip_df_dict["SiteID"],
                    gpp_obs_w_filtered.size,
                )
            elif (gpp_obs_m_filtered.size < 3) and (
                func_ix in [15, 16, 17, 18, 19, 28, 33, 38]  # , 43]
            ):
                # if there are less than 3 monthly good quality data points,
                # then skip the monthly metrices
                logger.warning(
                    "%s : only %d points left to calculate monthly metrices, skipping",
                    ip_df_dict["SiteID"],
                    gpp_obs_m_filtered.size,
                )
            elif (gpp_obs_y_filtered.size < 3) and (
                func_ix in [20, 21, 22, 23, 24, 29, 34, 39]  # , 44]
            ):
                # if there are less than 3 yearly good quality data points,
                # then skip the yearly metrices
                logger.warning(
                    "%s : only %d points left to calculate yearly metrices, skipping",
                    ip_df_dict["SiteID"],
                    gpp_obs_y_filtered.size,
                )
            else:
                # calculate the metrices
                try:
                    metric_func()
                # if a runtime warning is produced, store the warning in a list
                except Warning as e:
                    runtime_error_list.append(e)
                except Exception as exception:
                    runtime_error_list.append(exception)

    # log and print all the runtime warnings
    for er in runtime_error_list:
        logger.warning(
            (
                "%s occured for %s while calculating model performance metrices,"
                " Check the input forcing data and GPPobs and GPPsim timeseries"
            ),
            er,
            ip_df_dict["SiteID"],
        )
        warnings.warn(
            (
                f"{er} for {ip_df_dict['SiteID']} while calculating model performance metrices."
                " Check the input forcing data and GPPobs and GPPsim timeseries"
            )
        )

    return (
        model_perform_metric_dict,
        gpp_d_dict,
        gpp_w_dict,
        gpp_m_dict,
        gpp_y_dict,
        gpp_y_filtered_dict,
        good_gpp_d_idx,
        good_gpp_w_idx,
        good_gpp_m_idx,
        good_gpp_y_idx,
    )


def prep_results(ip_df_dict, model_op, settings_dict, xbest, p_names):
    """
    collect result data in a dictionary, resample data to different
    temporal resolution and calculate model performance metrices

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    model_op (dict): dictionary with PModel output with acclimation
    settings_dict (dict): dictionary with settings
    xbest (list or dict): array of scalars of optimized parameter values (in case of all year/
                           per PFT/ global optimization); dictionary with scalars of optimized
                           parameter values (in case of site year optimization)
    p_names (list): list of optimized parameter names

    returns:
    result_dict (dict): dictionary with all the results
    """

    if settings_dict["model_name"] == "P_model":
        # get the exact same subdaily gpp_obs and gpp_sim as in the optimization
        (
            gpp_obs,
            gpp_sim,
            _,
            et_obs,
            et_sim,
            _,
            drop_gpp_data_indices,
            drop_et_data_indices,
        ) = filter_data(
            model_op,
            ip_df_dict,
            settings_dict["data_filtering"],
            settings_dict["CO2_var"],
            et_var_name=settings_dict["et_var_name"],
        )

        (
            model_perform_metric_dict,
            gpp_d_dict,
            gpp_w_dict,
            gpp_m_dict,
            gpp_y_dict,
            gpp_y_filtered_dict,
            good_gpp_d_idx,
            good_gpp_w_idx,
            good_gpp_m_idx,
            good_gpp_y_idx,
        ) = calc_gpp_model_perform(
            ip_df_dict,
            model_op,
            gpp_obs,
            gpp_sim,
            drop_gpp_data_indices,
            "GPPp_opt_fW",
            # no_of_param,
        )

        (
            model_perform_metric_dict_no_moisture_stress,
            gpp_d_dict_no_moisture_stress,
            gpp_w_dict_no_moisture_stress,
            gpp_m_dict_no_moisture_stress,
            gpp_y_dict_no_moisture_stress,
            *_,
        ) = calc_gpp_model_perform(
            ip_df_dict,
            model_op,
            gpp_obs,
            model_op["GPPp_opt"][~drop_gpp_data_indices],
            drop_gpp_data_indices,
            "GPPp_opt",
            # no_of_param,
        )

    elif settings_dict["model_name"] == "LUE_model":
        # get the exact same subdaily gpp_obs and gpp_sim as in the optimization
        (
            gpp_obs,
            gpp_sim,
            _,
            et_obs,
            et_sim,
            _,
            drop_gpp_data_indices,
            drop_et_data_indices,
        ) = filter_lue_data(
            model_op,
            ip_df_dict,
            settings_dict["data_filtering"],
            settings_dict["CO2_var"],
            et_var_name=settings_dict["et_var_name"],
        )

        (
            model_perform_metric_dict,
            gpp_d_dict,
            gpp_w_dict,
            gpp_m_dict,
            gpp_y_dict,
            gpp_y_filtered_dict,
            good_gpp_d_idx,
            good_gpp_w_idx,
            good_gpp_m_idx,
            good_gpp_y_idx,
        ) = calc_gpp_model_perform(
            ip_df_dict,
            model_op,
            gpp_obs,
            gpp_sim,
            drop_gpp_data_indices,
            "gpp_lue",
            # no_of_param,
        )

    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
            "is not implemented"
        )

    # calculate model performance metrices to simulate et
    et_evaluator = RegressionMetric(et_obs, et_sim, decimal=5)

    et_nse = et_evaluator.NSE()
    et_cod = et_evaluator.COD()
    et_r2 = (et_evaluator.PCC()) ** 2
    et_kge = et_evaluator.KGE()
    et_rmse = et_evaluator.RMSE()

    param_val_dict = {}  # empty dictionary to store parameter values
    if isinstance(xbest, list):  # all year/ per PFT/ global optimization
        # collect actual optimized parameter values
        # (not xbest which is scalar of parameters) in a dictionary
        opti_param_vals = get_params(ip_df_dict, p_list=p_names, p_vec=xbest)
        for item in p_names:
            if item == "K_W":
                param_val_dict[item] = (
                    opti_param_vals[item] * -1.0
                )  # make K_W values negative
            elif item == "Kappa_VPD":
                param_val_dict[item] = (
                    opti_param_vals[item] * -1.0
                )  # make Kappa_VPD values negative
            else:
                param_val_dict[item] = opti_param_vals[item]
    else:  # site year optimization
        # for each site year in a site
        for year, param_scalar in xbest.items():
            year_param_val_dict = {}

            # if the site year could not be optimized, then all the parameter values are nan
            if np.isnan(param_scalar).any():
                for item in p_names:
                    year_param_val_dict[item] = np.nan

            # if the site year could be optimized
            else:
                # collect actual optimized parameter values
                # (not xbest which is scalar of parameters) in a dictionary
                opti_param_vals = get_params(
                    ip_df_dict, p_list=p_names, p_vec=param_scalar
                )

                for item in p_names:
                    if item == "K_W":  # make K_W values negative
                        year_param_val_dict[item] = opti_param_vals[item] * -1.0
                    elif item == "Kappa_VPD":
                        year_param_val_dict[item] = (
                            opti_param_vals[item] * -1.0
                        )  # make Kappa_VPD values negative
                    else:  # all other parameters
                        year_param_val_dict[item] = opti_param_vals[item]

            # store the parameter values for each year as a dictionary
            param_val_dict[year] = year_param_val_dict

    # collect everything in a dictionary
    # all the observed and simulated data are collected here without any filtering
    result_dict = {}

    result_dict["SiteID"] = ip_df_dict["SiteID"]
    result_dict["PFT"] = ip_df_dict["PFT"]
    result_dict["KG"] = ip_df_dict["KG"]
    result_dict["avg_temp"] = round(ip_df_dict["TA_GF"].mean(), 3)
    result_dict["avg_precip"] = round(ip_df_dict["P_GF"].mean(), 3)
    result_dict["arid_ind"] = round(
        ip_df_dict["P_GF"].mean() / ip_df_dict["PET"].mean(), 3
    )
    result_dict["exp_name"] = settings_dict["exp_name"]
    result_dict["Temp_res"] = ip_df_dict["Temp_res"]
    result_dict[f"Time_{ip_df_dict['Temp_res']}"] = ip_df_dict["Time"]
    result_dict["Time_daily"] = gpp_d_dict["Time"]
    result_dict["Time_weekly"] = gpp_w_dict["Time"]
    result_dict["Time_monthly"] = gpp_m_dict["Time"]
    result_dict["Time_yearly"] = gpp_y_dict["Time"]
    result_dict["Time_yearly_filtered"] = gpp_y_filtered_dict["Time"]
    result_dict[f"GPP_NT_{ip_df_dict['Temp_res']}"] = ip_df_dict["GPP_NT"]
    result_dict["GPP_NT_daily"] = gpp_d_dict["GPP_obs"]
    result_dict["GPP_NT_weekly"] = gpp_w_dict["GPP_obs"]
    result_dict["GPP_NT_monthly"] = gpp_m_dict["GPP_obs"]
    result_dict["GPP_NT_yearly"] = gpp_y_dict["GPP_obs"]
    result_dict["GPP_NT_yearly_filtered"] = gpp_y_filtered_dict["GPP_obs"]

    if settings_dict["model_name"] == "P_model":  # store P Model specific results
        result_dict[f"GPP_sim_{ip_df_dict['Temp_res']}"] = model_op["GPPp_opt_fW"]
        result_dict[f"GPP_sim_no_moisture_{ip_df_dict['Temp_res']}"] = model_op[
            "GPPp_opt"
        ]
        gpp_sim_d_no_moisture_stress = gpp_d_dict_no_moisture_stress["GPP_sim"]  # type: ignore
        gpp_sim_w_no_moisture_stress = gpp_w_dict_no_moisture_stress["GPP_sim"]  # type: ignore
        gpp_sim_m_no_moisture_stress = gpp_m_dict_no_moisture_stress["GPP_sim"]  # type: ignore
        gpp_sim_y_no_moisture_stress = gpp_y_dict_no_moisture_stress["GPP_sim"]  # type: ignore

        result_dict["GPP_sim_no_moisture_daily"] = gpp_sim_d_no_moisture_stress
        result_dict["GPP_sim_no_moisture_weekly"] = gpp_sim_w_no_moisture_stress
        result_dict["GPP_sim_no_moisture_monthly"] = gpp_sim_m_no_moisture_stress
        result_dict["GPP_sim_no_moisture_yearly"] = gpp_sim_y_no_moisture_stress

        result_dict["NSE_no_moisture_Stress"] = {
            f"NSE_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["nse_sd"],  # type: ignore
            "NSE_d": model_perform_metric_dict_no_moisture_stress["nse_d"],  # type: ignore
            "NSE_w": model_perform_metric_dict_no_moisture_stress["nse_w"],  # type: ignore
            "NSE_m": model_perform_metric_dict_no_moisture_stress["nse_m"],  # type: ignore
            "NSE_y": model_perform_metric_dict_no_moisture_stress["nse_y"],  # type: ignore
        }
        result_dict["COD_no_moisture_Stress"] = {
            f"COD_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["cod_sd"],  # type: ignore
            "COD_d": model_perform_metric_dict_no_moisture_stress["cod_d"],  # type: ignore
            "COD_w": model_perform_metric_dict_no_moisture_stress["cod_w"],  # type: ignore
            "COD_m": model_perform_metric_dict_no_moisture_stress["cod_m"],  # type: ignore
            "COD_y": model_perform_metric_dict_no_moisture_stress["cod_y"],  # type: ignore
        }
        result_dict["R2_no_moisture_Stress"] = {
            f"R2_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["r2_sd"],  # type: ignore
            "R2_d": model_perform_metric_dict_no_moisture_stress["r2_d"],  # type: ignore
            "R2_w": model_perform_metric_dict_no_moisture_stress["r2_w"],  # type: ignore
            "R2_m": model_perform_metric_dict_no_moisture_stress["r2_m"],  # type: ignore
            "R2_y": model_perform_metric_dict_no_moisture_stress["r2_y"],  # type: ignore
        }
        result_dict["KGE_no_moisture_Stress"] = {
            f"KGE_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["kge_sd"],  # type: ignore
            "KGE_d": model_perform_metric_dict_no_moisture_stress["kge_d"],  # type: ignore
            "KGE_w": model_perform_metric_dict_no_moisture_stress["kge_w"],  # type: ignore
            "KGE_m": model_perform_metric_dict_no_moisture_stress["kge_m"],  # type: ignore
            "KGE_y": model_perform_metric_dict_no_moisture_stress["kge_y"],  # type: ignore
        }
        result_dict["RMSE_no_moisture_Stress"] = {
            f"RMSE_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["rmse_sd"],  # type: ignore
            "RMSE_d": model_perform_metric_dict_no_moisture_stress["rmse_d"],  # type: ignore
            "RMSE_w": model_perform_metric_dict_no_moisture_stress["rmse_w"],  # type: ignore
            "RMSE_m": model_perform_metric_dict_no_moisture_stress["rmse_m"],  # type: ignore
            "RMSE_y": model_perform_metric_dict_no_moisture_stress["rmse_y"],  # type: ignore
        }
        result_dict["corr_coeff_no_moisture_Stress"] = {
            f"corr_coeff_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["corr_coeff_sd"],  # type: ignore
            "corr_coeff_d": model_perform_metric_dict_no_moisture_stress["corr_coeff_d"],  # type: ignore
            "corr_coeff_w": model_perform_metric_dict_no_moisture_stress["corr_coeff_w"],  # type: ignore
            "corr_coeff_m": model_perform_metric_dict_no_moisture_stress["corr_coeff_m"],  # type: ignore
            "corr_coeff_y": model_perform_metric_dict_no_moisture_stress["corr_coeff_y"],  # type: ignore
        }
        result_dict["variability_coeff_no_moisture_Stress"] = {
            f"variability_coeff_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["variability_coeff_sd"],  # type: ignore
            "variability_coeff_d": model_perform_metric_dict_no_moisture_stress["variability_coeff_d"],  # type: ignore
            "variability_coeff_w": model_perform_metric_dict_no_moisture_stress["variability_coeff_w"],  # type: ignore
            "variability_coeff_m": model_perform_metric_dict_no_moisture_stress["variability_coeff_m"],  # type: ignore
            "variability_coeff_y": model_perform_metric_dict_no_moisture_stress["variability_coeff_y"],  # type: ignore
        }
        result_dict["bias_coeff_no_moisture_Stress"] = {
            f"bias_coeff_{ip_df_dict['Temp_res']}": model_perform_metric_dict_no_moisture_stress["bias_coeff_sd"],  # type: ignore
            "bias_coeff_d": model_perform_metric_dict_no_moisture_stress["bias_coeff_d"],  # type: ignore
            "bias_coeff_w": model_perform_metric_dict_no_moisture_stress["bias_coeff_w"],  # type: ignore
            "bias_coeff_m": model_perform_metric_dict_no_moisture_stress["bias_coeff_m"],  # type: ignore
            "bias_coeff_y": model_perform_metric_dict_no_moisture_stress["bias_coeff_y"],  # type: ignore
        }
    elif settings_dict["model_name"] == "LUE_model":  # store LUE Model specific results
        result_dict[f"GPP_sim_{ip_df_dict['Temp_res']}"] = model_op["gpp_lue"]
        result_dict[f"fT_{ip_df_dict['Temp_res']}"] = model_op["fT"]
        result_dict[f"fVPD_{ip_df_dict['Temp_res']}"] = model_op["fVPD"]
        result_dict[f"fVPD_part_{ip_df_dict['Temp_res']}"] = model_op["fVPD_part"]
        result_dict[f"fCO2_part_{ip_df_dict['Temp_res']}"] = model_op["fCO2_part"]
        result_dict[f"fL_{ip_df_dict['Temp_res']}"] = model_op["fL"]
        result_dict[f"fCI_{ip_df_dict['Temp_res']}"] = model_op["fCI"]
        result_dict["ci"] = model_op["ci"]

    result_dict["GPP_sim_daily"] = gpp_d_dict["GPP_sim"]
    result_dict["GPP_sim_weekly"] = gpp_w_dict["GPP_sim"]
    result_dict["GPP_sim_monthly"] = gpp_m_dict["GPP_sim"]
    result_dict["GPP_sim_yearly"] = gpp_y_dict["GPP_sim"]
    result_dict["GPP_sim_yearly_filtered"] = gpp_y_filtered_dict["GPP_sim"]
    result_dict[f"GPP_drop_idx_{ip_df_dict['Temp_res']}"] = (
        drop_gpp_data_indices.astype(float)
    )
    result_dict["good_gpp_d_idx"] = good_gpp_d_idx.astype(float)
    result_dict["good_gpp_w_idx"] = good_gpp_w_idx.astype(float)
    result_dict["good_gpp_m_idx"] = good_gpp_m_idx.astype(float)
    result_dict["good_gpp_y_idx"] = good_gpp_y_idx.astype(float)
    result_dict[f"ET_{ip_df_dict['Temp_res']}"] = ip_df_dict["ET"]
    result_dict[f"ET_sim_{ip_df_dict['Temp_res']}"] = model_op["wai_results"]["et"]
    result_dict[f"ET_drop_idx_{ip_df_dict['Temp_res']}"] = drop_et_data_indices.astype(
        float
    )
    result_dict[f"WAI_{ip_df_dict['Temp_res']}"] = model_op["wai_results"]["wai"]
    result_dict[f"WAI_nor_{ip_df_dict['Temp_res']}"] = model_op["wai_results"][
        "wai_nor"
    ]
    result_dict[f"fW_Horn_{ip_df_dict['Temp_res']}"] = model_op["wai_results"]["fW"]
    result_dict["Par_num"] = len(p_names)
    result_dict["Opti_par_val"] = param_val_dict
    result_dict["NSE"] = {
        f"NSE_{ip_df_dict['Temp_res']}": model_perform_metric_dict["nse_sd"],
        "NSE_d": model_perform_metric_dict["nse_d"],
        "NSE_w": model_perform_metric_dict["nse_w"],
        "NSE_m": model_perform_metric_dict["nse_m"],
        "NSE_y": model_perform_metric_dict["nse_y"],
    }
    result_dict["COD"] = {
        f"COD_{ip_df_dict['Temp_res']}": model_perform_metric_dict["cod_sd"],
        "COD_d": model_perform_metric_dict["cod_d"],
        "COD_w": model_perform_metric_dict["cod_w"],
        "COD_m": model_perform_metric_dict["cod_m"],
        "COD_y": model_perform_metric_dict["cod_y"],
    }
    result_dict["R2"] = {
        f"R2_{ip_df_dict['Temp_res']}": model_perform_metric_dict["r2_sd"],
        "R2_d": model_perform_metric_dict["r2_d"],
        "R2_w": model_perform_metric_dict["r2_w"],
        "R2_m": model_perform_metric_dict["r2_m"],
        "R2_y": model_perform_metric_dict["r2_y"],
    }
    result_dict["KGE"] = {
        f"KGE_{ip_df_dict['Temp_res']}": model_perform_metric_dict["kge_sd"],
        "KGE_d": model_perform_metric_dict["kge_d"],
        "KGE_w": model_perform_metric_dict["kge_w"],
        "KGE_m": model_perform_metric_dict["kge_m"],
        "KGE_y": model_perform_metric_dict["kge_y"],
    }
    result_dict["RMSE"] = {
        f"RMSE_{ip_df_dict['Temp_res']}": model_perform_metric_dict["rmse_sd"],
        "RMSE_d": model_perform_metric_dict["rmse_d"],
        "RMSE_w": model_perform_metric_dict["rmse_w"],
        "RMSE_m": model_perform_metric_dict["rmse_m"],
        "RMSE_y": model_perform_metric_dict["rmse_y"],
    }
    # result_dict["AIC"] = {
    #     f"AIC_{ip_df_dict['Temp_res']}": model_perform_metric_dict["aic_sd"],
    #     "AIC_d": model_perform_metric_dict["aic_d"],
    #     "AIC_w": model_perform_metric_dict["aic_w"],
    #     "AIC_m": model_perform_metric_dict["aic_m"],
    #     "AIC_y": model_perform_metric_dict["aic_y"],
    # }
    result_dict["corr_coeff"] = {
        f"corr_coeff_{ip_df_dict['Temp_res']}": model_perform_metric_dict[
            "corr_coeff_sd"
        ],
        "corr_coeff_d": model_perform_metric_dict["corr_coeff_d"],
        "corr_coeff_w": model_perform_metric_dict["corr_coeff_w"],
        "corr_coeff_m": model_perform_metric_dict["corr_coeff_m"],
        "corr_coeff_y": model_perform_metric_dict["corr_coeff_y"],
    }
    result_dict["variability_coeff"] = {
        f"variability_coeff_{ip_df_dict['Temp_res']}": model_perform_metric_dict[
            "variability_coeff_sd"
        ],
        "variability_coeff_d": model_perform_metric_dict["variability_coeff_d"],
        "variability_coeff_w": model_perform_metric_dict["variability_coeff_w"],
        "variability_coeff_m": model_perform_metric_dict["variability_coeff_m"],
        "variability_coeff_y": model_perform_metric_dict["variability_coeff_y"],
    }
    result_dict["bias_coeff"] = {
        f"bias_coeff_{ip_df_dict['Temp_res']}": model_perform_metric_dict[
            "bias_coeff_sd"
        ],
        "bias_coeff_d": model_perform_metric_dict["bias_coeff_d"],
        "bias_coeff_w": model_perform_metric_dict["bias_coeff_w"],
        "bias_coeff_m": model_perform_metric_dict["bias_coeff_m"],
        "bias_coeff_y": model_perform_metric_dict["bias_coeff_y"],
    }
    result_dict["ET_model_performance"] = {
        f"ET_NSE_{ip_df_dict['Temp_res']}": et_nse,
        f"ET_COD_{ip_df_dict['Temp_res']}": et_cod,
        f"ET_R2_{ip_df_dict['Temp_res']}": et_r2,
        f"ET_KGE_{ip_df_dict['Temp_res']}": et_kge,
        f"ET_RMSE_{ip_df_dict['Temp_res']}": et_rmse,
    }

    return result_dict
