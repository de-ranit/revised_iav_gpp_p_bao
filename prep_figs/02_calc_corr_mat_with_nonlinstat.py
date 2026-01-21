#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calculate parameter correlation matrix using non-linear statistics

author: rde
first created: Wed Dec 03 2025 15:18:32 CET
"""
import sys
import os
import json
from pathlib import Path
from functools import partial
import re
import logging
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns

# import ipdb

# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_params import get_params
from src.common.get_data import get_data
from src.common.forward_run_model import scale_back_params
from src.common.forward_run_model import forward_run_model
from src.p_model.p_model_cost_function import filter_data
from src.lue_model.lue_model_cost_function import filter_lue_data


LEVEL_RE = re.compile(r"\]\s*(?P<level>INFO|WARNING|ERROR|CRITICAL)\b", re.IGNORECASE)


def split_log(src_path):
    src = Path(src_path)
    if not src.exists():
        raise SystemExit(f"Log file not found: {src}")

    out_info = src.with_name(src.stem + "_info.log")
    out_warn = src.with_name(src.stem + "_warning.log")
    out_err = src.with_name(src.stem + "_error.log")

    with src.open("r", encoding="utf-8", errors="replace") as fh, out_info.open(
        "w", encoding="utf-8"
    ) as finfo, out_warn.open("w", encoding="utf-8") as fwarn, out_err.open(
        "w", encoding="utf-8"
    ) as ferr:
        for line in fh:
            m = LEVEL_RE.search(line)
            if not m:
                # lines without explicit level -> send to info
                finfo.write(line)
                continue
            lvl = m.group("level").upper()
            if lvl == "INFO":
                finfo.write(line)
            elif lvl == "WARNING":
                fwarn.write(line)
            elif lvl in ("ERROR", "CRITICAL"):
                ferr.write(line)
            else:
                finfo.write(line)


def calculate_residual(
    xbest,
    ip_df_dict,
    ip_df_daily_wai,
    wai_output,
    time_info,
    settings_dict,
    opti_param_names,
):

    # forward run the model with given parameter values
    model_op, opti_param_names, _ = forward_run_model(
        ip_df_dict,
        ip_df_daily_wai,
        wai_output,
        time_info,
        settings_dict,
        xbest,
        opti_param_names=opti_param_names,
    )

    # calculate residuals
    if settings_dict["model_name"] == "P_model":
        # get the exact same subdaily gpp_obs and gpp_sim as in the optimization
        (
            gpp_obs,
            gpp_sim,
            _,
            et_obs,
            et_sim,
            _,
            _,
            _,
        ) = filter_data(
            model_op,
            ip_df_dict,
            settings_dict["data_filtering"],
            settings_dict["CO2_var"],
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
            _,
            _,
        ) = filter_lue_data(
            model_op,
            ip_df_dict,
            settings_dict["data_filtering"],
            settings_dict["CO2_var"],
        )

    # calculate residuals
    gpp_residuals = gpp_obs - gpp_sim
    et_residuals = et_obs - et_sim

    # calculate variances
    gpp_obs_variance = np.sum((gpp_obs - np.mean(gpp_obs)) ** 2.0)
    et_obs_variance = np.sum((et_obs - np.mean(et_obs)) ** 2.0)

    # calculate sse
    gpp_sse = np.dot(gpp_residuals, gpp_residuals)
    et_sse = np.dot(et_residuals, et_residuals)

    # tweak the residuals such that least_squares will give the same cost as in the optimization
    # which is (1 - NNSE) for GPP + (1 - NNSE) for ET
    gpp_residuals = np.sqrt(2.0) * gpp_residuals / np.sqrt(gpp_obs_variance + gpp_sse)
    et_residuals = np.sqrt(2.0) * et_residuals / np.sqrt(et_obs_variance + et_sse)

    return np.concatenate([gpp_residuals, et_residuals])


def cov_to_cor(cov_matrix):
    """
    Convert covariance matrix to correlation matrix

    parameters:
    cov_matrix (np.array): covariance matrix

    returns:
    corr_mat (np.array): correlation matrix
    """
    diag = np.diag(cov_matrix)
    corr_mat = cov_matrix / np.sqrt(np.outer(diag, diag))

    return corr_mat


def nonlinstats(jac, final_cost_val, site_name, logger):

    # approximate hessian matrix
    hessian = np.dot(jac.T, jac)

    try:  # inverting the hessian matrix
        hessian_inv = np.linalg.inv(hessian)

    except np.linalg.LinAlgError:  # if matrix is singular and non-invertible
        hessian_inv = np.linalg.pinv(hessian)
        hessian_inv = (hessian_inv + hessian_inv.T) / 2

        logger.warning(
            (
                "%s: Jacobian matrix product is singular and non-invertible."
                " Moore-Penrose pseudoinverse is applied."
            ),
            site_name,
        )

    # covariance matrix = inverse of hessian
    cov_matrix = hessian_inv

    # correlation matrix
    corr_mat = cov_to_cor(cov_matrix)

    # residual standard error
    s2 = final_cost_val / (jac.shape[0] - jac.shape[1])

    # standard error of parameters/ parameter uncertainty
    se = np.zeros(jac.shape[1])
    for i in range(len(se)):
        se[i] = np.sqrt(cov_matrix[i, i] * s2)

    # eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    return {
        "cov_matrix": cov_matrix,
        "corr_mat": corr_mat,
        "s2": s2,
        "se": se,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }


def plot_corr_matrix(p_names, corr_mat, site_id, dir_to_store):

    labels = p_names

    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_mat,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"shrink": 0.8},
        mask=mask,
    )
    ax.collections[0].cmap.set_bad("grey")

    ax.set_title(f"Parameter correlation — {site_id}", fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    dir_to_save_plots = os.path.join(
        f"./jac_nonlinstats/{dir_to_store}", "corr_matrix_per_site_plots"
    )
    os.makedirs(dir_to_save_plots, exist_ok=True)
    out_fname = f"{dir_to_save_plots}/{site_id}_param_correlation.png"
    plt.savefig(out_fname, dpi=300, bbox_inches="tight")

    plt.close(fig)


def calc_nonlinstat(site_idx, site_list, exp_names_dict, new_exp_name):

    site_name = site_list[site_idx]
    print(f"Calculating non-linear statistics for site: {site_name}")

    os.makedirs(os.path.join("./jac_nonlinstats", new_exp_name), exist_ok=True)

    # configure the logger: to log various information to a file
    logging.basicConfig(
        filename=(f"./jac_nonlinstats/{new_exp_name}/" "calc_jac.log"),
        filemode="a",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # first choose optimization with lowest cost
    cmaes_big_pop_opti_dict_path = f"../../02_optim_PModel/opti_results/{exp_names_dict['cmaes_big_pop']}/opti_dicts/{site_name}_opti_dict.json"
    with open(cmaes_big_pop_opti_dict_path, "r") as f:
        cmaes_big_pop_opti_dict = json.load(f)

    cmaes_big_pop_and_lbfgs_b_path = f"../opti_lbfgs_b/opti_lbfgs/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/lbgfgs_b_dicts/{site_name}_None_lbfgs_b_results.npy"
    cmaes_big_pop_and_lbfgs_b_opti_dict = np.load(
        cmaes_big_pop_and_lbfgs_b_path, allow_pickle=True
    ).item()

    cmaes_default_pop_opti_dict_path = f"../opti_results/{exp_names_dict['cmaes_default_pop']}/opti_dicts/{site_name}_opti_dict.json"
    with open(cmaes_default_pop_opti_dict_path, "r") as f:
        cmaes_default_pop_opti_dict = json.load(f)

    cost_arr = np.array(
        [
            cmaes_big_pop_opti_dict["fbest"],
            cmaes_big_pop_and_lbfgs_b_opti_dict["fun"],
            cmaes_default_pop_opti_dict["fbest"],
        ]
    )
    
    idx = int(np.argmin(cost_arr))

    # get xbest depending on which optimization had lowest cost
    if idx == 0:
        xbest = cmaes_big_pop_opti_dict["xbest"]

    elif idx == 1:
        site_kg = site_info_df.loc[site_info_df["SiteID"] == site_name, "KG"].values[0]
        site_elev = site_info_df.loc[
            site_info_df["SiteID"] == site_name, "elev"
        ].values[0]
        params = get_params({"KG": site_kg, "elev": site_elev})

        p_names = cmaes_default_pop_opti_dict["opti_param_names"]

        p_ubound_scaled = []
        p_lbound_scaled = []
        for p in p_names:
            p_ubound_scaled.append(params[p]["ub"] / params[p]["ini"])
            p_lbound_scaled.append(params[p]["lb"] / params[p]["ini"])

        multipliers = np.array(
            [ub - lb for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)]
        )
        zero = np.array(
            [-lb / (ub - lb) for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)]
        )
        xbest = list((cmaes_big_pop_and_lbfgs_b_opti_dict["x"] / multipliers) + zero)
        
    elif idx == 2:
        xbest = cmaes_default_pop_opti_dict["xbest"]

    # open settings dict
    settings_dict_file_path = (
        f"../opti_results/{exp_names_dict['cmaes_default_pop']}/settings_dict.json"
    )
    with open(settings_dict_file_path, "r") as fh:
        settings_dict = json.load(fh)  # load settings dict

    # get forcing data for the site
    ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
        site_name, settings_dict
    )

    params = get_params(ip_df_dict)
    p_names = cmaes_default_pop_opti_dict["opti_param_names"]

    # scale back xbest to actual parameter values
    xbest_actual = scale_back_params(xbest, p_names, params)
    settings_dict["scale_coord"] = False

    # get the scaled value of parameter bounds (ub/initial or lb/initial)
    p_ubound_scaled = []
    p_lbound_scaled = []
    for p in p_names:
        p_ubound_scaled.append(params[p]["ub"] / params[p]["ini"])
        p_lbound_scaled.append(params[p]["lb"] / params[p]["ini"])

    # make partial function of calculate_residual with fixed arguments
    residual_hand = partial(
        calculate_residual,
        ip_df_dict=ip_df_dict,
        ip_df_daily_wai=ip_df_daily_wai,
        wai_output=wai_output,
        time_info=time_info,
        settings_dict=settings_dict,
        opti_param_names=p_names,
    )

    lsq_loss_ini = ((np.array(residual_hand(xbest_actual)) ** 2.0)).sum() * 0.5

    # if (site_name == "US-Myb") or (site_name == "US-Me1"):
    #     xbest_actual[11] = round(xbest_actual[11], 5)
    
    result_lsq = least_squares(
        fun=residual_hand,
        x0=xbest_actual,
        jac="3-point",
        bounds=(p_lbound_scaled, p_ubound_scaled),
        method="trf",
        ftol=1e-8,
        max_nfev=100,
        verbose=2,
    )

    percentage_difference = (lsq_loss_ini - result_lsq.cost) / lsq_loss_ini * 100

    logger.info(
        (
            "%s: Initial loss: %.4f, Final loss: %.4f,"
            " Percentage difference: %.4f%%, nfev: %d, stop: %s",
        ),
        ip_df_dict["SiteID"],
        lsq_loss_ini,
        result_lsq.cost,
        percentage_difference,
        result_lsq.nfev,
        result_lsq.message,
    )

    nonlin_stats_op_dict = nonlinstats(
        result_lsq["jac"], result_lsq.cost, site_name, logger
    )
    nonlin_stats_op_dict["jacobian"] = result_lsq["jac"]
    nonlin_stats_op_dict["grad"] = result_lsq["grad"]

    # save the jacobian and non-linear statistics results
    dir_to_save_results = os.path.join(
        f"./jac_nonlinstats/{new_exp_name}", "nonlinstats_op_dicts"
    )
    os.makedirs(dir_to_save_results, exist_ok=True)
    np.save(
        f"{dir_to_save_results}/{ip_df_dict['SiteID']}_nonlinstats.npy",
        nonlin_stats_op_dict,
    )

    plot_corr_matrix(p_names, nonlin_stats_op_dict["corr_mat"], site_name, new_exp_name)

    # split the log file into info, warning, and error logs
    split_log(f"./jac_nonlinstats/{new_exp_name}/calc_jac.log")


if __name__ == "__main__":

    per_site_lue_exp_names = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_default_pop_cmaes",
    }

    site_info_df = pd.read_csv("../site_info/SiteInfo_BRKsite_list.csv")
    site_list = site_info_df["SiteID"].tolist()
    site_list = [
        x
        for x in site_list
        if x != "CG-Tch" and x != "MY-PSO" and x != "GH-Ank" and x != "US-LWW"
    ]

    site_idx = int(sys.argv[1]) - 1
    
    calc_nonlinstat(
        site_idx,
        site_list,
        per_site_lue_exp_names,
        new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever",
    )
