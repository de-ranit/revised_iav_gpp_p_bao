#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Further optimize model parameters using L-BFGS-B algorithm
starting from the best solution obtained from CMA-ES optimization with big
population size (for site year or per site optimization) The same for global
and per PFT optimization are integrated with CMA-ES based optimization.

author: rde
first created: Thu Oct 30 2025 16:59:44 CET
"""
# disable possible multithreading from the
# OPENBLAS and MKL linear algebra backends
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import glob
from pathlib import Path
import re
import importlib
import json
import logging
import logging.config
from functools import partial
import numpy as np
from optimparallel import minimize_parallel
from scipy.optimize import Bounds

# import ipdb

# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_data import get_data
from src.p_model.pmodel_plus import pmodel_plus
from src.common.get_params import get_params
from src.common.forward_run_model import scale_back_params
from src.p_model.p_model_cost_function import p_model_cost_function
from src.lue_model.lue_model_cost_function import lue_model_cost_function

LEVEL_RE = re.compile(r"\]\s*(?P<level>INFO|WARNING|ERROR|CRITICAL)\b", re.IGNORECASE)


def split_log(src_path: Path):
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


def opti_lbfgs(opti_res_path, site_yr_idx, et_var_name="ET"):

    # directory to store jacobian and non-linear statistics results
    dir_to_store = str(opti_res_path.parent.relative_to(Path("..") / "opti_results"))
    os.makedirs(os.path.join("./opti_lbfgs", dir_to_store), exist_ok=True)

    # configure the logger: to log various information to a file
    logging.basicConfig(
        filename=(f"./opti_lbfgs/{dir_to_store}/" "opti_lbfgs.log"),
        filemode="a",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # get the list of result files
    opti_res_file_list = glob.glob(f"{opti_res_path}/*.json")
    opti_res_file_list.sort()  # sort the files by site ID

    # get name of the result file for the site
    opti_res_file = opti_res_file_list[site_yr_idx]

    with open(
        opti_res_file, "r", encoding="utf-8"
    ) as opti_dict_json_file:  # read the optimization results
        xbest_dict = json.load(opti_dict_json_file)

    site_id = Path(opti_res_file).stem.split("_", 1)[0]
    site_yr = xbest_dict["site_year"]

    if site_yr is not None:
        print(f"optimizing for site year: {site_id}_{site_yr}")
    else:
        print(f"optimizing for site: {site_id}")

    if (xbest_dict["xbest"] is None) or (np.isnan(np.array(xbest_dict["xbest"])).any()):
        logger.error(
            "No CMAES optimization results found for site or site year: %s_%s",
            site_id,
            site_yr,
        )
        split_log(f"./opti_lbfgs/{dir_to_store}/opti_lbfgs.log")

        lbfgs_b_op_dict = {"x": None, "fun": None, "site_year": site_yr}
        # save the jacobian and non-linear statistics results
        dir_to_save_results = os.path.join(
            f"./opti_lbfgs/{dir_to_store}", "lbgfgs_b_dicts"
        )
        os.makedirs(dir_to_save_results, exist_ok=True)
        if site_yr is not None:
            np.save(
                f"{dir_to_save_results}/{site_id}_{site_yr}_lbfgs_b_results.npy",
                lbfgs_b_op_dict,
            )
        else:
            np.save(
                f"{dir_to_save_results}/{site_id}_lbfgs_b_results.npy",
                lbfgs_b_op_dict,
            )

        sys.exit("No CMAES optimization results found, exiting...")
    else:
        pass
        # get experiment name to open the settings dict
        exp_name = str(
            Path(opti_res_file).parent.parent.relative_to(Path("..") / "opti_results")
        )
        settings_dict_file_path = f"../opti_results/{exp_name}/settings_dict.json"

        with open(settings_dict_file_path, "r") as fh:
            settings_dict = json.load(fh)  # load settings dict

        # set the et variable name if not existing in settings dict
        if "et_var_name" not in settings_dict.keys():
            settings_dict["et_var_name"] = et_var_name

        # get forcing data for the site
        ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
            site_id, settings_dict
        )

        # get the parameters
        # parameter bounds are needed to calculate scalar of bounds for optimization
        params = get_params(ip_df_dict)

        xbest = xbest_dict["xbest"]
        p_names = xbest_dict["opti_param_names"]
        xbest_actual = scale_back_params(xbest, p_names, params)
        settings_dict["scale_coord"] = False

        # get the scaled value of parameter bounds (ub/initial or lb/initial)
        p_ubound_scaled = []
        p_lbound_scaled = []
        for p in p_names:
            p_ubound_scaled.append(params[p]["ub"] / params[p]["ini"])
            p_lbound_scaled.append(params[p]["lb"] / params[p]["ini"])

        ##############
        # Define costhand and parameters in case of P Model
        if settings_dict["model_name"] == "P_model":
            # run pmodelPlus without acclimation
            model_op_no_acclim_sd = pmodel_plus(
                ip_df_dict, params, settings_dict["CO2_var"]
            )

            # define the cost function to be optimized as a partial
            # function of p_model_cost_function
            if site_yr is None:  # in case of all year optimization
                costhand = partial(
                    p_model_cost_function,
                    p_names=p_names,
                    ip_df_dict=ip_df_dict,
                    model_op_no_acclim_sd=model_op_no_acclim_sd,
                    ip_df_daily_wai=ip_df_daily_wai,
                    wai_output=wai_output,
                    time_info=time_info,
                    fpar_var_name=settings_dict["fPAR_var"],
                    co2_var_name=settings_dict["CO2_var"],
                    data_filtering=settings_dict["data_filtering"],
                    cost_func=settings_dict["cost_func"],
                    consider_yearly_cost=settings_dict["cost_iav"],
                    et_var_name=settings_dict["et_var_name"],
                )
            else:  # in case of site year optimization
                costhand = partial(
                    p_model_cost_function,
                    p_names=p_names,
                    ip_df_dict=ip_df_dict,
                    model_op_no_acclim_sd=model_op_no_acclim_sd,
                    ip_df_daily_wai=ip_df_daily_wai,
                    wai_output=wai_output,
                    time_info=time_info,
                    fpar_var_name=settings_dict["fPAR_var"],
                    co2_var_name=settings_dict["CO2_var"],
                    data_filtering=settings_dict["data_filtering"],
                    cost_func=settings_dict["cost_func"],
                    site_year=site_yr,
                    et_var_name=settings_dict["et_var_name"],
                )

        # Define costhand and parameters in case of LUE Model
        elif settings_dict["model_name"] == "LUE_model":
            # generate synthetic data to calculate 3rd and 4th
            # component of LUE model cost function
            synthetic_data = {
                "TA": np.linspace(-5.0, 40.0, time_info["nstepsday"] * 365),  # deg C
                "VPD": np.linspace(4500, 0.0, time_info["nstepsday"] * 365),  # Pa
                "CO2": np.linspace(400.0, 400.0, time_info["nstepsday"] * 365),  # PPM
                "wai_nor": np.linspace(
                    0.0, 1.0, time_info["nstepsday"] * 365
                ),  # - (or mm/mm)
                "fPAR": np.linspace(0.0, 1.0, time_info["nstepsday"] * 365),  # -
                "PPFD": np.linspace(
                    0.0, 600.0, time_info["nstepsday"] * 365
                ),  # umol photons m-2s-1
            }

            if site_yr is None:  # in case of all year optimization
                costhand = partial(
                    lue_model_cost_function,
                    param_names=p_names,
                    ip_df_dict=ip_df_dict,
                    ip_df_daily_wai=ip_df_daily_wai,
                    wai_output=wai_output,
                    nstepsday=time_info["nstepsday"],
                    fpar_var_name=settings_dict["fPAR_var"],
                    co2_var_name=settings_dict["CO2_var"],
                    data_filtering=settings_dict["data_filtering"],
                    cost_func=settings_dict["cost_func"],
                    synthetic_data=synthetic_data,
                    consider_yearly_cost=settings_dict["cost_iav"],
                    et_var_name=settings_dict["et_var_name"],
                )
            else:  # in case of site year optimization
                costhand = partial(
                    lue_model_cost_function,
                    param_names=p_names,
                    ip_df_dict=ip_df_dict,
                    ip_df_daily_wai=ip_df_daily_wai,
                    wai_output=wai_output,
                    nstepsday=time_info["nstepsday"],
                    fpar_var_name=settings_dict["fPAR_var"],
                    co2_var_name=settings_dict["CO2_var"],
                    data_filtering=settings_dict["data_filtering"],
                    cost_func=settings_dict["cost_func"],
                    synthetic_data=synthetic_data,
                    site_year=site_yr,
                    et_var_name=settings_dict["et_var_name"],
                )

        ini_costval = costhand(xbest_actual)

        bounds = Bounds(np.array(p_lbound_scaled), np.array(p_ubound_scaled))

        lbfgs_b_op = minimize_parallel(
            fun=costhand,
            x0=xbest_actual,
            bounds=bounds,
            options={"maxfun": 20000, "ftol": 1e-8},  # 20000
            parallel={
                "loginfo": True,
                "time": True,
                "verbose": False,
                "max_workers": int(sys.argv[2]),
            },
        )

        percentage_reduction_in_cost = (
            (ini_costval - lbfgs_b_op.fun) / ini_costval * 100.0
        )

        lbfgs_b_op_dict = {
            "initial_cost": ini_costval,
            "fun": lbfgs_b_op.fun,
            "percentage_reduction_in_cost": percentage_reduction_in_cost,
            "jac": lbfgs_b_op.jac,
            "nfev": lbfgs_b_op.nfev,
            "njev": lbfgs_b_op.njev,
            "nit": lbfgs_b_op.nit,
            "status": lbfgs_b_op.status,
            "message": lbfgs_b_op.message,
            "x": lbfgs_b_op.x,
            "success": lbfgs_b_op.success,
            "loginfo": lbfgs_b_op.loginfo,
            "time": lbfgs_b_op.time,
        }

        logger.info(
            (
                "%s: "
                "Starting cost value: %.4f, Final cost value: %.4f, Percentage cost reduction: %4f, nfev: %d, njev: %d, nit: %d, status: %d, "
                "message: %s, success: %s, time elapsed: %.2f seconds"
            ),
            f"{ip_df_dict['SiteID']} - {site_yr}",
            ini_costval,
            lbfgs_b_op.fun,
            percentage_reduction_in_cost,
            lbfgs_b_op.nfev,
            lbfgs_b_op.njev,
            lbfgs_b_op.nit,
            lbfgs_b_op.status,
            lbfgs_b_op.message,
            str(lbfgs_b_op.success),
            lbfgs_b_op.time["elapsed"],
        )

        split_log(f"./opti_lbfgs/{dir_to_store}/opti_lbfgs.log")

        # save the jacobian and non-linear statistics results
        dir_to_save_results = os.path.join(
            f"./opti_lbfgs/{dir_to_store}", "lbgfgs_b_dicts"
        )
        os.makedirs(dir_to_save_results, exist_ok=True)
        if site_yr is not None:
            np.save(
                f"{dir_to_save_results}/{ip_df_dict['SiteID']}_{site_yr}_lbfgs_b_results.npy",
                lbfgs_b_op_dict,
            )
        else:
            np.save(
                f"{dir_to_save_results}/{ip_df_dict['SiteID']}_lbfgs_b_results.npy",
                lbfgs_b_op_dict,
            )


if __name__ == "__main__":

    result_paths = importlib.import_module("opti_result_path_coll")

    opti_res_path_coll = {
        "per_site_yr_p_model_res_path": result_paths.per_site_yr_p_model_res_path,
        "per_site_p_model_res_path": result_paths.per_site_p_model_res_path,
        "per_site_p_model_res_path_iav": result_paths.per_site_p_model_res_path_iav,
        "per_site_yr_lue_model_res_path": result_paths.per_site_yr_lue_model_res_path,
        "per_site_lue_model_res_path": result_paths.per_site_lue_model_res_path,
        "per_site_lue_model_res_path_iav": result_paths.per_site_lue_model_res_path_iav,
    }

    #### parallel
    # run as python -u opti_lbfgs_from_cmaes_syr_or_site.py <site_yr_idx> <n_workers> <opti_res_path_key>
    opti_res_path = opti_res_path_coll[sys.argv[3]]
    get_site_yr_idx = int(sys.argv[1]) - 1
    opti_lbfgs(opti_res_path, site_yr_idx=get_site_yr_idx, et_var_name="ET")

    #### non parallel (run for a specific site or site year)
    # for get_site_yr_idx in range(0, 1438):
    #     opti_lbfgs(opti_res_path, site_yr_idx=get_site_yr_idx, et_var_name="ET")
