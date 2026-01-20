#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
forward run per site optimization/ per PFT optimization/ global optimization 
with calibrated parameters obtained from L-BFGS-B algorithm
starting from the best solution obtained from CMA-ES optimization

author: rde
first created: Thu Nov 13 2025 14:06:38 CET
"""

import os
import sys
import glob
from pathlib import Path
from multiprocessing.pool import Pool
from functools import partial
import re
import importlib
import json
import logging
import logging.config
import numpy as np
import pandas as pd


import ipdb

# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_data import get_data
from src.common.forward_run_model import forward_run_model
from src.common.forward_run_model import save_n_plot_model_results


def calc_idx_for_jobarray(job_array_no, ini_idx, n_sites, max_n):
    """
    calculate the start and end index of site list for each job array

    Parameters:
    job_array_no (int): job array index == $SLURM_ARRAY_TASK_ID
    ini_idx (int): initial index of site list (filename_list)
    n_sites (int): no. of sites per job
    max_n (int): total no. of sites

    Returns:
    start_site_idx (int): start index of site list for each job array
    end_site_idx (int): end index of site list for each job array
    """

    if job_array_no == 1:  # for first job array
        start_site_idx = ini_idx
        end_site_idx = ini_idx + n_sites
    else:  # for other job arrays
        start_site_idx = ((job_array_no - 1) * n_sites) + ini_idx
        end_site_idx = (job_array_no * n_sites) + ini_idx
        if end_site_idx > max_n:
            end_site_idx = max_n
        else:
            pass
    return start_site_idx, end_site_idx


def forward_run_lbfgs(site_id_idx, site_id_list, opti_res_path, et_var_name="ET"):

    site_name = site_id_list[site_id_idx]

    dir_to_store = str(opti_res_path.parent.relative_to(Path("..") / "opti_results"))
    os.makedirs(os.path.join("./model_results", dir_to_store), exist_ok=True)

    # configure the logger: to log various information to a file
    logging.basicConfig(
        filename=(f"./model_results/{dir_to_store}/" "for_run_opti_lbfgs.log"),
        filemode="a",
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d,%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    settings_dict_file_path = f"../opti_results/{dir_to_store}/settings_dict.json"

    with open(settings_dict_file_path, "r") as fh:
        settings_dict = json.load(fh)  # load settings dict

    # set the et variable name if not existing in settings dict
    if "et_var_name" not in settings_dict.keys():
        settings_dict["et_var_name"] = et_var_name

    # always false for l-bfgs-b forward run
    settings_dict["scale_coord"] = False

    if settings_dict["opti_type"] == "all_year":
        # forward run the model with optimized parameters for all years of each site
        opti_dict_path = Path(
            "opti_lbfgs",
            dir_to_store,
            "lbgfgs_b_dicts",
        )
        opti_dict_path_filename = os.path.join(
            # opti_dict_path, f"{site_name}_None_lbfgs_b_results.npy"
            opti_dict_path,
            f"{site_name}_lbfgs_b_results.npy",
        )  # filename where optimization results are saved
        try:

            opti_dict = np.load(opti_dict_path_filename, allow_pickle=True).item()

            xbest = opti_dict["x"]  # get the optimized parameters for the site
            xbest = xbest.tolist()

            if (xbest is None) or (
                np.isnan(np.array(xbest)).any()
            ):  # if the optimization was not successful, skip the site
                logger.warning(
                    "%s : optimization was not successful (skipping forward run)",
                    site_name,
                )
            else:
                # get the forcing data and other information for the site to forward run p-model
                ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
                    site_name, settings_dict
                )

                # forward run model with optimized parameters
                model_op, p_names, xbest_actual = forward_run_model(
                    ip_df_dict,
                    ip_df_daily_wai,
                    wai_output,
                    time_info,
                    settings_dict,
                    xbest,
                )

                # save and plot model results
                save_n_plot_model_results(
                    ip_df_dict, model_op, settings_dict, xbest_actual, p_names
                )

        # if the optimization results file does not exist, skip the site
        except FileNotFoundError:
            logger.warning(
                "%s : optimization was not successful (skipping forward run)", site_name
            )
    # forward run the model with optimized parameters per PFT
    elif settings_dict["opti_type"] == "per_pft":
        # get the forcing data and other information for the site to forward run p-model
        ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
            site_name, settings_dict
        )

        site_pft = ip_df_dict["PFT"]

        opti_dict_path = Path(
            "opti_lbfgs",
            dir_to_store,
            "opti_dicts",
        )
        opti_dict_path_filename = os.path.join(
            opti_dict_path, f"{site_pft}_opti_dict.npy"
        )  # filename where optimization results are saved

        opti_dict = np.load(opti_dict_path_filename, allow_pickle=True).item()

        xbest = opti_dict["x"]  # get the optimized parameters for the site
        xbest = xbest.tolist()

        cmeas_opti_filename = (
            f"../opti_results/{dir_to_store}/opti_dicts/{site_pft}_opti_dict.json"
        )
        with open(cmeas_opti_filename, "r") as fh:
            cmeas_opti_dict = json.load(fh)  # load cmeas opti dict
        opti_dict["opti_param_names"] = cmeas_opti_dict["opti_param_names"]

        # if (site_pft == "SAV") and (settings_dict["model_name"] == "LUE_model"):
        #     xbest = cmeas_opti_dict["xbest"]
        #     settings_dict["scale_coord"] = True

        # forward run p-model with optimized parameters
        model_op, p_names, xbest_actual = forward_run_model(
            ip_df_dict,
            ip_df_daily_wai,
            wai_output,
            time_info,
            settings_dict,
            xbest,
            opti_dict["opti_param_names"],
        )

        # save and plot model results
        save_n_plot_model_results(
            ip_df_dict, model_op, settings_dict, xbest_actual, p_names
        )

    # forward run the model with parameters optimized using data of all sites (global optimization)
    elif settings_dict["opti_type"] == "global_opti":
        # get the forcing data and other information for the site to forward run p-model
        ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
            site_name, settings_dict
        )

        opti_dict_path = Path(
            "opti_lbfgs",
            dir_to_store,
        )
        opti_dict_path_filename = os.path.join(
            opti_dict_path, "global_opti_opti_dict.npy"
        )  # filename where optimization results are saved

        opti_dict = np.load(opti_dict_path_filename, allow_pickle=True).item()

        xbest = opti_dict["x"]  # get the optimized parameters for the site
        xbest = xbest.tolist()

        if settings_dict["model_name"] == "P_model":
            opti_dict["opti_param_names"] = [
                "acclim_window",
                "W_I",
                "K_W",
                "alpha",
                "AWC",
                "theta",
                "alphaPT",
                "meltRate_temp",
                "meltRate_netrad",
                "sn_a",
            ]
        elif settings_dict["model_name"] == "LUE_model":
            opti_dict["opti_param_names"] = [
                "LUE_max",
                "T_opt",
                "K_T",
                "alpha_fT_Horn",
                "Kappa_VPD",
                "Ca_0",
                "C_Kappa",
                "c_m",
                "gamma_fL_TAL",
                "mu_fCI",
                "W_I",
                "K_W",
                "alpha",
                "AWC",
                "theta",
                "alphaPT",
                "meltRate_temp",
                "meltRate_netrad",
                "sn_a",
            ]

        # forward run p-model with optimized parameters
        model_op, p_names, xbest_actual = forward_run_model(
            ip_df_dict,
            ip_df_daily_wai,
            wai_output,
            time_info,
            settings_dict,
            xbest,
            opti_dict["opti_param_names"],
        )

        # save and plot model results
        save_n_plot_model_results(
            ip_df_dict, model_op, settings_dict, xbest_actual, p_names
        )


if __name__ == "__main__":

    result_paths = importlib.import_module("opti_result_path_coll")

    opti_res_path_coll = {
        "per_site_p_model_res_path": result_paths.per_site_p_model_res_path,
        "per_site_p_model_res_path_iav": result_paths.per_site_p_model_res_path_iav,
        "per_site_lue_model_res_path": result_paths.per_site_lue_model_res_path,
        "per_site_lue_model_res_path_iav": result_paths.per_site_lue_model_res_path_iav,
    }

    opti_res_path = opti_res_path_coll[
        sys.argv[3]
    ]  # get the experiment name from command line argument

    site_list = pd.read_csv("../site_info/SiteInfo_BRKsite_list.csv", low_memory=False)[
        "SiteID"
    ].tolist()

    # # ########## not parallel version ##############
    # for ix in range(len(site_list)):
    #     forward_run_lbfgs(ix, site_list, opti_res_path, et_var_name="ET")
    # # # #############################################

    j_array_no = int(sys.argv[1])  # job array index == $SLURM_ARRAY_TASK_ID
    ini_idx = 0  # initial index of site list (filename_list)
    n_sites = int(sys.argv[2])  # no. of sites per job
    max_n = len(site_list)  # total no. of sites 202 in BRK15

    # calculate the start and end index of site list for each job array
    start_site_idx, end_site_idx = calc_idx_for_jobarray(
        j_array_no, ini_idx, n_sites, max_n
    )

    ################################################################################
    # use partial to make the function take only one argument
    forward_run_syr_lbfgs_partial = partial(
        forward_run_lbfgs,
        site_id_list=site_list,
        opti_res_path=opti_res_path,
        et_var_name="ET",
    )

    # forward run each site/ site year in parallel
    # create a process pool with many workers
    with Pool(n_sites) as pool_obj:
        pool_obj.map(
            forward_run_syr_lbfgs_partial,
            range(start_site_idx, end_site_idx, 1),
        )
        pool_obj.close()
