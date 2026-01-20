#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
forward run per site year optimization with calibrated parameters obtained from L-BFGS-B algorithm
starting from the best solution obtained from CMA-ES optimization

author: rde
first created: Mon Nov 10 2025 15:18:27 CET
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



# import ipdb

# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_data import get_data
from src.common.forward_run_model import forward_run_model
from src.common.forward_run_model import save_n_plot_model_results
from src.postprocess.plot_param_site_year import plot_opti_param_site_year


def filter_logs(ip_log_filename, clean_log_filename, msg_start, msg_end=None):
    """
    multiple log lines are written for each site (from each iteration of the otimizer),
    this function removes the duplicate lines and keeps only the last log per site

    Parameters:
    ip_log_filename (str): input log filename
    clean_log_filename (str): output log filename
    msg_start (int): index at which the actual log message starts (after timestamp and log level)
    msg_end (int): index at which the actual log message ends

    Returns:
    Create and save a .log file with unique log lines
    """
    seen_logs = set()  # create an empty set to store the unique log lines

    # read the original logfile
    with open(ip_log_filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # write the unique log lines to a new file
    with open(clean_log_filename, "w", encoding="utf-8") as f:
        for line in reversed(lines):
            subset_line = line[
                msg_start:msg_end
            ]  # get only the log message (excluding the timestamp and log level)
            if (
                subset_line not in seen_logs
            ):  # if the log message is not already in the set,
                # write it to the new file and add it to the set
                seen_logs.add(subset_line)
                f.write(line)



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


def forward_run_syr_lbfgs(site_id_idx, site_id_list, opti_res_path, et_var_name="ET"):

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

    # get forcing data for the site
    ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
        site_name, settings_dict
    )

    if settings_dict["model_name"] == "P_model":
        # list of variables in P-Model output
        model_op_var = [
            "kcPa_opt",
            "koPa_opt",
            "kmPa_opt",
            "GammaStarM_opt",
            "viscosityH2oStar_opt",
            "xiPaM_opt",
            "Ca_opt",
            "ciM_opt",
            "phi0_opt",
            "vcmaxPmodelM1_opt",
            "Ac1_opt",
            "JmaxM1_opt",
            "Jp1_opt",
            "AJp1_opt",
            "GPPp_opt",
            "TA_GF_opt",
            "Iabs_opt",
            "phi0",
            "TA_OptK",
            "TA_K",
            "GammaStarM",
            "kmPa",
            "Iabs",
            "xiPa",
            "ci",
            "vcmaxOpt",
            "JmaxOpt",
            "vcmaxAdjusted",
            "JmaxAdjusted",
            "Ac1Opt",
            "J",
            "AJp1Opt",
            "GPPp_opt_fW",
        ]
    elif settings_dict["model_name"] == "LUE_model":
        # list of variables in LUE Model output
        model_op_var = [
            "gpp_lue",
            "fT",
            "fVPD",
            "fVPD_part",
            "fCO2_part",
            "fW",
            "fL",
            "fCI",
            "ci",
            "wai_results",
        ]
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
            "is not implemented"
        )

    # initialize dictionaries to store model outputs
    model_op = {}
    for var in model_op_var:
        model_op[var] = np.zeros(ip_df_dict["Time"].size)

    # list of variables in WAI output
    wai_var = [
        "snomelt",
        "etsub",
        "pu",
        "sno",
        "wai",
        "wai_nor",
        "et",
        "etsno",
        "etsub_pot",
        "snofall",
        "pl",
        "fW",
    ]

    # initialize dictionaries to store WAI outputs
    wai_op = {}
    for var in wai_var:
        wai_op[var] = np.zeros(ip_df_dict["Time"].size)

    # add WAI output to model output dictionary
    model_op["wai_results"] = wai_op

    # initialize dictionary to store optimized parameters
    x_best_dict = {}

    # path to optimization results
    opti_dict_path = Path(
        "opti_lbfgs",
        dir_to_store,
        "lbgfgs_b_dicts",
    )
    opti_dict_path_filename = os.path.join(
        opti_dict_path, f"{site_name}_*_lbfgs_b_results.npy"
    )  # filename where optimization results are saved

    # get the list of optimization results (one file per year)
    opti_dict_filename_list = glob.glob(opti_dict_path_filename)
    opti_dict_filename_list = sorted(opti_dict_filename_list, key=str.lower)

    # if no optimization results are found, raise error
    if not opti_dict_filename_list:
        raise FileNotFoundError(
            (
                f"no optimization results found for {site_name} in {opti_dict_path}, "
                "Have you performed optimization for this site/ experiment?"
            )
        )

    # initialize p_names with None, if optimization was successful for at least one site year,
    # p_names will be updated. If optimization was not successful for any site year, p_names will
    # remain None
    p_names = None

    # loop over the optimization results for each site year
    for opti_site_years_filename in opti_dict_filename_list:
        opti_dict = np.load(opti_site_years_filename, allow_pickle=True).item()
        
        if isinstance(opti_dict["x"], np.ndarray):
            xbest = opti_dict["x"].tolist()
        else:
            xbest = opti_dict["x"]

        m = re.search(r"_(\d{4})(?=_lbfgs)", opti_site_years_filename)
        site_year = m.group(1) if m else None
        site_year = float(site_year)
        
        if (xbest is None) or (
            np.isnan(np.array(xbest)).any()
        ): 
            # if the optimization was not successful, skip the site year
            logger.warning(
                "%s (%s): optimization was not successful (skipping forward run)",
                site_name,
                site_year,
            )

            # set model output to NaN for the site year
            site_year_mask = ip_df_dict["year"] == site_year

            for k, v in model_op.items():
                if k == "wai_results":
                    for k1 in v:
                        v[k1][site_year_mask] = np.nan
                else:
                    v[site_year_mask] = np.nan

            # set the optimized parameters to NaN for the site year
            x_best_dict[site_year] = [np.nan]
            
        else:
            # if the optimization was successful, run the model with optimized parameters
            model_op_site_yr, p_names, xbest_actual = forward_run_model(
                ip_df_dict,
                ip_df_daily_wai,
                wai_output,
                time_info,
                settings_dict,
                xbest,
            )

            # store the model outputs for the site year
            site_year_mask = ip_df_dict["year"] == site_year

            for k, v in model_op.items():
                if k == "wai_results":
                    for k1 in v:
                        v[k1][site_year_mask] = model_op_site_yr[k][k1][site_year_mask]
                else:
                    v[site_year_mask] = model_op_site_yr[k][site_year_mask]

            # store the optimized parameters for the site year
            x_best_dict[site_year] = xbest_actual

    # if optimization was successful for at least one site year
    if p_names is not None:
        # save and plot model results
        result_dict = save_n_plot_model_results(
            ip_df_dict, model_op, settings_dict, x_best_dict, p_names
        )
        # plot optimized parameters for each site year
        plot_opti_param_site_year(result_dict, settings_dict)
    else:
        pass

    # clean the log file
    input_filename = os.path.join(
        "model_results",
        dir_to_store,
        "for_run_opti_lbfgs.log",
    )
    output_filename = os.path.join(
        "model_results",
        dir_to_store,
        "for_run_opti_lbfgs_clean.log",
    )
    filter_logs(
        input_filename, output_filename, 21
    )  # 21 is index at which "[%(asctime)s]" (which is only
    # differenc

if __name__ == "__main__":

    result_paths = importlib.import_module("opti_result_path_coll")

    opti_res_path_coll = {
        "per_site_yr_p_model_res_path": result_paths.per_site_yr_p_model_res_path,
        "per_site_yr_lue_model_res_path": result_paths.per_site_yr_lue_model_res_path,
    }

    opti_res_path = opti_res_path_coll[sys.argv[3]] # get experiment name from command line argument

    site_list = pd.read_csv("../site_info/SiteInfo_BRKsite_list.csv", low_memory=False)["SiteID"].tolist()

    ########## not parallel version ##############
    # for ix in range(len(site_list)):
    #     forward_run_syr_lbfgs(ix, site_list, opti_res_path, et_var_name="ET")
    #############################################

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
        forward_run_syr_lbfgs,
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

