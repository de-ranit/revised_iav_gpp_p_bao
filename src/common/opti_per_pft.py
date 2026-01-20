#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Perform the optimization of the model parameters for a given PFT

author: rde
first created: 2023-12-15
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import logging
import concurrent.futures
import multiprocessing
import json
import time
from functools import partial
import bottleneck as bn
import cma
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

from src.common.get_data import get_data
from src.p_model.pmodel_plus import pmodel_plus
from src.common.get_params import get_params
from src.p_model.p_model_cost_function import p_model_cost_function
from src.lue_model.lue_model_cost_function import lue_model_cost_function
from src.common.forward_run_model import scale_back_params
from src.common.opti_per_site_or_site_year import (
    get_cmaes_options,
    HiddenPrints,
    save_opti_results,
)


logger = logging.getLogger(__name__)


def pft_cost_fn(
    p_values_scalar,
    info_dict,
    site_list,
    p_names_tuple,
    ip_df_dict_tuple,
    ip_df_daily_wai_tuple,
    wai_output_tuple,
    fpar_var_tuple,
    co2_var_tuple,
    data_filtering_tuple,
    cost_func_tuple,
    model_name,
    nstepsday_tuple=None,
    time_info_tuple=None,
    model_op_no_acclim_sd_tuple=None,
    synthetic_data_tuple=None,
):
    """
    Cost function for the optimization of the model parameters for a given PFT.
    This function evaluates the cost function for each site in parallel and sums up
    all the costs to get the total cost for the PFT.

    Parameters:
    p_values_scalar (list): scalar values of parameters produced by the optimizer
    site_list (list): list of all site ID in a PFT
    p_names_tuple (tuple): names of parameters to be optimized
                           (in a list) repeated len(site_list) times in a tuple
    ip_df_dict_tuple (tuple): input dictionary of forcing variables for each site
    ip_df_daily_wai_tuple (tuple): input dictionary of daily variables needed
                                   for WAI spinup for each site
    wai_output_tuple (tuple): dictionary to collect WAI calculation results for each site
    fpar_var_tuple (tuple): fPAR variable name repeated len(site_list) times in a tuple
    co2_var_tuple (tuple): CO2 variable name repeated len(site_list) times in a tuple
    data_filtering_tuple (tuple): data filtering method repeated len(site_list) times
                                  in a tuple
    cost_func_tuple (tuple): cost function name repeated len(site_list) times in a tuple
    model_name (str): name of the model to be optimized
    nstepsday_tuple (tuple) [only for LUE Model]: number of steps per day repeated
                                                  len(site_list) times in a tuple
    time_info_tuple (tuple) [only for P Model]: dictionary containing time information
                                                repeated len(site_list) times in a tuple
    model_op_no_acclim_sd_tuple (tuple) [only for P Model]: model output without
                                                            acclimation for each site
    synthetic_data_tuple (tuple) [only for LUE Model]: synthetic data repeated
                                                       len(site_list) times in a tuple

    Returns:
    total_cost (float): total cost for the PFT
    """

    # limit the number of workers to 10 (if at least 10 cores are available),
    # as more workers create more overhead and takes longer time (probably)
    # e.g., for a PFT with 48 sites, allocating 48 workers doesn't
    # increase speed even when running on a machine with > 48 cores
    no_of_cores = multiprocessing.cpu_count()  # get the number of cores
    max_no_of_workers = 10 if no_of_cores > 10 else no_of_cores
    no_of_workers = (
        len(site_list) if len(site_list) < max_no_of_workers else max_no_of_workers
    )

    # repeat the scalar of parameter values for each site in a tuple
    p_values_scalar_tuple = tuple(p_values_scalar for _ in site_list)

    # evaluate the cost function for each site in parallel
    # and get a list of cost values for each site
    if model_name == "P_model":
        # ensure that model_op_no_acclim_sd_tuple and time_info_tuple are not None
        # for P Model
        if (model_op_no_acclim_sd_tuple is not None) & (time_info_tuple is not None):
            with concurrent.futures.ProcessPoolExecutor(no_of_workers) as executor:
                costs = list(
                    executor.map(
                        p_model_cost_function,
                        p_values_scalar_tuple,
                        p_names_tuple,
                        ip_df_dict_tuple,
                        model_op_no_acclim_sd_tuple,  # type: ignore
                        ip_df_daily_wai_tuple,
                        wai_output_tuple,
                        time_info_tuple,  # type: ignore
                        fpar_var_tuple,
                        co2_var_tuple,
                        data_filtering_tuple,
                        cost_func_tuple,
                        chunksize=1,  # increasing chunksize takes more time
                    )
                )
        else:
            raise ValueError(
                f"model_op_no_acclim_sd_tuple and time_info_tuple should not be None"
                f"for {model_name}"
            )
    elif model_name == "LUE_model":
        # ensure that nstepsday_tuple and synthetic_data_tuple are not None in case of LUE Model
        if (nstepsday_tuple is not None) & (synthetic_data_tuple is not None):
            with concurrent.futures.ProcessPoolExecutor(no_of_workers) as executor:
                costs = list(
                    executor.map(
                        lue_model_cost_function,
                        p_values_scalar_tuple,
                        p_names_tuple,
                        ip_df_dict_tuple,
                        ip_df_daily_wai_tuple,
                        wai_output_tuple,
                        nstepsday_tuple,  # type: ignore
                        fpar_var_tuple,
                        co2_var_tuple,
                        data_filtering_tuple,
                        cost_func_tuple,
                        synthetic_data_tuple,  # type: ignore
                        chunksize=1,  # increasing chunksize takes more time
                    )
                )
        else:
            raise ValueError(
                f"nstepsday_tuple and synthetic_data_tuple should not be None"
                f"for {model_name}"
            )
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {model_name}"
            "is not implemented"
        )

    # sum up the cost values for all sites to get the total cost for the PFT
    total_cost = bn.nansum(costs)

    if info_dict["do_l-bfgs-b_from_prev"]:
        print(info_dict["Nfeval"], p_values_scalar, total_cost)
        print("--------------------------------------------------")
        logger.info(
            "PFT %s, Function evaluation %d, Parameters: %s, Total cost: %.4f",
            info_dict["pft"],
            info_dict["Nfeval"],
            str(p_values_scalar),
            total_cost,
        )
        info_dict["Nfeval"] += 1

    return total_cost


def opti_per_pft(pft, sites_per_pft_dict, settings_dict):
    """
    Optimize the model parameters for a given PFT.

    Parameters:
    pft (str): PFT name
    sites_per_pft_dict (dict): dictionary containing the Site ID(s) in each PFT
    settings_dict (dict): dictionary containing the experiment settings

    Returns:
    save the optimization results in a .json file
    """

    # get the list of sites in the PFT
    site_list = sites_per_pft_dict[pft]

    # remove the sites with all bad fPAR data
    site_list = [
        sites
        for sites in site_list
        if sites not in ["CG-Tch", "MY-PSO", "GH-Ank", "US-LWW"]
    ]

    ip_data_collection = {}  # collect input data for each site in a dictionary
    kg_collection = []  # collect the KG values for each site in a list

    # get the input data for each site in the PFT
    for site_name in site_list:
        # get the input data for a site
        ip_df_dict, ip_df_daily_wai, wai_output, time_info = get_data(
            site_name, settings_dict
        )

        if settings_dict["model_name"] == "P_model":  # in case of P model
            # get the constant parameters to run the model without acclimation
            params = get_params(ip_df_dict)

            # run pmodelPlus without acclimation
            model_op_no_acclim_sd = pmodel_plus(
                ip_df_dict, params, settings_dict["CO2_var"]
            )

            # collect the input data and model output for each site in a dictionary
            ip_data_collection[site_name] = {
                "ip_df_dict": ip_df_dict,
                "ip_df_daily_wai": ip_df_daily_wai,
                "wai_output": wai_output,
                "time_info": time_info,
                "model_op_no_acclim_sd": model_op_no_acclim_sd,
            }
        elif settings_dict["model_name"] == "LUE_model":  # in case of LUE model
            # collect the input data for each site in a dictionary
            ip_data_collection[site_name] = {
                "ip_df_dict": ip_df_dict,
                "ip_df_daily_wai": ip_df_daily_wai,
                "wai_output": wai_output,
                "time_info": time_info,
            }
        else:
            raise ValueError(
                f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
                "is not implemented"
            )

        # collect the KG values for each site in a list
        kg_collection.append(ip_df_dict["KG"][0])

    # create tuple of input data to use as arguments for the cost function
    ip_df_dict_tuple = tuple(
        ip_data_collection[site_name]["ip_df_dict"] for site_name in site_list
    )
    ip_df_daily_wai_tuple = tuple(
        ip_data_collection[site_name]["ip_df_daily_wai"] for site_name in site_list
    )
    wai_output_tuple = tuple(
        ip_data_collection[site_name]["wai_output"] for site_name in site_list
    )
    fpar_var_tuple = tuple(settings_dict["fPAR_var"] for site_name in site_list)
    co2_var_tuple = tuple(settings_dict["CO2_var"] for site_name in site_list)
    data_filtering_tuple = tuple(
        settings_dict["data_filtering"] for site_name in site_list
    )
    cost_func_tuple = tuple(settings_dict["cost_func"] for site_name in site_list)

    # initialize function evaluation counter for logging during L-BFGS-B
    settings_dict["Nfeval"] = -1
    settings_dict["pft"] = pft

    # list of parameters to be optimized
    if settings_dict["model_name"] == "P_model":
        # list of parameters to be optimized in all cases
        p_names = [
            "acclim_window",
            "W_I",
            "K_W",
            "AWC",
            "theta",
            "alphaPT",
            "meltRate_temp",
            "meltRate_netrad",
            "sn_a",
        ]

        if "B" in kg_collection:  # if there is a site with KG starts with B in the PFT
            # then optimize the alpha (lag) parameter
            # get the parameter bounds and initial values
            params = get_params({"KG": "B", "elev": 101.3})
            # add alpha to the list of parameters to be optimized
            # insert alpha at 4th position;
            # to remain compatible with old optimization results
            p_names.insert(3, "alpha")
            # p_names.append("alpha")
        else:
            params = get_params({"KG": "A", "elev": 101.3})

        # define the cost function to be optimized for P model
        costhand = partial(
            pft_cost_fn,
            info_dict=settings_dict,
            site_list=site_list,
            p_names_tuple=(p_names,) * len(site_list),
            ip_df_dict_tuple=ip_df_dict_tuple,
            ip_df_daily_wai_tuple=ip_df_daily_wai_tuple,
            wai_output_tuple=wai_output_tuple,
            fpar_var_tuple=fpar_var_tuple,
            co2_var_tuple=co2_var_tuple,
            data_filtering_tuple=data_filtering_tuple,
            cost_func_tuple=cost_func_tuple,
            model_name=settings_dict["model_name"],
            time_info_tuple=tuple(
                ip_data_collection[site_name]["time_info"] for site_name in site_list
            ),
            model_op_no_acclim_sd_tuple=tuple(
                ip_data_collection[site_name]["model_op_no_acclim_sd"]
                for site_name in site_list
            ),
        )

    elif settings_dict["model_name"] == "LUE_model":
        # list of parameters to be optimized in all cases
        p_names = [
            "LUE_max",
            "T_opt",
            "K_T",
            "Kappa_VPD",
            "Ca_0",
            "C_Kappa",
            "c_m",
            "gamma_fL_TAL",
            "mu_fCI",
            "W_I",
            "K_W",
            "AWC",
            "theta",
            "alphaPT",
            "meltRate_temp",
            "meltRate_netrad",
            "sn_a",
        ]

        # generate complete list of parameters to be optimized based on KG values
        if "B" in kg_collection:
            # get the parameter bounds and initial values
            # when there is at least one site with KG starts with B in the PFT
            params = get_params({"KG": "B", "elev": 101.3})
            # add alpha to the list of parameters to be optimized
            p_names.append("alpha")

        if any(kg in kg_collection for kg in ["C", "D", "E"]):
            # get the parameter bounds and initial values
            # when there is at least one site with KG starts with C/D/E in the PFT
            params = get_params({"KG": "C", "elev": 101.3})
            # add alpha_fT_Horn to the list of parameters to be optimized
            p_names.append("alpha_fT_Horn")

        if ("B" in kg_collection) & any(kg in kg_collection for kg in ["C", "D", "E"]):
            # when both B and C/D/E sites are present in a PFT, ensure both alpha and
            # alpha_fT_Horn are dictionaries with bounds and initial values
            params = get_params({"KG": "C", "elev": 101.3})
            params["alpha"] = {"ini": 0.9899452, "ub": 1.0, "lb": 0.0}

        if all(kg not in kg_collection for kg in ["B", "C", "D", "E"]):
            # get the parameter bounds and initial values
            # when there is no site with KG starts with B/C/D/E in the PFT
            params = get_params({"KG": "A", "elev": 101.3})

        # generate synthetic data to calculate 3rd and 4th
        # component of LUE model cost function
        nstepsday = ip_data_collection[site_list[0]]["time_info"]["nstepsday"]
        synthetic_data = {
            "TA": np.linspace(-5.0, 40.0, nstepsday * 365),  # deg C
            "VPD": np.linspace(4500, 0.0, nstepsday * 365),  # Pa
            "CO2": np.linspace(400.0, 400.0, nstepsday * 365),  # PPM
            "wai_nor": np.linspace(0.0, 1.0, nstepsday * 365),  # - (or mm/mm)
            "fPAR": np.linspace(0.0, 1.0, nstepsday * 365),  # -
            "PPFD": np.linspace(0.0, 600.0, nstepsday * 365),  # umol photons m-2s-1
        }

        # define the cost function to be optimized for LUE model
        costhand = partial(
            pft_cost_fn,
            info_dict=settings_dict,
            site_list=site_list,
            p_names_tuple=(p_names,) * len(site_list),
            ip_df_dict_tuple=ip_df_dict_tuple,
            ip_df_daily_wai_tuple=ip_df_daily_wai_tuple,
            wai_output_tuple=wai_output_tuple,
            fpar_var_tuple=fpar_var_tuple,
            co2_var_tuple=co2_var_tuple,
            data_filtering_tuple=data_filtering_tuple,
            cost_func_tuple=cost_func_tuple,
            model_name=settings_dict["model_name"],
            nstepsday_tuple=(nstepsday,) * len(site_list),
            synthetic_data_tuple=(synthetic_data,) * len(site_list),
        )
    else:
        raise ValueError(
            f"model_name should be either P_model or LUE_model, {settings_dict['model_name']}"
            "is not implemented"
        )

    # get the scaled value of parameter bounds (ub/initial or lb/initial)
    p_ubound_scaled = []
    p_lbound_scaled = []
    for p in p_names:
        p_ubound_scaled.append(params[p]["ub"] / params[p]["ini"])  # type: ignore
        p_lbound_scaled.append(params[p]["lb"] / params[p]["ini"])  # type: ignore

    if (settings_dict["do_l-bfgs-b_from_prev"]) and (
        settings_dict["xbest_dict_file_path"] is not None
    ):
        xbest_dict_file = f"{settings_dict['xbest_dict_file_path']}{pft}_opti_dict.json"

        with open(xbest_dict_file, "r") as f:
            xbest_dict = json.load(f)

        xbest_cmaes = xbest_dict["xbest"]

        xbest_actual = scale_back_params(xbest_cmaes, p_names, params)

        ini_costval = costhand(p_values_scalar=xbest_actual)

        bounds = Bounds(np.array(p_lbound_scaled), np.array(p_ubound_scaled))

        start_time = time.time()
        lbfgs_b_op = minimize(
            fun=costhand,
            x0=xbest_actual,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "disp": True,
                "maxfun": 30000,
                "ftol": 1e-8,
                "maxcor": 10,
                "gtol": 1e-05,
                "eps": 1e-08,
                "maxls": 20,
            },
        )

        time_delta = time.time() - start_time

        percentage_reduction_in_cost = (ini_costval - lbfgs_b_op.fun) / ini_costval * 100.0

        lbfgs_b_op_dict = {
            "initial_cost": ini_costval,
            "fbest": lbfgs_b_op.fun,
            "percentage_reduction_in_cost": percentage_reduction_in_cost,
            "jac": lbfgs_b_op.jac,
            "nfev": lbfgs_b_op.nfev,
            "njev": lbfgs_b_op.njev,
            "nit": lbfgs_b_op.nit,
            "status": lbfgs_b_op.status,
            "message": lbfgs_b_op.message,
            "xbest": lbfgs_b_op.x,
            "success": lbfgs_b_op.success,
        }

        logger.info(
            (
                "PFT %s, Starting cost value: %.4f, Final cost value: %.4f, Percentage cost reduction: %4f, nfev: %d, njev: %d, nit: %d, status: %d, "
                "message: %s, success: %s, time elapsed: %.2f seconds"
            ),
            pft,
            ini_costval,
            lbfgs_b_op.fun,
            percentage_reduction_in_cost,
            lbfgs_b_op.nfev,
            lbfgs_b_op.njev,
            lbfgs_b_op.nit,
            lbfgs_b_op.status,
            lbfgs_b_op.message,
            str(lbfgs_b_op.success),
            time_delta,
        )

        opti_res_save_dir = f"./opti_results/{settings_dict['model_name']}/{settings_dict['exp_name']}/opti_dicts/"
        os.makedirs(opti_res_save_dir, exist_ok=True)

        np.save(
            f"{opti_res_save_dir}{pft}_opti_dict.npy",
            lbfgs_b_op_dict,
        )

    elif (settings_dict["do_l-bfgs-b_from_prev"]) and (
        settings_dict["xbest_dict_file_path"] is None
    ):
        raise ValueError(
            "`xbest_dict_file_path` should be provided when"
            "`do_l-bfgs-b_from_prev` is set to True"
        )
    else:

        # evaluate the cost function for initial guess of parameters
        # costvalue = costhand(p_values_scalar=list(np.ones(len(p_names))))

        # get the options for the CMA-ES optimizer
        opts, sigma0 = get_cmaes_options(
            settings_dict["scale_coord"],
            p_lbound_scaled,
            p_ubound_scaled,
            pft,
            settings_dict,
        )

        # run the CMA-ES optimizer
        with HiddenPrints():
            if settings_dict["scale_coord"]:
                ###### hints about scale coordinates ####################################
                # if parameters have different ranges, then it's useful to scale
                # parameters between 0 and 1, and the cost function should be modified accordingly
                # https://github.com/CMA-ES/pycma/issues/210
                # https://github.com/CMA-ES/pycma/issues/248
                # https://cma-es.github.io/cmaes_sourcecode_page.html#practical
                # bounds and rescaling section of
                # https://github.com/CMA-ES/pycma/blob/development/notebooks/notebook-usecases-basics.ipynb
                ##############################################################################
                # scale the functions to make parameters range between 0 and 1

                # using stable version of cmaes (v3.3.0)
                # scaled_costhand = cma.ScaleCoordinates(
                #     costhand,
                #     multipliers=[
                #         ub - lb for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)
                #     ],  # (ub - lb) for each parameter
                #     zero=[
                #         -lb/(ub-lb) for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)
                #     ],  # -lb/(ub-lb) for each parameter
                # )

                # using development version of cmaes (v3.3.0.1)
                scaled_costhand = cma.ScaleCoordinates(
                    costhand,
                    multipliers=[
                        ub - lb for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)
                    ],  # (ub - lb) for each parameter
                    lower=p_lbound_scaled,  # lower bound for each parameter
                )

                # change the bounds to [0, 1] for each parameters
                opts["bounds"] = [np.zeros(len(p_names)), np.ones(len(p_names))]

                # initial guess for parameters scalar to be optimized
                p_values_scalar = np.array([1.0] * len(p_names))
                multipliers = np.array([ub - lb for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)])
                zero = np.array(
                    [-lb / (ub - lb) for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)]
                )
                p_values_scalar = list((p_values_scalar / multipliers) + zero) # coordinate transformed
                
                # # initial guess for parameters scalar to be optimized
                # p_values_scalar = [0.5] * len(p_names)

                # run the optimization
                cma_es = cma.CMAEvolutionStrategy(
                    p_values_scalar, sigma0, opts
                )  # X0 (initial guess) is p_values_scalar,
                # sigma0 (initial standard deviation/ step size) , opts (options)
                res = cma_es.optimize(scaled_costhand)

            else:  # if no scaling is needed
                # initial guess for parameters scalar to be optimized
                p_values_scalar = [1.0] * len(p_names)

                cma_es = cma.CMAEvolutionStrategy(
                    p_values_scalar, sigma0, opts
                )  # X0 (initial guess) is p_values_scalar * 2.0 (i.e., 1.0),
                # sigma0 (initial standard deviation/ step size) , opts (options)
                res = cma_es.optimize(costhand)

        # collect the optimization results in a dictionary
        # descriptions from https://github.com/CMA-ES/pycma/blob/master/cma/evolution_strategy.py#L977
        op_opti = {}
        op_opti["site_year"] = np.nan  # add the site year in case of site year optimization
        op_opti["xbest"] = res.result.xbest  # best solution evaluated
        op_opti["fbest"] = res.result.fbest  # objective function value of best solution
        op_opti[
            "evals_best"
        ] = res.result.evals_best  # evaluation count when xbest was evaluated
        op_opti[
            "evaluations"
        ] = res.result.evaluations  # number of function evaluations done
        op_opti[
            "xfavorite"
        ] = res.result.xfavorite  # distribution mean in "phenotype" space,
        # to be considered as current best estimate of the optimum
        op_opti[
            "stop"
        ] = (
            res.result.stop
        )  # stop criterion reached (termination conditions in a dictionary)
        op_opti["stds"] = res.result.stds  # effective standard deviations, can be used to
        #   compute a lower bound on the expected coordinate-wise distance
        #   to the true optimum, which is (very) approximately stds[i] *
        #   dimension**0.5 / min(mueff, dimension) / 1.5 / 5 ~ std_i *
        #   dimension**0.5 / min(popsize / 2, dimension) / 5, where
        #   dimension = CMAEvolutionStrategy.N and mueff =
        #   CMAEvolutionStrategy.sp.weights.mueff ~ 0.3 * popsize
        op_opti["opti_param_names"] = p_names  # list of parameters optimized

        # save the optimization results in a .json file
        save_opti_results(op_opti, settings_dict, pft)
