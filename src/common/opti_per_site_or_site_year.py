#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this module performs optimization of model parameters using CMAES algorithm
for each site or each site year

author: rde
first created: 2023-11-07
"""

import os
import sys
from pathlib import Path
import json
import logging
from functools import partial
import cma  # development version (v 3.3.0.1)
import numpy as np

from src.p_model.pmodel_plus import pmodel_plus
from src.common.get_params import get_params
from src.p_model.p_model_cost_function import p_model_cost_function
from src.lue_model.lue_model_cost_function import lue_model_cost_function

logger = logging.getLogger(__name__)


class HiddenPrints:
    """
    supress the stdouts to console (https://stackoverflow.com/a/45669280)

    Example usage:
    with HiddenPrints():
        print("This will not be printed")
    print("This will be printed as usual")
    """

    def __init__(self):
        self._original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_cmaes_options(
    scale_coord=True,
    p_lbound_scaled=None,
    p_ubound_scaled=None,
    append=None,
    settings_dict=None,
    site_year=None,
):
    """
    set options for CMAES Optimization

    Parameters:
    p_lbound_scaled (list): lower bound of parameters (scaled).
    p_ubound_scaled (list): upper bound of parameters (scaled).
    append (string): Site ID or PFT name (to create unique CMAES output filenames).
    settings_dict (dict): Dictionary of settings used for the model
                          optimization experiment (to get experiment name).
    site_year (float): year to be optimized in case of site year optimization

    Returns:
    opts (dict): options for CMAES optimization.
    sigma0 (float): initial standard deviation/ step size.
    """

    if (
        (p_lbound_scaled is None)
        or (p_ubound_scaled is None)
        or (append is None)
        or (settings_dict is None)
    ):
        bounds = None
        verb_filenameprefix = None
    else:
        bounds = [p_lbound_scaled, p_ubound_scaled]
        if site_year is None:  # all year optimization
            verb_filenameprefix = (
                f"outcmaes/{settings_dict['model_name']}/"
                f"{settings_dict['exp_name']}/{append}_"
            )
        else:  # site year optimization
            verb_filenameprefix = (
                f"outcmaes/{settings_dict['model_name']}/{settings_dict['exp_name']}/"
                f"{append}_{int(site_year)}_"
            )

    # options for optimizer
    # https://github.com/CMA-ES/pycma/blob/master/cma/evolution_strategy.py#L415
    # accessible via cma.CMAOptions() or cma.CMAOptions('verb') or cma.CMAOptions('tol') etc.
    opts = {
        "seed": 199340,
        "bounds": bounds,  # lower (=bounds[0])
        # and upper domain boundaries, each a scalar or a list/vector'
        # "maxiter": 500,  #'100 + 150 * (N+3)**2 // popsize**0.5  #v maximum number of iterations'
        # "maxfevals": 50000,  #'inf
        # v maximum number of function evaluations'
        # "popsize": 1000,  #'4 + 3 * np.log(N)
        # population size, AKA lambda, int(popsize) is the number of new solution per iteration'
        # "popsize_factor": 1,  # multiplier for popsize,
        # convenience option to increase default popsize'
        "tolx": 1e-6,  #'1e-11
        # v termination criterion: tolerance in x-changes'
        "tolfun": 1e-6,  #'1e-11
        # v termination criterion: tolerance in function value, quite useful'
        # "tolflatfitness": 2, #'1  #v iterations tolerated with flat fitness before termination',
        "verb_log": 1,  ##v
        # verbosity: write data to files every verb_log iteration,
        # writing can be time critical on fast to evaluate functions'
        # "verb_plot": 0, #'0  #v in fmin(): plot() is called every verb_plot iteration'
        "verbose": 3,  #'3  #v verbosity
        # e.g. of initial/final message, -1 is very quiet,
        # -9 maximally quiet, may not be fully implemented',
        # "verb_disp": 10, #'100  #v verbosity: display console output every verb_disp iteration'
        "verb_filenameprefix": verb_filenameprefix,  # output path and filenames prefix'
    }

    # set the step size accordingly
    # the search space will roughly be (initial +/- step size)
    # the step size can be approximately set as 1/4th of parameter range
    if scale_coord:
        sigma0 = 0.25  # initial standard deviation/ step size
    else:
        sigma0 = 0.5  # initial standard deviation/ step size

    return opts, sigma0


def save_opti_results(opti_dict, settings_dict, append):
    """
    convert data to json serializeable data format and
    save the optimization results as json file

    parameters:
    opti_dict (dict): optimization results
    settings_dict (dict): Dictionary of settings used for the model optimization experiment
    append (str): Site ID or PFT name (to create unique filename
    for saving the optimization results)

    returns:
    save the optimization results as json file
    """
    # convert the numpy arrays returned by optimizer to list or
    # np.int64 to int for json serialization
    for k, v in opti_dict.items():
        if isinstance(v, np.ndarray):
            opti_dict[k] = v.tolist()
        elif isinstance(v, np.int64):  # type: ignore
            opti_dict[k] = int(v)

    # save the optimization results as json file
    opti_dict_path = Path(
        "opti_results",
        settings_dict["model_name"],
        settings_dict["exp_name"],
        "opti_dicts",
    )  # Path to save the optimization results
    os.makedirs(
        opti_dict_path, exist_ok=True
    )  # create the directory if it does not exist
    # (in case outcmeas is not created due to skipping optimization in case of no good quality data)
    opti_dict_path_filename = os.path.join(
        opti_dict_path, f"{append}_opti_dict.json"
    )  # filename to save the optimization results
    with open(
        opti_dict_path_filename, "w", encoding="utf-8"
    ) as opti_dict_json_file:  # save the optimization results
        json.dump(opti_dict, opti_dict_json_file, indent=4, separators=(", ", ": "))


def optimize_model(
    ip_df_dict, ip_df_daily_wai, wai_output, time_info, settings_dict, site_year=None
):
    """
    Optimizes the Model parameters using CMAES algorithm (https://github.com/CMA-ES/pycma).

    Parameters:
    ip_df_dict (dict): Input forcing data in dictionary format.
    ip_df_daily_wai (dict): Daily forcing data to calculate WAI (for faster WAI spinup).
    wai_output (dict): arrays of zeros to store wai output.
    time_info (dict): timsteps (24 for hourly, 48 for half-hourly) and index of noon data.
    settings_dict (dict): Dictionary of settings used for the model optimization experiment.
    site_year (float): year to be optimized in case of site year optimization

    Returns:
    op_opti (dict): contains result of CMAES optimization
                    (such as best parameter vector, stop criteria etc.).
    """

    # get the parameters: global constant parameters are needed in pmodel_plus
    # parameter bounds are needed to calculate scalar of bounds for optimization
    params = get_params(ip_df_dict)

    # Define costhand and parameters in case of P Model
    if settings_dict["model_name"] == "P_model":
        # run pmodelPlus without acclimation
        model_op_no_acclim_sd = pmodel_plus(
            ip_df_dict, params, settings_dict["CO2_var"]
        )

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
        # also optimize alpha in fW_Horn for arid sites
        if ip_df_dict["KG"][0] == "B":
            # insert alpha at 4th position;
            # to remain compatible with old optimization results
            p_names.insert(3, "alpha")
            # p_names.append("alpha")
        else:
            pass

        # define the cost function to be optimized as a partial
        # function of p_model_cost_function
        if site_year is None:  # in case of all year optimization
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
                site_year=site_year,
                et_var_name=settings_dict["et_var_name"],
            )

    # Define costhand and parameters in case of LUE Model
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
        # also optimize alpha in fW_Horn for arid sites
        if ip_df_dict["KG"][0] == "B":
            p_names.append("alpha")
        # also optimize alpha_fT_Horn in fT_Horn for temperate,
        # continental and polar sites
        elif ip_df_dict["KG"][0] in ["C", "D", "E"]:
            p_names.append("alpha_fT_Horn")
        else:
            pass

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
        # define the cost function to be optimized as a partial
        # function of lue_model_cost_function
        if site_year is None:  # in case of all year optimization
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
                site_year=site_year,
                et_var_name=settings_dict["et_var_name"],
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
        p_ubound_scaled.append(params[p]["ub"] / params[p]["ini"])
        p_lbound_scaled.append(params[p]["lb"] / params[p]["ini"])

    # get the costvalue with initial guess of parameters
    costvalue = costhand(p_values_scalar=list(np.ones(len(p_names))))

    # if the cost function returns nan due to no good quality data present after filtering,
    # return nan for the optimization results
    if np.isnan(costvalue):
        if site_year is None:
            logger.warning(
                "%s: no good quality data present after filtering (aborting optimization)",
                ip_df_dict["SiteID"],
            )
        else:
            logger.warning(
                "%s (%s): no good quality data present after filtering (aborting optimization)",
                ip_df_dict["SiteID"],
                str(int(site_year)),
            )
        op_opti = {
            "site_year": int(site_year) if site_year is not None else None,
            "xbest": np.array([np.nan]),
            "fbest": np.nan,
            "evals_best": np.nan,
            "evaluations": np.nan,
            "xfavorite": np.array([np.nan]),
            "stop": {
                "no_data": (
                    "no good quality data present after filtering"
                    "(not a CMAES stop criteria)"
                )
            },
            "stds": np.array([np.nan]),
        }
    else:  # run the optimization
        if site_year is None:
            opts, sigma0 = get_cmaes_options(
                settings_dict["scale_coord"],
                p_lbound_scaled,
                p_ubound_scaled,
                ip_df_dict["SiteID"],
                settings_dict,
            )
        else:
            opts, sigma0 = get_cmaes_options(
                settings_dict["scale_coord"],
                p_lbound_scaled,
                p_ubound_scaled,
                ip_df_dict["SiteID"],
                settings_dict,
                site_year,
            )

        # verbose 3 prints the optimization results to
        # console as well as produce .dat files
        # verbose -1 and -9 prints less to no outputs to console
        # and also don't produce .dat files
        # I want to get .dat files, but don't want to get stdouts
        # to console while running on cluster, so I set
        # verbose to 3 in cmaes_options to produce .dat files and used HiddenPrints()
        # to suppress the stdouts to console
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

                # using stable version of cmaes (v3.3.0)
                # scaled_costhand = cma.ScaleCoordinates(
                #     costhand,
                #     multipliers=[
                #         ub - lb for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)
                #     ], # (ub - lb) for each parameter
                #     zero=[
                #         -lb/(ub-lb) for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)
                #     ], # -lb/(ub-lb) for each parameter
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
                
                # initial guess for parameters scalar to be optimized
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
        # descriptions from:
        # https://github.com/CMA-ES/pycma/blob/master/cma/evolution_strategy.py#L977
        op_opti = {}
        op_opti["site_year"] = (
            int(site_year) if site_year is not None else None
        )  # add the site year in case of site year optimization
        op_opti["xbest"] = res.result.xbest  # best solution evaluated
        op_opti["fbest"] = res.result.fbest  # objective function value of best solution
        op_opti["evals_best"] = (
            res.result.evals_best
        )  # evaluation count when xbest was evaluated
        op_opti["evaluations"] = (
            res.result.evaluations
        )  # number of function evaluations done
        op_opti["xfavorite"] = (
            res.result.xfavorite
        )  # distribution mean in "phenotype" space,
        # to be considered as current best estimate of the optimum
        op_opti["stop"] = (
            res.result.stop
        )  # stop criterion reached (termination conditions in a dictionary)
        op_opti["stds"] = (
            res.result.stds
        )  # effective standard deviations, can be used to
        #   compute a lower bound on the expected coordinate-wise distance
        #   to the true optimum, which is (very) approximately stds[i] *
        #   dimension**0.5 / min(mueff, dimension) / 1.5 / 5 ~ std_i *
        #   dimension**0.5 / min(popsize / 2, dimension) / 5, where
        #   dimension = CMAEvolutionStrategy.N and mueff =
        #   CMAEvolutionStrategy.sp.weights.mueff ~ 0.3 * popsize
        op_opti["opti_param_names"] = p_names  # list of parameters optimized

        if site_year is not None:
            logger.info(
            "%s (%s): optimization completed in %s function evaluations, stop criteria: %s",
            ip_df_dict["SiteID"],
            str(int(site_year)),
            str(op_opti["evaluations"]),
            str(op_opti["stop"]),
        )
        else:
            logger.info(
                "%s: optimization completed in %s function evaluations, stop criteria: %s",
                ip_df_dict["SiteID"],
                str(op_opti["evaluations"]),
                str(op_opti["stop"]),
            )

        # if the optimization stopped due to flat fitness (can hapen depending
        # on formulation of cost function) and the site will not be optimized, log it
        try:  # need to use try, as in
            # other cases there will be no "tolflatfitness" in op_opti["stop"]
            if op_opti["stop"]["tolflatfitness"] == 1:
                logger.info(
                    "%s: model couldn't be optimized due to flat fitness",
                    ip_df_dict["SiteID"],
                )
        except KeyError:
            pass

    # save the optimization results as json file
    if site_year is None:
        save_opti_results(op_opti, settings_dict, ip_df_dict["SiteID"])
    else:
        save_opti_results(
            op_opti, settings_dict, f"{ip_df_dict['SiteID']}_{int(site_year)}"
        )
