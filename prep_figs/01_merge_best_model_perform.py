#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
select best results based on lowest cost value among different optimization experiments:
1. CMAES with big population size
2. CMAES with big population size + L-BFGS-B
3. CMAES with default population size

author: rde
first created: Fri Nov 28 2025 17:29:16 CET
"""
import sys
import os
import glob
import json
import shutil
import numpy as np
import pandas as pd
# import ipdb
import logging


# add the path where modules of experiments are stored
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MAIN_DIR = os.path.dirname(SCRIPT_PATH)
sys.path.append(MAIN_DIR)

from src.common.get_params import get_params


def collect_best_results(exp_names_dict, opti_type, new_exp_name, lue_iav=False):

    site_info_df = pd.read_csv("../site_info/SiteInfo_BRKsite_list.csv")
    site_list = site_info_df["SiteID"].tolist()

    if opti_type == "syr":
        for site_name in site_list:

            site_kg = site_info_df.loc[site_info_df["SiteID"] == site_name, "KG"].values[0]
            site_elev = site_info_df.loc[site_info_df["SiteID"] == site_name, "elev"].values[0]

            cmaes_big_pop_opti_dict_path = f"../../02_optim_PModel/opti_results/{exp_names_dict['cmaes_big_pop']}/opti_dicts/{site_name}_*_opti_dict.json"

            cmaes_big_pop_opti_dict_path_coll = glob.glob(cmaes_big_pop_opti_dict_path)
            cmaes_big_pop_opti_dict_path_coll.sort()

            for site_yr_file in cmaes_big_pop_opti_dict_path_coll:

                with open(site_yr_file, "r") as f:
                    cmaes_big_pop_opti_dict = json.load(f)

                cmaes_big_pop_and_lbfgs_b_path = f"../opti_lbfgs_b/opti_lbfgs/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/lbgfgs_b_dicts/{site_name}_{cmaes_big_pop_opti_dict['site_year']}_lbfgs_b_results.npy"
                cmaes_big_pop_and_lbfgs_b_opti_dict = np.load(
                    cmaes_big_pop_and_lbfgs_b_path, allow_pickle=True
                ).item()

                cmaes_default_pop_opti_dict_path = f"../opti_results/{exp_names_dict['cmaes_default_pop']}/opti_dicts/{site_name}_{cmaes_big_pop_opti_dict['site_year']}_opti_dict.json"
                with open(cmaes_default_pop_opti_dict_path, "r") as f:
                    cmaes_default_pop_opti_dict = json.load(f)

                cost_arr = np.array(
                    [
                        cmaes_big_pop_opti_dict["fbest"],
                        cmaes_big_pop_and_lbfgs_b_opti_dict["fun"],
                        cmaes_default_pop_opti_dict["fbest"],
                    ]
                )

                if any(isinstance(x, float) and np.isnan(x) for x in cost_arr.flat):
                    save_path = f"../opti_results/{new_exp_name}/opti_dicts/{site_name}_{cmaes_big_pop_opti_dict['site_year']}_opti_dict.json"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    with open(save_path, "w") as f:
                        json.dump(
                            cmaes_default_pop_opti_dict,
                            f,
                            indent=4,
                            separators=(", ", ": "),
                        )

                else:
                    idx = int(np.argmin(cost_arr))

                    exp_name_dict = {
                        0: "cmaes_big_pop",
                        1: "cmaes_big_pop_and_lbfgs_b",
                        2: "cmaes_default_pop",
                    }
                    print(
                        f"Best opti for site {site_name} year {cmaes_big_pop_opti_dict['site_year']} is idx {exp_name_dict[idx]} with cost {cost_arr[idx]}"
                    )

                    save_path = f"../opti_results/{new_exp_name}/opti_dicts/{site_name}_{cmaes_big_pop_opti_dict['site_year']}_opti_dict.json"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    logging.basicConfig(
                        filename=(f"../opti_results/{new_exp_name}/selected_exp.log"),
                        filemode="a",
                        level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
                        datefmt="%Y-%m-%d,%H:%M:%S",
                    )
                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"Best opti for site {site_name} year {cmaes_big_pop_opti_dict['site_year']} is idx {exp_name_dict[idx]} with cost {cost_arr[idx]}"
                    )

                    if idx == 0:
                        cmaes_big_pop_opti_dict["opti_exp_name"] = "cmaes_big_pop"
                        with open(save_path, "w") as f:
                            json.dump(
                                cmaes_big_pop_opti_dict,
                                f,
                                indent=4,
                                separators=(", ", ": "),
                            )

                    elif idx == 1:
                        params = get_params({"KG": site_kg, "elev": site_elev})

                        p_names = cmaes_default_pop_opti_dict["opti_param_names"]

                        p_ubound_scaled = []
                        p_lbound_scaled = []
                        for p in p_names:
                            p_ubound_scaled.append(params[p]["ub"] / params[p]["ini"])
                            p_lbound_scaled.append(params[p]["lb"] / params[p]["ini"])

                        multipliers = np.array([ub - lb for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)])
                        zero = np.array(
                            [-lb / (ub - lb) for ub, lb in zip(p_ubound_scaled, p_lbound_scaled)]
                        )
                        p_values_scalar = list((cmaes_big_pop_and_lbfgs_b_opti_dict["x"] / multipliers) + zero) # coordinate transformed
                       
                        new_opti_dict = {
                            "site_year": cmaes_big_pop_opti_dict["site_year"],
                            "xbest": p_values_scalar,
                            "fbest": cmaes_big_pop_and_lbfgs_b_opti_dict["fun"],
                            "stop": cmaes_big_pop_and_lbfgs_b_opti_dict["message"],
                            "opti_param_names": cmaes_big_pop_opti_dict[
                                "opti_param_names"
                            ],
                            "opti_exp_name": "cmaes_big_pop_and_lbfgs_b",
                        }
                        with open(save_path, "w") as f:
                            json.dump(
                                new_opti_dict, f, indent=4, separators=(", ", ": ")
                            )

                    elif idx == 2:
                        cmaes_default_pop_opti_dict["opti_exp_name"] = (
                            "cmaes_default_pop"
                        )
                        with open(save_path, "w") as f:
                            json.dump(
                                cmaes_default_pop_opti_dict,
                                f,
                                indent=4,
                                separators=(", ", ": "),
                            )

    elif opti_type == "allyr":
        for site_name in site_list:

            if lue_iav:
                cmaes_big_pop_opti_dict_path = f"../opti_results/{exp_names_dict['cmaes_big_pop']}/opti_dicts/{site_name}_opti_dict.json"
            else:
                cmaes_big_pop_opti_dict_path = f"../../02_optim_PModel/opti_results/{exp_names_dict['cmaes_big_pop']}/opti_dicts/{site_name}_opti_dict.json"

            try:
                with open(cmaes_big_pop_opti_dict_path, "r") as f:
                    cmaes_big_pop_opti_dict = json.load(f)
            except FileNotFoundError:
                cmaes_big_pop_opti_dict = {"fbest": np.nan}

            if lue_iav:
                cmaes_big_pop_and_lbfgs_b_path = f"../opti_lbfgs_b/opti_lbfgs/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/lbgfgs_b_dicts/{site_name}_lbfgs_b_results.npy"
            else:
                cmaes_big_pop_and_lbfgs_b_path = f"../opti_lbfgs_b/opti_lbfgs/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/lbgfgs_b_dicts/{site_name}_None_lbfgs_b_results.npy"
            try:
                cmaes_big_pop_and_lbfgs_b_opti_dict = np.load(
                    cmaes_big_pop_and_lbfgs_b_path, allow_pickle=True
                ).item()
            except FileNotFoundError:
                cmaes_big_pop_and_lbfgs_b_opti_dict = {"fun": np.nan}

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

            save_path = f"../model_results/{new_exp_name}/serialized_model_results/"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            logging.basicConfig(
                filename=(f"../model_results/{new_exp_name}/selected_exp.log"),
                filemode="a",
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d,%H:%M:%S",
            )
            logger = logging.getLogger(__name__)

            if any(isinstance(x, float) and np.isnan(x) for x in cost_arr.flat):
                logger.info(f"{site_name} No optimization result found.")
            else:
                idx = int(np.argmin(cost_arr))

                exp_name_dict = {
                    0: "cmaes_big_pop",
                    1: "cmaes_big_pop_and_lbfgs_b",
                    2: "cmaes_default_pop",
                }
                print(
                    f"Best opti for site {site_name} is idx {exp_name_dict[idx]} with cost {cost_arr[idx]}"
                )

                logger.info(
                    f"Best opti for site {site_name} is idx {exp_name_dict[idx]} with cost {cost_arr[idx]}"
                )

                if idx == 0:
                    if lue_iav:
                        res_dict_path = f"../model_results/{exp_names_dict['cmaes_big_pop']}/serialized_model_results/{site_name}_result.npy"
                    else:
                        res_dict_path = f"../../02_optim_PModel/model_results/{exp_names_dict['cmaes_big_pop']}/serialized_model_results/{site_name}_result.npy"

                    shutil.copy2(res_dict_path, save_path)
                elif idx == 1:
                    res_dict_path = f"../opti_lbfgs_b/model_results/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/serialized_model_results/{site_name}_result.npy"
                    shutil.copy2(res_dict_path, save_path)
                elif idx == 2:
                    res_dict_path = f"../model_results/{exp_names_dict['cmaes_default_pop']}/serialized_model_results/{site_name}_result.npy"
                    shutil.copy2(res_dict_path, save_path)

    elif opti_type == "perPFT":

        unique_pft = site_info_df["PFT"].unique().tolist()

        collect_best_opti = {}
        for pft in unique_pft:

            cmaes_big_pop_opti_dict_path = f"../../02_optim_PModel/opti_results/{exp_names_dict['cmaes_big_pop']}/opti_dicts/{pft}_opti_dict.json"

            with open(cmaes_big_pop_opti_dict_path, "r") as f:
                cmaes_big_pop_opti_dict = json.load(f)

            cmaes_big_pop_and_lbfgs_b_path = f"../opti_lbfgs_b/opti_lbfgs/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/opti_dicts/{pft}_opti_dict.npy"
            cmaes_big_pop_and_lbfgs_b_opti_dict = np.load(
                cmaes_big_pop_and_lbfgs_b_path, allow_pickle=True
            ).item()

            cmaes_default_pop_opti_dict_path = f"../opti_results/{exp_names_dict['cmaes_default_pop']}/opti_dicts/{pft}_opti_dict.json"
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

            collect_best_opti[pft] = idx

            exp_name_dict = {
                0: "cmaes_big_pop",
                1: "cmaes_big_pop_and_lbfgs_b",
                2: "cmaes_default_pop",
            }
            print(
                f"Best opti for PFT {pft} is idx {exp_name_dict[idx]} with cost {cost_arr[idx]}"
            )

        for site_name in site_list:
            site_pft = site_info_df.loc[
                site_info_df["SiteID"] == site_name, "PFT"
            ].values[0]

            best_opti_exp = collect_best_opti[site_pft]

            if best_opti_exp == 0:
                res_dict_path = f"../../02_optim_PModel/model_results/{exp_names_dict['cmaes_big_pop']}/serialized_model_results/{site_name}_result.npy"
                save_path = f"../model_results/{new_exp_name}/serialized_model_results/{site_name}_result.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy2(res_dict_path, save_path)
                print(f"Copied result for site {site_name} from cmaes_big_pop")

            elif best_opti_exp == 1:
                res_dict_path = f"../opti_lbfgs_b/model_results/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/serialized_model_results/{site_name}_result.npy"
                save_path = f"../model_results/{new_exp_name}/serialized_model_results/{site_name}_result.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy2(res_dict_path, save_path)
                print(
                    f"Copied result for site {site_name} from cmaes_big_pop_and_lbfgs_b"
                )

            elif best_opti_exp == 2:
                res_dict_path = f"../model_results/{exp_names_dict['cmaes_default_pop']}/serialized_model_results/{site_name}_result.npy"
                save_path = f"../model_results/{new_exp_name}/serialized_model_results/{site_name}_result.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy2(res_dict_path, save_path)
                print(f"Copied result for site {site_name} from cmaes_default_pop")

    elif opti_type == "global":
        cmaes_big_pop_opti_dict_path = f"../../02_optim_PModel/opti_results/{exp_names_dict['cmaes_big_pop']}/opti_dicts/global_opti_opti_dict.json"

        with open(cmaes_big_pop_opti_dict_path, "r") as f:
            cmaes_big_pop_opti_dict = json.load(f)

        cmaes_big_pop_and_lbfgs_b_path = f"../opti_lbfgs_b/opti_lbfgs/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/global_opti_opti_dict.npy"
        cmaes_big_pop_and_lbfgs_b_opti_dict = np.load(
            cmaes_big_pop_and_lbfgs_b_path, allow_pickle=True
        ).item()

        cmaes_default_pop_opti_dict_path = f"../opti_results/{exp_names_dict['cmaes_default_pop']}/opti_dicts/global_opti_opti_dict.json"
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

        exp_name_dict = {
            0: "cmaes_big_pop",
            1: "cmaes_big_pop_and_lbfgs_b",
            2: "cmaes_default_pop",
        }
        print(f"Best global opti is {exp_name_dict[idx]} with cost {cost_arr[idx]}")

        for site_name in site_list:

            if idx == 0:
                res_dict_path = f"../../02_optim_PModel/model_results/{exp_names_dict['cmaes_big_pop']}/serialized_model_results/{site_name}_result.npy"
                save_path = f"../model_results/{new_exp_name}/serialized_model_results/{site_name}_result.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy2(res_dict_path, save_path)
                print(f"Copied result for site {site_name} from cmaes_big_pop")
            elif idx == 1:
                res_dict_path = f"../opti_lbfgs_b/model_results/{exp_names_dict['cmaes_big_pop_and_lbfgs_b']}/serialized_model_results/{site_name}_result.npy"
                save_path = f"../model_results/{new_exp_name}/serialized_model_results/{site_name}_result.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy2(res_dict_path, save_path)
                print(
                    f"Copied result for site {site_name} from cmaes_big_pop_and_lbfgs_b"
                )
            elif idx == 2:
                res_dict_path = f"../model_results/{exp_names_dict['cmaes_default_pop']}/serialized_model_results/{site_name}_result.npy"
                save_path = f"../model_results/{new_exp_name}/serialized_model_results/{site_name}_result.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                shutil.copy2(res_dict_path, save_path)
                print(f"Copied result for site {site_name} from cmaes_default_pop")


if __name__ == "__main__":

    per_site_yr_lue_exp_names = {
        "cmaes_big_pop": "LUE_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "LUE_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_default_pop_cmaes",
    }

    per_site_lue_exp_names = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_default_pop_cmaes",
    }

    per_site_iav_lue_exp_names = {
        "cmaes_big_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_cost_iav_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_cost_iav_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_cost_iav_default_pop_cmaes",
    }

    per_pft_lue_exp_names = {
        "cmaes_big_pop": "LUE_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "LUE_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_default_pop_cmaes",
    }

    glob_lue_exp_names = {
        "cmaes_big_pop": "LUE_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "LUE_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "LUE_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_default_pop_cmaes",
    }

    per_site_yr_p_exp_names = {
        "cmaes_big_pop": "P_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "P_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "P_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_default_pop_cmaes",
    }

    per_site_p_exp_names = {
        "cmaes_big_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_default_pop_cmaes",
    }

    per_site_iav_p_exp_names = {
        "cmaes_big_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_try_annual_cost_fn",
        "cmaes_default_pop": "P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_cost_iav_default_pop_cmaes",
    }

    per_pft_p_exp_names = {
        "cmaes_big_pop": "P_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "P_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "P_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_default_pop_cmaes",
    }

    glob_p_exp_names = {
        "cmaes_big_pop": "P_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_cmaes",
        "cmaes_big_pop_and_lbfgs_b": "P_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_big_pop_and_lbfgs_b_cmaes",
        "cmaes_default_pop": "P_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_default_pop_cmaes",
    }

    collect_best_results(
        per_site_yr_lue_exp_names,
        opti_type="syr",
        new_exp_name="LUE_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever",
    )
    # collect_best_results(
    #     per_site_lue_exp_names,
    #     opti_type="allyr",
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever",
    # )
    # collect_best_results(
    #     per_site_iav_lue_exp_names,
    #     opti_type="allyr",
    #     new_exp_name="LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_cost_iav_best_opti_ever",
    #     lue_iav=True,
    # )
    # collect_best_results(
    #     per_pft_lue_exp_names,
    #     opti_type="perPFT",
    #     new_exp_name="LUE_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever",
    # )
    # collect_best_results(
    #     glob_lue_exp_names,
    #     opti_type="global",
    #     new_exp_name="LUE_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever",
    # )

    # collect_best_results(
    #     per_site_yr_p_exp_names,
    #     opti_type="syr",
    #     new_exp_name="P_model/site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever",
    # )
    # collect_best_results(
    #     per_site_p_exp_names,
    #     opti_type="allyr",
    #     new_exp_name="P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever",
    # )
    # collect_best_results(
    #     per_site_iav_p_exp_names,
    #     opti_type="allyr",
    #     new_exp_name="P_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_cost_iav_best_opti_ever",
    # )
    # collect_best_results(
    #     per_pft_p_exp_names,
    #     opti_type="perPFT",
    #     new_exp_name="P_model/per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever",
    # )
    # collect_best_results(
    #     glob_p_exp_names,
    #     opti_type="global",
    #     new_exp_name="P_model/global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_best_opti_ever",
    # )
