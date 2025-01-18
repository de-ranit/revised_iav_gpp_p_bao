#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
this module gives a dictionary with parameter values

author: rde
first created: 2023-11-07
"""


def get_params(ip_df_dict, p_list=None, p_vec=None):
    """
    generate a dictionary with default model parameter values/
    lower bound, upper bound, initial value when optimized

    parameters:
    ip_df_dict (dict): input dictionary of forcing variables
    p_list (list): list of parameters to be optimized
    p_vec (list): parameter vector given by optimizer

    returns:
    params (dict): dictionary with parameter values
    """
    params = {}

    ################################
    # global parameters for P model
    # ref: https://github.com/geco-bern/rpmodel/blob/master/R/subroutines.R
    # ref: https://github.com/GiuliaMengoli/P-model_subDaily/blob/main/R/pmodelPlus.R
    ################################
    # Parameters to calculate temperature dependence function of quantum efficiency
    params["phi0_coeff_a"] = (1.0/8.0) * 0.352
    params["phi0_coeff_b"] = (1.0/8.0) * 0.022
    params["phi0_coeff_c"] = (1.0/8.0) * 0.00034
    params["kL"] = 0.0065  # mean adiabatic lapse rate (K m-2)
    params["kG"] = 9.80665  # gravitational acceleration (m s-2)
    params["kMa"] = 0.028963  # molecular weight for dry air (kg mol-1)
    params["kR"] = 8.3145  # universal gas constant (J mol-1 K-1)
    params["kPo"] = 101325.0  # Standard atmopsheric pressure at 0 m a.s.l. (Pa)
    params["kTo"] = 298.15  # base temperature (K)
    params["kco"] = 2.09476e5  # O2 partial pressure with standard Atmosphere, i.e., kPo (Pa)
    params["Kc25"] = 39.97  # Michaelis-Menten const carboxylation at 25°C (Pa)
    params["EaKc"] = 79430.0  # Activation energy for carboxylation (J mol-1)
    params["Ko25"] = 27480.0  # Michaelis-Menten constant for oxygenation at 25°C (Pa)
    params["EaKo"] = 36380.0  # Activation energy for oxygenation (J mol-1)
    params["gamma25"] = 4.332  # Photorespiratory compensation point at 25°C (Pa)
    params["EaGamma"] = 37830.0  # Activation energy for photorespiratory compensation (J mol-1)
    params["sea_level_elev"] = 0.0  # sea level elevation (m)
    params["beta"] = 146.0  # the ratio of cost factors for carboxylation and
    # transpiration capacities at 25°C (-)
    params["c"] = 0.41  # the cost factor for electron-transport capacity (unitless)
    params["Ha"] = 65330.0  # Activation energy for Vcmax (J mol-1)
    params["Haj"] = 43900.0 # Activation energy for Jmax (J mol-1)
    params["Rgas"] = 8.314  # Universal gas constant (J mol-1 K-1)

    ########################################
    # parameters to be optimized for P model
    ########################################
    # acclimation window in days in P model
    params["acclim_window"] = {"ini": 18.0, "ub": 100.0, "lb": 1.0}
    # params["scalar_phi0"] = {"ini": 1.0, "ub": 2.0, "lb": 0.00001}

    ######################################################
    # parameters to be optimized for the partial senstivity
    # functions for the robust LUE model
    ######################################################
    # maximum LUE (µmol C/ µmol photons)
    # 1 gc/MJ = 1/(12.0107 * 2.04) µmol C/ µmol photons
    # cosidering 1J = 2.04 µmol photons; 1 gc = (1/12.0107) * 1e6 µmol C
    params["LUE_max"] = {"ini": 0.04, "ub": 0.13, "lb": 0.0}
    params["T_opt"] = {"ini": 10.0, "ub": 35.0, "lb": 5.0}  # optimal temperature (°C)
    params["K_T"] = {
        "ini": 2.0,
        "ub": 20.0,
        "lb": 1.0,
    }  # sensitivity to temperature changes (°C-1)
    # only use optimized alpha values for temperate, continental and polar sites
    # (KG starts with 'C', 'D' or 'E'), else just set alpha to 0
    if ip_df_dict["KG"][0] in ["C", "D", "E"]:
        # lag parameter for temperature effect (dimensionless)
        params["alpha_fT_Horn"] = {"ini": 0.29, "ub": 0.9, "lb": 0.0}
    else:
        params["alpha_fT_Horn"] = 0.0
    # sensitivity to VPD changes (Pa-1)
    # keep the bounds positive; inverse sign in the equation
    params["Kappa_VPD"] = {"ini": 5e-5, "ub": 0.01, "lb": 1e-5}
    # minimum optimal atmospheric CO2 concentration (ppm)
    params["Ca_0"] = {"ini": 380.0, "ub": 390.0, "lb": 340.0}
    # sensitivity to CO2 changes (dimensionless)
    params["C_Kappa"] = {"ini": 0.4, "ub": 10.0, "lb": 0.0}
    # CO2 fertilization intensity indicator (ppm)
    params["c_m"] = {"ini": 2000.0, "ub": 4000.0, "lb": 100.0}
    # light saturation curve indicator (µmol photons-1 m2s)
    # 1  MJ-1 m2 d = (24.0 * 3600.0) / (1e6 * 2.04) µmol photons-1 m2s; 1J = 2.04 µmol photons
    params["gamma_fL_TAL"] = {"ini": 0.002, "ub": 0.05, "lb": 0.0}
    # sensitivity to cloudiness index (dimensionless)
    params["mu_fCI"] = {"ini": 0.5, "ub": 1.0, "lb": 0.001}

    ########################################
    # parameters to be optimized for fW_Horn
    # (used in both P Model & LUE Model)
    ########################################
    # Optimal soil moisture (mm.mm-1)
    params["W_I"] = {"ini": 0.2595647, "ub": 0.99, "lb": 0.01}
    # Sensitivity to soil moisture changes (-)
    params["K_W"] = {
        "ini": 10.77005,
        "ub": 30.0,
        "lb": 5.0,
    }  # keep the bounds positive; inverse sign in the equation
    # lag parameter for soil moisture effect
    # only use optimized alpha values for arid sites (KG starts with 'B'),
    # else just set alpha to 0 (may change later)
    if ip_df_dict["KG"][0] == "B":
        params["alpha"] = {"ini": 0.9899452, "ub": 1.0, "lb": 0.0}
    else:
        params["alpha"] = 0.0

    ##############################
    # parameters for bucket model
    ##############################
    params["nloop_wai_spin"] = 5  # number of loops for spinup at daily scale
    params[
        "nloop_wai_act"
    ] = 1  # number of loops to calculate wai at sub-daily after spinup (when wai0 is stabilized)
    params["ca"] = float(
        ip_df_dict["elev"]
    )  # elevation of site; needed for calculation of psychrometer 'constant' (Gamma)
    # calculation of psychrometer 'constant' (Gamma)
    params["pa"] = 0.001  # specific heat of air in MJ/kg/K
    params["AWC"] = {"ini": 100.0, "ub": 1000.0, "lb": 1.0} # (mm)
    params["theta"] = {"ini": 0.05, "ub": 0.1, "lb": 0.0001} # (mm/hr)
    # params['alphaPT'] = 1.0 #scaler of alpha for ETp
    params["alphaPT"] = {"ini": 1.2, "ub": 5.0, "lb": 0.0}  # scaler of alpha for ETp
    params["meltRate_temp"] = {
        "ini": 0.125,
        "ub": 0.5,
        "lb": 0.0,
    }  # snowmelt rate [mm/°C/hr]
    params["meltRate_netrad"] = {
        "ini": 0.0375,
        "ub": 0.125,
        "lb": 0.0,
    }  # snowmelt rate based on radiation [mm/MJ/hr]
    params["sn_a"] = {
        "ini": 0.44,
        "ub": 3.0,
        "lb": 0.0,
    }  # sublimation resistance [-] 0.44

    ########################################################################################
    # replace default parameter values with values given by optimizer
    if (p_list is not None) and (p_vec is not None):
        for pn, pv in enumerate(p_list):
            # in PFT optimization, alpha may be optimized, but
            # if alpha or alpha_fT_Horn are not applicable to a site,
            # then alpha or alpha_fT_Horn is set to 0.0 and can't be substituted
            # by the optimized value
            if (
                (pv == "alpha")
                and isinstance(params[pv], float)
                or (pv == "alpha_fT_Horn")
                and isinstance(params[pv], float)
            ):
                params[pv] = params[pv]
            else:
                p_value = p_vec[pn] * params[pv]["ini"]
                params[pv] = p_value

    return params
