#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare the differences between fT used by Bao et al. (2022)
and this study

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Tue Jun 18 2024 14:51:00 CEST
"""

import os
from pathlib import Path
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams["font.family"] = 'STIXGeneral'
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2  # make the axes edge linewidth thicker


def f_temp_horn(temp, t_opt, k_t, alpha_ft):
    """
    calculate partial sensitivity function for temperature
    as of this study

    parameters:
    temp (array): temperature timeseries (degC)
    t_opt (float): parameter to calculate fT
                   (Optimal temperature in degC)
    k_t (float): parameter to calculate fT
                    (Sensitivity to temperature changes; degC-1)
    alpha_ft (float): lag parameter to calculate fT (dimensionless)

    returns:
    ft_horn_this_study (array): partial sensitivity function values for temperature
    """

    lag_step = 1  # number of previous timestep to be considered for the lag function

    # initialize t_f array
    t_f = np.zeros_like(temp)
    # calculate t_f
    for idx, tair in enumerate(temp):
        if idx == 0:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * tair
        else:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * t_f[idx - lag_step]

    # calculate ft_Horn
    # the scalar in numerator has been changed from 4 to 2 and
    # in denominator e(x**2) has been changed to (e(x))**2 to make the fT values between 0 and 1
    ft_eval_exp_num = -(t_f - t_opt) / k_t  # pylint: disable=unused-variable
    ft_eval_exp_deno = -(t_f - t_opt) / k_t  # pylint: disable=unused-variable
    ft_horn_this_study = (2.0 * ne.evaluate("exp(ft_eval_exp_num)")) / (
        1.0 + (ne.evaluate("exp(ft_eval_exp_deno)")) ** 2.0
    )

    return ft_horn_this_study

def f_temp_horn_old(temp, t_opt, k_t, alpha_ft):
    """
    calculate partial sensitivity function for temperature
    as of Bao et al. (2022)

    parameters:
    temp (array): temperature timeseries (degC)
    t_opt (float): parameter to calculate fT
                   (Optimal temperature in degC)
    k_t (float): parameter to calculate fT
                    (Sensitivity to temperature changes; degC-1)
    alpha_ft (float): lag parameter to calculate fT (dimensionless)

    returns:
    ft_horn_bao (array): partial sensitivity function values for temperature
    """

    lag_step = 1  # number of previous timestep to be considered for the lag function

    # initialize t_f array
    t_f = np.zeros_like(temp)
    # calculate t_f
    for idx, tair in enumerate(temp):
        if idx == 0:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * tair
        else:
            t_f[idx] = (1.0 - alpha_ft) * tair + alpha_ft * t_f[idx - lag_step]

    # calculate ft_Horn
    # the scalar in numerator has been changed from 4 to 2 and
    # in denominator e(x**2) has been changed to (e(x))**2 to make the fT values between 0 and 1
    ft_eval_exp_num = -(t_f - t_opt) / k_t  # pylint: disable=unused-variable
    ft_eval_exp_deno = (-(t_f - t_opt) / k_t)**2.0  # pylint: disable=unused-variable
    ft_horn_bao = (2.0 * ne.evaluate("exp(ft_eval_exp_num)")) / (
        1.0 + (ne.evaluate("exp(ft_eval_exp_deno)"))
    )

    return ft_horn_bao

if __name__ == "__main__":

    # create a temperature timeseries
    # and fT parameters
    t_air = np.linspace(-5.0, 40.0, 24 * 365)
    T_OPT_VAL = 10.0
    K_T_VAL = 2.0
    ALPHA_FT_VAL = 0.29

    ft_horn_old = f_temp_horn_old(t_air, T_OPT_VAL, K_T_VAL, ALPHA_FT_VAL)
    ft_horn = f_temp_horn(t_air, T_OPT_VAL, K_T_VAL, ALPHA_FT_VAL)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(t_air, ft_horn, label=r"$fT$ as used" + "\n" + "in this study")
    ax.plot(t_air, ft_horn_old, label=r"$fT$ from Bao" + "\n" + "et al. (2022, 2023)")

    ax.set_xlabel(r"Temperature [$^\circ$C]", fontdict={"size": 30})
    ax.set_ylabel(r"$fT$ [-]", fontdict={"size": 30})

    ax.legend(fontsize=22, loc="upper right")

    ax.tick_params(axis="both", which="major", labelsize=28)

    sns.despine(ax=ax, top=True, right=True)

    # save the figure
    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./supplement_figs/fs03.png", dpi=300, bbox_inches="tight")
    plt.savefig("./supplement_figs/fs03.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")
