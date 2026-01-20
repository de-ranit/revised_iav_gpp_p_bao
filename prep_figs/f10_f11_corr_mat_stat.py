#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

author: rde
first created: Fri Dec 05 2025 11:06:52 CET
"""
import os
from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import seaborn as sns

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


# ...existing code...
def plot_iqr_signed_annot(
    iqr,
    p25,
    p75,
    param_names,
    vmax=None,
    figsize=(10, 10),
    min_intensity=0.25,
):
    iqr = np.asarray(iqr, dtype=float)
    p25 = np.asarray(p25, dtype=float)
    p75 = np.asarray(p75, dtype=float)

    # masks for sign categories (only where finite)
    finite_mask = np.isfinite(iqr) & np.isfinite(p25) & np.isfinite(p75)
    pos_pos = finite_mask & (p25 > 0) & (p75 > 0)
    neg_neg = finite_mask & (p25 < 0) & (p75 < 0)
    opp = finite_mask & (p25 * p75 < 0)

    # hide strictly upper triangle
    mask_upper = np.triu(np.ones(iqr.shape, dtype=bool), k=1)

    # intensity normalization (0..1)
    if vmax is None:
        vmax = np.nanmax(iqr)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    norm = Normalize(vmin=0.0, vmax=vmax, clip=True)
    inten = norm(np.nan_to_num(iqr, nan=0.0))

    # base colors requested
    hex_pos = "#CC0000"  # both positive
    hex_opp = "#374057"  # opposite sign
    hex_neg = "#0039A6"  # both negative

    def hex_to_rgb01(h):
        h = h.lstrip("#")
        return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)

    base_pos = hex_to_rgb01(hex_pos)
    base_opp = hex_to_rgb01(hex_opp)
    base_neg = hex_to_rgb01(hex_neg)

    # compute color from base color and intensity but ensure minimum tint (no near-white)
    # weight runs from min_intensity .. 1.0
    weight = min_intensity + (1.0 - min_intensity) * inten  # shape matches iqr
    # color = white*(1-weight) + base_color*weight
    white = np.ones(3, dtype=float)

    img = np.zeros(iqr.shape + (4,), dtype=float)
    img[..., :] = 0.0  # transparent by default

    # apply colors for each mask
    if np.any(pos_pos):
        w = weight[pos_pos][:, None]
        img[pos_pos, :3] = white * (1.0 - w) + base_pos * w
        img[pos_pos, 3] = 1.0
    if np.any(neg_neg):
        w = weight[neg_neg][:, None]
        img[neg_neg, :3] = white * (1.0 - w) + base_neg * w
        img[neg_neg, 3] = 1.0
    if np.any(opp):
        w = weight[opp][:, None]
        img[opp, :3] = white * (1.0 - w) + base_opp * w
        img[opp, 3] = 1.0

    # make upper triangle invisible
    img[mask_upper, 3] = 0.0

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, interpolation="nearest", aspect="equal", origin="upper")

    nrows, ncols = iqr.shape
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(param_names, rotation=90, fontsize=16)
    ax.set_yticklabels(param_names, fontsize=16)

    # annotate visible lower-triangle finite cells with formatted IQR (3 decimals)
    for i in range(nrows):
        for j in range(ncols):
            if mask_upper[i, j]:
                continue
            v = iqr[i, j]
            if not np.isfinite(v):
                continue
            txt = f"{v:.3f}"
            rgba = img[i, j, :3]
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if lum < 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=8)

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    # # colorbar: show intensity (IQR magnitude) as a grayscale bar; hue meaning shown in legend
    # colorbar: create a greys colormap that starts at min_intensity (not white)
    greys = matplotlib.colormaps["Greys"]
    # sample greys at weights between min_intensity..1.0
    sample_weights = np.linspace(min_intensity, 1.0, 256)
    grey_colors = [greys(w) for w in sample_weights]
    custom_greys = ListedColormap(grey_colors, name="custom_greys")

    sm = ScalarMappable(cmap=custom_greys, norm=Normalize(vmin=0.0, vmax=vmax))

    sm.set_array(np.linspace(0, vmax, 256))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("IQR magnitude", fontsize=18)
    cbar.ax.tick_params(labelsize=12)

    # legend for hues
    blue_patch = Patch(
        color=hex_pos,
        label=r"both 25$^{\text{th}}$ \& 75$^{\text{th}}$ percentile values are positive",
    )
    gray_patch = Patch(
        color=hex_opp,
        label=r"25$^{\text{th}}$ \& 75$^{\text{th}}$ percentile values are negative"
        + "\n"
        + r"\& positive, respectively",
    )
    brown_patch = Patch(
        color=hex_neg,
        label=r"both 25$^{\text{th}}$ \& 75$^{\text{th}}$ percentile values are negative",
    )
    ax.legend(
        handles=[blue_patch, gray_patch, brown_patch], loc="upper right", fontsize=14
    )

    sns.despine(ax=ax, top=True, right=True)

    return fig, ax


def calc_corr_stat(exp_path, opti_dict_path):

    corr_mat_file_list = glob.glob(os.path.join(exp_path, "*_nonlinstats.npy"))

    filtered_corr_mat_file_list = [
        files
        for files in corr_mat_file_list
        if not (
            "CG-Tch" in files
            or "MY-PSO" in files
            or "GH-Ank" in files
            or "US-LWW" in files
        )
    ]
    filtered_corr_mat_file_list.sort()

    opti_dict_file_path = f"{opti_dict_path}/DE-Hai_opti_dict.json"
    with open(opti_dict_file_path, "r") as fh:
        opti_dict = json.load(fh)
    all_param_names = opti_dict["opti_param_names"]
    all_param_names = all_param_names + ["alpha"]

    collect_corr_df = []

    for file in filtered_corr_mat_file_list:

        site_id = os.path.basename(file).split("_")[0]

        nonlinstat_dict = np.load(file, allow_pickle=True).item()
        corr_mat = nonlinstat_dict["corr_mat"]

        site_opti_dict_file_path = f"{opti_dict_path}/{site_id}_opti_dict.json"
        with open(site_opti_dict_file_path, "r") as fh:
            site_opti_dict = json.load(fh)
        site_param_names = site_opti_dict["opti_param_names"]

        corr_df = pd.DataFrame(
            corr_mat,
            index=site_param_names,
            columns=site_param_names,
        )
        corr_df_all_param = corr_df.reindex(
            index=all_param_names,
            columns=all_param_names,
        )

        collect_corr_df.append(corr_df_all_param)

    param_label_dict = {
        "acclim_window": "$A_t$",
        "LUE_max": r"$\varepsilon_{max}$",
        "T_opt": r"$T_{opt}$",
        "K_T": r"$k_T$",
        "Kappa_VPD": r"$\kappa$",
        "Ca_0": r"$C_{a0}$",
        "C_Kappa": r"$C_{\kappa}$",
        "c_m": r"$C_m$",
        "gamma_fL_TAL": r"$\gamma$",
        "mu_fCI": r"$\mu$",
        "W_I": r"$W_I$",
        "K_W": r"$K_W$",
        "AWC": r"$AWC$",
        "theta": r"$\theta$",
        "alpha": r"$\alpha$",
        "alphaPT": r"$PET_{scalar}$",
        "meltRate_temp": r"$MR_{tair}$",
        "meltRate_netrad": r"$MR_{netrad}$",
        "sn_a": r"$sn_a$",
        "alpha_fT_Horn": r"$\alpha_{fT}$",
    }

    ###############
    stack_corr_arr = np.stack([df.values for df in collect_corr_df])
    median_corr_mat = np.nanmedian(stack_corr_arr, axis=0)

    median_corr_df = pd.DataFrame(
        median_corr_mat,
        index=all_param_names,
        columns=all_param_names,
    )

    new_param_order = (
        all_param_names[0:3]
        + ["alpha_fT_Horn"]
        + all_param_names[3:11]
        + ["alpha"]
        + all_param_names[11:17]
    )
    median_corr_df = median_corr_df.loc[new_param_order, new_param_order]
    median_corr_mat = median_corr_df.values

    # labels = all_param_names
    labels = [param_label_dict.get(name, name) for name in new_param_order]

    mask = np.triu(np.ones_like(median_corr_mat, dtype=bool), k=1)

    # prepare string annotation matrix to avoid showing "-0.00"
    decimals = 2
    nrows, ncols = median_corr_mat.shape
    annot = np.empty((nrows, ncols), dtype=object)
    for i in range(nrows):
        for j in range(ncols):
            if mask[i, j] or not np.isfinite(median_corr_mat[i, j]):
                annot[i, j] = ""
            else:
                s = f"{median_corr_mat[i, j]:.{decimals}f}"
                if s == f"-0.{''.join('0' for _ in range(decimals))}":
                    s = f"0.{''.join('0' for _ in range(decimals))}"
                annot[i, j] = s

    _, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(
        median_corr_mat,
        ax=ax,
        cmap="BrBG",  # "RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"shrink": 0.8},
        mask=mask,
        annot=annot,
        fmt="",
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label("Median correlation", fontsize=18)
    cbar.ax.tick_params(labelsize=12)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.2)

    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_edgecolor("black")
        ax.spines[sp].set_linewidth(1.2)

    ax.set_xticklabels(labels, rotation=90, fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        "./figures/f10_median_param_correlation_matrix_bao_model.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        "./figures/f10_median_param_correlation_matrix_bao_model.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")

    #############

    #############
    stack_corr_arr = np.stack([df.values for df in collect_corr_df])

    twenty_five_perc_corr_mat = np.nanpercentile(stack_corr_arr, 25, axis=0)
    seventy_five_perc_corr_mat = np.nanpercentile(stack_corr_arr, 75, axis=0)
    iqr_corr_mat = seventy_five_perc_corr_mat - twenty_five_perc_corr_mat

    iqr_corr_df = pd.DataFrame(
        iqr_corr_mat,
        index=all_param_names,
        columns=all_param_names,
    )
    twenty_five_perc_corr_df = pd.DataFrame(
        twenty_five_perc_corr_mat,
        index=all_param_names,
        columns=all_param_names,
    )
    seventy_five_perc_corr_df = pd.DataFrame(
        seventy_five_perc_corr_mat,
        index=all_param_names,
        columns=all_param_names,
    )

    new_param_order = (
        all_param_names[0:3]
        + ["alpha_fT_Horn"]
        + all_param_names[3:11]
        + ["alpha"]
        + all_param_names[11:17]
    )
    iqr_corr_df = iqr_corr_df.loc[new_param_order, new_param_order]
    iqr_corr_mat = iqr_corr_df.values

    twenty_five_perc_corr_df = twenty_five_perc_corr_df.loc[
        new_param_order, new_param_order
    ]
    twenty_five_perc_corr_mat = twenty_five_perc_corr_df.values

    seventy_five_perc_corr_df = seventy_five_perc_corr_df.loc[
        new_param_order, new_param_order
    ]
    seventy_five_perc_corr_mat = seventy_five_perc_corr_df.values

    labels = [param_label_dict.get(name, name) for name in new_param_order]

    _, ax = plot_iqr_signed_annot(
        iqr_corr_mat, twenty_five_perc_corr_mat, seventy_five_perc_corr_mat, labels
    )

    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        "./figures/f11_iqr_param_correlation_matrix_bao_model.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        "./figures/f11_iqr_param_correlation_matrix_bao_model.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")
    #############


if __name__ == "__main__":

    nonlinstat_path = "./jac_nonlinstats/LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_best_opti_ever/nonlinstats_op_dicts/"
    opti_dict_path = "../opti_results/LUE_model/all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_default_pop_cmaes/opti_dicts/"

    calc_corr_stat(nonlinstat_path, opti_dict_path)
