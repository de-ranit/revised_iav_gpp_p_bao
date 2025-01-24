#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot location of FLUXNET sites used in this study on a world map

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Sat Feb 03 2024 16:43:25 CET
"""

import os
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams["font.family"] = 'STIXGeneral'
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def plot_site_loc(site_info_df, remove_sites=None):
    """
    plot location of FLUXNET sites used in this study on a world map
    and color code them based on their PFT and size based on the number
    of years of data available

    Parameters:
    -----------
    site_info_df (pd.DataFrame) : dataframe containing site information
    remove_sites (list) : list of site IDs to be removed from the plot
    (if certain sites were not used because of bad data quality)

    Returns:
    --------
    None
    """

    # rempve bad sites if needed
    if remove_sites is not None:
        site_info_df = site_info_df[~site_info_df["SiteID"].isin(remove_sites)]

    # create a color map for PFTs
    pft_list = site_info_df["PFT"].unique()
    colors = matplotlib.colormaps["tab20"](np.linspace(0, 1, len(pft_list)))  # type: ignore
    pft_color_map = dict(zip(pft_list, colors))
    color_map = ListedColormap([pft_color_map[pft] for pft in pft_list])

    # create the plot
    _, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.Robinson()})

    # set different background color for ocean and land, and add coastlines
    # cite: https://scitools.org.uk/cartopy/docs/latest/citation.html
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), color="#A9F5EE")  # type: ignore
    ax.add_feature(cfeature.LAND.with_scale("50m"), color="#FBFBFB")  # type: ignore
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black")  # type: ignore

    # set global extent
    ax.set_global()  # type: ignore

    # set the marker size based on the number of years of data available
    marker_size_dict = {range(1, 6): 4, range(6, 11): 7, range(11, 50): 10}

    marker_size_list = []
    # for each site, plot the location on the map
    for sites in site_info_df.itertuples():
        marker_size = next(
            (
                size
                for years_range, size in marker_size_dict.items()
                if sites.no_of_years in years_range
            ),
            1,
        )  # define the marker size based on the number of years of data available
        ax.plot(
            sites.Lon,
            sites.Lat,
            "o",
            alpha=0.6,
            markersize=marker_size,
            transform=ccrs.Geodetic(),
            color=pft_color_map[sites.PFT],
            markeredgecolor="none",
        )
        marker_size_list.append(marker_size)

    # add a color bar of PFTs
    norm = BoundaryNorm(np.arange(len(pft_list) + 1) - 0.5, len(pft_list))
    sm = matplotlib.cm.ScalarMappable(cmap=color_map, norm=norm)
    # dummy array for the scalar mappable
    sm._A = []  # pylint: disable=W0212 # type: ignore
    cbar = plt.colorbar(
        sm,
        ticks=np.arange(len(pft_list)),
        orientation="horizontal",
        pad=0.05,
        aspect=50,
        ax=ax,
    )
    cbar.ax.set_xticklabels(pft_list)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label("PFT", fontsize=18, fontweight="bold")

    # add gridlines
    glines = ax.gridlines(  # type: ignore
        draw_labels=True, color="black", alpha=0.5, linestyle="--"
    )
    glines.xlabel_style = {"size": 16}
    glines.ylabel_style = {"size": 16}

    # create legend for marker sizes based on the number of years of data available
    no_of_sites_dict = dict(Counter(marker_size_list))
    leg_labs = [
        f"1 to 5 years ({no_of_sites_dict[4]} sites)",
        f"6 to 10 years ({no_of_sites_dict[7]} sites)",
        rf" $>$ 10 years ({no_of_sites_dict[10]} sites)",
    ]

    legend_handles = [
        Line2D(
            [],
            [],
            markeredgecolor="black",
            color="white",
            marker="o",
            markersize=size,
            linestyle="None",
            label=leg_labs[ix],
        )
        for ix, size in enumerate(marker_size_dict.values())
    ]

    # add the legend to the plot
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        title="Number of available site years ",
        title_fontsize=20,
        fontsize=18,
        frameon=True,
        ncol=3,
        bbox_to_anchor=(0.1, -0.4),
    )

    # add metadata to the pdf of the figure
    metadata = {
        "Title": "FLUXNET site locations",
        "Author": "rde (rde@bgc-jena.mpg.de)",
        "Subject": "Location of FLUXNET sites",
        "Keywords": "FLUXNET2015, EddyCovariance, Site location",
    }

    # save the figure
    fig_path = Path("supplement_figs")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(
        "./supplement_figs/fs04.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        "./supplement_figs/fs04.pdf",
        metadata=metadata,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


if __name__ == "__main__":

    # read the site information from the csv file
    site_info = pd.read_csv("../site_info/SiteInfo_BRKsite_list.csv")

    # plot the site locations on a world map
    plot_site_loc(site_info, ["CG-Tch", "MY-PSO", "GH-Ank", "US-LWW"])
