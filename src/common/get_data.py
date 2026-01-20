#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this module reads the forcing data from netCDF files and converts them for further use accordingly

author: rde, skoirala
first created: 2023-11-07
"""
import sys
import os
import glob

import pandas as pd
import numpy as np
import xarray as xr


def is_hourly_site(site_name):
    """
    to check if the current site is from one of the sites, where only hourly data is available

    parameters:
    site_name (str): name of the site

    returns:
    True/False (bool): True if the site is one of the hourly sites, else False
    """

    hourly_site_name_list = [
        "AU-Tum",
        "BR-Sa1",
        "NO-Blv",
        "US-Cop",
        "US-Ha1",
        "US-MMS",
        "US-Ne1",
        "US-Ne2",
        "US-Ne3",
        "US-PFa",
        "US-UMB",
    ]
    return site_name in hourly_site_name_list


def get_time_info(site_name, temp_res):
    """
    get time info (no. of time steps in a day, start and end index of noon data)
    based on temporal resolution of input data and site name

    parameters:
    site_name (str): name of the site
    temp_res (str): temporal resolution of input data

    returns:
    t_i (dict): dictionary with time info

    """
    t_i = {}
    if temp_res == "HalfHourly":
        if is_hourly_site(site_name):  # for sites with only hourly data
            nstepsday = 24  # for hourly sites, use 24 time steps per day
            start_hour_ind = 11  # start index of noon data (11:00 hrs.)
            end_hour_ind = 14  # end index of noon data (13:00 hrs.)
        else:  # for all other sites in half-hourly resolution
            nstepsday = 48  # for half-hourly sites, use 48 time steps per day
            start_hour_ind = 23  # start index of noon data (11:30 hrs.)
            end_hour_ind = 26  # end index of noon data (12:30 hrs.)
    elif temp_res == "Hourly":  # when running the model for hourly timestep
        nstepsday = 24  # for hourly sites, use 24 time steps per day
        start_hour_ind = 11  # start index of noon data (11:00 hrs.)
        end_hour_ind = 14  # end index of noon data (13:00 hrs.)
    elif temp_res == "Daily":
        nstepsday = 1  # for daily sites, use 1 time step per day
        start_hour_ind = 0  # ToDo: check this indices for implementing daily in P-model
        end_hour_ind = 0
    else:
        raise ValueError(
            f"temporal resolution {temp_res} not supported. Please use HalfHourly, Hourly or Daily."
        )

    t_i["nstepsday"] = nstepsday
    t_i["start_hour_ind"] = start_hour_ind
    t_i["end_hour_ind"] = end_hour_ind

    return t_i


def read_nc_data(infile, settings_dict):
    """
    read input data from netcdf file and collect required variables in a dictionary
    set physically unrealistic negative values to 0

    parameters:
    infile (str): path to input netcdf file and file name
    settings_dict (dict): dictionary with settings

    returns:
    ip_df_dict (dict): dictionary with input forcing data

    """
    ip_ds = xr.open_dataset(infile)

    ip_df_dict = {}
    ip_df_dict["Time"] = ip_ds["time"].values
    for _var in ip_ds.data_vars:
        ip_df_dict[_var] = ip_ds[_var].values.reshape(-1)

    # append required data from global attributes to the ip data dictionary
    ip_df_dict["Iabs"] = (
        ip_df_dict["PPFD_IN_GF"] * ip_df_dict[settings_dict["fPAR_var"]]
    )  # micromol photon/m2/s
    ip_df_dict["SiteID"] = ip_ds.attrs["Site_id"]
    ip_df_dict["Temp_res"] = ip_ds.attrs["temporal_resolution"]
    ip_df_dict["elev"] = float(ip_ds.attrs["Elevation"])
    ip_df_dict["KG"] = ip_ds.attrs["KG"]
    ip_df_dict["PFT"] = ip_ds.attrs["PFT"]
    ip_df_dict["prec_unit"] = ip_ds["P_GF"].unit

    ### fill missing values/ replace non-realistic values
    # ip_df_dict["PET"] = np.nan_to_num(
    #     ip_df_dict["PET"], nan=0.0
    # )  # fill gaps in ETp with 0 (should not be case anymore as netrad and tair are gap filled)
    ip_df_dict["PET"] = np.clip(
        ip_df_dict["PET"], a_min=0.0, a_max=None
    )  # fill negative ETp values with 0
    # ip_df_dict["Prec"] = ip_df_dict["Prec"].fillna(
    #     0.0
    # )  # fill gaps in Prec with 0 (should not be case anymore as P_GF will be used)
    ip_df_dict["NETRAD_GF"] = np.clip(
        ip_df_dict["NETRAD_GF"], a_min=0.0, a_max=None
    )  # fill negative Rn values with 0
    ip_df_dict["ET"] = np.clip(
        ip_df_dict["ET"], a_min=0.0, a_max=None
    )  # fill negative ET values with 0
    try:
        ip_df_dict["ET_CORR"] = np.clip(
            ip_df_dict["ET_CORR"], a_min=0.0, a_max=None
        )  # fill negative ET values with 0
    except KeyError:
        pass

    return ip_df_dict


def df_to_dict(df):
    """
    convert input pandas dataframe to dictionary with column name as keys
    and column values as numpy arrays

    parameters:
    df (pandas dataframe): input dataframe

    returns:
    data_dict (dict): dictionary with input data

    """
    data_dict = {}
    dfnames = df.head()
    for _df in dfnames:
        _dat = df[_df].to_numpy()
        data_dict[_df] = _dat
    return data_dict


def get_daily_data_for_wai(sub_daily_dict, et_var_name="ET"):
    """
    select varibales needed for calculation of WAI and resample
    them to daily timescale (for faster WAI spinup)

    parameters:
    sub_daily_dict (dict): dictionary with input forcing data at sub-daily scale

    returns:
    sub_daily_f (pandas dataframe): dataframe with input forcing data at daily scale

    """
    varibs_dict = {
        "Time": sub_daily_dict["Time"],
        "TA_GF": sub_daily_dict["TA_GF"],
        "PET": sub_daily_dict["PET"],
        "NETRAD_GF": sub_daily_dict["NETRAD_GF"],
        "P_GF": sub_daily_dict["P_GF"],
        "ET": sub_daily_dict[et_var_name],
        "ET_RANDUNC": sub_daily_dict["LE_RANDUNC"],
        "LE_QC": sub_daily_dict["LE_QC"],
    }  # variables needed for calculation of WAI

    sub_daily_f = pd.DataFrame(varibs_dict)

    # resample subset dataframe to daily scale
    sub_daily_f.set_index("Time", inplace=True)
    sub_daily_f.index = pd.to_datetime(sub_daily_f.index)

    # resample to daily scale by taking sum for some variables and mean for others
    # return sub_daily_f.resample("D").agg(
    #     {
    #         "TA_GF": "mean",
    #         "PET": "sum",
    #         "NETRAD_GF": "mean",
    #         "P_GF": "sum",
    #         "ET": "sum",
    #         "ET_RANDUNC": "sum",
    #         "LE_QC": "mean",
    #     }
    # )

    # resample to daily scale by taking mean for all variables
    return sub_daily_f.resample("D").mean()


def prep_wai_output(t_size):
    """
    prepare dictionary to store WAI output

    parameters:
    t_size (int): size of time dimension (total no. of timesteps) of input data

    returns:
    wai_output (dict): dictionary to store WAI output
    """
    snomelt_timeseries = np.zeros(t_size)
    etsub_timeseries = np.zeros_like(snomelt_timeseries)
    pu_timeseries = np.zeros_like(snomelt_timeseries)
    sno_timeseries = np.zeros_like(snomelt_timeseries)
    wai_timeseries = np.zeros_like(snomelt_timeseries)
    et_timeseries = np.zeros_like(snomelt_timeseries)
    w_timeseries = np.zeros_like(snomelt_timeseries)
    wai_output = {}
    wai_output["snomelt"] = snomelt_timeseries
    wai_output["etsub"] = etsub_timeseries
    wai_output["pu"] = pu_timeseries
    wai_output["sno"] = sno_timeseries
    wai_output["wai"] = wai_timeseries
    wai_output["wai_nor"] = w_timeseries
    wai_output["et"] = et_timeseries
    wai_output["etsno"] = wai_output["et"] + wai_output["etsub"]
    return wai_output


def get_data(site_name, settings_dict):
    """
    prepare input forcing data, input data for daily WAI calculation
    during spinup, dictionary to store WAI output and dictionary with time info

    parameters:
    site_name (str): name of the site
    settings_dict (dict): dictionary with settings

    returns:
    ip_df_dict (dict): dictionary with input forcing data
    ip_df_daily_wai (dict): dictionary with input forcing data for daily WAI calculation
    wai_output (dict): dictionary to store WAI output
    time_info (dict): dictionary with time info
    """

    # read input data
    forcing_data_filename = glob.glob(
        os.path.join(settings_dict["ip_data_path"], f"{site_name}.*.nc")
    )[0]
    ip_df_dict = read_nc_data(forcing_data_filename, settings_dict)

    if (ip_df_dict["Temp_res"] == "Daily") and (
        settings_dict["model_name"] == "P_model"
    ):
        sys.exit(
            (
                f"Model {settings_dict['model_name']} cannot be"
                "run with daily data. use HalfHourly or Hourly data"
            )
        )

    # determine time steps to be used for getting noon data
    time_info = get_time_info(site_name, ip_df_dict["Temp_res"])

    ip_df_daily_wai = df_to_dict(
        get_daily_data_for_wai(ip_df_dict, et_var_name=settings_dict["et_var_name"])
    )  # get varibales for WAI calculation at daily timestep
    # in case of daily data, it will have same values of variables from as ip_df_dict
    wai_output = prep_wai_output(
        ip_df_dict["Time"].size
    )  # get dictionary to store WAI outputs arrays

    return ip_df_dict, ip_df_daily_wai, wai_output, time_info
