#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
this script generates sbatch files to optimize each PFT and submits
them to a cluster using SLURM as a job scheduler

author: rde
first created: 2023-12-17
"""

import os
import time
from pathlib import Path
import pandas as pd


ROOT_PATH = Path(__file__).parent  # Get path of the current script
site_info = pd.read_csv(
    ROOT_PATH / "site_info/SiteInfo_BRKsite_list.csv"
)  # get the site list

# get a dictionary with sites per PFT
site_pft_grouped = site_info.groupby("PFT")["SiteID"].apply(list)
sites_per_pft_dict = site_pft_grouped.to_dict()

# get a list of PFTs
pft_list = list(sites_per_pft_dict.keys())

# submit a slurm job for each PFT
for k, v in sites_per_pft_dict.items():
    model_name = "LUE" # "P" # "LUE" 
    # generate a job name based on the PFT
    job_name = f"{k}_{model_name}_opti"

    ##################### EDIT THE FOLLOWING PARAMETERS #####################
    # specify the partition name
    PARTITION_NAME = "work" #ToDo: change according to your cluster
    # specify the number of cores available on the machine where main script will be run
    NO_OF_CORES = 64 #ToDo: change according to your cluster
    # max number of workers based on number of cores
    MAX_NO_OF_WORKERS = 10 if NO_OF_CORES > 10 else NO_OF_CORES
    # set the number of cores based on number of sites in a PFT
    CORE_COUNT = len(v) if len(v) < MAX_NO_OF_WORKERS else MAX_NO_OF_WORKERS
    # specify runtime in based on number of sites in a PFT
    RUN_TIME = "30-2:00:00" if k == "ENF" else "20-2:00:00"
    # specify user email id
    USER_EMAIL = "rde@bgc-jena.mpg.de" #ToDo: change according to your email id
    ##########################################################################

    # get the PFT number and pass
    # it as an argument to the python script
    pft_no = pft_list.index(k) + 1

    # generate the lines for sbatch file
    sbatch_code_list = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}                                # job name",
        f"#SBATCH --partition={PARTITION_NAME}                         # partition name",
        (
            "#SBATCH --nodes=1                                         # nodes requested"
            "(min-max format or single number if min and max are same)"
        ),
        "#SBATCH --ntasks=1                                            # tasks requested",
        (
            f"#SBATCH --cpus-per-task={CORE_COUNT}                     #"
            " cores requested per job"
        ),
        "#SBATCH --mem-per-cpu=5G                                     # memory per core/ CPU",
        "#SBATCH --output=./err_out_files/outfile_%x-%A_%a.log         # send stdout to outfile",
        "#SBATCH --error=./err_out_files/errfile_%x-%A_%a.log          # send stderr to errfile",
        (
            f"#SBATCH --time={RUN_TIME}                                #"
            " max time requested in D-hour:minute:second"
        ),
        f"#SBATCH --mail-user={USER_EMAIL}                             # email address of the user",
        (
            "#SBATCH --mail-type=FAIL                                  # when"
            "to email the user (in case of job fail)"
        ),
        "",
        "# Name of the conda environment and location of conda.sh",
        'ENV_NAME="gen_env_2024"',
        "# if the conda environment is already not active, load the conda environment",
        "if [[ $CONDA_DEFAULT_ENV != $ENV_NAME ]]; then",
        "    # get location of conda.sh",
        (
            "    PATH_TO_CONDA=$(conda info | grep -i 'base environment' | "
            'cut -d ":" -f 2 | cut -d " " -f 2)'
        ),
        '    PATH_TO_CONDA_SCRIPT="${PATH_TO_CONDA}/etc/profile.d/conda.sh"',
        "",
        "    # Initialize conda",
        "    source $PATH_TO_CONDA_SCRIPT ",
        "    # Activate the environment",
        "    conda activate $ENV_NAME",
        "fi",
        "",
        "# Change to directory where the python script is located",
        f"cd {ROOT_PATH}",
        "",
        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}",
        "",
        "# Run the python script",
        f"srun python -u main_opti_and_run_model.py {pft_no}",
    ]

    # create a folder to store the error and output files (if it does not exist)
    os.makedirs(Path("err_out_files"), exist_ok=True)

    # create a directory to store the sbatch files
    sbatch_file_path = Path("sbatch_files")
    os.makedirs(sbatch_file_path, exist_ok=True)
    sbatch_filename = os.path.join(sbatch_file_path, f"{k}_send_slurm_job.sh")

    # save the SBATCH files
    # Open the file in write mode
    with open(sbatch_filename, "w", encoding="utf-8") as file:
        # Write each string as a line of code
        for lines in sbatch_code_list:
            file.write(f"{lines}\n")

    # make the sbatch file executable and submit it to the cluster
    os.system(f"chmod u+x {sbatch_filename}")
    os.system(f"sbatch {sbatch_filename}")

    # wait for some time before submitting the next job
    if k != pft_list[-1]:  # no need to wait after submitting the last job
        time.sleep(10)
