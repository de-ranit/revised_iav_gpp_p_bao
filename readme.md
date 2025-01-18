[![Static Badge](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13729514-blue?logo=Zenodo&logoColor=white&logoSize=auto)](https://doi.org/10.5281/zenodo.13729514)
[![Static Badge](https://img.shields.io/badge/Preprint_DOI-10.22541%2Fessoar.172656939.93739740%2Fv1-blue)](https://doi.org/10.22541/essoar.172656939.93739740/v1)

# Addressing Challenges in Simulating Inter–annual Variability of Gross Primary Production
![alt text](https://github.com/de-ranit/iav_gpp_p_bao/blob/main/prep_figs/figures/f01.png?raw=true)
Created in BioRender. De, R. (2024) https://BioRender.com/i01x768  

# Description
This repository contains codes to perform analysis and reproduce figures of our research paper:

```
De, R., Bao, S., Koirala, S., Brenning, A., Reichstein, M., Tagesson, T., Liddell, M., Ibrom, A., Wolf, S., Šigut, L., Hörtnagl, L., Woodgate, W., Korkiakoski, M., Merbold, L., Black, T. A., Roland, M., Klosterhalfen, A., Blanken, P. D., Knox, S., Sabbatini, S., Gielen, B., Montagnani, L., Fensholt, R., Wohlfahrt, G., Desai, A. R., Paul-Limoges, E., Galvagno, M., Hammerle, A., Jocher, G., Ruiz Reverter, B., Holl, D., Chen, J., Vitale, L., Arain, M. A., and Carvalhais, N.: Addressing Challenges in Simulating Inter–annual Variability of Gross
Primary Production, ESS Open Archive, 2024, 1–42, https://doi.org/10.22541/essoar.172656939.93739740/v1, 2024.
```

We used majorly the following two models in our study. It is highly recommended to get acquainted with the following two research papers before using our codes.

1. Optimality-based model: P-model of Mengoli
```
Mengoli, G., Agustí-Panareda, A., Boussetta, S., Harrison, S. P., Trotta, C., and Prentice, I. C.: Ecosystem photosynthesis in
land-surface models: A first-principles approach incorporating acclimation, Journal of Advances in Modeling Earth Systems, 14,
https://doi.org/10.1029/2021MS002767, 2022
```

2. Semi-empirical model: Bao model
```
Bao, S., Wutzler, T., Koirala, S., Cuntz, M., Ibrom, A., Besnard, S., Walther, S., Šigut, L., Moreno, A., Weber, U., Wohlfahrt,695
G., Cleverly, J., Migliavacca, M., Woodgate, W., Merbold, L., Veenendaal, E., and Carvalhais, N.: Environment-sensitivity
functions for gross primary productivity in light use efficiency models, Agricultural and Forest Meteorology, 312, 108 708,
https://doi.org/10.1016/j.agrformet.2021.108708, 2022
```

# Disclaimer
The codes are written to be compatible with computing platforms and filestructure of [MPI-BGC, Jena](https://www.bgc-jena.mpg.de/). It maybe necessary to adapt the certain parts of codes to make them compatible with other computing platforms. All the data should be prepared in NetCDF format and variables should be named as per the code. While the actual data used for analysis is not shared in this repository due to large sizes, all the data source are cited in the relevant paper and openly accessible. Corresponding author (Ranit De, [rde@bgc-jena.mpg.de](mailto:rde@bgc-jena.mpg.de) or [de.ranit19@gmail.com](mailto:de.ranit19@gmail.com)) can be contacted in regards to code usage and data preparation. Any usage of codes are sole responsibility of the users.

# Structure 
- `site_info`: This folder contains two `.csv` files: (1) `SiteInfo_BRKsite_list.csv`, this one is necessary so that the code knows data for which all sites are available and can access site specific metadata for preparing results, such as data analysis and grouping of sites according to site characteristics, (2) `site_year_list.csv` lists all the site–years available for site–year specific optimization. This list also contains site–years which are not of good quality, and later gets excluded during data processing steps.
- `src`: This folder basically contains all source codes. It has four folders: (1) `common` folder contains all the scripts which are common for both the Optimality-based (P-model and its variations) and the semi-empirical model (Bao model and its variations), (2) `lue_model` contains model codes and cost function specific to the semi-empirical model (Bao model and its variations), (3) `p_model` contains model codes and cost function specific to the Optimality-based (P-model and its variations), and (4) `postprocess` contains all the scripts to prepare exploratory plots after parameterization and forward runs.
- `prep_figs`: This folder contains all the scripts to reproduce the figures which are presented in our research paper and its supplementary document. All modelling experiments and their relevant data must be available to reproduce the figures and their relative paths should be correctly mentioned at `result_path_coll.py`.

# How to run codes?
- Create a [conda environment and install dependencies](https://docs.conda.io/projects/conda/en/stable/commands/env/create.html). Dependencies are listed in `requirements.yml`.
- Open `model_settings.xlsx` and specify all the experiment parameters from dropdown or by typing as described in the worksheet.
- Run `main_opti_and_run_model.py` (except PFT specific optimization). For PFT specific optimization, run `submit_pft_opti_jobs.py`. If you want parallel processing on a high performance computing (HPC) platform, other settings are necessary based on the platform you are using. PFT specific optimization and global optimization can only be performed using parallel processing on a HPC as multi-site data must be used. See `send_slurm_job.sh` for a sample job submission recipie to a HPC platform using [`slurm`](https://slurm.schedmd.com/overview.html) as a job scheduler.

# How to cite?
**Research paper:**
  - BibTeX
```
@article{De_IAV_GPP_2024,
  author = {De, R. and Bao, S. and Koirala, S. and Brenning, A. and Reichstein, M. and Tagesson, T. and Liddell, M. and Ibrom, A. and Wolf, S. and Šigut, L. and Hörtnagl, L. and Woodgate, W. and Korkiakoski, M. and Merbold, L. and Black, T. A. and Roland, M. and Klosterhalfen, A. and Blanken, P. D. and Knox, S. and Sabbatini, S. and Gielen, B. and Montagnani, L. and Fensholt, R. and Wohlfahrt, G. and Desai, A. R. and Paul-Limoges, E. and Galvagno, M. and Hammerle, A. and Jocher, G. and Ruiz Reverter, B. and Holl, D. and Chen, J. and Vitale, L. and Arain, M. A. and Carvalhais, N.},
  title = {{Addressing Challenges in Simulating Inter–annual Variability of Gross Primary Production}},
  journal = {{ESS Open Archive}},
  volume = {2024},
  year = {2024},
  pages = {1--42},
  url = {https://essopenarchive.org/doi/full/10.22541/essoar.172656939.93739740/v1},
  doi = {10.22541/essoar.172656939.93739740/v1}
}
```
  - APA
```
De, R., Bao, S., Koirala, S., Brenning, A., Reichstein, M., Tagesson, T., Liddell, M., Ibrom, A., Wolf, S., Šigut, L., Hörtnagl, L., Woodgate, W., Korkiakoski, M., Merbold, L., Black, T. A., Roland, M., Klosterhalfen, A., Blanken, P. D., Knox, S., Sabbatini, S., Gielen, B., Montagnani, L., Fensholt, R., Wohlfahrt, G., Desai, A. R., Paul-Limoges, E., Galvagno, M., Hammerle, A., Jocher, G., Ruiz Reverter, B., Holl, D., Chen, J., Vitale, L., Arain, M. A., and Carvalhais, N.: Addressing Challenges in Simulating Inter–annual Variability of Gross
Primary Production, ESS Open Archive, 2024, 1–42, https://doi.org/10.22541/essoar.172656939.93739740/v1, 2024.
```

**This repository:**
  - BibTeX
```
@software{De2024Codes,
  author       = {De, Ranit},
  title        = {{Scripts for analyses presented in ``Addressing challenges in simulating inter–annual variability of gross primary production''}},
  month        = sep,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.2-preprint},
  doi          = {10.5281/zenodo.13729514},
  url          = {https://github.com/de-ranit/iav_gpp_p_bao}
}
```
  - APA
```
De, R. (2024). Scripts for analyses presented in "Addressing challenges in simulating inter–annual variability of gross primary production" (v1.1-preprint). Zenodo. https://doi.org/10.5281/zenodo.13729514
```

# License
[![GNU GPL v3.0][gnu-gpl-v3-shield]][gnu-gpl-v3]

This work is licensed under a
[GNU General Public License v3.0][gnu-gpl-v3].

[![GNU GPL v3.0][gnu-gpl-v3-image]][gnu-gpl-v3]

[gnu-gpl-v3]: https://www.gnu.org/licenses/gpl-3.0.en.html
[gnu-gpl-v3-image]: https://www.gnu.org/graphics/gplv3-127x51.png
[gnu-gpl-v3-shield]: https://img.shields.io/badge/License-GNU_GPL_v3.0-blue
