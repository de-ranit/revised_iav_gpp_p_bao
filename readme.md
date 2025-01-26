# Addressing Challenges in Simulating Inter–annual Variability of Gross Primary Production
<p align="center">
  <img src=https://raw.githubusercontent.com/de-ranit/iav_gpp_p_bao/refs/heads/main/prep_figs/figures/f01.png alt="workflow" width="600">
</p>

<p align="center">
  Created in BioRender. De, R. (2024) <a href=https://BioRender.com/i01x768>https://BioRender.com/i01x768</a>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.13729514">
    <img alt="ZenodoDOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.13729514-blue?logo=Zenodo&logoColor=white&logoSize=auto"
  ></a>

  <a href="https://doi.org/10.22541/essoar.172656939.93739740/v2">
    <img alt="PreprintDOI" src="https://img.shields.io/badge/Preprint_DOI-10.22541%2Fessoar.172656939.93739740%2Fv2-blue"
  ></a>
</p>

# Description
This repository contains codes to perform analysis and reproduce figures of our research paper:

```
De, R., Bao, S., Koirala, S., Brenning, A., Reichstein, M., Tagesson, T., Liddell, M., Ibrom, A., Wolf, S., Šigut, L., Hörtnagl, L., Woodgate, W., Korkiakoski, M., Merbold, L., Black, T. A., Roland, M., Klosterhalfen, A., Blanken, P. D., Knox, S., Sabbatini, S., Gielen, B., Montagnani, L., Fensholt, R., Wohlfahrt, G., Desai, A. R., Paul-Limoges, E., Galvagno, M., Hammerle, A., Jocher, G., Ruiz Reverter, B., Holl, D., Chen, J., Vitale, L., Arain, M. A., and Carvalhais, N. (2025). Addressing Challenges in Simulating Inter–annual Variability of Gross
Primary Production, ESS Open Archive, 1–75, https://doi.org/10.22541/essoar.172656939.93739740/v2
```

We used majorly the following two models in our study. It is highly recommended to get acquainted with the following two research papers before using our codes.

1. Optimality-based model: P-model of Mengoli
```
Mengoli, G., Agustí-Panareda, A., Boussetta, S., Harrison, S. P., Trotta, C., and Prentice, I. C. (2022). Ecosystem photosynthesis in
land-surface models: A first-principles approach incorporating acclimation, Journal of Advances in Modeling Earth Systems, 14,
https://doi.org/10.1029/2021MS002767
```

2. Semi-empirical model: Bao model
```
Bao, S., Wutzler, T., Koirala, S., Cuntz, M., Ibrom, A., Besnard, S., Walther, S., Šigut, L., Moreno, A., Weber, U., Wohlfahrt,695
G., Cleverly, J., Migliavacca, M., Woodgate, W., Merbold, L., Veenendaal, E., and Carvalhais, N. (2022). Environment-sensitivity
functions for gross primary productivity in light use efficiency models, Agricultural and Forest Meteorology, 312, 108 708,
https://doi.org/10.1016/j.agrformet.2021.108708
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
@article{De_IAV_GPP_2025,
  author = {De, R. and Bao, S. and Koirala, S. and Brenning, A. and Reichstein, M. and Tagesson, T. and Liddell, M. and Ibrom, A. and Wolf, S. and Šigut, L. and Hörtnagl, L. and Woodgate, W. and Korkiakoski, M. and Merbold, L. and Black, T. A. and Roland, M. and Klosterhalfen, A. and Blanken, P. D. and Knox, S. and Sabbatini, S. and Gielen, B. and Montagnani, L. and Fensholt, R. and Wohlfahrt, G. and Desai, A. R. and Paul-Limoges, E. and Galvagno, M. and Hammerle, A. and Jocher, G. and Ruiz Reverter, B. and Holl, D. and Chen, J. and Vitale, L. and Arain, M. A. and Carvalhais, N.},
  title = {{Addressing Challenges in Simulating Inter–annual Variability of Gross Primary Production}},
  journal = {{ESS Open Archive}},
  volume = {2025},
  year = {2025},
  pages = {1--75},
  url = {https://essopenarchive.org/doi/full/10.22541/essoar.172656939.93739740/v2},
  doi = {10.22541/essoar.172656939.93739740/v2}
}
```
  - APA
```
De, R., Bao, S., Koirala, S., Brenning, A., Reichstein, M., Tagesson, T., Liddell, M., Ibrom, A., Wolf, S., Šigut, L., Hörtnagl, L., Woodgate, W., Korkiakoski, M., Merbold, L., Black, T. A., Roland, M., Klosterhalfen, A., Blanken, P. D., Knox, S., Sabbatini, S., Gielen, B., Montagnani, L., Fensholt, R., Wohlfahrt, G., Desai, A. R., Paul-Limoges, E., Galvagno, M., Hammerle, A., Jocher, G., Ruiz Reverter, B., Holl, D., Chen, J., Vitale, L., Arain, M. A., and Carvalhais, N. (2025). Addressing Challenges in Simulating Inter–annual Variability of Gross
Primary Production, ESS Open Archive, 1–75, https://doi.org/10.22541/essoar.172656939.93739740/v2
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
De, R. (2024). Scripts for analyses presented in "Addressing challenges in simulating inter–annual variability of gross primary production" (v1.2-preprint). Zenodo. https://doi.org/10.5281/zenodo.13729514
```

# Change Log:
**v1.2-preprint**
- updated upper limit of LUEmax model parameter for Bao model and its variations.
- added an analysis on GPP uncertianty at annual scale.
- added statistical significance testing for model performance across PFTs and Bioclimatic regions.
- updated codes for figures as per reviewers' suggestions.
- updated license.

**v1.1-preprint**
- Updated readme with citations for the preprint and the Zenodo repository.

**v1.0-preprint**
- Initial code for submission to a Zenodo repository and publication of preprint.


# License
[![MIT License][MIT-License-shield]][MIT License]

This work is licensed under a
[MIT License][MIT License].

[MIT License]: https://github.com/de-ranit/iav_gpp_p_bao/blob/main/LICENSE
[MIT-License-shield]: https://img.shields.io/badge/License-MIT-blue
<a href="https://github.com/de-ranit/iav_gpp_p_bao/blob/main/LICENSE">
<img src=https://raw.githubusercontent.com/de-ranit/iav_gpp_p_bao/refs/heads/main/lic_logo/mit_license_logo.png alt="MIT-License-image" width="150"/>
</a>

<span style="font-size:6px;">Created by [ExcaliburZero](https://www.deviantart.com/excaliburzero/art/MIT-License-Logo-595847140), used under [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/)</span>