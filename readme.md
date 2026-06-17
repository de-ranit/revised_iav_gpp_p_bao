# Addressing Challenges in Simulating Inter–Annual Variability of Gross Primary Production 
(**Revised** for the publication: *Inter–Annual Variability of Model Parameters Improves Simulation of Annual Gross Primary Production; https://doi.org/10.1029/2025MS005116*)
<p align="center">
  <img src=https://raw.githubusercontent.com/de-ranit/iav_gpp_p_bao/refs/heads/main/prep_figs/figures/f01.png alt="workflow" width="600">
</p>

<p align="center">
  Created in BioRender. De, R. (2025) <a href=https://BioRender.com/i01x768>https://BioRender.com/i01x768</a>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.18326239">
    <img alt="ZenodoDOI" src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18326239-blue?logo=Zenodo&logoColor=white&logoSize=auto"
  ></a>

  <a href="https://doi.org/10.1029/2025MS005116">
    <img alt="ArticleDOI" src="https://img.shields.io/badge/Article_DOI-10.1029/2025MS005116-blue"
  ></a>

  <a href="https://doi.org/10.1029/2024MS004697">
    <img alt="ArticleDOI" src="https://img.shields.io/badge/Article_DOI-10.1029%2F2024MS004697-blue"
  ></a>
</p>

# Description
This repository contains **revised codes** of De et al. (2025, https://doi.org/10.1029/2024MS004697) to perform analysis and reproduce figures of our research paper:
> De, R., Brenning, A., Reichstein, M., Šigut, L., Ruiz Reverter, B., Korkiakoski, M., Paul-Limoges, E., Blanken, P. D., Black, T. A., Gielen, B., Tagesson, T., Wohlfahrt, G., Montagnani, L., Wolf, S., Chen, J., Liddell, M., Desai, A. R., Koirala, S. and Carvalhais, N. (2026). Inter–Annual Variability of Model Parameters Improves Simulation of Annual Gross Primary Production. *Journal of Advances in Modeling Earth Systems*, 18(6), e2025MS005116. https://doi.org/10.1029/2025MS005116


We used majorly the following two models in our study. It is highly recommended to get acquainted with the following two research papers before using our codes.

1. Optimality-based model: P-model of Mengoli
> Mengoli, G., Agustí-Panareda, A., Boussetta, S., Harrison, S. P., Trotta, C., and Prentice, I. C. (2022). Ecosystem Photosynthesis in Land-Surface Models: A First-Principles Approach Incorporating Acclimation. *Journal of Advances in Modeling Earth Systems*, 14(1), e2021MS002767. https://doi.org/10.1029/2021MS002767


2. Semi-empirical model: Bao model
> Bao, S., Wutzler, T., Koirala, S., Cuntz, M., Ibrom, A., Besnard, S., Walther, S., Šigut, L., Moreno, A., Weber, U., Wohlfahrt, G., Cleverly, J., Migliavacca, M., Woodgate, W., Merbold, L., Veenendaal, E., and Carvalhais, N. (2022). Environment-sensitivity functions for gross primary productivity in light use efficiency models. *Agricultural and Forest Meteorology*, 312, 108708. https://doi.org/10.1016/j.agrformet.2021.108708



# Disclaimer
The codes are written to be compatible with computing platforms and filestructure of [MPI-BGC, Jena](https://www.bgc-jena.mpg.de/) and [MPCDF](https://www.mpcdf.mpg.de/). It maybe necessary to adapt the certain parts of codes to make them compatible with other computing platforms. All the data should be prepared in NetCDF format and variables should be named as per the code. While the actual data used for analysis is not shared in this repository due to large sizes, all the data source are cited in the relevant paper and openly accessible. Corresponding author (Ranit De, [rde@bgc-jena.mpg.de](mailto:rde@bgc-jena.mpg.de) or [de.ranit19@gmail.com](mailto:de.ranit19@gmail.com)) can be contacted in regards to code usage and data preparation. Any usage of codes are sole responsibility of the users.


# Structure 
- `site_info`: This folder contains two `.csv` files: (1) `SiteInfo_BRKsite_list.csv`, this one is necessary so that the code knows data for which all sites are available and can access site specific metadata for preparing results, such as data analysis and grouping of sites according to site characteristics, (2) `site_year_list.csv` lists all the site–years available for site–year specific optimization. This list also contains site–years which are not of good quality, and later gets excluded during data processing steps.
- `src`: This folder basically contains all source codes. It has four folders: (1) `common` folder contains all the scripts which are common for both the Optimality-based (P-model and its variations) and the semi-empirical model (Bao model and its variations), (2) `lue_model` contains model codes and cost function specific to the semi-empirical model (Bao model and its variations), (3) `p_model` contains model codes and cost function specific to the Optimality-based (P-model and its variations), and (4) `postprocess` contains all the scripts to prepare exploratory plots after parameterization and forward runs.
- `opti_lbfgs_b`: This folder contains the code to further constrain model parameters obtained from CMA-ES (with a big population size) by using a gradient-based optimizer (L-BFGS-B).
- `prep_figs`: This folder contains all the scripts to reproduce the figures which are presented in our research paper and its supplementary document. All modelling experiments and their relevant data must be available to reproduce the figures and their relative paths should be correctly mentioned at `result_path_coll.py`.


# How to run codes?
- Create a [conda environment and install dependencies](https://docs.conda.io/projects/conda/en/stable/commands/env/create.html). Dependencies are listed in `requirements.yml`.
- Open `model_settings.xlsx` and specify all the experiment parameters from dropdown or by typing as described in the worksheet.
- Run `main_opti_and_run_model.py` (except PFT specific optimization). For PFT specific optimization, run `submit_pft_opti_jobs.py`. If you want parallel processing on a high performance computing (HPC) platform, other settings are necessary based on the platform you are using. PFT specific optimization and global optimization can only be performed using parallel processing on a HPC as multi-site data must be used. See `send_slurm_job.sh` for a sample job submission recipie to a HPC platform using [`slurm`](https://slurm.schedmd.com/overview.html) as a job scheduler.
- The codes under `opti_lbfgs_b` can be used to further constrain model parameters obtained from CMA-ES (with a big population size) by using a gradient-based optimizer (L-BFGS-B) for per site-year or per site optimization. For per PFT and global optimization, the required settings must be selected in `model_settings.xlsx`.

# How to cite?
**Research paper:**
  - BibTeX
```
@article{De_2026_paramval,
author = {De, Ranit and Brenning, Alexander and Reichstein, Markus and Šigut, Ladislav and Reverter, Borja Ruiz and Korkiakoski, Mika and Paul-Limoges, Eugénie and Blanken, Peter D. and Black, T. Andrew and Gielen, Bert and Tagesson, Torbern and Wohlfahrt, Georg and Montagnani, Leonardo and Wolf, Sebastian and Chen, Jiquan and Liddell, Michael and Desai, Ankur R. and Koirala, Sujan and Carvalhais, Nuno},
title = {{Inter–Annual Variability of Model Parameters Improves Simulation of Annual Gross Primary Production}},
journal = {Journal of Advances in Modeling Earth Systems},
volume = {18},
number = {6},
pages = {e2025MS005116},
doi = {10.1029/2025MS005116},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2025MS005116},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2025MS005116},
note = {e2025MS005116 2025MS005116},
month = {jun},
year = {2026}
}
```
  - APA
> De, R., Brenning, A., Reichstein, M., Šigut, L., Ruiz Reverter, B., Korkiakoski, M., Paul-Limoges, E., Blanken, P. D., Black, T. A., Gielen, B., Tagesson, T., Wohlfahrt, G., Montagnani, L., Wolf, S., Chen, J., Liddell, M., Desai, A. R., Koirala, S. and Carvalhais, N. (2026). Inter–Annual Variability of Model Parameters Improves Simulation of Annual Gross Primary Production. *Journal of Advances in Modeling Earth Systems*, 18(6), e2025MS005116. https://doi.org/10.1029/2025MS005116

**This repository:**
  - BibTeX
```
@software{de2026codes_gpp_iav,
  author       = {De, Ranit},
  title        = {{Revised scripts of analyses presented in ``Addressing challenges in simulating inter–Annual variability of gross primary production''}},
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.2-published},
  doi          = {10.5281/zenodo.18326239},
  url          = {https://github.com/de-ranit/revised_iav_gpp_p_bao}
}
```
  - APA
> De, R. (2026). Revised scripts of analyses presented in ``Addressing challenges in simulating inter–Annual variability of gross primary production'' (v1.2-published). *Zenodo*. https://doi.org/10.5281/zenodo.18326239

# Change Log:
**v1.2-published**
- updated readme with correct references after publication of our article
- no changes in actual code

**v1.1-preprint**
- Dynamic variable name for ET to use either ET or ET_CORR (energy-balance corrected)
- Usage of strict data filtering
- Constraining model parameters with L-BFGS-B
- Running CMA-ES with default lower population size
- Parameter correlation in Bao model
- Further analyses presented in https://doi.org/10.1029/2025MS005116

**v1.3-published (https://github.com/de-ranit/iav_gpp_p_bao)**
- Codes for the analyses presented in the Version of Record of https://doi.org/10.1029/2024MS004697

# License
[![MIT License][MIT-License-shield]][MIT License]

This work is licensed under a
[MIT License][MIT License].

[MIT License]: https://github.com/de-ranit/revised_iav_gpp_p_bao/blob/main/LICENSE
[MIT-License-shield]: https://img.shields.io/badge/License-MIT-blue
<a href="https://github.com/de-ranit/revised_iav_gpp_p_bao/blob/main/LICENSE">
<img src=https://raw.githubusercontent.com/de-ranit/revised_iav_gpp_p_bao/refs/heads/main/lic_logo/mit_license_logo.png alt="MIT-License-image" width="150"/>
</a>

<span style="font-size:6px;">License logo is created by [ExcaliburZero](https://www.deviantart.com/excaliburzero/art/MIT-License-Logo-595847140), used under [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/)</span>