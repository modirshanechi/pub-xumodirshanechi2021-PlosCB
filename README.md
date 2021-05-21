# pub-xumodirshanechi2021-PlosCB

This repository contains the code and data for the results reported in the article:

H.A. Xu*, A. Modirshanechi*, M.P. Lehmann, W. Gerstner**, and M.H. Herzog**, [“Novelty is not Surprise: Human exploratory and adaptive behavior in sequential decision-making”](https://www.biorxiv.org/content/10.1101/2020.09.24.311084v2.full), PLOS Computational Biology (2021)

\*  H.X. and A.M. contributed equally to this work.

** W.G. and M.H.H. contributed equally to this work.

Contact: alireza.modirshanechi@epfl.ch

# Dependencies

* [Julia](https://julialang.org/) (1.3) (to reproduce results of the behavioral modeling)
* [MATLAB](https://ch.mathworks.com/products/matlab.html) (2019a) (to reproduce results of the EEG analysis)

# Usage (Behavioral results)

To install the necessary Julia packages, follow these steps:

1.	Navigate into the “SurNoR_2020” folder.
2.	Open a julia terminal, press "]" to enter the package management mode.
3.	In the package management mode, type “activate .”.
4.	In the package management mode, type “instantiate”.

All Julia packages and dependencies will be installed automatically within this environment.

To reproduce the figures reported in the paper, run the corresponding scripts in the “figure_script” folder. For example, to reproduce panels of Fig 5 and see the corresponding statistics, type “include(“figure_script/Fig5.jl”)”.

# Usage (EEG results)

Navigate into the “SurNoR_2020/ figure_script/” folder. Open MATLAB 2019a. To reproduce the figures reported in the paper, place the corresponding scripts to your MATLAB path and run them. For example, run “Fig_9.m” to reproduce the results for the grand correlation analysis in Fig 9 of the paper.

# Data and source files

* The experimental data are saved as “BehavData_SwitchState.csv” and “EEG_Frontal.csv” in the “data” folder. Read “data/ ReadMe.text” for more information.
* Each script in the “figure_script” folder uses some of the data files saved in the “data” folder (e.g., the fitted parameters or the experimental data) and some of the functions in the “src” folder (e.g., to run a model). To reproduce and save data files in each subfolder of the “data” folder, read “ReadMe.text” in the corresponding subfolder. Functions and source files in the “src” folder are separately documented.
* To reproduce data for Bayesian Model Selection (Fig 5B), you need to install [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) – see [Stephan, et al. (2009)](https://doi.org/10.1016/j.neuroimage.2009.03.025). See “Fig5.jl” in the “figure_script” folder and “ReadMe.text” in the “data/Log_evidences” folder for details.
