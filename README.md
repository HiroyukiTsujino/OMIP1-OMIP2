OMIP1-OMIP2 comparison
========

  This repository contains scripts to process model outputs
  for an omip-1 and omip-2 intercomparison study,
  which is summarized as a paper submitted to
  Geoscientific Model Development (Tsujino et al. 2020).

  Note that this repository provides only scripts to process data.
  Data should be obtained separately, which is available from the following.
  
  * Brown University Digital Repository

    - Model outputs
       https://doi.org/10.26300/g2a0-5x34

    - Analysis & Reference data
       https://doi.org/10.26300/60wh-ak09

    - Note: Use the latest version if multiple versions are present.
            Check <https://climate.mri-jma.go.jp/~htsujino/omip1-omip2.html> for the latest information.


  * MRI server:

    - Model outputs
       https://climate.mri-jma.go.jp/~htsujino/userspace/OMIP1-OMIP2/repos/model_outputs/latest

    - Analysis & Reference data
       https://climate.mri-jma.go.jp/~htsujino/userspace/OMIP1-OMIP2/repos/evaluation/latest


  After obtaining data, make the following directories at the top directory for the data archives
    and place and extract the obtained data appropriately.

  * model    :  Model outputs (provided separately for each contributing model)
  * analysis :  Analysis data (analysis.tgz)
  * refdata  :  Reference data (reference.tgz)


Contents
--------

  * ANALYSIS: Additional processing of model outputs for performance evaluations.

  * DRAW_figs: Drawing figures for presentation.

  * REF_data: Processing observational data for evaluations.

  * python: Utilities of python scripts.

  * utils: Utilities of shell scripts.

  * MRI_data: Sample python scripts to process outputs from MRI.COM.

  * anl: Fortran programs to support analyses.

  * lib: Fortran library used by programs in anl.

  * setting: Settings to compile and run programs in "lib" and "anl" (used upon executing Setup.sh).


Usage Notes
--------

  * See README.md of each directory.


Contact
--------

  * Hiroyuki Tsujino, Shogo Urakawa (JMA-MRI)
