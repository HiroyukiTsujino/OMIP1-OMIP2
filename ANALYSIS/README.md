ANALYSIS
========

  * Preprocessing before evaluating performance of simulations.


First thing to do
--------

  * See README.md in the top directory for how to obtain and place the datasets.

  * Make symbolic links to data archive.
    Use sample/make_symbolic_link.sh if you like.

    - model    ---> ${Where_Archive_is_Extracted}/model
    - analysis ---> ${Where_Archive_is_Extracted}/analysis
    - refdata  ---> ${Where_Archive_is_Extracted}/refdata


SST analysis
--------

  * AMIP-SST directory

    - Compute climatology and time series of annual mean for PCMDI-SST.

  * MODEL_SST directory

    - Process results of OMIP simulations.


SSH analysis
--------

  * CMEMS-SSH directory

    - Compute climatology and time series of annual mean for CMEMS-SSH.

  * MODEL_SSH directory

    - Process results of OMIP simulations.


MLD analysis
--------

  * IFREMER-MLD

    - Compute climatology of deBoyer et al. (2004) MLD.

  * MODEL_MLD directory

    - Process results of OMIP simulations.

SSS analysis
--------

  * MODEL-SSS

    - Process results of OMIP simulations.
  

Zonal mean analysis
--------

  * MODEL-ZM

    - Process results of OMIP simulations.
  

MOC analysis
--------

  * MODEL-MOC

    - Process results of OMIP simulations.
  

Wind stress analysis
--------

  * WIND_STRESS

    - Process results of OMIP simulations.



Contacts
--------

  * Hiroyuki Tsujino (JMA-MRI)
