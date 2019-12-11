ANALYSIS
========

  * Preprocessing before evaluating performance of simulations.


First thing to do
--------

  * Make symbolic link to top directories of data archive.
    Use sample/make_symbolic_link.sh if you like.

    - model    ---> ${Where_Archive_is_Extracted}/model
    - analysis ---> ${Where_Archive_is_Extracted}/analysis
    - refdata  ---> ${Where_Archive_is_Extracted}/refdata


SST anaylsis
--------

  * AMIP-SST directory

    - Compute climatology and time sereis of annual mean for PCMDI-SST.

  * MODEL_SST directory

    - Process results of OMIP simulations.


SSH anaylsis
--------

  * CMEMS-SSH directory

    - Compute climatology and time sereis of annual mean for CMEMS-SSH.

  * MODEL_SSH directory

    - Process results of OMIP simulations.


MLD anaylsis
--------

  * IFREMER-MLD

    - Compute climatology of deBoyer et al. (2004) MLD.

  * MODEL_MLD directory

    - Process results of OMIP simulations.


Contacts
--------

  * Hiroyuki Tsujino (JMA-MRI)
