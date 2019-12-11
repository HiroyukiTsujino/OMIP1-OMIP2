MODEL_MLD
========

  * Performance evaluation of OMIP simulations.
  * Python scripts in this directory require anaconda3.


Contents
-------

  * Calculation of climatology (annual, monthly)

    - $ python ./MODEL_MLD_climatology_np.py omip1 1980 2009 all
    - $ python ./MODEL_MLD_climatology_np.py omip2 1980 2009 all


  * Calculation for Taylor diagram

    - $ python ./MODEL_IFREMER_comp_monclim.py omip1 1980 2009 1  # exclude high latitude North Atlantic
    - $ python ./MODEL_IFREMER_comp_monclim.py omip2 1980 2009 1  # exclude high latitude North Atlantic


* csv: Store summary data used for drawing.

    - samples: sample output used in the submitted paper


Contact
--------

  * Hiroyuki Tsujino (JMA-MRI)
