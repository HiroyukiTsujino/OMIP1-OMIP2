MODEL_SST
========

  * Additional analyses to evaluate performance of OMIP simulations.
  * Python scripts in this directory require anaconda3.

Contents
-------

  * Calculation of climatology (annual, monthly)

    - $ python ./MODEL_SST_climatology_np.py omip1 1980 2009 all (or modelname)
    - $ python ./MODEL_SST_climatology_np.py omip2 1980 2009 all (or modelname)


  * Calculation of time series of annual mean

    - $ python ./MODEL_SST_annual_np.py omip1 1948 2009 all (or modelname)
    - $ python ./MODEL_SST_annual_np.py omip2 1958 2018 all (or modelname)


  * Calculation for Taylor diagram

    - $ python ./MODEL_AMIP_comp_annclim.py omip1 1980 2009
    - $ python ./MODEL_AMIP_comp_annclim.py omip2 1980 2009

    - $ python ./MODEL_AMIP_comp_monclim.py omip1 1980 2009
    - $ python ./MODEL_AMIP_comp_monclim.py omip2 1980 2009

    - $ python ./MODEL_AMIP_comp_interannual.py omip1 1980 2009
    - $ python ./MODEL_AMIP_comp_interannual.py omip2 1980 2009


  * Calculation of stasistical properties

    - $ python ./MODEL_AMIP_comp_annual_wrt_annclim_rbar.py omip1 1980 2009
    - $ python ./MODEL_AMIP_comp_annual_wrt_annclim_rbar.py omip2 1980 2009

    - $ python ./MODEL_AMIP_comp_monthly_wrt_annclim_rbar.py omip1 1980 2009
    - $ python ./MODEL_AMIP_comp_monthly_wrt_annclim_rbar.py omip2 1980 2009


  * csv: Store summary data used for drawing.
    - samples: sample output used for the submitted paper


Contact
--------

  * Hiroyuki Tsujino (JMA-MRI)
