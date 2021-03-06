MODEL_SSH
========

  * Performance evaluation of OMIP simulations.
  * Python scripts in this directory require anaconda3.


Contents
-------

  * Calculation of climatology (annual, monthly)

    - $ python ./MODEL_SSH_climatology_np.py omip1 1993 2009 all
    - $ python ./MODEL_SSH_climatology_np.py omip2 1993 2009 all


  * Calculation of time series of annual mean

    - $ python ./MODEL_SSH_annual_np.py omip1 1948 2009 all
    - $ python ./MODEL_SSH_annual_np.py omip2 1958 2018 all


  * Calculation for Taylor diagram

    - $ python ./MODEL_CMEMS_comp_annclim.py omip1 1993 2009 yes ! use filtered CMEMS
    - $ python ./MODEL_CMEMS_comp_annclim.py omip2 1993 2009 yes ! use filtered CMEMS

    - $ python ./MODEL_CMEMS_comp_monclim.py omip1 1993 2009 yes ! use filtered CMEMS
    - $ python ./MODEL_CMEMS_comp_monclim.py omip2 1993 2009 yes ! use filtered CMEMS

    - $ python ./MODEL_CMEMS_comp_interannual.py omip1 1993 2009 yes ! use filtered CMEMS
    - $ python ./MODEL_CMEMS_comp_interannual.py omip2 1993 2009 yes ! use filtered CMEMS


  * Calculation of statistical properties

    - $ python ./MODEL_CMEMS_comp_annual_wrt_annclim_rbar.py omip1 1993 2009 yes
    - $ python ./MODEL_CMEMS_comp_annual_wrt_annclim_rbar.py omip2 1993 2009 yes

    - $ python ./MODEL_CMEMS_comp_monthly_wrt_annclim_rbar.py omip1 1993 2009 yes
    - $ python ./MODEL_CMEMS_comp_monthly_wrt_annclim_rbar.py omip2 1993 2009 yes

  * csv: Store summary data used for drawing.

    - samples: sample output used in the submitted paper


  * Compute total uncertainty (standard deviation) of the difference between OMIP-1 and OMIP-2 simulations

    (- $ python ./SSH_diff_std.py)
    - $ python ./SSH_diff_std_med.py (Mediterranean Sea excluded)

  * Compare the ensemble spread and the room-mean-square of bias.

    - $ python ./SSH_rmse_std_med.py (Mediterranean Sea excluded)


Contact
--------

  * Hiroyuki Tsujino (JMA-MRI)
