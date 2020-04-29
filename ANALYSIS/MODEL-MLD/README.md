MODEL_MLD
========

  * Performance evaluation of OMIP simulations.
  * Python scripts in this directory require anaconda3.


Contents
-------

  * Calculation of climatology (annual, monthly)

    - $ python ./MODEL_MLD_climatology_np.py omip1 1980 2009 all (or individual model name)
    - $ python ./MODEL_MLD_climatology_np.py omip2 1980 2009 all (or individual model name)


  * Calculation for Taylor diagram

    - $ python ./MODEL_IFREMER_comp_monclim.py omip1 1980 2009 1  # exclude high latitude North Atlantic and marginal seas around Antarctica
    - $ python ./MODEL_IFREMER_comp_monclim.py omip2 1980 2009 1  # exclude high latitude North Atlantic and marginal seas around Antarctica


  * Compute seasonal data (Winter: JFM for NH, JAS for SH, Summer: JAS for NH, JFM for SH)

    - $ python ./MODEL_MLD_seasonal_series_np.py omip1 1980 2009 all
    - $ python ./MODEL_MLD_seasonal_series_np.py omip2 1980 2009 all


  * Compare seasonal data with observation (to check the script "MODEL_MLD_seasonal_series_np.py")

    - $ python ./MODEL_IFREMER_comp_seasonal.py omip1 1980 2009 winter
    - $ python ./MODEL_IFREMER_comp_seasonal.py omip2 1980 2009 winter
    - $ python ./MODEL_IFREMER_comp_seasonal.py omip1 1980 2009 summer
    - $ python ./MODEL_IFREMER_comp_seasonal.py omip2 1980 2009 summer


  * Compare the ensemble spread and the room-mean-square of bias.

    - $ python ./MLD_rmse_std_seasonal.py


  * Compute the significance of the difference between omip-1 and omip-2 based on Wakamatsu et al. (2017).

    - $ python ./MLD_diff_std_seasonal.py


  * Compute winter MLD in the northern North Atlantic and the merginal seas around the Antarctica

    - $ python ./MLD_MOC_comp.py

      - the northern North Atlantic: (45.5-79.5N, 280-390E)
      - the merginal seas around the Antarctica: south of 60S


  * csv: Store summary data used for drawing.

    - samples: sample output used in the submitted paper


Contact
--------

  * Hiroyuki Tsujino (JMA-MRI)
