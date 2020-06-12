SITES-RBAR diagram
========

  * Note:
     - You need to load anaconda3 to run Python scripts in this directory.


Main part
--------

  * Table 3: r-scores of linear fits for model scatters between OMIP-1 and OMIP-2 in some metrics
     - vat_all.py          (Table D1)
     - circ_index.py       (Table D2) ----> Figure 28
        Check contents of "csv_spin", if vacant, you should either
          - run ../Spin_up/FigS1_vat_all.py
          - run ../Spin_up/FigS3_all.py
        or
          - copy contents of "csv_spin/samples" directory to "csv_spin".

     - SST_SSS_bias.py     (Table D3) ----> Figure 27
     - SSH_bias.py         (Table D3)
     - mld_bias_win_sum.py (Table D4)
     - ZMT_bias.py         (Table D5)
     - ZMS_bias.py         (Table D6)
        Check contents of "csv_clim", if vacant, you should either
          - run ../Climatology/SST_bias_allmodels.py 1
          - run ../Climatology/SST_bias_allmodels.py 2
          - run ../Climatology/SSS_bias_allmodels.py 1
          - run ../Climatology/SSS_bias_allmodels.py 2
          - run ../Climatology/SSH_bias_allmodels.py 1
          - run ../Climatology/SSH_bias_allmodels.py 2
          - run ../Climatology/MLD_Winter_bias_allmodels.py 1 1980 2009 1
          - run ../Climatology/MLD_Winter_bias_allmodels.py 2 1980 2009 1
          - run ../Climatology/MLD_Summer_bias_allmodels.py 1 1980 2009 1
          - run ../Climatology/MLD_Summer_bias_allmodels.py 2 1980 2009 1
          - run ../Climatology/ZMT_bias_allmodels.py 1
          - run ../Climatology/ZMT_bias_allmodels.py 2
          - run ../Climatology/ZMS_bias_allmodels.py 1
          - run ../Climatology/ZMS_bias_allmodels.py 2
        or
          - copy contents of "csv_clim/samples" directory to "csv_clim".

     - mld_Lab_Wed.py      (Table D4)
        Check contents of "csv_mld", if vacant, you either
          - run ../../../ANALYSIS/MODEL-MLD/MLD_deep_comp.py
        or
          - copy contents of "csv_mld/samples" directory to "csv_mld".

     - si_extent.py        (Table D7)
        Check contents of "csv_var", if vacant, you either
          - run ../Variability/siextent_all_month.py
        or
          - copy contents of "csv_var/samples" directory to "csv_var".


In appendix
--------

  * Figure E6: sites_rbar_sst.py
     - Check contents of "csv_sst".
       If vacant, you should either
         - trace the entire analyses at ANALYSYS/MODEL-SST directory.
       or
         - copy contents of "csv_sst/samples" directory to simply reproduce Figure 30 of the manuscript.

  * Figure E7: sites_rbar_zos.py
     - Check contents of "csv_zos".
       If vacant, you should either
         - trace the entire analyses at ANALYSYS/MODEL-SSH directory.
       or
         - copy contents of "csv_zos/samples" directory to simply reproduce Figure 31 of the manuscript.


Other Plots
--------

  * mld_circ_nh.py
     - Plot mld in the Lab sea and A-MOC

  * mld_circ_sh.py
     - Plot mld in the Weddell Sea and deep-bottom G-MOC


Contact
-------

  * Hiroyuki Tsujino
