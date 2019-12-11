WOA13v2/grid
========

  Generate vertical grid files to process WOA13v2 using libMXE/fortran.


Contents
--------

  * src 
    - dz_ctl.F90: Generate vertical grid width file for libMXE

  * script

    * Original 102 layer
       - make_dz_woa13_L101.sh : Runs dz_ctl.
       - NAMELIST.MXE.WOA13v2_1x1_L102

    * Standard 33 layer
       - NAMELIST.MXE.WOA13v2_1x1_L33
       - make_dz_woa13_L32.sh : Runs dz_ctl.
Note
--------

  * We set 0 m at the top, thus the number of cells
    is 32 and 101 for 33 layer and 102 layer data, respectively.

      dep      dp

 (1)  0m ---
          |
         dz(1) 5m (1)
          |
 (2) 10m ---
          |


Contact
--------

  * Hiroyuki Tsujino, Shogo Urakawa (JMA-MRI)
