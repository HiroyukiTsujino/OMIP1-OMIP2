SURF_FLUX/scow_grads
========

This directory contains shell srcipts/Fortran programs to process
 Scatterometer Climatology of Ocean Wind (SCOW) (Risien and Chelton 2008).

  1. Compute time series of monthly data in GrADS format

  2. Compute zonal mean

  3. Generate netCDF file for zonal mean

Programs/Scripts
--------

  * Generate monthly data
     Program:  src/nc2grads_scow.F90
     Namelist: namelist.scow

  * Zonal mean
     Script:   zave_tau[x,y]_basin.sh, zave_tau[x,y]_global.sh
     Program:  ../../anl/integ/zave_ctl.F90

  * Generate netCDF file for zonal mean (zonal wind stress only)
     Script:  zmtx_to_nc4.py
        - This requires anaconda3.

Development
--------

  * Developed at: Climate and Geochemistry Research Department,
                  Meteorological Research Institute,
                  Japan Meteorological Agency
  * Contact: Hiroyuki Tsujino (htsujino@mri-jma.go.jp)
