horizontal_mean
========

Calculate horizontal mean of WOA


Document
--------

  Simple horizontal mean


Usage
--------

  * annual mean

    - execute ./script/horizontal_mean_woa13v2.sh

  * monthly mean

    - execute ./script/horizontal_mean_woa13v2_monthly.sh


Source Program
---------

./src/horizontal_mean.F90

&nml_horz_mean
  flin_trc      : WOA climatology (1x1 L33)
  l_netcdf_in   : use netCDF or not for above
  flin_area     : areacello
  flin_basin    : basin mask
  flout         : horizontal mean
  flout_vol     : ocean volume of each vertical level
  l_netcdf_out  : use netCDF or not for above
  name_trc      : thetao or so
  unit          : units
  standard_name : standard name
  deflate_level : deflate_level
  name_area     : areacello
  name_vol      : volo


Contact
-------

  * Hiroyuki Tsujino (htsujino@mri-jma.go.jp)
