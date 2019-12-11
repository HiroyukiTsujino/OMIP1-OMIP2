zonal_mean
========

Calculate zonal mean of WOA on 1x1 grid.


Document
--------

  Simple (virtually unweighted) basin wide zonal mean


Source Program
---------

./src/zonal_mean.F90

&nml_zonal_mean
  flin_trc      : WOA climatology (1x1 L33)
  l_netcdf_in   : use netCDF or not for above
  flin_area     : areacello
  flin_basin    : basin mask
  flout         : Zonal mean
  l_netcdf_out  : use netCDF or not for above
  flout_basin   : Masks actually used
  name_trc      : thetao or so
  unit          : units
  standard_name : standard name
  deflate_level : deflate_level
  name_area     : areacello
  num_basin     : = 4
  basin_name(1) : "global_ocean"
  basin_name(2) : "atlantic_arctic_ocean"
  basin_name(3) : "indian_ocean"
  basin_name(4) : "pacific_ocean"


Contact
-------

  * Hiroyuki Tsujino (htsujino@mri-jma.go.jp)
