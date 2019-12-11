#!/bin/bash

set -e

flin_area="/denkei-shared/og/ocpublic/refdata/AMIP/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc"
name_area=areacello

flin_basin="../MASK/basinmask_01.gd"
flout_basin="../MASK/basinmask_zonalmean_01.gd"

item[1]=th; item_name[1]=thetao; unit[1]=degrees_celcius; standard_name[1]=basin_wide_zonal_mean_temperature
item[2]=s; item_name[2]=so; unit[2]=psu; standard_name[2]=basin_wide_zonal_mean_salinity

l_netcdf_in=.true.
l_netcdf_out=.true.
deflate_level=1

ln -sfn ../grid/script/NAMELIST.MXE.WOA13v2_1x1_L33.annual NAMELIST.MXE
ln -sfn ../grid/script/NAMELIST.MXE.WOA13v2_1x1_L33.const  NAMELIST.MXE.const
ln -sfn ./script/NAMELIST.MXE.OUT NAMELIST.MXE.OUT

for n in 1 2
do

flin_trc="../DATA/woa13_decav_${item[${n}]}"
flout="../DATA/woa13_decav_${item[${n}]}_basin"

./zonal_mean<<EOF
&nml_zonal_mean
  flin_trc      = "$flin_trc",
  flin_area     = "$flin_area",
  flin_basin    = "$flin_basin",
  flout         = "$flout",
  flout_basin   = "$flout_basin",
  l_netcdf_in   = ${l_netcdf_in},
  l_netcdf_out  = ${l_netcdf_out},
  name_trc      = "${item_name[${n}]}",
  name_area     = "${name_area}",
  unit          = "${unit[${n}]}",
  standard_name = "${standard_name[${n}]}",
  deflate_level = ${deflate_level},
  num_basin     = 4,
  basin_name(1) = "global_ocean",
  basin_name(2) = "Atlantic_Arctic",
  basin_name(3) = "Indian_ocean",
  basin_name(4) = "Pacific_ocean"
 /
EOF

done

exit 0
