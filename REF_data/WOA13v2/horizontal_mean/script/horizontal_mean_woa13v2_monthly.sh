#!/bin/bash

set -e

flin_area="/denkei-shared/og/ocpublic/refdata/AMIP/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc"
name_area=areacello
name_vol=volo

flin_basin="../MASK/basinmask_01.gd"
flout_basin="../MASK/basinmask_horizontalmean.gd"
flout_vol="../DATA/horizontal_volume.nc"

item[1]=th; item_name[1]=thetao; unit[1]=degrees_celcius; standard_name[1]=horizontal_mean_temperature
item[2]=s; item_name[2]=so; unit[2]=psu; standard_name[2]=horizontal_mean_salinity

l_netcdf_in=.true.
l_netcdf_out=.true.
deflate_level=0

ln -sfn ../grid/script/NAMELIST.MXE.WOA13v2_1x1_L33.monthly NAMELIST.MXE
ln -sfn ../grid/script/NAMELIST.MXE.WOA13v2_1x1_L33.const  NAMELIST.MXE.const
ln -sfn ./script/NAMELIST.MXE.OUT.monthly NAMELIST.MXE.OUT

for n in 1 2
do

flin_trc="../DATA/monthly/woa13_decav_${item[${n}]}"
flout="../DATA/monthly/woa13_decav_${item[${n}]}_horizontal"

./horizontal_mean<<EOF
&nml_horz_mean
  flin_trc      = "$flin_trc",
  flin_area     = "$flin_area",
  flin_basin    = "$flin_basin",
  flout         = "$flout",
  flout_basin   = "$flout_basin",
  flout_vol     = "$flout_vol",
  l_netcdf_in   = ${l_netcdf_in},
  l_netcdf_out  = ${l_netcdf_out},
  name_trc      = "${item_name[${n}]}",
  name_area     = "${name_area}",
  name_vol      = "${name_vol}",
  unit          = "${unit[${n}]}",
  standard_name = "${standard_name[${n}]}",
  deflate_level = ${deflate_level},
 /
EOF

done

exit 0
