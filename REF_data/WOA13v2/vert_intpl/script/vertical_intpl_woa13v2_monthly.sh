#!/bin/bash

if [ "x${1}" = "x" ];then
  echo "Usage ${0} item"
fi

set -e

item_name=${1}

indir=/denkei-shared/og1/htsujino/refdata/WOA/WOA13v2/grads/1deg_L102/monthly
outdir=/denkei-shared/og1/htsujino/refdata/WOA/WOA13v2/netCDF/1deg_L33/monthly

item[1]=th; item_name[1]=thetao
item[2]=s; item_name[2]=so

for n in 1 2
do

./vertical_intpl_ctl <<EOF
 &nml_vert_intp
  src_namelist='../grid/script/NAMELIST.MXE.WOA13v2_1x1_L102.monthly'
  dst_namelist='../grid/script/NAMELIST.MXE.WOA13v2_1x1_L33.monthly'
  shape='xyz',
  cgrid='U',
  file_in='${indir}/woa13_decav_${item[${n}]}',
  file_out='${outdir}/woa13_decav_${item[${n}]}',
  nvar=1,
  name_in='${item_name[${n}]}',
  l_netcdf_in=.false.,
  l_netcdf_out=.true.,
  deflate_level_in=0,
  deflate_level_out=1,
  undef_in=-9.99e33,
  undef_out=-9.99e33
 /
EOF

done
