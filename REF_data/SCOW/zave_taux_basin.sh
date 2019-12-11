#!/bin/bash

set -e

if [ x${1} = x ]; then
  echo "Usage: ${0} region_number"
  echo "  region_number = 1,2,3,4,9"
  exit
fi

ln -sf NAMELIST.MXE.SCOW.monthly NAMELIST.MXE

orgdir=../refdata/SCOW/grads
outdir=../refdata/SCOW/grads

exe=zave_ctl

region_number=${1}

file_region[1]=atl
file_region[2]=pac
file_region[3]=ind
file_region[4]=med
file_region[9]=so

echo ${file_region[${region_number}]}

./${exe}<<EOF
&zave_lst
  file_base="${orgdir}/taux",
  fileo_base="${outdir}/taux_${file_region[${region_number}]}_zm"
  l2d=.true.,
  cgrid="U",
  file_mask="../refdata/SCOW/mask/basin_index.gd"
  i_region_number=${region_number},
/
EOF

exit 0
