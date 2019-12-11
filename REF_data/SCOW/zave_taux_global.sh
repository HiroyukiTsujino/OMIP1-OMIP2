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

./${exe}<<EOF
&zave_lst
  file_base="${orgdir}/taux",
  fileo_base="${outdir}/taux_glb_zm"
  l2d=.true.,
  cgrid="U",
  file_mask="../refdata/SCOW/mask/basin_index.gd"
  i_region_number=-9,
  ex_region_number=8
/
EOF

exit 0
