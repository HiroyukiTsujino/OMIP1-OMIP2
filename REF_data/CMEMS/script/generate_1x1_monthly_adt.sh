#!/bin/bash -f

set -e

make generate_1x1_monthly

ln -sfn script/NAMELIST.MXE.1x1    .
ln -sfn script/NAMELIST.MXE.CMEMS  .
ln -sfn script/namelist.adt2onedeg .

./generate_1x1_monthly

rm NAMELIST.MXE.1x1
rm NAMELIST.MXE.CMEMS
rm namelist.adt2onedeg
