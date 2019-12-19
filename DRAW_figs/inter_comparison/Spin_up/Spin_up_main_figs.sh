#!/usr/bin/bash

if [ x"${1}" = x ]; then
   echo "Usage ${0} figure# or all"
   exit 99
fi

target=${1}
   
if [ x${1} = xall ]; then
   draw[1]="yes"; draw[2]="yes"; draw[3]="yes"; draw[4]="yes"
   draw[31]="yes"; draw[32]="yes"; draw[33]="yes"; draw[34]="yes"
else 
   draw[1]="no"; draw[2]="no"; draw[3]="no"; draw[4]="no"
   draw[31]="no"; draw[32]="no"; draw[33]="no"; draw[34]="no"
   draw[${target}]="yes"        
fi

set -e

module load anaconda3

if [ x${draw[1]} = xyes ]; then
  echo "Figure 1 ......."
  python ./FigS1_vat_all.py
  mv ./fig/FigS1_vat_ALL.pdf ../main_figs/fig01.pdf
  mv ./fig/FigS1_vat_ALL.png ../main_figs/fig01.png
fi

if [ x${draw[2]} = xyes ]; then
  echo "Figure 2 ......."
  python ./FigS2_all_std.py MMM
  mv ./fig/FigS2_MMM.pdf ../main_figs/fig02.pdf
  mv ./fig/FigS2_MMM.png ../main_figs/fig02.png
fi

if [ x${draw[3]} = xyes ]; then
  echo "Figure 3 ......."
  python ./FigS1_sivol_all.py
  mv ./fig/FigS1_sivol_ALL.pdf ../main_figs/fig03.pdf
  mv ./fig/FigS1_sivol_ALL.png ../main_figs/fig03.png
fi

if [ x${draw[4]} = xyes ]; then
  echo "Figure 4 ......."
  python ./FigS3_all.py
  mv ./fig/FigS3_ALL.pdf ../main_figs/fig04.pdf
  mv ./fig/FigS3_ALL.png ../main_figs/fig04.png
fi
  
if [ x${draw[31]} = xyes ]; then
  echo "Figure B1 (31)......."
  python ./FigS1_vat_mip_1958-2009.py
  mv ./fig/FigS1_vat_52yr.pdf ../main_figs/figB1.pdf
  mv ./fig/FigS1_vat_52yr.png ../main_figs/figB1.png
fi

if [ x${draw[32]} = xyes ]; then
  echo "Figure B2 (32)......."
  python ./FigS3_mip_1958-2009.py
  mv ./fig/FigS3_52yr.pdf ../main_figs/figB2.pdf
  mv ./fig/FigS3_52yr.png ../main_figs/figB2.png
fi

#echo "Figure B3 ......."
#
#python ./FigS3_mip_omip2_alpha.py
#
#mv ./fig/FigS3_omip2_alpha.pdf ../main_figs/figB3.pdf
#mv ./fig/FigS3_omip2_alpha.png ../main_figs/figB3.png

if [ x${draw[33]} = xyes ]; then
  echo "Figure B3 (33)......."
  python ./FigS1_vat_mip_aircore.py
  mv ./fig/FigS1_vat_aircore.pdf ../main_figs/figB3.pdf
  mv ./fig/FigS1_vat_aircore.png ../main_figs/figB3.png
fi

if [ x${draw[34]} = xyes ]; then
  echo "Figure B4 (34)......."
  python ./FigS3_mip_omip2_aircore.py
  mv ./fig/FigS3_omip2_aircore.pdf ../main_figs/figB4.pdf
  mv ./fig/FigS3_omip2_aircore.png ../main_figs/figB4.png
fi

echo "............done"
