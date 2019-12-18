#!/usr/bin/bash

set -e

if [ x"${1}" = x ]; then
   echo "Usage ${0} figure# or all"
   exit 99
fi

target=${1}
   
if [ x${1} = xall ]; then
   draw[17]="yes"; draw[18]="yes"; draw[19]="yes"; draw[20]="yes"
   draw[21]="yes"; draw[23]="yes"; draw[24]="yes"
else 
   draw[17]="no"; draw[18]="no"; draw[19]="no"; draw[20]="no"
   draw[21]="no"; draw[23]="no"; draw[24]="no"
   draw[${target}]="yes"        
fi

module load anaconda3

if [ x${draw[17]} = xyes ]; then
  echo "Figure 17 ......."
  python ./amoc_rapid_all.py
  mv ./fig/Fig1a_all.pdf ../main_figs/fig17.pdf
  mv ./fig/Fig1a_all.png ../main_figs/fig17.png
fi

if [ x${draw[18]} = xyes ]; then
  echo "Figure 18 ......."
  python ./drake_passage_all.py
  mv ./fig/Fig1b_all.pdf ../main_figs/fig18.pdf
  mv ./fig/Fig1b_all.png ../main_figs/fig18.png
fi

if [ x${draw[19]} = xyes ]; then
  echo "Figure 19 ......."
  python ./tosga_all.py
  mv ./fig/Fig1d_all.pdf ../main_figs/fig19.pdf
  mv ./fig/Fig1d_all.png ../main_figs/fig19.png
fi

if [ x${draw[20]} = xyes ]; then
  echo "Figure 20 ......."
  python ./siextent_all_month.py
  mv ./fig/Fig1e_all.pdf ../main_figs/fig20.pdf
  mv ./fig/Fig1e_all.png ../main_figs/fig20.png
fi

if [ x${draw[22]} = xyes ]; then
  echo "Figure 22 ......."
  python ./ohca_topbot_dtr_all.py
  mv ./fig/Fig1g_all.pdf ../main_figs/fig22.pdf
  mv ./fig/Fig1g_all.png ../main_figs/fig22.png
fi

if [ x${draw[23]} = xyes ]; then
  echo "Figure 23 ......."
  python ./zostoga_detrend_all.py
  mv ./fig/Fig1f_all.pdf ../main_figs/fig23.pdf
  mv ./fig/Fig1f_all.png ../main_figs/fig23.png
fi

if [ x${draw[24]} = xyes ]; then
  echo "Figure 24 ......."
  python ./VAT700_Trend.py MMM
  mv ./fig/VAT700_trend_MMM.pdf ../main_figs/fig24.pdf
  mv ./fig/VAT700_trend_MMM.png ../main_figs/fig24.png
fi

echo "............done"
