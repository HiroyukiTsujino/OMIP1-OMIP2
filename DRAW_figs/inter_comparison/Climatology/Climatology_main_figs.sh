#!/usr/bin/bash

if [ x"${1}" = x ]; then
   echo "Usage ${0} figure# or all"
   exit 99
fi

target=${1}
   
if [ x${1} = xall ]; then
   draw[5]="yes"; draw[6]="yes"; draw[7]="yes"; draw[8]="yes"
   draw[9]="yes"; draw[10]="yes"; draw[11]="yes"; draw[12]="yes"
   draw[13]="yes"; draw[14]="yes"; draw[15]="yes"; draw[16]="yes"
else 
   draw[5]="no"; draw[6]="no"; draw[7]="no"; draw[8]="no"
   draw[9]="no"; draw[10]="no"; draw[11]="no"; draw[12]="no"
   draw[13]="no"; draw[14]="no"; draw[15]="no"; draw[16]="no"
   draw[${target}]="yes"        
fi

set -e

module load anaconda3

echo "Start main part ....."

if [ x${draw[5]} = xyes ]; then
   echo "Figure 5 ......."
   python ./SST_bias_std.py MMM
   mv ./fig/SST_bias_MMM.pdf ../main_figs/fig05.pdf
   mv ./fig/SST_bias_MMM.png ../main_figs/fig05.png
fi

if [ x${draw[6]} = xyes ]; then
   echo "Figure 6 ......."
   python ./SSS_bias_std.py MMM
   mv ./fig/SSS_bias_MMM.pdf ../main_figs/fig06.pdf
   mv ./fig/SSS_bias_MMM.png ../main_figs/fig06.png
fi
   
if [ x${draw[7]} = xyes ]; then
   echo "Figure 7 ......."
   python ./SICONC_mmm.py
   mv ./fig/SICONC_MMM.pdf ../main_figs/fig07.pdf
   mv ./fig/SICONC_MMM.png ../main_figs/fig07.png
fi
   
if [ x${draw[8]} = xyes ]; then
   echo "Figure 8 ......."
   python ./SSH_bias_std.py MMM
   mv ./fig/SSH_bias_MMM.pdf ../main_figs/fig08.pdf
   mv ./fig/SSH_bias_MMM.png ../main_figs/fig08.png
fi

if [ x${draw[9]} = xyes ]; then
   echo "Figure 9 ......."
   python ./MLD_Winter_std.py MMM
   mv ./fig/MLD_Winter_MMM.pdf ../main_figs/fig09.pdf
   mv ./fig/MLD_Winter_MMM.png ../main_figs/fig09.png
fi

if [ x${draw[10]} = xyes ]; then
   echo "Figure 10 ......."
   python ./MLD_Summer_std.py MMM
   mv ./fig/MLD_Summer_MMM.pdf ../main_figs/fig10.pdf
   mv ./fig/MLD_Summer_MMM.png ../main_figs/fig10.png
fi
   
if [ x${draw[11]} = xyes ]; then
   echo "Figure 11 ......."
   python ./ZMT_bias_std.py MMM
   mv ./fig/ZMT_bias_MMM.pdf ../main_figs/fig11.pdf
   mv ./fig/ZMT_bias_MMM.png ../main_figs/fig11.png
fi

if [ x${draw[12]} = xyes ]; then
   echo "Figure 12 ......."
   python ./ZMS_bias_std.py MMM
   mv ./fig/ZMS_bias_MMM.pdf ../main_figs/fig12.pdf
   mv ./fig/ZMS_bias_MMM.png ../main_figs/fig12.png
fi

if [ x${draw[13]} = xyes ]; then
   echo "Figure 13 ......."
   python ./MOC_all_std.py MMM
   mv ./fig/MOC_MMM.pdf ../main_figs/fig13.pdf
   mv ./fig/MOC_MMM.png ../main_figs/fig13.png
fi
   
if [ x${draw[14]} = xyes ]; then
   echo "Figure 14 ......."
   python ./TAU_all.py MMM
   mv ./fig/TAUX_1999-2009_MMM.pdf ../main_figs/fig14.pdf
   mv ./fig/TAUX_1999-2009_MMM.png ../main_figs/fig14.png
fi
   
if [ x${draw[15]} = xyes ]; then
   echo "Figure 15 ......."
   python ./Heat_Transport_all.py MMM
   mv ./fig/heat_transport_MMM.pdf ../main_figs/fig15.pdf
   mv ./fig/heat_transport_MMM.png ../main_figs/fig15.png
fi

if [ x${draw[16]} = xyes ]; then
   echo "Figure 16 ......."
   python ./U140W_std.py MMM
   mv ./fig/U140W_MMM.pdf ../main_figs/fig16.pdf
   mv ./fig/U140W_MMM.png ../main_figs/fig16.png
fi
   
echo "............done"
echo
echo "Start appendix ....."

#if [ x${draw[35]} = xyes ]; then
#   echo "Figure 35 ......."
#   python ./Heat_Transport_omip2_alpha.py
#   mv ./fig/heat_transport_omip2_alpha.pdf ../main_figs/figB4.pdf
#   mv ./fig/heat_transport_omip2_alpha.png ../main_figs/figB4.png
#fi
#   
#if [ x${draw[36]} = xyes ]; then
#   echo "Figure 36 ......."
#   python ./U140W_omip2_alpha.py
#   mv ./fig/U140W_omip2_alpha.pdf ../main_figs/figB5.pdf
#   mv ./fig/U140W_omip2_alpha.png ../main_figs/figB5.png
#fi
   
if [ x${draw[39]} = xyes ]; then
   echo "Figure B5 (35)......."
   python ./SST_bias_aircore.py
   mv ./fig/SST_bias_aircore.pdf ../main_figs/figB5.pdf
   mv ./fig/SST_bias_aircore.png ../main_figs/figB5.png
fi

echo "............done"
