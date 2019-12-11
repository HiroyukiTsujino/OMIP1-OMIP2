#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure 5 ......."

python ./SST_bias_std.py MMM

mv ./fig/SST_bias_MMM.pdf ../main_figs/fig05.pdf
mv ./fig/SST_bias_MMM.png ../main_figs/fig05.png

echo "Figure 6 ......."

python ./SSS_bias_std.py MMM

mv ./fig/SSS_bias_MMM.pdf ../main_figs/fig06.pdf
mv ./fig/SSS_bias_MMM.png ../main_figs/fig06.png

echo "Figure 7 ......."

python ./SICONC_mmm.py

mv ./fig/SICONC_MMM.pdf ../main_figs/fig07.pdf
mv ./fig/SICONC_MMM.png ../main_figs/fig07.png

echo "Figure 8 ......."

python ./SSH_bias_std.py MMM

mv ./fig/SSH_bias_MMM.pdf ../main_figs/fig08.pdf
mv ./fig/SSH_bias_MMM.png ../main_figs/fig08.png

echo "Figure 9 ......."

python ./MLD_Winter_std.py MMM

mv ./fig/MLD_Winter_MMM.pdf ../main_figs/fig09.pdf
mv ./fig/MLD_Winter_MMM.png ../main_figs/fig09.png

echo "Figure 10 ......."

python ./MLD_Summer_std.py MMM

mv ./fig/MLD_Summer_MMM.pdf ../main_figs/fig10.pdf
mv ./fig/MLD_Summer_MMM.png ../main_figs/fig10.png

echo "Figure 11 ......."

python ./ZMT_bias_std.py MMM

mv ./fig/ZMT_bias_MMM.pdf ../main_figs/fig11.pdf
mv ./fig/ZMT_bias_MMM.png ../main_figs/fig11.png

echo "Figure 12 ......."

python ./ZMS_bias_std.py MMM

mv ./fig/ZMS_bias_MMM.pdf ../main_figs/fig12.pdf
mv ./fig/ZMS_bias_MMM.png ../main_figs/fig12.png

echo "Figure 13 ......."

python ./MOC_all_std.py MMM

mv ./fig/MOC_MMM.pdf ../main_figs/fig13.pdf
mv ./fig/MOC_MMM.png ../main_figs/fig13.png

echo "Figure 14 ......."

python ./TAU_all.py MMM

mv ./fig/TAUX_1999-2009_MMM.pdf ../main_figs/fig14.pdf
mv ./fig/TAUX_1999-2009_MMM.png ../main_figs/fig14.png

echo "Figure 15 ......."

python ./Heat_Transport_all.py MMM

mv ./fig/heat_transport_MMM.pdf ../main_figs/fig15.pdf
mv ./fig/heat_transport_MMM.png ../main_figs/fig15.png

echo "Figure 16 ......."

python ./U140W_std.py MMM

mv ./fig/U140W_MMM.pdf ../main_figs/fig16.pdf
mv ./fig/U140W_MMM.png ../main_figs/fig16.png

echo "............done"

echo "Figure 35 ......."

python ./Heat_Transport_omip2_alpha.py

mv ./fig/heat_transport_omip2_alpha.pdf ../main_figs/figB4.pdf
mv ./fig/heat_transport_omip2_alpha.png ../main_figs/figB4.png

echo "Figure 36 ......."

python ./U140W_omip2_alpha.py

mv ./fig/U140W_omip2_alpha.pdf ../main_figs/figB5.pdf
mv ./fig/U140W_omip2_alpha.png ../main_figs/figB5.png

echo "Figure 39 ......."

python ./SST_bias_aircore.py

mv ./fig/SST_bias_aircore.pdf ../main_figs/figB8.pdf
mv ./fig/SST_bias_aircore.png ../main_figs/figB8.png

echo "............done"
