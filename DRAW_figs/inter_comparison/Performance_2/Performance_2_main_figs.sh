#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure 27 ......."

python ./SST_SSS_bias.py

mv ./fig/SST_SSS_bias.pdf ../main_figs/fig27.pdf
mv ./fig/SST_SSS_bias.png ../main_figs/fig27.png

echo "Figure 28 ......."

python ./circ_index.py

mv ./fig/Circulation.pdf ../main_figs/fig28.pdf
mv ./fig/Circulation.png ../main_figs/fig28.png

echo "Figure E6 ......."

python ./sites_rbar_sst.py

mv ./fig/sst_sites_rbar.pdf ../main_figs/figE6.pdf
mv ./fig/sst_sites_rbar.png ../main_figs/figE6.png

echo "Figure E7 ......."

python ./sites_rbar_zos.py

mv ./fig/zos_sites_rbar.pdf ../main_figs/figE7.pdf
mv ./fig/zos_sites_rbar.png ../main_figs/figE7.png

echo "............done"
