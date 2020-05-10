#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure E6 ......."

python ./sites_rbar_sst.py

mv ./fig/sst_sites_rbar.pdf ../main_figs/figE6.pdf
mv ./fig/sst_sites_rbar.png ../main_figs/figE6.png

echo "Figure E7 ......."

python ./sites_rbar_zos.py

mv ./fig/zos_sites_rbar.pdf ../main_figs/figE7.pdf
mv ./fig/zos_sites_rbar.png ../main_figs/figE7.png

echo "............done"
