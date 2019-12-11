#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure 30 ......."

python ./sites_rbar_sst.py

mv ./fig/sst_sites_rbar.pdf ../main_figs/fig30.pdf
mv ./fig/sst_sites_rbar.png ../main_figs/fig30.png

echo "Figure 31 ......."

python ./sites_rbar_zos.py

mv ./fig/zos_sites_rbar.pdf ../main_figs/fig31.pdf
mv ./fig/zos_sites_rbar.png ../main_figs/fig31.png

echo "............done"
