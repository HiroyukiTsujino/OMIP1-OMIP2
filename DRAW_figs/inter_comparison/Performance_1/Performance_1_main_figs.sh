#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure 21 ......."

python ./taylor_siextent.py

mv ./fig/SIextent_Taylor.pdf ../main_figs/fig21.pdf
mv ./fig/SIextent_Taylor.png ../main_figs/fig21.png

echo "Figure 25 ......."

python ./SST_moncl_corr.py MMM

mv ./fig/toscor_monclim_MMM.pdf ../main_figs/fig25.pdf
mv ./fig/toscor_monclim_MMM.png ../main_figs/fig25.png

echo "Figure 26 ......."

python ./SST_interannual_corr.py MMM

mv ./fig/toscor_interannual_MMM.pdf ../main_figs/fig26.pdf
mv ./fig/toscor_interannual_MMM.png ../main_figs/fig26.png

echo "Figure 27 ......."

python ./SSH_moncl_corr.py MMM

mv ./fig/zoscor_monclim_MMM.pdf ../main_figs/fig27.pdf
mv ./fig/zoscor_monclim_MMM.png ../main_figs/fig27.png

echo "Figure 28 ......."

python ./SSH_interannual_corr.py MMM

mv ./fig/zoscor_interannual_MMM.pdf ../main_figs/fig28.pdf
mv ./fig/zoscor_interannual_MMM.png ../main_figs/fig28.png

echo "Figure 29 ......."

python ./MLD_moncl_corr_woNAWD.py MMM

mv ./fig/mldcor_monclim_woNAWD_MMM.pdf ../main_figs/fig29.pdf
mv ./fig/mldcor_monclim_woNAWD_MMM.png ../main_figs/fig29.png

echo "............done"
