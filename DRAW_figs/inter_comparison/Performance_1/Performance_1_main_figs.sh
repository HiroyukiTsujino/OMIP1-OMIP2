#!/usr/bin/bash

set -e

module load anaconda3

# Main part

echo "Figure 23 ......."

python ./taylor_siextent.py

mv ./fig/SIextent_Taylor.pdf ../main_figs/fig23.pdf
mv ./fig/SIextent_Taylor.png ../main_figs/fig23.png

# Appendix

echo "Figure E1 ......."

python ./SST_moncl_corr.py MMM

mv ./fig/toscor_monclim_MMM.pdf ../main_figs/figE1.pdf
mv ./fig/toscor_monclim_MMM.png ../main_figs/figE1.png

echo "Figure E2 ......."

python ./SST_interannual_corr.py MMM

mv ./fig/toscor_interannual_MMM.pdf ../main_figs/figE2.pdf
mv ./fig/toscor_interannual_MMM.png ../main_figs/figE2.png

echo "Figure E3 ......."

python ./SSH_moncl_corr.py MMM

mv ./fig/zoscor_monclim_MMM.pdf ../main_figs/figE3.pdf
mv ./fig/zoscor_monclim_MMM.png ../main_figs/figE3.png

echo "Figure E4 ......."

python ./SSH_interannual_corr.py MMM

mv ./fig/zoscor_interannual_MMM.pdf ../main_figs/figE4.pdf
mv ./fig/zoscor_interannual_MMM.png ../main_figs/figE4.png

echo "Figure E5 ......."

python ./MLD_moncl_corr_woNAWD.py MMM

mv ./fig/mldcor_monclim_woNAWD_MMM.pdf ../main_figs/figE5.pdf
mv ./fig/mldcor_monclim_woNAWD_MMM.png ../main_figs/figE5.png

echo "............done"
