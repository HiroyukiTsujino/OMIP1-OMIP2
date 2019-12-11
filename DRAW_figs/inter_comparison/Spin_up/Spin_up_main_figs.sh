#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure 1 ......."

python ./FigS1_vat_all.py

mv ./fig/FigS1_vat_ALL.pdf ../main_figs/fig01.pdf
mv ./fig/FigS1_vat_ALL.png ../main_figs/fig01.png

echo "Figure 2 ......."

python ./FigS1_sivol_all.py

mv ./fig/FigS1_sivol_ALL.pdf ../main_figs/fig02.pdf
mv ./fig/FigS1_sivol_ALL.png ../main_figs/fig02.png

echo "Figure 3 ......."

python ./FigS2_all_std.py MMM

mv ./fig/FigS2_MMM.pdf ../main_figs/fig03.pdf
mv ./fig/FigS2_MMM.png ../main_figs/fig03.png

echo "Figure 4 ......."

python ./FigS3_all.py

mv ./fig/FigS3_ALL.pdf ../main_figs/fig04.pdf
mv ./fig/FigS3_ALL.png ../main_figs/fig04.png

echo "Figure 32 ......."

python ./FigS1_vat_mip_1958-2009.py

mv ./fig/FigS1_vat_52yr.pdf ../main_figs/figB1.pdf
mv ./fig/FigS1_vat_52yr.png ../main_figs/figB1.png

echo "Figure 33 ......."

python ./FigS3_mip_1958-2009.py

mv ./fig/FigS3_52yr.pdf ../main_figs/figB2.pdf
mv ./fig/FigS3_52yr.png ../main_figs/figB2.png

echo "Figure 34 ......."

python ./FigS3_mip_omip2_alpha.py

mv ./fig/FigS3_omip2_alpha.pdf ../main_figs/figB3.pdf
mv ./fig/FigS3_omip2_alpha.png ../main_figs/figB3.png

echo "Figure 37 ......."

python ./FigS1_vat_mip_aircore.py

mv ./fig/FigS1_vat_aircore.pdf ../main_figs/figB6.pdf
mv ./fig/FigS1_vat_aricore.png ../main_figs/figB6.png

echo "Figure 38 ......."

python ./FigS3_mip_omip2_aircore.py

mv ./fig/FigS3_omip2_aircore.pdf ../main_figs/figB7.pdf
mv ./fig/FigS3_omip2_aircore.png ../main_figs/figB7.png

echo "............done"
