#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure 17 ......."

python ./amoc_rapid_all.py

mv ./fig/Fig1a_all.pdf ../main_figs/fig17.pdf
mv ./fig/Fig1a_all.png ../main_figs/fig17.png

echo "Figure 18 ......."

python ./drake_passage_all.py

mv ./fig/Fig1b_all.pdf ../main_figs/fig18.pdf
mv ./fig/Fig1b_all.png ../main_figs/fig18.png

echo "Figure 19 ......."

python ./tosga_all.py

mv ./fig/Fig1d_all.pdf ../main_figs/fig19.pdf
mv ./fig/Fig1d_all.png ../main_figs/fig19.png

echo "Figure 20 ......."

python ./siextent_all_month.py

mv ./fig/Fig1e_all.pdf ../main_figs/fig20.pdf
mv ./fig/Fig1e_all.png ../main_figs/fig20.png

echo "Figure 22 ......."

python ./ohca_topbot_dtr_all.py

mv ./fig/Fig1g_all.pdf ../main_figs/fig22.pdf
mv ./fig/Fig1g_all.png ../main_figs/fig22.png

echo "Figure 23 ......."

python ./zostoga_detrend_all.py

mv ./fig/Fig1f_all.pdf ../main_figs/fig23.pdf
mv ./fig/Fig1f_all.png ../main_figs/fig23.png

echo "Figure 24 ......."

python ./VAT700_Trend.py MMM

mv ./fig/VAT700_trend_MMM.pdf ../main_figs/fig24.pdf
mv ./fig/VAT700_trend_MMM.png ../main_figs/fig24.png

echo "............done"
