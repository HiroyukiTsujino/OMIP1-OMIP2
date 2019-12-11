#!/usr/bin/bash

show=${1}

set -e

module load anaconda3

echo "Figure S47-S48 ......."

for omip in 1 2
do
  python ./VAT700_Trend_allmodels.py ${omip} ${show}
done    

mv ./fig/VAT700_trend_omip1.png ../supple_figs/figS47.png
mv ./fig/VAT700_trend_omip2.png ../supple_figs/figS48.png

echo "............done"
