#!/usr/bin/bash

show=${1}

set -e

module load anaconda3

echo "Figure S1 and S2......."

for omip in 1 2
do
  python ./FigS1_vat_mip.py ${omip} ${show}
done

mv ./fig/FigS1_vat_OMIP1.png ../supple_figs/figS01.png
mv ./fig/FigS1_vat_OMIP2.png ../supple_figs/figS02.png

echo "Figure S3 - S6 ......."

for mip in 1 2
do
  for var in 1 2	   
  do
    python ./FigS2_allmodels.py ${mip} ${var} ${show}
  done
done   

mv ./fig/FigS2_OMIP1-thetao.png ../supple_figs/figS03.png
mv ./fig/FigS2_OMIP2-thetao.png ../supple_figs/figS04.png
mv ./fig/FigS2_OMIP1-so.png ../supple_figs/figS05.png
mv ./fig/FigS2_OMIP2-so.png ../supple_figs/figS06.png

echo "Figure S7 ......."

python ./FigS1_sivol_mip.py 1 ${show}

mv ./fig/FigS1_sivol_allmodels.png ../supple_figs/figS07.png

echo "Figure S8 ......."

python ./FigS3_mip.py 1 ${show}

mv ./fig/FigS3_omip1.png ../supple_figs/figS08.png

echo "Figure S9 ......."

python ./FigS3_mip.py 2 ${show}

mv ./fig/FigS3_omip2.png ../supple_figs/figS09.png

echo "............done"
