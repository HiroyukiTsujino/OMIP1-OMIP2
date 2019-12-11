#!/usr/bin/bash

set -e

module load anaconda3

echo "Figure S49-S51 ......."

for omip in 1 2 3
do
  python ./SST_moncl_corr_allmodels.py ${omip}
done    

mv ./fig/toscor_monclim_omip1.png   ../supple_figs/figS49.png
mv ./fig/toscor_monclim_omip2.png   ../supple_figs/figS50.png
mv ./fig/toscor_monclim_omip2-1.png ../supple_figs/figS51.png

echo "Figure S52-S54 ......."

for omip in 1 2 3
do
  python ./SST_interannual_corr_allmodels.py ${omip}
done    

mv ./fig/toscor_interannual_omip1.png   ../supple_figs/figS52.png
mv ./fig/toscor_interannual_omip2.png   ../supple_figs/figS53.png
mv ./fig/toscor_interannual_omip2-1.png ../supple_figs/figS54.png


echo "Figure S55-S57 ......."

for omip in 1 2 3
do
  python ./SSH_moncl_corr_allmodels.py ${omip}
done

mv ./fig/zoscor_monclim_omip1.png   ../supple_figs/figS55.png
mv ./fig/zoscor_monclim_omip2.png   ../supple_figs/figS56.png
mv ./fig/zoscor_monclim_omip2-1.png ../supple_figs/figS57.png

echo "Figure S58-60 ......."

for omip in 1 2 3
do
  python ./SSH_interannual_corr_allmodels.py ${omip}
done

mv ./fig/zoscor_interannual_omip1.png   ../supple_figs/figS58.png
mv ./fig/zoscor_interannual_omip2.png   ../supple_figs/figS59.png
mv ./fig/zoscor_interannual_omip2-1.png ../supple_figs/figS60.png

echo "Figure S61-S63 ......."

for omip in 1 2 3
do
  python ./MLD_moncl_corr_woNAWD_allmodels.py ${omip}
done

mv ./fig/mldcor_monclim_woNAWD_omip1.png   ../supple_figs/figS61.png
mv ./fig/mldcor_monclim_woNAWD_omip2.png   ../supple_figs/figS62.png
mv ./fig/mldcor_monclim_woNAWD_omip2-1.png ../supple_figs/figS63.png

echo "............done"
