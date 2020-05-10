#!/usr/bin/bash

set -e

module load anaconda3

#####if [ x${1} = xall ]; then

echo "Figures S10-S12 ......."

for omip in 1 2 3
do 
    python ./SST_bias_allmodels.py ${omip}
done

mv ./fig/SST_bias_allmodels_omip1bias.png ../supple_figs/figS10.png
mv ./fig/SST_bias_allmodels_omip2bias.png ../supple_figs/figS11.png
mv ./fig/SST_bias_allmodels_omip2-1.png   ../supple_figs/figS12.png

echo "Figures S13-S15 ......."

for omip in 1 2 3
do 
    python ./SSS_bias_allmodels.py ${omip}
done

mv ./fig/SSS_bias_allmodels_omip1bias.png ../supple_figs/figS13.png
mv ./fig/SSS_bias_allmodels_omip2bias.png ../supple_figs/figS14.png
mv ./fig/SSS_bias_allmodels_omip2-1.png   ../supple_figs/figS15.png

echo "Figure S16-S23 ......."

for month in 3 9
do 
  for omip in 1 2
  do
    python ./SICONC_NH_allmodels.py ${month} ${omip}
    python ./SICONC_SH_allmodels.py ${month} ${omip}
  done
done

mv ./fig/SICONC_NH_allmodels_omip1_MAR.png   ../supple_figs/figS16.png
mv ./fig/SICONC_NH_allmodels_omip2_MAR.png   ../supple_figs/figS17.png
mv ./fig/SICONC_NH_allmodels_omip1_SEP.png   ../supple_figs/figS18.png
mv ./fig/SICONC_NH_allmodels_omip2_SEP.png   ../supple_figs/figS19.png

mv ./fig/SICONC_SH_allmodels_omip1_SEP.png   ../supple_figs/figS20.png
mv ./fig/SICONC_SH_allmodels_omip2_SEP.png   ../supple_figs/figS21.png
mv ./fig/SICONC_SH_allmodels_omip1_MAR.png   ../supple_figs/figS22.png
mv ./fig/SICONC_SH_allmodels_omip2_MAR.png   ../supple_figs/figS23.png

echo "Figure S24-S26 ......."

for omip in 1 2 3
do 
    python ./SSH_bias_allmodels.py ${omip}
done

mv ./fig/SSH_bias_allmodels_omip1bias.png ../supple_figs/figS24.png
mv ./fig/SSH_bias_allmodels_omip2bias.png ../supple_figs/figS25.png
mv ./fig/SSH_bias_allmodels_omip2-1.png   ../supple_figs/figS26.png

#####fi

echo "Figure S27-S29 ......."

for omip in 1 2 3
do 
    python ./MLD_Winter_bias_allmodels.py ${omip} 1980 2009 1
done

mv ./fig/MLD_Winter_bias_allmodels_omip1bias.png ../supple_figs/figS27.png
mv ./fig/MLD_Winter_bias_allmodels_omip2bias.png ../supple_figs/figS28.png
mv ./fig/MLD_Winter_bias_allmodels_omip2-1.png   ../supple_figs/figS29.png

echo "Figure S30-S32 ......."

for omip in 1 2 3
do 
    python ./MLD_Summer_bias_allmodels.py ${omip} 1980 2009 1
done

mv ./fig/MLD_Summer_bias_allmodels_omip1bias.png ../supple_figs/figS30.png
mv ./fig/MLD_Summer_bias_allmodels_omip2bias.png ../supple_figs/figS31.png
mv ./fig/MLD_Summer_bias_allmodels_omip2-1.png   ../supple_figs/figS32.png

echo "Figure S33-S35 ......."

for omip in 1 2 3
do 
  python ./ZMT_bias_allmodels.py ${omip}
done

mv ./fig/ZMT_bias_omip1bias.png ../supple_figs/figS33.png
mv ./fig/ZMT_bias_omip2bias.png ../supple_figs/figS34.png
mv ./fig/ZMT_bias_omip2-1.png   ../supple_figs/figS35.png

echo "Figure S36-S38 ......."

for omip in 1 2 3
do 
  python ./ZMS_bias_allmodels.py ${omip}
done

mv ./fig/ZMS_bias_omip1bias.png ../supple_figs/figS36.png
mv ./fig/ZMS_bias_omip2bias.png ../supple_figs/figS37.png
mv ./fig/ZMS_bias_omip2-1.png   ../supple_figs/figS38.png

echo "Figure S39-S41 ......."

for omip in 1 2 3
do 
  python ./MOC_allmodels.py ${omip}
done

mv ./fig/MOC_omip1.png   ../supple_figs/figS39.png
mv ./fig/MOC_omip2.png   ../supple_figs/figS40.png
mv ./fig/MOC_omip2-1.png ../supple_figs/figS41.png

echo "Figure S42-S43 ......."

for omip in 1 2
do 
  python ./Heat_Transport_allmodels.py ${omip}
done

mv ./fig/heat_transport_OMIP1.png ../supple_figs/figS42.png
mv ./fig/heat_transport_OMIP2.png ../supple_figs/figS43.png

echo "Figure S44-S46 ......."

for omip in 1 2 3
do 
  python ./U140W_allmodels.py ${omip}
done    

mv ./fig/U140W_allmodels_omip1.png ../supple_figs/figS44.png
mv ./fig/U140W_allmodels_omip2.png ../supple_figs/figS45.png
mv ./fig/U140W_allmodels_omip2-1.png ../supple_figs/figS46.png

echo "............done"
