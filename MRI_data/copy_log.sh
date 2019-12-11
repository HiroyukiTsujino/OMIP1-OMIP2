#!/bin/sh

dir='ogsv007:/home/surakawa/GONDOLA_100/exp'

if [ ! -e "logdir" ]; then
    mkdir -p logdir/omip1
    mkdir -p logdir/omip2
fi

omip1dir=${dir}/run-20190621o1/logs_org/ANALY/annual
omip2dir=${dir}/run-20190522o2/logs_org/ANALY/annual

for item in t_ave s_ave t_have s_have thermosteric ice_integ drake itf thetaoga700 thetaoga2000 thetaoga2000-bottom
do
  rsync -av ${omip1dir}/hs_${item}.????  ./logdir/omip1/
  rsync -av ${omip2dir}/hs_${item}.????  ./logdir/omip2/
done

if [ ! -e "logmdir" ]; then
    mkdir -p logmdir/omip1
    mkdir -p logmdir/omip2
fi

omip1dir=${dir}/run-20190621o1/logs_org/MONIT/monthly
omip2dir=${dir}/run-20190522o2/logs_org/MONIT/monthly

for item in sst_ave sss_ave ice_integ
do
  rsync -av ${omip1dir}/hs_${item}.????  ./logmdir/omip1/
  rsync -av ${omip2dir}/hs_${item}.????  ./logmdir/omip2/
done

exit 0
