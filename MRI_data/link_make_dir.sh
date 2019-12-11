#!/bin/sh

dir='/denkei-shared/og1/surakawa/GONDOLA_100'

if [ ! -e "indir" ]; then
    mkdir indir
fi

ln -sfn ${dir}/run-20190621o1/ANALY/annual ./indir/omip1
ln -sfn ${dir}/run-20190522o2/ANALY/annual ./indir/omip2

if [ ! -e 'inmdir' ]; then
    mkdir inmdir
fi

ln -sfn ${dir}/run-20190621o1/ANALY/monthly ./inmdir/omip1
ln -sfn ${dir}/run-20190522o2/ANALY/monthly ./inmdir/omip2

if [ ! -e "outdir" ]; then
    mkdir -p outdir/omip1
    mkdir -p outdir/omip2
fi

exit 0

