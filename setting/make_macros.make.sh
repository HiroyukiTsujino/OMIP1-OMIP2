#!/bin/bash
# make用のビルド設定 macros.make をつくる

set -e

if [ -s macros.make ]; then
    exit 0
fi

echo "Create macros.make (config for Make)"
hostname=`hostname -s`
case ${hostname} in
    front*)
	cp machine/mri-front/* .
	;;
    ogsv007)
	cp machine/ogsv007-pgf90/* .
	;;
    ogsv009)
	cp machine/ogsv009-pgf90/* .
	;;
    ocsv*|ogsv*)
	cp machine/ocsv-pgf90/* .
	;;
    *)
	cp sample/macros.make .
	;;
esac

exit 0
