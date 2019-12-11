#!/usr/bin/bash

#----- Where you have extracted data archive -----
#
ARC_DIR=/denkei-shared/og1/htsujino/OMIP
#
#-------------------------------------------------

set -e

ln -s ${ARC_DIR}/model .
ln -s ${ARC_DIR}/analysis .
ln -s ${ARC_DIR}/refdata .
