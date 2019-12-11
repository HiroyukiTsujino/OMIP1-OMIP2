#!/usr/bin/bash

#----- Where you have extracted data archive -----
#
#ARC_DIR=/denkei-shared/og1/htsujino/OMIP
ARC_DIR=/worke/htsujino/OMIP
#
#-------------------------------------------------

set -e

ln -sfn ${ARC_DIR}/model .
ln -sfn ${ARC_DIR}/analysis .
ln -sfn ${ARC_DIR}/refdata .
