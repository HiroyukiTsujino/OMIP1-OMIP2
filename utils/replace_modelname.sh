#!/bin/bash

#----------------------
oldname="LICOM3-CICE4"
newname="CAS-LICOM3"
#----------------------

set -e

target_file=${1}

echo ${target_file}

sed -e "
    s%${oldname}%${newname}%g
    " ${target_file} > tmp_file

mv tmp_file ${target_file}
