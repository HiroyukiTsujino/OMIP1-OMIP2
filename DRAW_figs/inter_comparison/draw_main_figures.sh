#!/usr/bin/bash

set -e

## Figures for section 3

pushd Spin_up
echo
echo "Drawing figures for section 3"

./Spin_up_main_figs.sh

echo
popd

## Figures for section 4

pushd Climatology
echo
echo "Drawing figures for section 4"

./Climatology_main_figs.sh

echo
popd

## Figures for section 5

pushd Variability
echo
echo "Drawing figures for section 5"

./Variability_main_figs.sh

echo
popd

## Figures for section 6 (Taylor diagram)

pushd Performance_1
echo
echo "Drawing figures for section 6 (Taylor diagrams)"

./Performance_1_main_figs.sh

echo
popd

## Figures for section 6 (SITES and RBAR)

pushd Performance_2
pwd
echo
echo "Drawing figures for section 6 (SITES & RBAR)"

./Performance_2_main_figs.sh

echo
popd
