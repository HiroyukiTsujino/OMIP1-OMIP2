#!/bin/bash

set -e

exe=dz_ctl
make ${exe}
./${exe} <<EOF
&grid_level
  km       = 32,
  depth_cm = 10.d2, 20.d2, 30.d2, 50.d2, 75.d2, 100.d2, 125.d2, 150.d2, 200.d2, 250.d2, 300.d2, 400.d2, 500.d2, 600.d2, 700.d2, 800.d2, 900.d2, 1000.d2, 1100.d2, 1200.d2, 1300.d2, 1400.d2, 1500.d2, 1750.d2, 2000.d2, 2500.d2, 3000.d2, 3500.d2, 4000.d2, 4500.d2, 5000.d2, 5500.d2,
/
EOF

mv depth.txt depth-L33m1.txt
mv dz_cm.d dz_cm-L33m1.d

exit 0
