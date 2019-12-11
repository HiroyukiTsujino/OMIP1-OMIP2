#!/bin/bash

set -e

exe=dz_ctl
make ${exe}
./${exe} <<EOF
&grid_level
  km       = 101,
  depth_cm = 5.d2, 10.d2, 15.d2, 20.d2, 25.d2, 30.d2, 35.d2, 40.d2, 45.d2, 50.d2, 55.d2, 60.d2, 65.d2, 70.d2, 75.d2, 80.d2, 85.d2, 90.d2, 95.d2, 100.d2, 125.d2, 150.d2, 175.d2, 200.d2, 225.d2, 250.d2, 275.d2, 300.d2, 325.d2, 350.d2, 375.d2, 400.d2, 425.d2, 450.d2, 475.d2, 500.d2, 550.d2, 600.d2, 650.d2, 700.d2, 750.d2, 800.d2, 850.d2, 900.d2, 950.d2, 1000.d2, 1050.d2, 1100.d2, 1150.d2, 1200.d2, 1250.d2, 1300.d2, 1350.d2, 1400.d2, 1450.d2, 1500.d2, 1550.d2, 1600.d2, 1650.d2, 1700.d2, 1750.d2, 1800.d2, 1850.d2, 1900.d2, 1950.d2, 2000.d2, 2100.d2, 2200.d2, 2300.d2, 2400.d2, 2500.d2, 2600.d2, 2700.d2, 2800.d2, 2900.d2, 3000.d2, 3100.d2, 3200.d2, 3300.d2, 3400.d2, 3500.d2, 3600.d2, 3700.d2, 3800.d2, 3900.d2, 4000.d2, 4100.d2, 4200.d2, 4300.d2, 4400.d2, 4500.d2, 4600.d2, 4700.d2, 4800.d2, 4900.d2, 5000.d2, 5100.d2, 5200.d2, 5300.d2, 5400.d2, 5500.d2,
/
EOF

mv depth.txt depth-L102m1.txt
mv dz_cm.d dz_cm-L102m1.d

exit 0
