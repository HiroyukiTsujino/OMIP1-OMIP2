# -*- coding: utf-8 -*-
import sys
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print 'Usage: test.py filename sector_name'
    sys.exit()

file = sys.argv[1]

nc = netCDF4.Dataset(file,'r')

sector = nc.variables['sector'][:]
mfo    = nc.variables['mfo'][:]

time_var = nc.variables['time']
dtime = num2date(time_var[:],time_var.units)

nc.close()

if len(sys.argv) == 2:
    print 'Sector name list:'
    for i in range(len(sector)):
        print ' ' + ''.join(sector[i])
    sys.exit()
else:
    sector_name = sys.argv[2]

n = -1
for i in range(len(sector)):
    name = ''.join(sector[i])
    if name == sector_name:
        n = i
        print 'drawing ' + name
        break

if n == -1:
    sys.exit('Error: not found')


mfo_df = pd.DataFrame(index=dtime)
mfo_df = pd.concat([mfo_df,pd.Series([row[n]/1e9 for row in mfo],name=sector_name,index=dtime)],axis=1)

mfo_df = mfo_df.set_index([mfo_df.index.year,mfo_df.index])
mfo_df.index.names = ['year','date']


mfo_df.mean(level='year').plot()
plt.title(''.join(sector[n]))

plt.show()
