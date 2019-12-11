# -*- coding: utf-8 -*-
import sys
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print 'Usage: amoc_rapid_obs.py filename'
    sys.exit()

file = sys.argv[1]

nc = netCDF4.Dataset(file,'r')
amoc_rapid = nc.variables['moc_mar_hc10'][:]
time_var = nc.variables['time']
cftime = num2date(time_var[:],time_var.units)
nc.close()

col = pd.Index(['RAPID'],name='hydrographic section')
rapid_df = pd.DataFrame(amoc_rapid,index=cftime,columns=col)
print rapid_df
rapid_df = rapid_df.set_index([rapid_df.index.year,rapid_df.index])
rapid_df.index.names = ['year','date']

rapid_df.mean(level='year').plot()
plt.title('AMOC_RAPID')

plt.show()
