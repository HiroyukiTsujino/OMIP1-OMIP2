# -*- coding: utf-8 -*-
import sys
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print 'Usage: amoc_rapid.py filename'
    sys.exit()

file = sys.argv[1]

nc = netCDF4.Dataset(file,'r')

amoc_rapid = nc.variables['amoc_rapid'][:]
time_var = nc.variables['time']
time_var_day = np.array(time_var)
time_var_day = time_var_day * 365 - 1948 * 365
#print time_var_day
#print amoc_rapid
dtime = num2date(time_var_day[:],'days since 1948-01-01 00:00:00')

nc.close()

col = pd.Index(['FSU-COAPS'],name='institution')
rapid_df = pd.DataFrame(amoc_rapid,index=dtime,columns=col)
print rapid_df
#rapid_df = pd.concat([rapid_df,pd.Series(amoc_rapid,name='AMOC-RAPID',index=dtime)])
#print rapid_df
rapid_df = rapid_df.set_index([rapid_df.index.year,rapid_df.index])
rapid_df.index.names = ['year','date']

rapid_df.mean(level='year').plot()
plt.title('AMOC_RAPID')

plt.show()
