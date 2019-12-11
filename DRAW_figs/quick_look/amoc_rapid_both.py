# -*- coding: utf-8 -*-
import sys
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    print 'Usage: amoc_rapid.py filename_core filename_jra'
    sys.exit()

filec = sys.argv[1]
filej = sys.argv[2]

ncc = netCDF4.Dataset(filec,'r')
ncj = netCDF4.Dataset(filej,'r')

amoc_rapid_core = ncc.variables['amoc_rapid'][:]
#time_var_core = ncc.variables['time']
#time_var_core_day = np.array(time_var_core)
#time_var_core_day = time_var_core_day * 365 - 1948 * 365
#dtime_core_noleap = num2date(time_var_core_day[:],'days since 1948-01-01 00:00:00',calendar='365_day')
#dtime_core = num2date(time_var_core_day[:],'days since 1948-01-01 00:00:00')
dtime_core = pd.date_range('1948-01-01','2009-12-01',freq='MS')

print dtime_core

amoc_rapid_jra = ncj.variables['amoc_rapid'][:]
#time_var_jra = ncj.variables['time']
#time_var_jra_day = np.array(time_var_jra)
#time_var_jra_day = time_var_jra_day * 365 - 1948 * 365
#dtime_jra = num2date(time_var_jra_day[:],'days since 1948-01-01 00:00:00',calendar='365_day')
dtime_jra = pd.date_range('1958-01-01','2018-12-01',freq='MS')

ncc.close()
ncj.close()

col = pd.Index(['omip1'],name='forcing')
rapid_df_core = pd.DataFrame(amoc_rapid_core,index=dtime_core,columns=col)
print rapid_df_core
#rapid_df = pd.concat([rapid_df,pd.Series(amoc_rapid,name='AMOC-RAPID',index=dtime)])
#print rapid_df
rapid_df_core = rapid_df_core.set_index([rapid_df_core.index.year,rapid_df_core.index])
rapid_df_core.index.names = ['year','date']

# OMIP2

col = pd.Index(['omip2'],name='forcing')
rapid_df_jra = pd.DataFrame(amoc_rapid_jra,index=dtime_jra,columns=col)
print rapid_df_jra
#rapid_df = pd.concat([rapid_df,pd.Series(amoc_rapid,name='AMOC-RAPID',index=dtime)])
#print rapid_df
rapid_df_jra = rapid_df_jra.set_index([rapid_df_jra.index.year,rapid_df_jra.index])
rapid_df_jra.index.names = ['year','date']

rapid_df_merged=pd.concat([rapid_df_core,rapid_df_jra],axis=1)

print rapid_df_merged

rapid_df_merged.mean(level='year').plot()
plt.title('AMOC_RAPID')

plt.show()
