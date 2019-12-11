# -*- coding: utf-8 -*-
import sys
import json
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

######

metainfo=json.load(open('amoc_rapid_omip1.json'))

template = 'Institution {0:3d} is {1:s}'
dtime = pd.date_range('1948-01-01','2009-12-01',freq='MS')

i=0

for inst in metainfo.keys():

    print template.format(i,inst)

    fac=float(metainfo[inst]['factor'])
    infile = metainfo[inst]['path'] + '/' + metainfo[inst]['fname']

    print infile, fac

    nc = netCDF4.Dataset(infile,'r')
    amoc_rapid = nc.variables['amoc_rapid'][:]
    nc.close()

    col = pd.Index([inst],name='institution')
    rapid_df = pd.DataFrame(amoc_rapid*fac,index=dtime,columns=col)
    rapid_df = rapid_df.set_index([rapid_df.index.year,rapid_df.index])
    rapid_df.index.names = ['year','date']

    if i == 0:
      rapid_df_all=rapid_df
    else:
      rapid_df_all=pd.concat([rapid_df_all,rapid_df],axis=1)

    print rapid_df_all
    i=i+1


file_rapid = "RAPID/moc_transports.nc"
nco = netCDF4.Dataset(file_rapid,'r')
amoc_rapid_obs = nco.variables['moc_mar_hc10'][:]
time_var_obs = nco.variables['time']
cftime = num2date(time_var_obs[:],time_var_obs.units)
nco.close()

col = pd.Index(['RAPID'],name='institution')
rapid_obs_df = pd.DataFrame(amoc_rapid_obs,index=cftime,columns=col)
print rapid_obs_df
rapid_obs_df = rapid_obs_df.set_index([rapid_obs_df.index.year, rapid_obs_df.index])
rapid_obs_df.index.names = ['year','date']
rapid_obs_annual=rapid_obs_df.mean(level='year')


rapid_annual=rapid_df_all.mean(level='year')
rapid_annual_all=pd.concat([rapid_annual,rapid_obs_annual],axis=1)

print rapid_annual_all

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
rapid_annual_all.plot(y=rapid_annual_all.columns[i],color='darkgrey',linewidth=4,ax=axes,ylim=[7,23])
rapid_annual_all.plot(y=rapid_annual_all.columns[0:i],ax=axes,ylim=[7,23],title='AMOC_RAPID_OMIP1 (CORE)')
#
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)
#
plt.show()
