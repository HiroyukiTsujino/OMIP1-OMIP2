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

metainfo=json.load(open('json/zostoga_omip2.json'))

outfile = './fig/Fig1f_omip2_asis.png'

template = 'Institution {0:3d} is {1:s}'
dtime = pd.date_range('1958-01-01','2018-12-31',freq='A-DEC')

i=0

for inst in metainfo.keys():

    print (template.format(i,inst))

    factor=float(metainfo[inst]['factor'])
    infile = metainfo[inst]['path'] + '/' + metainfo[inst]['fname']

    print (infile, factor)

    nc = netCDF4.Dataset(infile,'r')
    zostoga_tmp = nc.variables['zostoga'][:]
    #zostoga = zostoga_tmp - zostoga_tmp[0]
    zostoga = zostoga_tmp
    nc.close()

    col = pd.Index([inst],name='institution')
    zostoga_df = pd.DataFrame(zostoga*factor,index=dtime,columns=col)
    zostoga_df = zostoga_df.set_index([zostoga_df.index.year])
    zostoga_df.index.names = ['year']

    if i == 0:
      zostoga_df_all=zostoga_df
    else:
      zostoga_df_all=pd.concat([zostoga_df_all,zostoga_df],axis=1)

    print (zostoga_df_all)
    i=i+1

zostoga_mean=zostoga_df_all.mean(axis=1)
col = pd.Index(['MMM'],name='institution')
zostoga_mean_df = pd.DataFrame(zostoga_mean,columns=col)
zostoga_df_all=pd.concat([zostoga_df_all,zostoga_mean_df],axis=1)
print (zostoga_df_all)

file_zostoga = "../Global_SL_Budget/original/GLOBAL_SL_Budget.nc"
nco = netCDF4.Dataset(file_zostoga,'r')
zostoga_obs = nco.variables['steric'][:] * 1.0e-3 + zostoga_df_all.loc[2005,'MMM']
time_var_obs = nco.variables['time']
cftime = num2date(time_var_obs[:],time_var_obs.units)
nco.close()

col = pd.Index(['Global_SLB'],name='institution')
zostoga_obs_df = pd.DataFrame(zostoga_obs,index=cftime,columns=col)
print (zostoga_obs_df)
zostoga_obs_df = zostoga_obs_df.set_index([zostoga_obs_df.index.year])
zostoga_obs_df.index.names = ['year']

zostoga_df_all=pd.concat([zostoga_df_all,zostoga_obs_df],axis=1)

print (zostoga_df_all)

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
zostoga_df_all.plot(y=zostoga_df_all.columns[i+1],color='darkgrey',linewidth=4,ax=axes,ylim=[-0.05,0.05])
zostoga_df_all.plot(y=zostoga_df_all.columns[i],color='darkblue',linewidth=4,ax=axes,ylim=[-0.05,0.05])
zostoga_df_all.plot(y=zostoga_df_all.columns[0:i],ax=axes,ylim=[-0.05,0.05],title='Global mean thermosteric sea level OMIP2 (JRA55-do)')
#
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
