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

metainfo = [json.load(open('json/zostoga_omip1.json')),
            json.load(open('json/zostoga_omip2.json'))]

outfile = './fig/Fig1f.png'

zostoga_model = []
template = 'Institution {0:3d} is {1:s}'

for omip in range(2):

    i=0

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')
    else:
        dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')

    for inst in metainfo[omip].keys():

        print (template.format(i,inst))

        factor=float(metainfo[omip][inst]['factor'])
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']

        print (infile, factor)

        nc = netCDF4.Dataset(infile,'r')
        zostoga_tmp = nc.variables['zostoga'][:]
        if (omip==0):
            zostoga = zostoga_tmp - (zostoga_tmp[57:61]).mean()
            print(dtime[57])
        else:
            zostoga = zostoga_tmp - (zostoga_tmp[47:51]).mean()
            print(dtime[47])

        nc.close()

        col = pd.Index([inst + '-OMIP' + str(omip+1)],name='institution')
        zostoga_df = pd.DataFrame(zostoga*factor,index=dtime,columns=col)
        zostoga_df = zostoga_df.set_index([zostoga_df.index.year])
        zostoga_df.index.names = ['year']

        if i == 0:
            zostoga_df_all=zostoga_df
        else:
            zostoga_df_all=pd.concat([zostoga_df_all,zostoga_df],axis=1)

        print (zostoga_df_all)
        i=i+1

    i=i+1
    zostoga_mean=zostoga_df_all.mean(axis=1)
    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    zostoga_mean_df = pd.DataFrame(zostoga_mean,columns=col)
    zostoga_model_tmp=pd.concat([zostoga_df_all,zostoga_mean_df],axis=1)

    zostoga_model += [zostoga_model_tmp]
    print (zostoga_model[omip])

#######

file_zostoga = "../Global_SL_Budget/original/GLOBAL_SL_Budget.nc"
nco = netCDF4.Dataset(file_zostoga,'r')
zostoga_obs = nco.variables['steric'][:] * 1.0e-3
zostoga_obs_norm = zostoga_obs - (zostoga_obs[0:4]).mean()
time_var_obs = nco.variables['time']
cftime = num2date(time_var_obs[:],time_var_obs.units)
nco.close()

col = pd.Index(['Global_SLB'],name='institution')
zostoga_obs_df = pd.DataFrame(zostoga_obs_norm,index=cftime,columns=col)
zostoga_obs_df = zostoga_obs_df.set_index([zostoga_obs_df.index.year])
zostoga_obs_df.index.names = ['year']
print (zostoga_obs_df)

zostoga_all=pd.concat([zostoga_model[0],zostoga_model[1],zostoga_obs_df],axis=1)

print (zostoga_all)

# draw figures

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
zostoga_all.plot(y=zostoga_all.columns[2*i],  color='darkgrey',linewidth=5,ax=axes,ylim=[-0.012,0.017],title='Global mean thermosteric sea level relative to 2005-2009 mean')
zostoga_all.plot(y=zostoga_all.columns[i-1],  color='darkred' ,linewidth=2,ax=axes,ylim=[-0.012,0.017])
zostoga_all.plot(y=zostoga_all.columns[2*i-1],color='darkblue',linewidth=2,ax=axes,ylim=[-0.012,0.017])

axes.set_ylabel('[m]',fontsize=12)
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
