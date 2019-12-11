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

metainfo = json.load(open("./json/vat_omip1.json"))
model_list = metainfo.keys()
var_list = [ "thetaoga_700", "thetaoga_2000", "thetaoga_2000_bottom", "thetaoga_all" ]
volume_list = np.array([ 2.338e17, 6.216e17, 7.593e17, 1.381e18 ])
degC_to_ZJ = volume_list * 3.99e3 * 1.036e3 * 1.0e-21

template = 'Institution {0:3d} is {1:s}'
dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')

DF = []
nvar = 0
for var in var_list:

    i=0
    for inst in metainfo.keys():

        print (template.format(i,inst))

        factor=float(metainfo[inst][var]['factor'])
        infile = metainfo[inst][var]['path'] + '/' + metainfo[inst][var]['fname']
        vname = metainfo[inst][var]['varname']
        print (infile, factor)

        nc = netCDF4.Dataset(infile,'r')
        if ( inst == "AWI-FESOM"):
            vat_tmp = nc.variables[vname][:]
        else:
            vat_tmp = nc.variables[vname][310:372]
            
        vat_tmp = vat_tmp * degC_to_ZJ[nvar]
        vat = vat_tmp - (vat_tmp[57:61]).mean()
        nc.close()

        col = pd.Index([inst],name='institution')
        vat_df = pd.DataFrame(vat*factor,index=dtime,columns=col)
        vat_df = vat_df.set_index([vat_df.index.year])
        vat_df.index.names = ['year']

        if i == 0:
            vat_df_all=vat_df
        else:
            vat_df_all=pd.concat([vat_df_all,vat_df],axis=1)

        print (vat_df_all)
        i=i+1

    vat_mean=vat_df_all.mean(axis=1)
    col = pd.Index(['MMM'],name='institution')
    vat_mean_df = pd.DataFrame(vat_mean,columns=col)
    vat_df_all=pd.concat([vat_df_all,vat_mean_df],axis=1)
    print (vat_df_all)

    file_vat = "../Zanna_OHC/OHC_GF_1870_2018.nc"
    nco = netCDF4.Dataset(file_vat,'r')
    if (nvar == 0):
        vat_obs = nco.variables['OHC_700m'][:]
    if (nvar == 1):
        vat_obs = nco.variables['OHC_2000m'][:]
    if (nvar == 2):
        vat_obs = nco.variables['OHC_below_2000m'][:]
    if (nvar == 3):
        vat_obs = nco.variables['OHC_full_depth'][:]

    vat_obs_norm = vat_obs - (vat_obs[-15:-9]).mean()
    time_var_obs = nco.variables['time'][:]
    #cftime = num2date(time_var_obs[:],time_var_obs.units)
    nco.close()

    col = pd.Index(['Zanna_2019'],name='institution')
    #print(vat_obs_norm)
    #print(time_var_obs)
    vat_obs_df = pd.DataFrame(vat_obs_norm,index=time_var_obs,columns=col)
    print (vat_obs_df)
    #vat_obs_df = vat_obs_df.set_index([vat_obs_df.index.year])
    vat_obs_df.index.names = ['year']

    vat_df_all=pd.concat([vat_df_all,vat_obs_df],axis=1)

    print (vat_df_all)

    DF += [vat_df_all]
    nvar += 1
    if (nvar == 5):
        break

# draw figures

title=["Ocean heat content anomaly (0-700m)",
       "Ocean heat content anomaly (0-2000m)",
       "Ocean heat content anomaly (2000m-bottom)",
       "Ocean heat content anomaly (0m-bottom)" ]
ylim = [ [-250, 100], [-300, 150], [-150, 150], [-400, 300] ]

fig  = plt.figure(figsize = (9,15))
fig.suptitle("OHCA OMIP1" , fontsize=20)
outfile = './fig/Fig1g_omip1.png'

axes = [ plt.subplot(4,1,1),
         plt.subplot(4,1,2),
         plt.subplot(4,1,3),
         plt.subplot(4,1,4) ]

nv = 0
for var in var_list:
    DF[nv].plot(y=DF[nv].columns[i+1],color='darkgrey',linewidth=8,ax=axes[nv],ylim=ylim[nv],xlim=[1948,2018])
    DF[nv].plot(y=DF[nv].columns[i],color='darkred',linewidth=4,ax=axes[nv],ylim=ylim[nv],xlim=[1948,2018])
    DF[nv].plot(y=DF[nv].columns[0:i],ax=axes[nv],ylim=ylim[nv],xlim=[1948,2018],title=title[nv])
    axes[nv].set_ylabel('meter',fontsize=12)
    axes[nv].legend(bbox_to_anchor=(1.05,1.0))
    nv += 1

plt.subplots_adjust(left=0.1,right=0.8,hspace=0.3)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
