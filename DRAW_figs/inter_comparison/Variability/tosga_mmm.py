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

metainfo = [json.load(open('json/tosga_omip1.json')),
            json.load(open('json/tosga_omip2.json'))]

outfile = './fig/Fig1d_mmm.png'

# COBE-SST

file_cobe = "../COBE-SST/netCDF/COBESST_glbm_monthly_194801-201812.nc"
nccobe = netCDF4.Dataset(file_cobe,'r')
tosga_cobe = nccobe.variables['tosga'][:]
time_var_cobe = nccobe.variables['time']
cftime = num2date(time_var_cobe[:],time_var_cobe.units)
nccobe.close()

col = pd.Index(['COBE-SST'],name='institution')
tosga_cobe_df = pd.DataFrame(tosga_cobe,index=cftime,columns=col)
print (tosga_cobe_df)
tosga_cobe_df = tosga_cobe_df.set_index([tosga_cobe_df.index.year, tosga_cobe_df.index])
tosga_cobe_df.index.names = ['year','date']
tosga_cobe_annual=tosga_cobe_df.mean(level='year')

# AMIP-SST

file_amip = "../AMIP-SST/netCDF/AMIPSST_glbm_monthly_194801-201812.nc"
ncamip = netCDF4.Dataset(file_amip,'r')
tosga_amip_tmp = ncamip.variables['tosga'][:]
tosga_amip = np.where(tosga_amip_tmp==0.0,np.NaN,tosga_amip_tmp)

time_var_amip = ncamip.variables['time']
cftime = num2date(time_var_amip[:],time_var_amip.units)
ncamip.close()

col = pd.Index(['PCMDI-SST'],name='institution')
tosga_amip_df = pd.DataFrame(tosga_amip,index=cftime,columns=col)
print (tosga_amip_df)
tosga_amip_df = tosga_amip_df.set_index([tosga_amip_df.index.year, tosga_amip_df.index])
tosga_amip_df.index.names = ['year','date']
tosga_amip_annual=tosga_amip_df.mean(level='year')

###############

tosga_annual_model = []
tosga_mean_df = []
template = 'Institution {0:3d} is {1:s}'

for omip in range(2):

    i=0

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-01',freq='MS')
    else:
        dtime = pd.date_range('1958-01-01','2018-12-01',freq='MS')

    for inst in metainfo[omip].keys():

        print (template.format(i,inst))

        offset=float(metainfo[omip][inst]['offset'])
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']

        print (infile, offset)

        nc = netCDF4.Dataset(infile,'r')
        if (inst == 'Kiel-NEMO'):
            tosga = nc.variables['tosga'][:,0,0]
        else:
            tosga = nc.variables['tosga'][:]
        nc.close()

        col = pd.Index([inst],name='institution')
        tosga_df = pd.DataFrame(tosga + offset,index=dtime,columns=col)
        tosga_df = tosga_df.set_index([tosga_df.index.year,tosga_df.index])
        tosga_df.index.names = ['year','date']

        if i == 0:
            tosga_df_all=tosga_df
        else:
            tosga_df_all=pd.concat([tosga_df_all,tosga_df],axis=1)
            
        print (tosga_df_all)
        i=i+1

# merge

    tosga_annual_model_tmp=tosga_df_all.mean(level='year')

    tosga_annual_model += [tosga_annual_model_tmp]

    tosga_annual_tmp=tosga_df_all.mean(level='year')
    tosga_mean=tosga_annual_tmp.mean(axis=1)
    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    tosga_mean_df_tmp = pd.DataFrame(tosga_mean,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    tosga_std=tosga_annual_model_tmp.std(axis=1)
    tosga_std_df_tmp = pd.DataFrame(tosga_std,columns=col)

    tosga_mean_df_tmp = pd.concat([tosga_mean_df_tmp,tosga_std_df_tmp],axis=1)

    tosga_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = tosga_mean_df_tmp.iloc[:,0] - tosga_mean_df_tmp.iloc[:,1]
    tosga_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = tosga_mean_df_tmp.iloc[:,0] + tosga_mean_df_tmp.iloc[:,1]

    tosga_mean_df += [tosga_mean_df_tmp]

tosga_annual_all=pd.concat([tosga_cobe_annual,tosga_amip_annual,tosga_mean_df[0],tosga_mean_df[1],tosga_annual_model[0],tosga_annual_model[1]],axis=1)
print (tosga_annual_all)

fig  = plt.figure(figsize = (15,15))
axes = fig.add_subplot(1,1,1)
tosga_annual_all.plot(y=tosga_annual_all.columns[1],ax=axes,color='darkgrey',linewidth=3,ylim=[17.8,18.8])
#tosga_annual_all.plot(y=tosga_annual_all.columns[1],ax=axes,color='dimgrey',linewidth=2,ylim=[17.7,18.9])
#axes.fill_between(x=tosga_annual_all.index,y1=tosga_annual_all['OMIP1-min'],y2=tosga_annual_all['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
#axes.fill_between(x=tosga_annual_all.index,y1=tosga_annual_all['OMIP2-min'],y2=tosga_annual_all['OMIP2-max'],alpha=0.5,facecolor='lightblue')
tosga_annual_all.plot(y=tosga_annual_all.columns[2],ax=axes,color='darkred',linewidth=3,ylim=[17.8,18.8])
tosga_annual_all.plot(y=tosga_annual_all.columns[6],ax=axes,color='darkblue',linewidth=3,ylim=[17.8,18.8])

axes.set_ylabel(r'[$^\circ \mathrm{C}$]',fontsize=18)
axes.legend(bbox_to_anchor=(1.25,1.0),fontsize=16)
plt.title('Global mean SST',fontsize=20)
plt.xlabel("[year]",fontsize=18)
plt.tick_params(labelsize=16)
plt.subplots_adjust(left=0.1,right=0.8)
#
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
