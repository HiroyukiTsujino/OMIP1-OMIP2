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

metainfo=json.load(open('json/tosga_omip2.json'))

outfile = './fig/Fig1d_omip2.png'

template = 'Institution {0:3d} is {1:s}'
dtime = pd.date_range('1958-01-01','2018-12-01',freq='MS')

i=0

for inst in metainfo.keys():

    print (template.format(i,inst))

    offset=float(metainfo[inst]['offset'])
    infile = metainfo[inst]['path'] + '/' + metainfo[inst]['fname']

    print (infile, offset)

    nc = netCDF4.Dataset(infile,'r')
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


#file_tosga = "../TOSGA/moc_transports.nc"
#nco = netCDF4.Dataset(file_tosga,'r')
#amoc_tosga_obs = nco.variables['moc_mar_hc10'][:]
#time_var_obs = nco.variables['time']
#cftime = num2date(time_var_obs[:],time_var_obs.units)
#nco.close()

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

col = pd.Index(['AMIP-SST'],name='institution')
tosga_amip_df = pd.DataFrame(tosga_amip,index=cftime,columns=col)
print (tosga_amip_df)
tosga_amip_df = tosga_amip_df.set_index([tosga_amip_df.index.year, tosga_amip_df.index])
tosga_amip_df.index.names = ['year','date']
tosga_amip_annual=tosga_amip_df.mean(level='year')



# merge

tosga_annual=tosga_df_all.mean(level='year')
tosga_annual_all=pd.concat([tosga_annual,tosga_cobe_annual],axis=1)
tosga_annual_all=pd.concat([tosga_annual_all,tosga_amip_annual],axis=1)

print (tosga_annual_all)

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
tosga_annual_all.plot(y=tosga_annual_all.columns[i+1],ax=axes,color='black',linewidth=4,ylim=[17,19])
tosga_annual_all.plot(y=tosga_annual_all.columns[i],ax=axes,color='darkgrey',linewidth=4,ylim=[17,19])
tosga_annual_all.plot(y=tosga_annual_all.columns[0:i],ax=axes,ylim=[17,19],title='Global mean SST OMIP2 (JRA55-do)')

axes.set_ylabel(r'$^\circ \mathrm{C}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
