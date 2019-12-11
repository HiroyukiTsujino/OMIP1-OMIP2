# -*- coding: utf-8 -*-
import sys
import glob
import json
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

######

metainfo=json.load(open('json/drake_passage_omip1.json'))

outfile = './fig/Fig1b_omip1.png'

template = 'Institution {0:3d} is {1:s}'
dtime1 = np.arange(1948,2010,1)
dtime5 = np.arange(1700,2010,1)
dtime6 = np.arange(1638,2010,1)

i=0

for inst in metainfo.keys():

    print (template.format(i,inst))

    fac=float(metainfo[inst]['factor'])
    vname=metainfo[inst]['name']
    nth_line=int(metainfo[inst]['line'])
    infile = metainfo[inst]['path'] + '/' + metainfo[inst]['fname']
    total_cycle=int(metainfo[inst]['cycle'])

    print (infile, fac, nth_line, vname)
    col = pd.Index([inst],name='institution')

    if inst == 'AWI-FESOM':

        drake_df = pd.DataFrame()
        for loop in range(6):
            loopfile=metainfo[inst]['path'] + str(loop+1) + '/' + 'mfo_Oyr_FESOM1.4_historical_loop' + str(loop+1) + '_gr_194801-200912.nc'
            nc = netCDF4.Dataset(loopfile,'r')
            drake_tmp= nc.variables[vname][:,:]
            drake   = drake_tmp[nth_line-1,:]
            time_var = nc.variables['time']
            cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
            year     = np.array([tmp.year for tmp in cftime]) - 62 * (5-loop)
            nc.close()

            #print(year)
            df_tmp = pd.DataFrame(drake*fac,index=year,columns=col)
            drake_df = pd.concat([drake_df,df_tmp])

        print(drake_df)

    else:
        nc = netCDF4.Dataset(infile,'r')
        if nth_line > 0:
            drake_tmp = nc.variables[vname][:,:]
            drake = drake_tmp[:,nth_line-1]
        else:
            drake_tmp = nc.variables[vname][:]
            drake = drake_tmp[:]
        nc.close()

        print (drake)
        print ('length of data =', len(drake))


        if total_cycle == 5:
            drake_df = pd.DataFrame(drake*fac,index=dtime5,columns=col)
        elif total_cycle == 1:
            drake_df = pd.DataFrame(drake*fac,index=dtime1,columns=col)
        else:
            drake_df = pd.DataFrame(drake*fac,index=dtime6,columns=col)

    drake_df.index.names = ['year']

    if i == 0:
      drake_df_all=drake_df
    else:
      drake_df_all=pd.concat([drake_df_all,drake_df],axis=1)

    print (drake_df_all)
    i=i+1

drake_mean=drake_df_all.mean(axis=1)
for yr in drake_mean.index:
    if (yr < 1948):
        drake_mean.loc[yr] = np.NaN

col = pd.Index(['MMM'],name='institution')
drake_mean_df = pd.DataFrame(drake_mean,columns=col)
drake_df_all=pd.concat([drake_df_all,drake_mean_df],axis=1)
print (drake_df_all)

# draw figures

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
drake_df_all.plot(y=drake_df_all.columns[i],ax=axes,ylim=[100,200],color='darkred',linewidth=4)
drake_df_all.plot(y=drake_df_all.columns[0:i],ax=axes,ylim=[100,200],title='DRAKE_PASSAGE_OMIP1 (CORE)')
#
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)
#
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
