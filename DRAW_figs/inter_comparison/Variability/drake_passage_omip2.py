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

metainfo=json.load(open('json/drake_passage_omip2.json'))

outfile = './fig/Fig1b_omip2.png'

template = 'Institution {0:3d} is {1:s}'
dtime1 = np.arange(1958,2019,1)
dtime5 = np.arange(1714,2019,1)
dtime6fsu = np.arange(1668,2019,1)
dtime6 = np.arange(1653,2019,1)

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
            loopfile=metainfo[inst]['path'] + str(loop+1) + '/' + 'mfo_Oyr_FESOM1.4_historical_loop' + str(loop+1) + '_gr_195801-201812.nc'
            nc = netCDF4.Dataset(loopfile,'r')
            drake_tmp= nc.variables[vname][:,:]
            drake   = drake_tmp[nth_line-1,:]
            time_var = nc.variables['time']
            cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
            year     = np.array([tmp.year for tmp in cftime]) - 61 * (5-loop)
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

        if inst == 'FSU-COAPS':
            drake_fsu = np.zeros([366])
            drake_fsu[:] = np.nan
            for cyc in range(5):
                isto = 0 + cyc * 61
                iedo = isto + 57
                istf = 0 + cyc * 58
                iedf = istf + 57
                print (isto, iedo)
                print (istf, iedf)
                drake_fsu[isto:iedo+1] = drake[istf:iedf+1]

            cyc = 5
            isto = 0 + cyc * 61
            iedo = isto + 61
            istf = 0 + cyc * 58
            iedf = istf + 61
            print (isto, iedo)
            print (istf, iedf)
            drake_fsu[isto:iedo+1] = drake[istf:iedf+1]
            print (drake_fsu)

         
        print (drake)

        if total_cycle == 5:
            drake_df = pd.DataFrame(drake*fac,index=dtime5,columns=col)
        elif total_cycle == 1:
            drake_df = pd.DataFrame(drake*fac,index=dtime1,columns=col)
        else:
            if inst == 'FSU-COAPS':
                drake_df = pd.DataFrame(drake_fsu*fac,index=dtime6,columns=col)
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
    if (yr < 1958):
        drake_mean.loc[yr] = np.NaN

col = pd.Index(['MMM'],name='institution')
drake_mean_df = pd.DataFrame(drake_mean,columns=col)
drake_df_all=pd.concat([drake_df_all,drake_mean_df],axis=1)
print (drake_df_all)

# draw figures

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
drake_df_all.plot(y=drake_df_all.columns[i],ax=axes,ylim=[100,200],color='darkblue',linewidth=4)
drake_df_all.plot(y=drake_df_all.columns[0:i],ax=axes,ylim=[100,200],title='DRAKE_PASSAGE_OMIP2 (JRA55-do)')
#
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)
#
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
