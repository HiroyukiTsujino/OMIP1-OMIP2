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

metainfo=json.load(open('json/ITF_passage_omip2.json'))

outfile = './fig/Fig1c_omip2.png'

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

        ITF_df = pd.DataFrame()
        for loop in range(6):
            loopfile=metainfo[inst]['path'] + str(loop+1) + '/' + 'mfo_Oyr_FESOM1.4_historical_loop' + str(loop+1) + '_gr_195801-201812.nc'
            nc = netCDF4.Dataset(loopfile,'r')
            ITF_tmp  = nc.variables[vname][:,:]
            ITF      = ITF_tmp[nth_line-1,:]
            time_var = nc.variables['time']
            cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
            year     = np.array([tmp.year for tmp in cftime]) - 61 * (5-loop)
            nc.close()

            #print(year)
            df_tmp = pd.DataFrame(ITF*fac,index=year,columns=col)
            ITF_df = pd.concat([ITF_df,df_tmp])

        print(ITF_df)

    else:

        nc = netCDF4.Dataset(infile,'r')

        if nth_line > 0:
            ITF_tmp = nc.variables[vname][:,:]
            ITF = ITF_tmp[:,nth_line-1]
        else:
            ITF_tmp = nc.variables[vname][:]
            ITF = ITF_tmp[:]

        nc.close()

        if inst == 'FSU-COAPS':
            ITF_fsu = np.zeros([366])
            ITF_fsu[:] = np.nan
            for cyc in range(5):
                isto = 0 + cyc * 61
                iedo = isto + 57
                istf = 0 + cyc * 58
                iedf = istf + 57
                print (isto, iedo)
                print (istf, iedf)
                ITF_fsu[isto:iedo+1] = ITF[istf:iedf+1]

            cyc = 5
            isto = 0 + cyc * 61
            iedo = isto + 61
            istf = 0 + cyc * 58
            iedf = istf + 61
            print (isto, iedo)
            print (istf, iedf)
            ITF_fsu[isto:iedo+1] = ITF[istf:iedf+1]
            print (ITF_fsu)

         
        print (ITF)

        if total_cycle == 5:
            ITF_df = pd.DataFrame(ITF*fac,index=dtime5,columns=col)
        elif total_cycle == 1:
            ITF_df = pd.DataFrame(ITF*fac,index=dtime1,columns=col)
        else:
            if inst == 'FSU-COAPS':
                ITF_df = pd.DataFrame(ITF_fsu*fac,index=dtime6,columns=col)
            else:
                ITF_df = pd.DataFrame(ITF*fac,index=dtime6,columns=col)
          
    ITF_df.index.names = ['year']

    if i == 0:
      ITF_df_all=ITF_df
    else:
      ITF_df_all=pd.concat([ITF_df_all,ITF_df],axis=1)

    print (ITF_df_all)
    i=i+1


ITF_mean=ITF_df_all.mean(axis=1)
for yr in ITF_mean.index:
    if (yr < 1958):
        ITF_mean.loc[yr] = np.NaN

col = pd.Index(['MMM'],name='institution')
ITF_mean_df = pd.DataFrame(ITF_mean,columns=col)
ITF_df_all=pd.concat([ITF_df_all,ITF_mean_df],axis=1)
print (ITF_df_all)

# draw figures

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
ITF_df_all.plot(y=ITF_df_all.columns[i],ax=axes,ylim=[2,24],color='darkblue',linewidth=4,title='Indonesian through flow OMIP2 (JRA55-do)')
ITF_df_all.plot(y=ITF_df_all.columns[0:i],ax=axes,ylim=[2,24])
#
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)
#
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
