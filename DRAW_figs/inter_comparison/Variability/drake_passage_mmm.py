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

metainfo = [json.load(open('json/drake_passage_omip1.json')),
            json.load(open('json/drake_passage_omip2.json'))]

outfile = './fig/Fig1b.png'

dtime_all = pd.date_range('1948-01-01','2018-12-31',freq='AS-JAN')
yr_all = 71

drake_annual_model = []

template = 'Institution {0:3d} is {1:s}'

for omip in range(2):

    i=0

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')
        yr_cyc = 62
    else:
        dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')
        yr_cyc = 61


    for inst in metainfo[omip].keys():

        print (template.format(i,inst))

        fac=float(metainfo[omip][inst]['factor'])
        vname=metainfo[omip][inst]['name']
        nth_line=int(metainfo[omip][inst]['line'])
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']
        total_cycle=int(metainfo[omip][inst]['cycle'])
        
        print (infile, fac, nth_line, vname)

        nc = netCDF4.Dataset(infile,'r')

        if nth_line > 0:
            drake_tmp = nc.variables[vname][:,:]
            if inst == 'AWI-FESOM':
                drake = drake_tmp[nth_line-1,:]
            else:
                drake = drake_tmp[:,nth_line-1]
        else:
            drake_tmp = nc.variables[vname][:]
            drake = drake_tmp[:]

        nc.close()

        print ('length of data =', len(drake))
        num_data = len(drake)
        drake_lastcyc = np.array(np.zeros(yr_cyc),dtype=np.float64)
        drake_lastcyc[0:yr_cyc] = drake[num_data-yr_cyc:num_data]

        col = pd.Index([inst + '-OMIP' + str(omip+1) ],name='institution')

        drake_df = pd.DataFrame(drake_lastcyc*fac,index=dtime,columns=col)

        drake_df.index.names = ['year']

        if i == 0:
            drake_df_all=drake_df
        else:
            drake_df_all=pd.concat([drake_df_all,drake_df],axis=1)
            
        print (drake_df_all)

        i=i+1

    i=i+1
    drake_mean=drake_df_all.mean(axis=1)
    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    drake_mean_df = pd.DataFrame(drake_mean,columns=col)
    drake_annual_model_tmp=pd.concat([drake_df_all,drake_mean_df],axis=1)
    drake_annual_model += [drake_annual_model_tmp]
    print (drake_annual_model[omip])


drake_obs1 = np.array(np.zeros(yr_all),dtype=np.float64)
drake_obs1[:] = 134.0
drake_obs1_df = pd.DataFrame(pd.Series(drake_obs1,name='Cunningham et al. (2003)',index=dtime_all))
drake_obs2 = np.array(np.zeros(yr_all),dtype=np.float64)
drake_obs2[:] = 173.3
drake_obs2_df = pd.DataFrame(pd.Series(drake_obs2,name='Donohue et al. (2016)',index=dtime_all))

drake_annual_all=pd.concat([drake_annual_model[0],drake_annual_model[1],drake_obs1_df,drake_obs2_df],axis=1)

# draw figures

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
drake_annual_all.plot(y=drake_annual_all.columns[1*i-1],ax=axes,ylim=[100,200],color='darkred',linewidth=4,title='Drake Passage (last cycle)')
drake_annual_all.plot(y=drake_annual_all.columns[2*i-1],ax=axes,ylim=[100,200],color='darkblue',linewidth=4)
drake_annual_all.plot(y=drake_annual_all.columns[2*i],ax=axes,ylim=[100,200],color='grey',linewidth=4)
drake_annual_all.plot(y=drake_annual_all.columns[2*i+1],ax=axes,ylim=[100,200],color='darkgrey',linewidth=4)
#
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.25,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
