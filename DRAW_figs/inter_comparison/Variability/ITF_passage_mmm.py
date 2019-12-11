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

metainfo = [json.load(open('json/ITF_passage_omip1.json')),
            json.load(open('json/ITF_passage_omip2.json'))]

outfile = './fig/Fig1c.png'

dtime_all = pd.date_range('1948-01-01','2018-12-31',freq='AS-JAN')
yr_all = 71

ITF_annual_model = []

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
            ITF_tmp = nc.variables[vname][:,:]
            if inst == 'AWI-FESOM':
                ITF = ITF_tmp[nth_line-1,:]
            else:
                ITF = ITF_tmp[:,nth_line-1]

        else:
            ITF_tmp = nc.variables[vname][:]
            ITF = ITF_tmp[:]

        nc.close()

        print ('length of data =', len(ITF))
        num_data = len(ITF)
        ITF_lastcyc = np.array(np.zeros(yr_cyc),dtype=np.float64)
        ITF_lastcyc[0:yr_cyc] = ITF[num_data-yr_cyc:num_data]

        col = pd.Index([inst + '-OMIP' + str(omip+1) ],name='institution')

        ITF_df = pd.DataFrame(ITF_lastcyc*fac,index=dtime,columns=col)
        ITF_df.index.names = ['year']

        if i == 0:
            ITF_df_all=ITF_df
        else:
            ITF_df_all=pd.concat([ITF_df_all,ITF_df],axis=1)

        print (ITF_df_all)
        i=i+1

    i=i+1
    ITF_mean=ITF_df_all.mean(axis=1)
    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    ITF_mean_df = pd.DataFrame(ITF_mean,columns=col)
    ITF_annual_model_tmp=pd.concat([ITF_df_all,ITF_mean_df],axis=1)
    ITF_annual_model += [ITF_annual_model_tmp]
    print (ITF_annual_model[omip])


ITF_obs1 = np.array(np.zeros(yr_all),dtype=np.float64)
ITF_obs1[:] = 10.7
ITF_obs1_df = pd.DataFrame(pd.Series(ITF_obs1,name='Observational lower bound',index=dtime_all))
ITF_obs2 = np.array(np.zeros(yr_all),dtype=np.float64)
ITF_obs2[:] = 18.7
ITF_obs2_df = pd.DataFrame(pd.Series(ITF_obs2,name='Observational upper bound',index=dtime_all))
ITF_obsm= np.array(np.zeros(yr_all),dtype=np.float64)
ITF_obsm[:] = 15.0
ITF_obsm_df = pd.DataFrame(pd.Series(ITF_obsm,name='Sprintall et al. (2009)',index=dtime_all))

ITF_annual_all=pd.concat([ITF_annual_model[0],ITF_annual_model[1],ITF_obs1_df,ITF_obs2_df,ITF_obsm_df],axis=1)

# draw figures

fig  = plt.figure(figsize = (15,9))
axes = fig.add_subplot(1,1,1)
ITF_annual_all.plot(y=ITF_annual_all.columns[1*i-1],ax=axes,ylim=[5,20],color='darkred',linewidth=4,title='Indonesian through flow (last cycle)')
ITF_annual_all.plot(y=ITF_annual_all.columns[2*i-1],ax=axes,ylim=[5,20],color='darkblue',linewidth=4)
ITF_annual_all.plot(y=ITF_annual_all.columns[2*i]  ,ax=axes,ylim=[5,20],color='grey',linewidth=4)
ITF_annual_all.plot(y=ITF_annual_all.columns[2*i+1],ax=axes,ylim=[5,20],color='grey',linewidth=4)
ITF_annual_all.plot(y=ITF_annual_all.columns[2*i+2],ax=axes,ylim=[5,20],color='darkgrey',linewidth=4)

axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.25,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
