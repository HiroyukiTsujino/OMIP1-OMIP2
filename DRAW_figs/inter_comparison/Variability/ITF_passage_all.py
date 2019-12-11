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

lineinfo = json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1c_all.png'
suptitle = 'Indonesian Throughflow Transport'

#dtime_all = pd.date_range('1948-01-01','2018-12-31',freq='AS-JAN')
dtime_all = np.array(np.linspace(1948,2018,71))
yr_all = 71

template = 'Institution {0:3d} is {1:s}'

ITF_annual_model = []
ITF_mean_df = []

lincol = []
linsty = []
modnam = []
nummodel = []

for omip in range(2):

    if (omip == 0):
        #dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')
        dtime = np.array(np.linspace(1948,2009,62))
        yr_cyc = 62
    else:
        dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')
        dtime = np.array(np.linspace(1958,2018,61))
        yr_cyc = 61

    coltmp = []
    stytmp = []
    namtmp = []

    i=0
    for inst in metainfo[omip].keys():

        print (template.format(i,inst))

        coltmp +=[lineinfo[inst]["color"]]
        stytmp +=[lineinfo[inst]["style"]]
        namtmp +=[inst]

        fac=float(metainfo[omip][inst]['factor'])
        vname=metainfo[omip][inst]['name']
        nth_line=int(metainfo[omip][inst]['line'])
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']
        total_cycle=int(metainfo[omip][inst]['cycle'])

        print (infile, fac, nth_line, vname)
        col = pd.Index([inst + '-OMIP' + str(omip+1) ],name='institution')

        if (inst == 'AWI-FESOM'):

            ITF_tmp_df = pd.DataFrame()

            if ( omip == 0 ):
                for loop in range(6):
                    loopfile=metainfo[omip][inst]['path'] + str(loop+1) + '/' + 'mfo_Oyr_FESOM1.4_historical_loop' + str(loop+1) + '_gr_194801-200912.nc'
                    nc = netCDF4.Dataset(loopfile,'r')
                    ITF_tmp= nc.variables[vname][:,:]
                    ITF   = ITF_tmp[nth_line-1,:]
                    time_var = nc.variables['time']
                    cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
                    year     = np.array([tmp.year for tmp in cftime]) - 62 * (5-loop)
                    nc.close()

                    #print(year)
                    df_tmp = pd.DataFrame(ITF*fac,index=year,columns=col)
                    ITF_tmp_df = pd.concat([ITF_tmp_df,df_tmp])

            else:
                for loop in range(6):
                    loopfile=metainfo[omip][inst]['path'] + str(loop+1) + '/' + 'mfo_Oyr_FESOM1.4_historical_loop' + str(loop+1) + '_gr_195801-201812.nc'
                    nc = netCDF4.Dataset(loopfile,'r')
                    ITF_tmp= nc.variables[vname][:,:]
                    ITF   = ITF_tmp[nth_line-1,:]
                    time_var = nc.variables['time']
                    cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
                    year     = np.array([tmp.year for tmp in cftime]) - 61 * (5-loop)
                    nc.close()

                    #print(year)
                    df_tmp = pd.DataFrame(ITF*fac,index=year,columns=col)
                    ITF_tmp_df = pd.concat([ITF_tmp_df,df_tmp])

            ITF = ITF_tmp_df.iloc[:,0]
            print ('length of data =', len(ITF))
            num_data = len(ITF)
            ITF_lastcyc = np.array(np.zeros(yr_cyc),dtype=np.float64)
            ITF_lastcyc[0:yr_cyc] = ITF[num_data-yr_cyc:num_data]
            ITF_df = pd.DataFrame(ITF_lastcyc*fac,index=dtime,columns=col)
            ITF_df.index.names = ['year']
            #print(ITF_df)

        else:
        
            nc = netCDF4.Dataset(infile,'r')

            if nth_line > 0:
                ITF_tmp = nc.variables[vname][:,:]
                if (inst == 'AWI-FESOM' or inst == 'CMCC-NEMO' or inst == 'GFDL-MOM'):
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

    ITF_annual_model += [ITF_df_all]

    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    ITF_mean = ITF_df_all.mean(axis=1)
    ITF_mean_df_tmp = pd.DataFrame(ITF_mean,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    ITF_std = ITF_df_all.std(axis=1)
    ITF_std_df_tmp = pd.DataFrame(ITF_std,columns=col)

    ITF_mean_df_tmp = pd.concat([ITF_mean_df_tmp,ITF_std_df_tmp],axis=1)

    ITF_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = ITF_mean_df_tmp.iloc[:,0] - ITF_mean_df_tmp.iloc[:,1]
    ITF_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = ITF_mean_df_tmp.iloc[:,0] + ITF_mean_df_tmp.iloc[:,1]

    ITF_mean_df += [ITF_mean_df_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    modnam += [namtmp]
    nummodel += [i]


ITF_obs1 = np.array(np.zeros(yr_all),dtype=np.float64)
ITF_obs1[:] = 10.7
ITF_obs1_df = pd.DataFrame(pd.Series(ITF_obs1,name='Observational lower bound',index=dtime_all))
ITF_obs2 = np.array(np.zeros(yr_all),dtype=np.float64)
ITF_obs2[:] = 18.7
ITF_obs2_df = pd.DataFrame(pd.Series(ITF_obs2,name='Observational upper bound',index=dtime_all))
ITF_obsm= np.array(np.zeros(yr_all),dtype=np.float64)
ITF_obsm[:] = 15.0
ITF_obsm_df = pd.DataFrame(pd.Series(ITF_obsm,name='Sprintall et al. (2009)',index=dtime_all))

ITF_annual_all=pd.concat([ITF_obs1_df,ITF_obs2_df,ITF_obsm_df,ITF_mean_df[0],ITF_mean_df[1],ITF_annual_model[0],ITF_annual_model[1]],axis=1)

# draw figures

fig  = plt.figure(figsize = (15,9))

fig.suptitle( suptitle, fontsize=20 )

# OMIP1
axes = fig.add_subplot(1,3,1)
ITF_annual_all.plot(y=ITF_annual_all.columns[2],ax=axes,ylim=[5,23],color='darkgrey',linewidth=2,title='(a) OMIP1')
for ii in range(nummodel[0]):
    #print(ii)
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    #print(ii,linecol,linesty)
    ITF_annual_model[0].plot(y=ITF_annual_model[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[5,23],label=inst)
    axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
    axes.legend(bbox_to_anchor=(4.08,0.8))
    plt.subplots_adjust(left=0.1,right=0.82)

# OMIP2
axes = fig.add_subplot(1,3,2)
ITF_annual_all.plot(y=ITF_annual_all.columns[2],ax=axes,ylim=[5,23],color='darkgrey',linewidth=2,title='(b) OMIP2',legend=False)
for ii in range(nummodel[1]):
    #print(ii)
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    #print(ii,linecol,linesty)
    ITF_annual_model[1].plot(y=ITF_annual_model[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[5,23],legend=False)
    #axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
    axes.set_ylabel('')
    #axes.legend(bbox_to_anchor=(0.0,0.0),loc='lower left')
    plt.subplots_adjust(left=0.1,right=0.82)

# MMM

axes = fig.add_subplot(1,3,3)
ITF_annual_all.plot(y=ITF_annual_all.columns[2],color='darkgrey',linewidth=2,ax=axes,ylim=[5,23],title='(c) MMM')
ITF_annual_all.plot(y=ITF_annual_all.columns[0],color='grey',linewidth=2,ax=axes,ylim=[5,23])
ITF_annual_all.plot(y=ITF_annual_all.columns[1],color='grey',linewidth=2,ax=axes,ylim=[5,23])
axes.fill_between(x=ITF_annual_all.index,y1=ITF_annual_all['OMIP1-min'],y2=ITF_annual_all['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=ITF_annual_all.index,y1=ITF_annual_all['OMIP2-min'],y2=ITF_annual_all['OMIP2-max'],alpha=0.5,facecolor='lightblue')
ITF_annual_all.plot(y=ITF_annual_all.columns[3],color='darkred' ,linewidth=2,ax=axes,ylim=[5,23])
ITF_annual_all.plot(y=ITF_annual_all.columns[7],color='darkblue',linewidth=2,ax=axes,ylim=[5,23])
#axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{m}^{-3}$',fontsize=12)
axes.set_ylabel('')
axes.legend(bbox_to_anchor=(1.8,1.0))
plt.subplots_adjust(left=0.1,right=0.82)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
