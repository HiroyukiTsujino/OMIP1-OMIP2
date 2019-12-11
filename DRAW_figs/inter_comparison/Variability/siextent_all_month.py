# -*- coding: utf-8 -*-
#
import sys
import json
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

######

metainfo = [json.load(open('json/siextent_omip1.json')),
            json.load(open('json/siextent_omip2.json'))]

lineinfo=json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1e_all'
suptitle = 'Sea ice extent'

template = 'Institution {0:3d} is {1:s}'

ystr = 1980
yend = 2009
nyr = yend - ystr + 1

month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

#########################################################
# observation

file_siextent = "../refdata/NSIDC_SII/nsidc_si_index_monthly_NH_SH.nc"
nco = netCDF4.Dataset(file_siextent,'r')
siextentn_obs = nco.variables['siextentn'][:]
siextents_obs = nco.variables['siextents'][:]
time_var_obs = nco.variables['time']
cftime = num2date(time_var_obs[:],time_var_obs.units)
nco.close()

col = pd.Index(['NSIDC_SII'],name='institution')
siextentn_obs_df = pd.DataFrame(siextentn_obs,index=cftime,columns=col)
siextents_obs_df = pd.DataFrame(siextents_obs,index=cftime,columns=col)
print (siextentn_obs_df)
print (siextents_obs_df)
siextentn_obs_df = siextentn_obs_df.set_index([siextentn_obs_df.index.year,siextentn_obs_df.index.month,siextentn_obs_df.index])
siextents_obs_df = siextents_obs_df.set_index([siextents_obs_df.index.year,siextents_obs_df.index.month,siextents_obs_df.index])
siextentn_obs_df.index.names = ['year','month','date']
siextents_obs_df.index.names = ['year','month','date']

#########################################################

siextentn_model_omip1 = []
siextents_model_omip1 = []
siextentn_mean_omip1 = []
siextents_mean_omip1 = []

siextentn_model_omip2 = []
siextents_model_omip2 = []
siextentn_mean_omip2 = []
siextents_mean_omip2 = []

lincol = []
linsty = []
modnam = []
nummodel = []

for omip in range(2):

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-01',freq='MS')
    else:
        dtime = pd.date_range('1958-01-01','2018-12-01',freq='MS')

    coltmp = []
    stytmp = []
    namtmp = []

    i=0

    for inst in metainfo[omip].keys():

        print (template.format(i,inst))

        coltmp +=[lineinfo[inst]["color"]]
        stytmp +=[lineinfo[inst]["style"]]
        namtmp +=[inst]

        factor=float(metainfo[omip][inst]['factor'])

        print (inst, factor)

        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fnamen']
        nc = netCDF4.Dataset(infile,'r')
        if (inst == 'AWI-FESOM'):
            nc.set_auto_mask(False)
            siextentn = nc.variables['siextentn'][:]
        elif (inst == 'Kiel-NEMO'):
            nc.set_auto_maskandscale(False)
            siextentn = nc.variables['siextn'][:,0,0]
        else:
            siextentn = nc.variables['siextentn'][:]
            
        nc.close()

        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fnames']
        nc = netCDF4.Dataset(infile,'r')
        if inst == 'AWI-FESOM':
            nc.set_auto_mask(False)
            siextents = nc.variables['siextents'][:]
        elif inst == 'Kiel-NEMO':
            nc.set_auto_maskandscale(False)
            siextents = nc.variables['siexts'][:,0,0]
        else:
            siextents = nc.variables['siextents'][:]

        nc.close()

        col = pd.Index([inst + '-OMIP' +str(omip+1)],name='institution')
        siextentn_df = pd.DataFrame(siextentn*factor,index=dtime,columns=col)
        siextentn_df = siextentn_df.set_index([siextentn_df.index.year,siextentn_df.index.month,siextentn_df.index])
        siextentn_df.index.names = ['year','month','date']

        siextents_df = pd.DataFrame(siextents*factor,index=dtime,columns=col)
        siextents_df = siextents_df.set_index([siextents_df.index.year,siextents_df.index.month,siextents_df.index])
        siextents_df.index.names = ['year','month','date']

        if i == 0:
            siextentn_df_all=siextentn_df
            siextents_df_all=siextents_df
        else:
            siextentn_df_all=pd.concat([siextentn_df_all,siextentn_df],axis=1)
            siextents_df_all=pd.concat([siextents_df_all,siextents_df],axis=1)

        print (siextentn_df_all)
        print (siextents_df_all)

        i+=1

    # construct multi model mean

    # NH
    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    siextentn_mmm=siextentn_df_all.mean(axis=1)
    siextentn_mmm_df = pd.DataFrame(siextentn_mmm,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    siextentn_mms=siextentn_df_all.std(axis=1)
    siextentn_mms_df = pd.DataFrame(siextentn_mms,columns=col)

    siextentn_df_mean = pd.concat([siextentn_mmm_df,siextentn_mms_df],axis=1)
    siextentn_df_mean['OMIP' + str(omip+1) + '-min'] = siextentn_df_mean.iloc[:,0] - siextentn_df_mean.iloc[:,1]
    siextentn_df_mean['OMIP' + str(omip+1) + '-max'] = siextentn_df_mean.iloc[:,0] + siextentn_df_mean.iloc[:,1]

    # SH
    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    siextents_mmm=siextents_df_all.mean(axis=1)
    siextents_mmm_df = pd.DataFrame(siextents_mmm,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    siextents_mms=siextents_df_all.std(axis=1)
    siextents_mms_df = pd.DataFrame(siextents_mms,columns=col)

    siextents_df_mean = pd.concat([siextents_mmm_df,siextents_mms_df],axis=1)
    siextents_df_mean['OMIP' + str(omip+1) + '-min'] = siextents_df_mean.iloc[:,0] - siextents_df_mean.iloc[:,1]
    siextents_df_mean['OMIP' + str(omip+1) + '-max'] = siextents_df_mean.iloc[:,0] + siextents_df_mean.iloc[:,1]

    #####
    
    for mon in range(1,13):

        monarg='month == ' + str(mon)

        siextentn_df_all_mon = siextentn_df_all.query(monarg)
        siextentn_df_mean_mon = siextentn_df_mean.query(monarg)
        siextents_df_all_mon = siextents_df_all.query(monarg)
        siextents_df_mean_mon = siextents_df_mean.query(monarg)

        if (omip == 0):
            siextentn_model_omip1 += [siextentn_df_all_mon]
            siextents_model_omip1 += [siextents_df_all_mon]
            siextentn_mean_omip1 += [siextentn_df_mean_mon]
            siextents_mean_omip1 += [siextents_df_mean_mon]
        else:
            siextentn_model_omip2 += [siextentn_df_all_mon]
            siextents_model_omip2 += [siextents_df_all_mon]
            siextentn_mean_omip2 += [siextentn_df_mean_mon]
            siextents_mean_omip2 += [siextents_df_mean_mon]

    lincol += [coltmp]
    linsty += [stytmp]
    modnam += [namtmp]
    nummodel += [i]

###################

siextentn_df = []
siextents_df = []

for mon in range(1,13):
    monarg='month == ' + str(mon)
    siextentn_obs_df_mon=siextentn_obs_df.query(monarg)
    siextents_obs_df_mon=siextents_obs_df.query(monarg)
    siextentn_mon=pd.concat([siextentn_obs_df_mon,siextentn_mean_omip1[mon-1],siextentn_mean_omip2[mon-1],siextentn_model_omip1[mon-1],siextentn_model_omip2[mon-1]],axis=1)
    siextents_mon=pd.concat([siextents_obs_df_mon,siextents_mean_omip1[mon-1],siextents_mean_omip2[mon-1],siextents_model_omip1[mon-1],siextents_model_omip2[mon-1]],axis=1)
    siextentn_mon = siextentn_mon.reset_index()
    siextentn_mon.drop(['month','date'], axis='columns', inplace=True)
    siextentn_mon_clean=siextentn_mon.set_index('year')
    print (siextentn_mon_clean)
    siextents_mon = siextents_mon.reset_index()
    siextents_mon.drop(['month','date'], axis='columns', inplace=True)
    siextents_mon_clean=siextents_mon.set_index('year')
    print (siextents_mon_clean)
    siextentn_df += [siextentn_mon_clean]
    siextents_df += [siextents_mon_clean]

for mon in range(1,13):
    siextentn_model_omip1[mon-1] = siextentn_model_omip1[mon-1].reset_index()
    siextentn_model_omip1[mon-1].drop(['month','date'], axis='columns', inplace=True)
    siextentn_model_omip1[mon-1] = siextentn_model_omip1[mon-1].set_index('year')
    siextents_model_omip1[mon-1] = siextents_model_omip1[mon-1].reset_index()
    siextents_model_omip1[mon-1].drop(['month','date'], axis='columns', inplace=True)
    siextents_model_omip1[mon-1] = siextents_model_omip1[mon-1].set_index('year')

    siextentn_model_omip2[mon-1] = siextentn_model_omip2[mon-1].reset_index()
    siextentn_model_omip2[mon-1].drop(['month','date'], axis='columns', inplace=True)
    siextentn_model_omip2[mon-1] = siextentn_model_omip2[mon-1].set_index('year')
    siextents_model_omip2[mon-1] = siextents_model_omip2[mon-1].reset_index()
    siextents_model_omip2[mon-1].drop(['month','date'], axis='columns', inplace=True)
    siextents_model_omip2[mon-1] = siextents_model_omip2[mon-1].set_index('year')
    

########################
# Statistical Analysis

ystr=1980
yend=2009

hemi = 'NH'

for mon in [3, 9]:

    obs=siextentn_df[mon-1].loc[str(ystr):str(yend),'NSIDC_SII'].values

    obs_sum = obs.sum()
    obs_mean = obs_sum / (yend-ystr+1)
    varobs_sum = ((obs-obs_mean)**2).sum() 
    varobs = varobs_sum / (yend-ystr+1)
    stdobs = np.sqrt(varobs)
    print(obs.std(),stdobs)

    for omip in range(2):
        dict_interannual={}
        for ii in range(nummodel[omip]):

            inst=modnam[omip][ii]
            simnam=inst+'-OMIP'+str(omip+1)
            if (omip == 0):
                sim=siextentn_model_omip1[mon-1].loc['1980':'2009',simnam].values
            else:
                sim=siextentn_model_omip2[mon-1].loc['1980':'2009',simnam].values

            sim_sum = sim.sum()
            sim_mean = sim_sum / (yend-ystr+1)
            varsim_sum = ((sim-sim_mean)**2).sum()
            varsim = varsim_sum / (yend-ystr+1)
            stdsim = np.sqrt(varsim)
            print('standard deviation ', sim.std(), stdsim)

            corr_sum = ((obs-obs_mean)*(sim-sim_mean)).sum()
            corr = corr_sum / (yend-ystr+1) / stdsim / stdobs
            print('Correlation ', np.corrcoef(obs,sim),corr)

            dist_sum = (((obs-obs_mean) - (sim-sim_mean))**2).sum()
            dist = dist_sum / (yend-ystr+1)
            print('Distance (raw)          ', dist)

            dist_tmp = varobs + varsim - 2.0 * stdobs * stdsim * corr
            print('Distance (confirmation) ', dist_tmp)
        
            dict_interannual[inst]=[corr,stdsim]

        dict_interannual['Reference']=[1.0,stdobs]
        summary=pd.DataFrame(dict_interannual,index=['Correlation','Standard deviation'])
        summary_t=summary.T
        print (summary_t)
        summary_t.to_csv('csv/siextent_'+ hemi + '_' + month[mon-1] + '_OMIP' + str(omip+1) + '.csv')

    
hemi = 'SH'

for mon in [3, 9]:

    obs=siextents_df[mon-1].loc[str(ystr):str(yend),'NSIDC_SII'].values

    obs_sum = obs.sum()
    obs_mean = obs_sum / (yend-ystr+1)
    varobs_sum = ((obs-obs_mean)**2).sum() 
    varobs = varobs_sum / (yend-ystr+1)
    stdobs = np.sqrt(varobs)
    print(obs.std(),stdobs)

    for omip in range(2):
        dict_interannual={}
        for ii in range(nummodel[omip]):

            inst=modnam[omip][ii]
            simnam=inst+'-OMIP'+str(omip+1)
            if (omip == 0):
                sim=siextents_model_omip1[mon-1].loc['1980':'2009',simnam].values
            else:
                sim=siextents_model_omip2[mon-1].loc['1980':'2009',simnam].values

            sim_sum = sim.sum()
            sim_mean = sim_sum / (yend-ystr+1)
            varsim_sum = ((sim-sim_mean)**2).sum()
            varsim = varsim_sum / (yend-ystr+1)
            stdsim = np.sqrt(varsim)
            print('standard deviation ', sim.std(), stdsim)

            corr_sum = ((obs-obs_mean)*(sim-sim_mean)).sum()
            corr = corr_sum / (yend-ystr+1) / stdsim / stdobs
            print('Correlation ', np.corrcoef(obs,sim),corr)

            dist_sum = (((obs-obs_mean) - (sim-sim_mean))**2).sum()
            dist = dist_sum / (yend-ystr+1)
            print('Distance (raw)          ', dist)

            dist_tmp = varobs + varsim - 2.0 * stdobs * stdsim * corr
            print('Distance (confirmation) ', dist_tmp)
        
            dict_interannual[inst]=[corr,stdsim]

        dict_interannual['Reference']=[1.0,stdobs]
        summary=pd.DataFrame(dict_interannual,index=['Correlation','Standard deviation'])
        summary_t=summary.T
        print (summary_t)
        summary_t.to_csv('csv/siextent_'+ hemi + '_' + month[mon-1] + '_OMIP' + str(omip+1) + '.csv')

    
########################
# Draw Figures

fig  = plt.figure(figsize = (11,8))
fig.suptitle( suptitle, fontsize=18 )

# March NH

mon = 3

axes = fig.add_subplot(4,3,1)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[0],ax=axes,ylim=[10,20],color='black',linewidth=2,legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextentn_model_omip1[mon-1].plot(y=siextentn_model_omip1[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[10,20],label=inst)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)
leg = axes.legend(bbox_to_anchor=(3.3,0.5),loc='upper left')
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
axes.set_title('(a) '+month[mon-1]+' NH OMIP1',{'fontsize':10,'verticalalignment':'top'})
#plt.subplots_adjust(left=0.1,right=0.8,bottom=0.1,top=0.95)

axes = fig.add_subplot(4,3,2)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[0],ax=axes,ylim=[10,20],color='black',linewidth=2,legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    inst=modnam[1][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextentn_model_omip2[mon-1].plot(y=siextentn_model_omip2[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[10,20],legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_title('(b) '+month[mon-1]+' NH OMIP2',{'fontsize':10, 'verticalalignment':'top'})
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)
#axes.legend(bbox_to_anchor=(0.0,0.0),loc='lower left')

axes = fig.add_subplot(4,3,3)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[0],ax=axes,ylim=[10,20],color='darkgreen',linewidth=2)
axes.fill_between(x=siextentn_df[mon-1].index,y1=siextentn_df[mon-1]['OMIP1-min'],y2=siextentn_df[mon-1]['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextentn_df[mon-1].index,y1=siextentn_df[mon-1]['OMIP2-min'],y2=siextentn_df[mon-1]['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[1],ax=axes,ylim=[10,20],color='darkred',linewidth=2)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[5],ax=axes,ylim=[10,20],color='darkblue',linewidth=2)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_title('(c) '+month[mon-1]+' NH MMM',{'fontsize':10, 'verticalalignment':'top'})
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
leg = axes.legend(bbox_to_anchor=(1.01,1.0),loc='upper left')
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
plt.subplots_adjust(left=0.1,right=0.83,bottom=0.05,top=0.90)

# September NH

mon = 9

axes = fig.add_subplot(4,3,4)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[0],ax=axes,ylim=[0,10],color='black',linewidth=2,legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextentn_model_omip1[mon-1].plot(y=siextentn_model_omip1[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[0,10],legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)
axes.set_title('(d) '+month[mon-1]+' NH OMIP1',{'fontsize':10, 'verticalalignment':'top'})

axes = fig.add_subplot(4,3,5)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[0],ax=axes,ylim=[0,10],color='black',linewidth=2,legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextentn_model_omip2[mon-1].plot(y=siextentn_model_omip2[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[0,10],legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_title('(e) '+month[mon-1]+' NH OMIP2',{'fontsize':10, 'verticalalignment':'top'})

axes = fig.add_subplot(4,3,6)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[0],ax=axes,ylim=[0,10],color='darkgreen',linewidth=2,legend=False)
axes.fill_between(x=siextentn_df[mon-1].index,y1=siextentn_df[mon-1]['OMIP1-min'],y2=siextentn_df[mon-1]['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextentn_df[mon-1].index,y1=siextentn_df[mon-1]['OMIP2-min'],y2=siextentn_df[mon-1]['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[1],ax=axes,ylim=[0,10],color='darkred',linewidth=2,legend=False)
siextentn_df[mon-1].plot(y=siextentn_df[mon-1].columns[5],ax=axes,ylim=[0,10],color='darkblue',linewidth=2,legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_title('(f) '+month[mon-1]+' NH MMM',{'fontsize':10, 'verticalalignment':'top'})
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
#axes.legend(bbox_to_anchor=(1.07,1.0))
plt.subplots_adjust(left=0.1,right=0.8,bottom=0.05,top=0.90)

# SH Feb

mon = 3

axes = fig.add_subplot(4,3,7)
siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[0],ax=axes,ylim=[0,7],color='black',linewidth=2,legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextents_model_omip1[mon-1].plot(y=siextents_model_omip1[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[0,7],legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)
axes.set_title('(g) '+month[mon-1]+' SH OMIP1',{'fontsize':10, 'verticalalignment':'top'})

axes = fig.add_subplot(4,3,8)
siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[0],ax=axes,ylim=[0,7],color='black',linewidth=2,legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextents_model_omip2[mon-1].plot(y=siextentn_model_omip2[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[0,7],legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_title('(h) '+month[mon-1]+' SH OMIP2',{'fontsize':10, 'verticalalignment':'top'})
    
axes = fig.add_subplot(4,3,9)

siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[0],ax=axes,ylim=[0,7],color='darkgreen',linewidth=2,legend=False)
axes.fill_between(x=siextents_df[mon-1].index,y1=siextents_df[mon-1]['OMIP1-min'],y2=siextents_df[mon-1]['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextents_df[mon-1].index,y1=siextents_df[mon-1]['OMIP2-min'],y2=siextents_df[mon-1]['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[1],ax=axes,ylim=[0,7],color='darkred',linewidth=2,legend=False)
siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[5],ax=axes,ylim=[0,7],color='darkblue',linewidth=2,legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('')
axes.set_title('(i) '+month[mon-1]+' SH MMM',{'fontsize':10, 'verticalalignment':'top'})
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
#axes.legend(bbox_to_anchor=(1.07,1.0))
#plt.subplots_adjust(left=0.1,right=0.8,bottom=0.05,top=0.90)


### SH Aug

mon = 9

axes = fig.add_subplot(4,3,10)
siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[0],ax=axes,ylim=[12,27],color='black',linewidth=2,legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextents_model_omip1[mon-1].plot(y=siextents_model_omip1[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[12,27],legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('year')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)
axes.set_title('(j) '+month[mon-1]+' SH OMIP1',{'fontsize':10, 'verticalalignment':'top'})

axes = fig.add_subplot(4,3,11)

siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[0],ax=axes,ylim=[12,27],color='black',linewidth=2,legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    if (linesty == 'dashed'):
        lwidth=1.2
    else:
        lwidth=1.0
    siextents_model_omip2[mon-1].plot(y=siextents_model_omip2[mon-1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[12,27],legend=False)

axes.tick_params(labelsize=9)
axes.set_xlabel('year')
axes.set_title('(k) '+month[mon-1]+' SH OMIP2',{'fontsize':10, 'verticalalignment':'top'})

axes = fig.add_subplot(4,3,12)

siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[0],ax=axes,ylim=[12,27],color='darkgreen',linewidth=2,legend=False)
axes.fill_between(x=siextents_df[mon-1].index,y1=siextents_df[mon-1]['OMIP1-min'],y2=siextents_df[mon-1]['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextents_df[mon-1].index,y1=siextents_df[mon-1]['OMIP2-min'],y2=siextents_df[mon-1]['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[1],ax=axes,ylim=[12,27],color='darkred',linewidth=2,legend=False)
siextents_df[mon-1].plot(y=siextents_df[mon-1].columns[5],ax=axes,ylim=[12,27],color='darkblue',linewidth=2,legend=False)

axes.set_title('(l) '+month[mon-1]+' SH MMM',{'fontsize':10, 'verticalalignment':'top'})
axes.tick_params(labelsize=9)
axes.set_xlabel('year')
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
#axes.legend(bbox_to_anchor=(1.07,1.0))

plt.subplots_adjust(left=0.05,right=0.82,bottom=0.06,top=0.92,hspace=0.25,wspace=0.15)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
