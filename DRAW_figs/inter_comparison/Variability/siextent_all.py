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

outfile = './fig/Fig1e_all.png'
suptitle = 'Sea ice extent'

template = 'Institution {0:3d} is {1:s}'

#########################################################
# observation

file_siextent = "../NSIDC_SII/20190801/nsidc_si_index_monthly_NH_SH.nc"
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

siextentn_model_sep = []
siextentn_model_mar = []

siextents_model_sep = []
siextents_model_mar = []

siextentn_mean_sep = []
siextentn_mean_mar = []

siextents_mean_sep = []
siextents_mean_mar = []

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
    
    siextentn_df_all_sep=siextentn_df_all.query('month == 9')
    siextentn_df_mean_sep=siextentn_df_mean.query('month == 9')

    siextents_df_all_sep=siextents_df_all.query('month == 9')
    siextents_df_mean_sep=siextents_df_mean.query('month == 9')

    siextentn_model_sep += [siextentn_df_all_sep]
    siextents_model_sep += [siextents_df_all_sep]

    siextentn_mean_sep += [siextentn_df_mean_sep]
    siextents_mean_sep += [siextents_df_mean_sep]
    
    siextentn_df_all_mar=siextentn_df_all.query('month == 3')
    siextentn_df_mean_mar=siextentn_df_mean.query('month == 3')

    siextents_df_all_mar=siextents_df_all.query('month == 3')
    siextents_df_mean_mar=siextents_df_mean.query('month == 3')

    siextentn_model_mar += [siextentn_df_all_mar]
    siextents_model_mar += [siextents_df_all_mar]

    siextentn_mean_mar += [siextentn_df_mean_mar]
    siextents_mean_mar += [siextents_df_mean_mar]

    lincol += [coltmp]
    linsty += [stytmp]
    modnam += [namtmp]
    nummodel += [i]

###################
# September

siextentn_obs_df_sep=siextentn_obs_df.query('month == 9')
siextents_obs_df_sep=siextents_obs_df.query('month == 9')
siextentn_sep=pd.concat([siextentn_obs_df_sep,siextentn_mean_sep[0],siextentn_mean_sep[1],siextentn_model_sep[0],siextentn_model_sep[1]],axis=1)
siextents_sep=pd.concat([siextents_obs_df_sep,siextents_mean_sep[0],siextents_mean_sep[1],siextents_model_sep[0],siextents_model_sep[1]],axis=1)

siextentn_sep = siextentn_sep.reset_index()
siextentn_sep.drop(['month','date'], axis='columns', inplace=True)
siextentn_sep_clean=siextentn_sep.set_index('year')
print (siextentn_sep_clean)

siextents_sep = siextents_sep.reset_index()
siextents_sep.drop(['month','date'], axis='columns', inplace=True)
siextents_sep_clean=siextents_sep.set_index('year')
print (siextents_sep_clean)


###################
# March

siextentn_obs_df_mar=siextentn_obs_df.query('month == 3')
siextents_obs_df_mar=siextents_obs_df.query('month == 3')
siextentn_mar=pd.concat([siextentn_obs_df_mar,siextentn_mean_mar[0],siextentn_mean_mar[1],siextentn_model_mar[0],siextentn_model_mar[1]],axis=1)
siextents_mar=pd.concat([siextents_obs_df_mar,siextents_mean_mar[0],siextents_mean_mar[1],siextents_model_mar[0],siextents_model_mar[1]],axis=1)

siextentn_mar = siextentn_mar.reset_index()
siextentn_mar.drop(['month','date'], axis='columns', inplace=True)
siextentn_mar_clean=siextentn_mar.set_index('year')
print (siextentn_mar_clean)

siextents_mar = siextents_mar.reset_index()
siextents_mar.drop(['month','date'], axis='columns', inplace=True)
siextents_mar_clean=siextents_mar.set_index('year')
print (siextents_mar_clean)


for omip in range(2):
    siextentn_model_sep[omip] = siextentn_model_sep[omip].reset_index()
    siextentn_model_sep[omip].drop(['month','date'], axis='columns', inplace=True)
    siextentn_model_sep[omip]=siextentn_model_sep[omip].set_index('year')
    siextentn_model_mar[omip] = siextentn_model_mar[omip].reset_index()
    siextentn_model_mar[omip].drop(['month','date'], axis='columns', inplace=True)
    siextentn_model_mar[omip]=siextentn_model_mar[omip].set_index('year')
    siextents_model_sep[omip] = siextents_model_sep[omip].reset_index()
    siextents_model_sep[omip].drop(['month','date'], axis='columns', inplace=True)
    siextents_model_sep[omip]=siextents_model_sep[omip].set_index('year')
    siextents_model_mar[omip] = siextents_model_mar[omip].reset_index()
    siextents_model_mar[omip].drop(['month','date'], axis='columns', inplace=True)
    siextents_model_mar[omip]=siextents_model_mar[omip].set_index('year')


###################
# Draw Figures

fig  = plt.figure(figsize = (15,12))
fig.suptitle( suptitle, fontsize=20 )

# March NH

axes = fig.add_subplot(4,3,1)
siextentn_mar_clean.plot(y=siextentn_mar_clean.columns[0],ax=axes,ylim=[10,20],color='black',linewidth=2,title='(a) Mar NH OMIP1',legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    siextentn_model_mar[0].plot(y=siextentn_model_mar[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[10,20],label=inst)

axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)
leg = axes.legend(bbox_to_anchor=(4.2,0.6))
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

#plt.subplots_adjust(left=0.1,right=0.8,bottom=0.1,top=0.95)

axes = fig.add_subplot(4,3,2)
siextentn_mar_clean.plot(y=siextentn_mar_clean.columns[0],ax=axes,ylim=[10,20],color='black',linewidth=2,title='(b) Mar NH OMIP2',legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    inst=modnam[1][ii]
    siextentn_model_mar[1].plot(y=siextentn_model_mar[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[10,20],legend=False)

axes.set_xlabel('')
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)
#axes.legend(bbox_to_anchor=(0.0,0.0),loc='lower left')

axes = fig.add_subplot(4,3,3)
siextentn_mar_clean.plot(y=siextentn_mar_clean.columns[0],ax=axes,ylim=[10,20],color='black',linewidth=2,title='(c) Mar NH MMM')
axes.fill_between(x=siextentn_mar_clean.index,y1=siextentn_mar_clean['OMIP1-min'],y2=siextentn_mar_clean['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextentn_mar_clean.index,y1=siextentn_mar_clean['OMIP2-min'],y2=siextentn_mar_clean['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextentn_mar_clean.plot(y=siextentn_mar_clean.columns[1],ax=axes,ylim=[10,20],color='red',linewidth=2)
siextentn_mar_clean.plot(y=siextentn_mar_clean.columns[5],ax=axes,ylim=[10,20],color='blue',linewidth=2)

axes.set_xlabel('')
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
leg = axes.legend(bbox_to_anchor=(1.7,1.0))
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
plt.subplots_adjust(left=0.1,right=0.8,bottom=0.05,top=0.90)

# September NH

axes = fig.add_subplot(4,3,4)
siextentn_sep_clean.plot(y=siextentn_sep_clean.columns[0],ax=axes,ylim=[0,10],color='black',linewidth=2,title='(d) Sep NH OMIP1',legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    siextentn_model_sep[0].plot(y=siextentn_model_sep[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[0,10],legend=False)

axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)

axes = fig.add_subplot(4,3,5)
siextentn_sep_clean.plot(y=siextentn_sep_clean.columns[0],ax=axes,ylim=[0,10],color='black',linewidth=2,title='(e) Sep NH OMIP2',legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    siextentn_model_sep[1].plot(y=siextentn_model_sep[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[0,10],legend=False)

axes.set_xlabel('')

axes = fig.add_subplot(4,3,6)
siextentn_sep_clean.plot(y=siextentn_sep_clean.columns[0],ax=axes,ylim=[0,10],color='black',linewidth=2,title='(f) Sep NH MMM',legend=False)
axes.fill_between(x=siextentn_sep_clean.index,y1=siextentn_sep_clean['OMIP1-min'],y2=siextentn_sep_clean['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextentn_sep_clean.index,y1=siextentn_sep_clean['OMIP2-min'],y2=siextentn_sep_clean['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextentn_sep_clean.plot(y=siextentn_sep_clean.columns[1],ax=axes,ylim=[0,10],color='red',linewidth=2,legend=False)
siextentn_sep_clean.plot(y=siextentn_sep_clean.columns[5],ax=axes,ylim=[0,10],color='blue',linewidth=2,legend=False)

axes.set_xlabel('')
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
#axes.legend(bbox_to_anchor=(1.07,1.0))
plt.subplots_adjust(left=0.1,right=0.8,bottom=0.05,top=0.90)

# March SH

axes = fig.add_subplot(4,3,7)
siextents_mar_clean.plot(y=siextents_mar_clean.columns[0],ax=axes,ylim=[0,7],color='black',linewidth=2,title='(g) Mar SH OMIP1',legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    siextents_model_mar[0].plot(y=siextents_model_mar[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[0,7],legend=False)

axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)

axes = fig.add_subplot(4,3,8)
siextents_mar_clean.plot(y=siextents_mar_clean.columns[0],ax=axes,ylim=[0,7],color='black',linewidth=2,title='(h) Mar SH OMIP2',legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    siextents_model_mar[1].plot(y=siextentn_model_mar[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[0,7],legend=False)

axes.set_xlabel('')

    
axes = fig.add_subplot(4,3,9)

siextents_mar_clean.plot(y=siextents_mar_clean.columns[0],ax=axes,ylim=[0,7],color='black',linewidth=2,title='(i) Mar SH MMM',legend=False)
axes.fill_between(x=siextents_mar_clean.index,y1=siextents_mar_clean['OMIP1-min'],y2=siextents_mar_clean['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextents_mar_clean.index,y1=siextents_mar_clean['OMIP2-min'],y2=siextents_mar_clean['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextents_mar_clean.plot(y=siextents_mar_clean.columns[1],ax=axes,ylim=[0,7],color='red',linewidth=2,legend=False)
siextents_mar_clean.plot(y=siextents_mar_clean.columns[5],ax=axes,ylim=[0,7],color='blue',linewidth=2,legend=False)

axes.set_xlabel('')
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
#axes.legend(bbox_to_anchor=(1.07,1.0))
plt.subplots_adjust(left=0.1,right=0.8,bottom=0.05,top=0.90)

axes = fig.add_subplot(4,3,10)
siextents_sep_clean.plot(y=siextents_sep_clean.columns[0],ax=axes,ylim=[12,27],color='black',linewidth=2,title='(j) Sep SH OMIP1',legend=False)
for ii in range(nummodel[0]):
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    siextents_model_sep[0].plot(y=siextents_model_sep[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[12,27],legend=False)

axes.set_xlabel('year')
axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=10)

axes = fig.add_subplot(4,3,11)

siextents_sep_clean.plot(y=siextents_sep_clean.columns[0],ax=axes,ylim=[12,27],color='black',linewidth=2,title='(k) Sep SH OMIP2',legend=False)
for ii in range(nummodel[1]):
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    siextents_model_sep[1].plot(y=siextents_model_sep[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[12,27],legend=False)

axes.set_xlabel('year')

axes = fig.add_subplot(4,3,12)

siextents_sep_clean.plot(y=siextents_sep_clean.columns[0],ax=axes,ylim=[12,27],color='black',linewidth=2,title='(l) Sep SH MMM',legend=False)
axes.fill_between(x=siextents_sep_clean.index,y1=siextents_sep_clean['OMIP1-min'],y2=siextents_sep_clean['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=siextents_sep_clean.index,y1=siextents_sep_clean['OMIP2-min'],y2=siextents_sep_clean['OMIP2-max'],alpha=0.5,facecolor='lightblue')
siextents_sep_clean.plot(y=siextents_sep_clean.columns[1],ax=axes,ylim=[12,27],color='red',linewidth=2,legend=False)
siextents_sep_clean.plot(y=siextents_sep_clean.columns[5],ax=axes,ylim=[12,27],color='blue',linewidth=2,legend=False)

axes.set_xlabel('year')
#axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
#axes.legend(bbox_to_anchor=(1.07,1.0))
plt.subplots_adjust(left=0.1,right=0.8,bottom=0.05,top=0.90)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
