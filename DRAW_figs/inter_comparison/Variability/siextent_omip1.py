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

metainfo=json.load(open('json/siextent_omip1.json'))
lineinfo=json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1e_omip1.png'

template = 'Institution {0:3d} is {1:s}'
dtime = pd.date_range('1948-01-01','2009-12-01',freq='MS')

i=0

for inst in metainfo.keys():

    print (template.format(i,inst))

    factor=float(metainfo[inst]['factor'])

    print (inst, factor)

    infile = metainfo[inst]['path'] + '/' + metainfo[inst]['fnamen']
    nc = netCDF4.Dataset(infile,'r')
    if inst == 'AWI-FESOM':
        nc.set_auto_mask(False)
        siextentn = nc.variables['siextentn'][:]
    else:
        siextentn = nc.variables['siextentn'][:]

    nc.close()

    infile = metainfo[inst]['path'] + '/' + metainfo[inst]['fnames']
    nc = netCDF4.Dataset(infile,'r')
    if inst == 'AWI-FESOM':
        nc.set_auto_mask(False)
        siextents = nc.variables['siextents'][:]
    else:
        siextents = nc.variables['siextents'][:]

    nc.close()

    col = pd.Index([inst],name='institution')
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
    i=i+1


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


###################
# September

siextentn_obs_df_sep=siextentn_obs_df.query('month == 9')
siextentn_df_all_sep=siextentn_df_all.query('month == 9')

siextents_obs_df_sep=siextents_obs_df.query('month == 9')
siextents_df_all_sep=siextents_df_all.query('month == 9')

siextentn_sep=pd.concat([siextentn_df_all_sep,siextentn_obs_df_sep],axis=1)
siextents_sep=pd.concat([siextents_df_all_sep,siextents_obs_df_sep],axis=1)

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
siextentn_df_all_mar=siextentn_df_all.query('month == 3')

siextents_obs_df_mar=siextents_obs_df.query('month == 3')
siextents_df_all_mar=siextents_df_all.query('month == 3')

siextentn_mar=pd.concat([siextentn_df_all_mar,siextentn_obs_df_mar],axis=1)
siextents_mar=pd.concat([siextents_df_all_mar,siextents_obs_df_mar],axis=1)

siextentn_mar = siextentn_mar.reset_index()
siextentn_mar.drop(['month','date'], axis='columns', inplace=True)
siextentn_mar_clean=siextentn_mar.set_index('year')
print (siextentn_mar_clean)

siextents_mar = siextents_mar.reset_index()
siextents_mar.drop(['month','date'], axis='columns', inplace=True)
siextents_mar_clean=siextents_mar.set_index('year')
print (siextents_mar_clean)


###################
# Draw Figures

fig  = plt.figure(figsize = (12,15))
axes = fig.add_subplot(2,1,1)
axes.set_xlabel('year')

siextentn_mar_clean.plot(y=siextentn_mar_clean.columns[i],ax=axes,ylim=[0,20],color='darkgrey',linewidth=4,title='Sea Ice extent NH OMIP1 (CORE) Mar and Sep')
ii=0
for inst in metainfo.keys():
    coltmp=lineinfo[inst]["color"]
    stytmp=lineinfo[inst]["style"]
    siextentn_mar_clean.plot(y=siextentn_mar_clean.columns[ii],ax=axes,ylim=[0,20],color=coltmp,linewidth=2,linestyle=stytmp)
    ii+=1
    
siextentn_sep_clean.plot(y=siextentn_sep_clean.columns[i],ax=axes,ylim=[0,20],color='darkgrey',linewidth=4,linestyle='solid',label='_nolegend_')
ii=0
for inst in metainfo.keys():
    coltmp=lineinfo[inst]["color"]
    stytmp=lineinfo[inst]["style"]
    siextentn_sep_clean.plot(y=siextentn_sep_clean.columns[ii],ax=axes,ylim=[0,20],color=coltmp,linewidth=2,linestyle=stytmp,label='_nolegend_')
    ii+=1

axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.05,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

axes = fig.add_subplot(2,1,2)
axes.set_xlabel('year')

siextents_mar_clean.plot(y=siextents_mar_clean.columns[i],ax=axes,ylim=[0,27],color='darkgrey',linewidth=4,title='Sea Ice extent SH OMIP1 (CORE) Mar and Sep')
ii=0
for inst in metainfo.keys():
    coltmp=lineinfo[inst]["color"]
    stytmp=lineinfo[inst]["style"]
    siextents_mar_clean.plot(y=siextents_mar_clean.columns[ii],ax=axes,ylim=[0,27],color=coltmp,linewidth=2,linestyle=stytmp)
    ii+=1
    
siextents_sep_clean.plot(y=siextents_sep_clean.columns[i],ax=axes,ylim=[0,27],color='darkgrey',linewidth=4,label='_nolegend_')
ii=0
for inst in metainfo.keys():
    coltmp=lineinfo[inst]["color"]
    stytmp=lineinfo[inst]["style"]
    siextents_sep_clean.plot(y=siextents_sep_clean.columns[ii],ax=axes,ylim=[0,27],color=coltmp,linewidth=2,linestyle=stytmp,label='_nolegend_')
    ii+=1

axes.set_ylabel(r'$\times 10^{6} \mathrm{km}^{2}$',fontsize=12)
axes.legend(bbox_to_anchor=(1.05,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
