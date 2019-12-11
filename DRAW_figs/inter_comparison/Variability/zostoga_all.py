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

metainfo = [json.load(open('json/zostoga_omip1.json')),
            json.load(open('json/zostoga_omip2.json'))]

lineinfo = json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1f_all.png'
suptitle = 'Thermosteric Sea Level anomaly relative to 2005-2009 mean'

template = 'Institution {0:3d} is {1:s}'

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)

zostoga_model = []
zostoga_mean_df = []

lincol = []
linsty = []
modnam = []
nummodel = []

for omip in range(2):

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')
        zostoga_ann = np.array(np.zeros(62),dtype=np.float64)
    else:
        dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')
        zostoga_ann = np.array(np.zeros(61),dtype=np.float64)

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
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']

        print (infile, factor)

        nc = netCDF4.Dataset(infile,'r')

        if ( inst == "GFDL-MOM" ):
            zostoga_mon = nc.variables['zostoga'][:]
            if (omip == 0):
                num_yr = 62
            else:
                num_yr = 61
            
            nd = 0
            for yr in range(0,num_yr):
                wgt_tmp = 0
                for mn in range(1,13):
                    wgt_tmp = wgt_tmp + mon_days[mn-1]
                    zostoga_ann[yr] = zostoga_ann[yr] + zostoga_mon[nd] * mon_days[mn-1]
                    nd += 1

                zostoga_ann[yr] = zostoga_ann[yr] / wgt_tmp
                print(yr, nd, zostoga_ann[yr])

            zostoga_tmp = np.array(zostoga_ann)

        elif ( inst == "Kiel-NEMO" ):
            zostoga_tmp = nc.variables['zostoga'][:,0,0]
        else:
            zostoga_tmp = nc.variables['zostoga'][:]

        if (omip==0):
            zostoga = zostoga_tmp - (zostoga_tmp[57:61]).mean()
            print(dtime[57])
        else:
            zostoga = zostoga_tmp - (zostoga_tmp[47:51]).mean()
            print(dtime[47])

        nc.close()

        col = pd.Index([inst + '-OMIP' + str(omip+1)],name='institution')
        zostoga_df = pd.DataFrame(zostoga*factor,index=dtime,columns=col)
        zostoga_df = zostoga_df.set_index([zostoga_df.index.year])
        zostoga_df.index.names = ['year']

        if i == 0:
            zostoga_df_all=zostoga_df
        else:
            zostoga_df_all=pd.concat([zostoga_df_all,zostoga_df],axis=1)

        print (zostoga_df_all)
        i+=1

    zostoga_model += [zostoga_df_all]

    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    zostoga_mean = zostoga_df_all.mean(axis=1)
    zostoga_mean_tmp = pd.DataFrame(zostoga_mean,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    zostoga_std = zostoga_df_all.std(axis=1)
    zostoga_std_tmp = pd.DataFrame(zostoga_std,columns=col)

    zostoga_mean_df_tmp = pd.concat([zostoga_mean_tmp,zostoga_std_tmp],axis=1)

    zostoga_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = zostoga_mean_df_tmp.iloc[:,0] - zostoga_mean_df_tmp.iloc[:,1]
    zostoga_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = zostoga_mean_df_tmp.iloc[:,0] + zostoga_mean_df_tmp.iloc[:,1]

    zostoga_mean_df += [zostoga_mean_df_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    modnam += [namtmp]
    nummodel += [i]

    print (zostoga_model[omip])

#######

file_zostoga = "../Global_SL_Budget/original/GLOBAL_SL_Budget.nc"
nco = netCDF4.Dataset(file_zostoga,'r')
zostoga_obs = nco.variables['steric'][:] * 1.0e-3
zostoga_obs_norm = zostoga_obs - (zostoga_obs[0:4]).mean()
time_var_obs = nco.variables['time']
cftime = num2date(time_var_obs[:],time_var_obs.units)
nco.close()

col = pd.Index(['Global_SLB'],name='institution')
zostoga_obs_df = pd.DataFrame(zostoga_obs_norm,index=cftime,columns=col)
zostoga_obs_df = zostoga_obs_df.set_index([zostoga_obs_df.index.year])
zostoga_obs_df.index.names = ['year']
print (zostoga_obs_df)

zostoga_all=pd.concat([zostoga_obs_df,zostoga_mean_df[0],zostoga_mean_df[1],zostoga_model[0],zostoga_model[1]],axis=1)

print (zostoga_all)

# draw figures

fig = plt.figure(figsize = (15,9))
fig.suptitle( suptitle, fontsize=20 )

# OMIP1
axes = fig.add_subplot(1,3,1)
zostoga_all.plot(y=zostoga_all.columns[0],ax=axes,ylim=[-0.03,0.03],color='darkgrey',linewidth=2,title='(a) OMIP1',legend=False)
for ii in range(nummodel[0]):
    #print(ii)
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    #print(ii,linecol,linesty)
    zostoga_model[0].plot(y=zostoga_model[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[-0.03,0.03],label=inst)
    axes.set_ylabel('[m]',fontsize=12)
    axes.legend(bbox_to_anchor=(4.05,0.8))
    plt.subplots_adjust(left=0.1,right=0.9)

# OMIP2
axes = fig.add_subplot(1,3,2)
zostoga_all.plot(y=zostoga_all.columns[0],ax=axes,ylim=[-0.03,0.03],color='darkgrey',linewidth=2,title='(b) OMIP2',legend=False)
for ii in range(nummodel[1]):
    #print(ii)
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    inst=modnam[1][ii]
    #print(ii,linecol,linesty)
    zostoga_model[1].plot(y=zostoga_model[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[-0.03,0.03],legend=False)
    #axes.set_ylabel('[m]',fontsize=12)
    #axes.legend(bbox_to_anchor=(0.0,0.0),loc='lower left')
    plt.subplots_adjust(left=0.1,right=0.9)

# MMM
axes = fig.add_subplot(1,3,3)
zostoga_all.plot(y=zostoga_all.columns[0],color='darkgrey',linewidth=4,ax=axes,ylim=[-0.03,0.03],title='(c) MMM')
axes.fill_between(x=zostoga_all.index,y1=zostoga_all['OMIP1-min'],y2=zostoga_all['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=zostoga_all.index,y1=zostoga_all['OMIP2-min'],y2=zostoga_all['OMIP2-max'],alpha=0.5,facecolor='lightblue')
zostoga_all.plot(y=zostoga_all.columns[1],color='darkred' ,linewidth=2,ax=axes,ylim=[-0.03,0.03])
zostoga_all.plot(y=zostoga_all.columns[5],color='darkblue',linewidth=2,ax=axes,ylim=[-0.03,0.03])

#axes.set_ylabel('[m]',fontsize=12)
axes.legend(bbox_to_anchor=(1.65,0.95))
plt.subplots_adjust(left=0.1,right=0.8)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
