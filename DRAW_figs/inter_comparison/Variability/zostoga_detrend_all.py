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

timey = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]

lineinfo = json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1f_all'
suptitle = 'Thermosteric Sea Level anomaly relative to 2005-2009 mean'

template = 'Institution {0:3d} is {1:s}'

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)

zostoga_model = []
zostoga_dtr_model = []
zostoga_mean_df = []
zostoga_dtr_mean_df = []

lincol = []
linsty = []
modnam = []
nummodel = []

for omip in range(2):

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')
        zostoga_ann = np.array(np.zeros(62),dtype=np.float64)
        zostoga_dtr_tmp = np.empty(len(timey[omip]))
    else:
        dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')
        zostoga_ann = np.array(np.zeros(61),dtype=np.float64)
        zostoga_dtr_tmp = np.empty(len(timey[omip]))

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

        zostoga_tmp = zostoga_tmp * factor

        zostoga_fit = np.polyfit(timey[omip],zostoga_tmp,1)
        print(zostoga_fit[0],zostoga_fit[1])
        for yr in range(len(timey[omip])):
            #print(timey[omip][yr],vat_tmp[yr],vat_fit[0]*timey[omip][yr]+vat_fit[1])
            zostoga_dtr_tmp[yr] = zostoga_tmp[yr]-(zostoga_fit[0]*timey[omip][yr]+zostoga_fit[1])

        if (omip==0):
            zostoga = zostoga_tmp - (zostoga_tmp[57:61]).mean()
            zostoga_dtr = zostoga_dtr_tmp - (zostoga_dtr_tmp[57:61]).mean()
            print(dtime[57])
        else:
            zostoga = zostoga_tmp - (zostoga_tmp[47:51]).mean()
            zostoga_dtr = zostoga_dtr_tmp - (zostoga_dtr_tmp[47:51]).mean()
            print(dtime[47])

        nc.close()

        col = pd.Index([inst + '-OMIP' + str(omip+1)],name='institution')
        zostoga_df = pd.DataFrame(zostoga,index=dtime,columns=col)
        zostoga_df = zostoga_df.set_index([zostoga_df.index.year])
        zostoga_df.index.names = ['year']
        zostoga_dtr_df = pd.DataFrame(zostoga_dtr,index=dtime,columns=col)
        zostoga_dtr_df = zostoga_dtr_df.set_index([zostoga_dtr_df.index.year])
        zostoga_dtr_df.index.names = ['year']

        if i == 0:
            zostoga_df_all=zostoga_df
            zostoga_dtr_df_all=zostoga_dtr_df
        else:
            zostoga_df_all=pd.concat([zostoga_df_all,zostoga_df],axis=1)
            zostoga_dtr_df_all=pd.concat([zostoga_dtr_df_all,zostoga_dtr_df],axis=1)

        print (zostoga_df_all)
        i+=1

    zostoga_model += [zostoga_df_all]
    zostoga_dtr_model += [zostoga_dtr_df_all]

    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    zostoga_mean = zostoga_df_all.mean(axis=1)
    zostoga_mean_tmp = pd.DataFrame(zostoga_mean,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    zostoga_std = zostoga_df_all.std(axis=1,ddof=0)
    zostoga_std_tmp = pd.DataFrame(zostoga_std,columns=col)

    zostoga_mean_df_tmp = pd.concat([zostoga_mean_tmp,zostoga_std_tmp],axis=1)

    zostoga_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = zostoga_mean_df_tmp.iloc[:,0] - zostoga_mean_df_tmp.iloc[:,1]
    zostoga_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = zostoga_mean_df_tmp.iloc[:,0] + zostoga_mean_df_tmp.iloc[:,1]

    zostoga_mean_df += [zostoga_mean_df_tmp]

    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    zostoga_dtr_mean = zostoga_dtr_df_all.mean(axis=1)
    zostoga_dtr_mean_tmp = pd.DataFrame(zostoga_dtr_mean,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    zostoga_dtr_std = zostoga_dtr_df_all.std(axis=1,ddof=0)
    zostoga_dtr_std_tmp = pd.DataFrame(zostoga_dtr_std,columns=col)

    zostoga_dtr_mean_df_tmp = pd.concat([zostoga_dtr_mean_tmp,zostoga_dtr_std_tmp],axis=1)

    zostoga_dtr_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = zostoga_dtr_mean_df_tmp.iloc[:,0] - zostoga_dtr_mean_df_tmp.iloc[:,1]
    zostoga_dtr_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = zostoga_dtr_mean_df_tmp.iloc[:,0] + zostoga_dtr_mean_df_tmp.iloc[:,1]

    zostoga_dtr_mean_df += [zostoga_dtr_mean_df_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    modnam += [namtmp]
    nummodel += [i]

    print (zostoga_model[omip])

#######

file_zostoga = "../refdata/Global_SL_Budget/GLOBAL_SL_Budget.nc"
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
zostoga_dtr_all=pd.concat([zostoga_obs_df,zostoga_dtr_mean_df[0],zostoga_dtr_mean_df[1],zostoga_dtr_model[0],zostoga_dtr_model[1]],axis=1)

print (zostoga_all)

# draw figures

fig = plt.figure(figsize = (11,8))
fig.suptitle( suptitle, fontsize=20 )

# OMIP1
axes = fig.add_subplot(2,3,1)
zostoga_all.plot(y=zostoga_all.columns[0],ax=axes,ylim=[-0.03,0.03],color='darkgrey',linewidth=4,legend=False)
axes.set_title('(a) OMIP1',{'fontsize':12, 'verticalalignment':'top'})
axes.tick_params(labelsize=8)
for ii in range(nummodel[0]):
    #print(ii)
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    #print(ii,linecol,linesty)
    zostoga_model[0].plot(y=zostoga_model[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[-0.03,0.03],label=inst)
    axes.set_xlabel('')
    axes.set_ylabel('[m]',fontsize=10)
    leg = axes.legend(bbox_to_anchor=(3.45,0.7),loc='upper left')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)

# OMIP2
axes = fig.add_subplot(2,3,2)
zostoga_all.plot(y=zostoga_all.columns[0],ax=axes,ylim=[-0.03,0.03],color='darkgrey',linewidth=4,legend=False)
axes.set_title('(b) OMIP2',{'fontsize':12, 'verticalalignment':'top'})
axes.tick_params(labelsize=8)
for ii in range(nummodel[1]):
    #print(ii)
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    inst=modnam[1][ii]
    #print(ii,linecol,linesty)
    zostoga_model[1].plot(y=zostoga_model[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[-0.03,0.03],legend=False)
    axes.set_xlabel('')
    axes.set_ylabel('')
    #axes.set_ylabel('[m]',fontsize=12)
    #axes.legend(bbox_to_anchor=(0.0,0.0),loc='lower left')

# MMM
axes = fig.add_subplot(2,3,3)
zostoga_all.plot(y=zostoga_all.columns[0],color='darkgreen',linewidth=4,ax=axes,ylim=[-0.03,0.03])
axes.set_title('(c) MMM',{'fontsize':12, 'verticalalignment':'top'})
axes.tick_params(labelsize=8)
axes.fill_between(x=zostoga_all.index,y1=zostoga_all['OMIP1-min'],y2=zostoga_all['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=zostoga_all.index,y1=zostoga_all['OMIP2-min'],y2=zostoga_all['OMIP2-max'],alpha=0.5,facecolor='lightblue')
zostoga_all.plot(y=zostoga_all.columns[1],color='darkred' ,linewidth=2,ax=axes,ylim=[-0.03,0.03])
zostoga_all.plot(y=zostoga_all.columns[5],color='darkblue',linewidth=2,ax=axes,ylim=[-0.03,0.03])
axes.set_xlabel('')
axes.set_ylabel('')
#axes.set_ylabel('[m]',fontsize=12)
leg = axes.legend(bbox_to_anchor=(1.01,0.95),loc='upper left')
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)


# OMIP1
axes = fig.add_subplot(2,3,4)
zostoga_dtr_all.plot(y=zostoga_dtr_all.columns[0],ax=axes,ylim=[-0.03,0.03],color='darkgrey',linewidth=4,legend=False)
axes.set_title('(d) OMIP1 detrended',{'fontsize':12, 'verticalalignment':'top'})
axes.tick_params(labelsize=8)
for ii in range(nummodel[0]):
    #print(ii)
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    #print(ii,linecol,linesty)
    zostoga_dtr_model[0].plot(y=zostoga_dtr_model[0].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[-0.03,0.03],legend=False)
    axes.set_xlabel('year',fontsize=10)
    axes.set_ylabel('[m]',fontsize=10)
    #axes.legend(bbox_to_anchor=(4.05,0.8))
    #plt.subplots_adjust(left=0.1,right=0.9)

# OMIP2
axes = fig.add_subplot(2,3,5)
zostoga_dtr_all.plot(y=zostoga_dtr_all.columns[0],ax=axes,ylim=[-0.03,0.03],color='darkgrey',linewidth=4,legend=False)
axes.set_title('(e) OMIP2 detrended',{'fontsize':12, 'verticalalignment':'top'})
axes.tick_params(labelsize=8)
for ii in range(nummodel[1]):
    #print(ii)
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    inst=modnam[1][ii]
    #print(ii,linecol,linesty)
    zostoga_dtr_model[1].plot(y=zostoga_dtr_model[1].columns[ii],ax=axes,color=linecol,linewidth=1,linestyle=linesty,ylim=[-0.03,0.03],legend=False)
    axes.set_xlabel('year',fontsize=10)
    axes.set_ylabel('')
    #axes.set_ylabel('[m]',fontsize=12)
    #axes.legend(bbox_to_anchor=(0.0,0.0),loc='lower left')
    #plt.subplots_adjust(left=0.1,right=0.9)

# MMM
axes = fig.add_subplot(2,3,6)
zostoga_dtr_all.plot(y=zostoga_dtr_all.columns[0],color='darkgreen',linewidth=4,ax=axes,ylim=[-0.03,0.03],legend=False)
axes.set_title('(f) MMM detrended',{'fontsize':12, 'verticalalignment':'top'})
axes.tick_params(labelsize=8)
axes.fill_between(x=zostoga_dtr_all.index,y1=zostoga_dtr_all['OMIP1-min'],y2=zostoga_dtr_all['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=zostoga_dtr_all.index,y1=zostoga_dtr_all['OMIP2-min'],y2=zostoga_dtr_all['OMIP2-max'],alpha=0.5,facecolor='lightblue')
zostoga_dtr_all.plot(y=zostoga_dtr_all.columns[1],color='darkred' ,linewidth=2,ax=axes,ylim=[-0.03,0.03],legend=False)
zostoga_dtr_all.plot(y=zostoga_dtr_all.columns[5],color='darkblue',linewidth=2,ax=axes,ylim=[-0.03,0.03],legend=False)
axes.set_xlabel('year',fontsize=10)
axes.set_ylabel('')

plt.subplots_adjust(left=0.08,right=0.8,top=0.92,bottom=0.08,wspace=0.22)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
