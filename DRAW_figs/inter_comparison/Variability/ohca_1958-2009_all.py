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

metainfo = [json.load(open('json/vat_omip1_1958-2009.json')),
            json.load(open('json/vat_omip2_1958-2009.json'))]

lineinfo = json.load(open('../json/inst_color_style-52yr.json'))

outfile = './fig/Fig1g_1958-2009.png'
suptitle = 'Ocean heat content anomaly relative to 2005-2009 mean'

template = 'Institution {0:3d} is {1:s}'

var_list = [ "thetaoga_700", "thetaoga_2000", "thetaoga_2000_bottom", "thetaoga_all" ]
volume_list = np.array([ 2.338e17, 6.216e17, 7.593e17, 1.381e18 ])
degC_to_ZJ = volume_list * 3.99e3 * 1.036e3 * 1.0e-21

thetaoga_model_df = []
thetaoga_mean_df = []

lincol = []
linsty = []
modnam = []
nummodel = []

num_df = 0
for omip in range(2):

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')
        vat_tmp = np.zeros(62)
    else:
        dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')
        vat_tmp = np.zeros(61)

    nvar = 0
    for var in var_list:

        coltmp = []
        stytmp = []
        namtmp = []

        i=0
        for inst in metainfo[omip].keys():

            vat_tmp[:] = np.nan

            print (template.format(i,inst))
            coltmp += [lineinfo[inst]["color"]]
            stytmp += [lineinfo[inst]["style"]]
            namtmp +=[inst]

            factor = float(metainfo[omip][inst][var]['factor'])
            infile = metainfo[omip][inst][var]['path'] + '/' + metainfo[omip][inst][var]['fname']
            vname = metainfo[omip][inst][var]['varname']
            print (infile, factor)

            nc = netCDF4.Dataset(infile,'r')

            if ( omip == 0 ):
                if ( inst == "MIROC-COCO4-9" ):
                    vat_tmp[0:62] = nc.variables[vname][248:310]
                elif ( inst == "MIROC-COCO4-9_52yr" ):
                    vat_tmp[10:62] = nc.variables[vname][208:260]
                elif ( inst == "MRI.COM" ):
                    vat_tmp[0:62] = nc.variables[vname][310:372]
                elif ( inst == "MRI.COM_52yr" ):
                    vat_tmp[10:62] = nc.variables[vname][260:312]
                else:
                    vat_tmp = nc.variables[vname][310:372]
            else:
                if ( inst == "MIROC-COCO4-9" ):
                    vat_tmp[0:61] = nc.variables[vname][305:366]
                elif ( inst == "MIROC-COCO4-9_52yr" ):
                    vat_tmp[0:52] = nc.variables[vname][260:312]
                elif ( inst == "MRI.COM" ):
                    vat_tmp[0:61] = nc.variables[vname][305:366]
                elif ( inst == "MRI.COM_52yr" ):
                    vat_tmp[0:52] = nc.variables[vname][260:312]
                else:
                    vat_tmp = nc.variables[vname][305:366]

            vat_tmp = vat_tmp * degC_to_ZJ[nvar]

            if ( omip == 0 ):
                vat = vat_tmp - (vat_tmp[57:61]).mean()
            else:
                vat = vat_tmp - (vat_tmp[47:52]).mean()

            nc.close()

            col = pd.Index([inst],name='institution')
            vat_df = pd.DataFrame(vat*factor,index=dtime,columns=col)
            vat_df = vat_df.set_index([vat_df.index.year])
            vat_df.index.names = ['year']

            if i == 0:
                vat_df_all=vat_df
            else:
                vat_df_all=pd.concat([vat_df_all,vat_df],axis=1)

            print (vat_df_all)
            i+=1

        thetaoga_model_df += [vat_df_all]

        col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
        vat_mean = vat_df_all.mean(axis=1)
        vat_mean_tmp = pd.DataFrame(vat_mean,columns=col)

        col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
        vat_std = vat_df_all.std(axis=1)
        vat_std_tmp = pd.DataFrame(vat_std,columns=col)

        vat_mean_df_tmp = pd.concat([vat_mean_tmp,vat_std_tmp],axis=1)

        vat_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = vat_mean_df_tmp.iloc[:,0] - vat_mean_df_tmp.iloc[:,1]
        vat_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = vat_mean_df_tmp.iloc[:,0] + vat_mean_df_tmp.iloc[:,1]

        thetaoga_mean_df += [vat_mean_df_tmp]

        lincol += [coltmp]
        linsty += [stytmp]
        modnam += [namtmp]
        nummodel += [i]

        file_vat = "../Zanna_OHC/OHC_GF_1870_2018.nc"
        nco = netCDF4.Dataset(file_vat,'r')
        if (nvar == 0):
            vat_obs = nco.variables['OHC_700m'][:]
        if (nvar == 1):
            vat_obs = nco.variables['OHC_2000m'][:]
        if (nvar == 2):
            vat_obs = nco.variables['OHC_below_2000m'][:]
        if (nvar == 3):
            vat_obs = nco.variables['OHC_full_depth'][:]
        
        vat_obs_norm = vat_obs - (vat_obs[-15:-9]).mean()
        time_var_obs = nco.variables['time'][:]
        nco.close()

        col = pd.Index(['Zanna_2019'],name='institution')
        vat_obs_df = pd.DataFrame(vat_obs_norm,index=time_var_obs,columns=col)
        print (vat_obs_df)
        vat_obs_df.index.names = ['year']

        thetaoga_mean_df[num_df] = pd.concat([vat_obs_df,thetaoga_mean_df[num_df]],axis=1)

        num_df += 1
        nvar += 1
        print(num_df,nvar)
        
# draw figures

title=["(a) OMIP1 (0-700m)", "(b) OMIP2 (0-700m)", 
       "(c) OMIP1 (0-2000m)","(d) OMIP2 (0-2000m)",
       "(e) OMIP1 (2000m-bottom)", "(f) OMIP2 (2000m-bottom)",
       "(g) OMIP1 (0m-bottom)", "(h) OMIP2 (0m-bottom)"]
ylim = [ [-250, 100], [-300, 150], [-250, 250], [-400, 300] ]

fig  = plt.figure(figsize = (10,12))
fig.suptitle(suptitle , fontsize=20)

axes = [ plt.subplot(4,2,1),
         plt.subplot(4,2,2),
         plt.subplot(4,2,3),
         plt.subplot(4,2,4),
         plt.subplot(4,2,5),
         plt.subplot(4,2,6),
         plt.subplot(4,2,7),
         plt.subplot(4,2,8) ]

nv = 0
for var in var_list:
    for omip in range(2):
        ndf = 4*omip + nv
        nax = 2*nv + omip
        nm=nummodel[ndf]
        if (omip == 0):
            axes[nax].set_ylabel('ZJ',fontsize=12)

        if (omip == 0):
            if (nv == 0):
                thetaoga_mean_df[ndf].plot(y=thetaoga_mean_df[ndf].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax])
                for ii in range(nummodel[ndf]):
                    linecol=lincol[ndf][ii]
                    linesty=linsty[ndf][ii]
                    inst=modnam[omip][ii]
                    thetaoga_model_df[ndf].plot(y=thetaoga_model_df[ndf].columns[ii],ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],color=linecol,linewidth=1,linestyle=linesty,label=inst)

                axes[nax].legend(bbox_to_anchor=(2.95,0.5))
            else:
                thetaoga_mean_df[ndf].plot(y=thetaoga_mean_df[ndf].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax],legend=False)
                for ii in range(nummodel[ndf]):
                    linecol=lincol[ndf][ii]
                    linesty=linsty[ndf][ii]
                    inst=modnam[omip][ii]
                    thetaoga_model_df[ndf].plot(y=thetaoga_model_df[ndf].columns[ii],ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],color=linecol,linewidth=1,linestyle=linesty,legend=False)
                
        if (omip == 1):
            thetaoga_mean_df[ndf].plot(y=thetaoga_mean_df[ndf].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax],legend=False)
            for ii in range(nummodel[ndf]):
                linecol=lincol[ndf][ii]
                linesty=linsty[ndf][ii]
                inst=modnam[omip][ii]
                thetaoga_model_df[ndf].plot(y=thetaoga_model_df[ndf].columns[ii],ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],color=linecol,linewidth=1,linestyle=linesty,legend=False)

        if (nv < 3):
            axes[nax].set_xlabel('')

    #ndf1 = nv
    #ndf2 = nv + 4
    #nax = 3*nv + 2
    #if (nv == 0):
    #    thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[0],color='darkgrey',linewidth=8,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018])
    #    thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[1],color='darkred',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018])
    #    thetaoga_mean_df[ndf2].plot(y=thetaoga_mean_df[ndf2].columns[1],color='darkblue',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax])
    #    axes[nax].legend(bbox_to_anchor=(1.2,1.0))
    #else:
    #    thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[0],color='darkgrey',linewidth=8,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
    #    thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[1],color='darkred',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
    #    thetaoga_mean_df[ndf2].plot(y=thetaoga_mean_df[ndf2].columns[1],color='darkblue',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax],legend=False)
    
    if (nv < 3):
        axes[nax].set_xlabel('')


    nv += 1

plt.subplots_adjust(left=0.1,right=0.75,hspace=0.34)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
