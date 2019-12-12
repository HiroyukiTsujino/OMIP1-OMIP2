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

metainfo = [json.load(open('json/vat_omip1.json')),
            json.load(open('json/vat_omip2.json'))]

timey = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]

lineinfo = json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1g_all'
suptitle = 'Ocean heat content anomaly relative to 2005-2009 mean'

template = 'Institution {0:3d} is {1:s}'

var_list = [ "thetaoga_700", "thetaoga_2000", "thetaoga_2000_bottom", "thetaoga_all", "thetaoga_all_dtr" ]
volume_list = np.array([ 2.338e17, 6.216e17, 7.593e17, 1.381e18, 1.381e18  ])
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
    else:
        dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')

    nvar = 0
    for var in var_list:

        coltmp = []
        stytmp = []
        namtmp = []

        i=0
        for inst in metainfo[omip].keys():

            print (template.format(i,inst))
            coltmp += [lineinfo[inst]["color"]]
            stytmp += [lineinfo[inst]["style"]]
            namtmp +=[inst]

            if (var == 'thetaoga_all_dtr'):
                factor = float(metainfo[omip][inst]['thetaoga_all']['factor'])
                infile = metainfo[omip][inst]['thetaoga_all']['path'] + '/' + metainfo[omip][inst]['thetaoga_all']['fname']
                vname = metainfo[omip][inst]['thetaoga_all']['varname']
            else:
                factor = float(metainfo[omip][inst][var]['factor'])
                infile = metainfo[omip][inst][var]['path'] + '/' + metainfo[omip][inst][var]['fname']
                vname = metainfo[omip][inst][var]['varname']

            print (infile, factor)

            nc = netCDF4.Dataset(infile,'r')
            if ( inst == "AWI-FESOM" or inst == "EC-Earth3-NEMO" or inst == "Kiel-NEMO" ):
                if ( inst == "Kiel-NEMO" ):
                    vat_tmp = nc.variables[vname][:,0,0]
                elif ( inst == "EC-Earth3-NEMO" ):
                    if ( var == 'thetaoga_all' or var == 'thetaoga_all_dtr' ):
                        vat_tmp = nc.variables[vname][:]
                    else:
                        vat_tmp = nc.variables[vname][:,0,0]
                else:
                    vat_tmp = nc.variables[vname][:]
            else:
                if ( omip == 0 ):
                    if ( inst == "MIROC-COCO4.9" ):
                        vat_tmp = nc.variables[vname][248:310]
                    elif ( inst == "GFDL-MOM" ):
                        vat_tmp = nc.variables[vname][300:362]
                    else:
                        vat_tmp = nc.variables[vname][310:372]
                else:
                    if ( inst == "FSU-HYCOM"):
                        vat_tmp = nc.variables[vname][290:351]
                    elif ( inst == "GFDL-MOM" ):
                        vat_tmp = nc.variables[vname][302:363]
                    else:
                        vat_tmp = nc.variables[vname][305:366]
                    
            vat_tmp = vat_tmp * factor
                        
            if (var == 'thetaoga_all_dtr'):            
                vat_fit = np.polyfit(timey[omip],vat_tmp,1)
                print(vat_fit[0],vat_fit[1])
                for yr in range(len(timey[omip])):
                    #print(timey[omip][yr],vat_tmp[yr],vat_fit[0]*timey[omip][yr]+vat_fit[1])
                    vat_tmp[yr] = vat_tmp[yr]-(vat_fit[0]*timey[omip][yr]+vat_fit[1])

            #vat_tmp = vat_tmp * degC_to_ZJ[nvar]

            if ( omip == 0 ):
                vat = vat_tmp - (vat_tmp[57:61]).mean()
            else:
                vat = vat_tmp - (vat_tmp[47:52]).mean()

            nc.close()

            col = pd.Index([inst],name='institution')

            vat_df = pd.DataFrame(vat,index=dtime,columns=col)

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

        file_vat = "../refdata/Zanna_OHC/OHC_GF_1870_2018.nc"
        nco = netCDF4.Dataset(file_vat,'r')
        if (nvar == 0):
            vat_obs = nco.variables['OHC_700m'][:]
        if (nvar == 1):
            vat_obs = nco.variables['OHC_2000m'][:]
        if (nvar == 2):
            vat_obs = nco.variables['OHC_below_2000m'][:]
        if (nvar == 3 or nvar == 4):
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
        print("num_df, nvar ",num_df,nvar)

#######################################################
# Reference data except for Zanna

###########
# Ishii

file_ishii = "../refdata/Ishii_v7.2/OHC_0-700.txt"
names=['year','Ishii_700m','Ishii_700m_se']
ishii_700m_df=pd.read_table(file_ishii,sep='\s+',names=names,index_col='year')
ishii_700m_df['Ishii_700m'] = ishii_700m_df['Ishii_700m'] - ishii_700m_df.loc[2005:2009,'Ishii_700m'].mean()

file_ishii = "../refdata/Ishii_v7.2/OHC_0-2000.txt"
names=['year','Ishii_2000m','Ishii_2000m_se']
ishii_2000m_df=pd.read_table(file_ishii,sep='\s+',names=names,index_col='year')
ishii_2000m_df['Ishii_2000m'] = ishii_2000m_df['Ishii_2000m'] - ishii_2000m_df.loc[2005:2009,'Ishii_2000m'].mean()

###########
# Chen

file_chen = "../refdata/Chen_OHC/IAP_OHC_estimate_update.txt"
names=['year', 'month', 'Chen_700m','Chen_700m_smth', 'Chen_700m_se', 'Chen_700-2000m','Chen_700-2000m_smth', 'Chen_700-2000m_se']
chen_df=pd.read_table(file_chen,sep='\s+',names=names,skiprows=18)
chen_df['Chen_700m'] = chen_df['Chen_700m'] * 10.0
chen_df['Chen_700-2000m'] = chen_df['Chen_700-2000m'] * 10.0
chen_df['Chen_2000m'] = chen_df['Chen_700m'] + chen_df['Chen_700-2000m']
#time_chen=pd.to_datetime(chen_df.year*10000+chen_df.month*100+1,format='%Y%m%d')
chen_df.index=chen_df.year
chen_df_ann=chen_df.mean(level='year',skipna=False)
chen_df_ann.drop(['year','month','Chen_700m_smth','Chen_700-2000m_smth'], axis='columns', inplace=True)
chen_df_ann.drop(2019, inplace=True)
chen_df_ann['Chen_700m'] = chen_df_ann['Chen_700m'] - chen_df_ann.loc[2005:2009,'Chen_700m'].mean()
chen_df_ann['Chen_2000m'] = chen_df_ann['Chen_2000m'] - chen_df_ann.loc[2005:2009,'Chen_2000m'].mean()
chen_df_ann['Chen_700-2000m'] = chen_df_ann['Chen_700-2000m'] - chen_df_ann.loc[2005:2009,'Chen_700-2000m'].mean()

#print(chen_df_ann)

#sys.exit()

###########
# NOAA

file_noaa = "../refdata/NOAA_OHC/NOAA_h22-w0-700m.dat.txt"
names=['year', 'NOAA_700m','NOAA_700m_se', 'NOAA_700m_NH', 'NOAA_700m_NH_se', 'NOAA_700m_SH', 'NOAA_700m_SH_se']
noaa_700m_df=pd.read_table(file_noaa,sep='\s+',names=names,skiprows=1)
noaa_700m_df['year'] = np.floor(noaa_700m_df['year'])
#print(noaa_700m_df['year'])
noaa_700m_df['NOAA_700m'] = noaa_700m_df['NOAA_700m'] * 10.0
noaa_700m_df.index=noaa_700m_df.year
noaa_700m_df.drop(['year','NOAA_700m_NH','NOAA_700m_NH_se','NOAA_700m_SH','NOAA_700m_SH_se'], axis='columns', inplace=True)
noaa_700m_df['NOAA_700m'] = noaa_700m_df['NOAA_700m'] - noaa_700m_df.loc[2005:2009,'NOAA_700m'].mean()

file_noaa = "../refdata/NOAA_OHC/NOAA_h22-w0-2000m.dat.txt"
names=['year', 'NOAA_2000m','NOAA_2000m_se', 'NOAA_2000m_NH', 'NOAA_2000m_NH_se', 'NOAA_2000m_SH', 'NOAA_2000m_SH_se']
noaa_2000m_df=pd.read_table(file_noaa,sep='\s+',names=names,skiprows=1)
noaa_2000m_df['year'] = np.floor(noaa_2000m_df['year'])
#print(noaa_2000m_df['year'])
noaa_2000m_df['NOAA_2000m'] = noaa_2000m_df['NOAA_2000m'] * 10.0
noaa_2000m_df.index=noaa_2000m_df.year
noaa_2000m_df.drop(['year','NOAA_2000m_NH','NOAA_2000m_NH_se','NOAA_2000m_SH','NOAA_2000m_SH_se'], axis='columns', inplace=True)
noaa_2000m_df['NOAA_2000m'] = noaa_2000m_df['NOAA_2000m'] - noaa_2000m_df.loc[2005:2009,'NOAA_2000m'].mean()

#print(noaa_df)

#sys.exit()

#######
# merge data

ohc700m_all=pd.concat([ishii_700m_df['Ishii_700m'],chen_df_ann['Chen_700m'],noaa_700m_df['NOAA_700m']],axis=1)

print (ohc700m_all)

ohc2000m_all=pd.concat([ishii_2000m_df['Ishii_2000m'],chen_df_ann['Chen_2000m'],noaa_2000m_df['NOAA_2000m']],axis=1)

print (ohc2000m_all)


# draw figures

title=["(a) OMIP1 (0-700m)", "(b) OMIP2 (0-700m)", "(c) MMM (0-700m)",
       "(d) OMIP1 (0-2000m)","(e) OMIP2 (0-2000m)", "(f) MMM (0-2000m)",
       "(g) OMIP1 (2000m-bottom)", "(h) OMIP2 (2000m-bottom)", "(i) MMM (2000m-bottom)",
       "(j) OMIP1 (0m-bottom)", "(k) OMIP2 (0m-bottom)", "(l) MMM (0m-bottom)",
       "(m) (j) is detrended", "(n) (k) is detrended", "(o) (l) is detrended"]
ylim = np.array([ [-250, 100], [-300, 150], [-250, 250], [-400, 300], [-120, 120] ], dtype=np.float64)
ylimvat = ylim.copy()
for j in range(5):
    for i in range(2):
        ylimvat[j,i] = ylimvat[j,i] / degC_to_ZJ[j]
    
fig  = plt.figure(figsize = (8,11))
fig.suptitle(suptitle , fontsize=18)

axes = [ plt.subplot(5,3,1),
         plt.subplot(5,3,2),
         plt.subplot(5,3,3),
         plt.subplot(5,3,4),
         plt.subplot(5,3,5),
         plt.subplot(5,3,6),
         plt.subplot(5,3,7),
         plt.subplot(5,3,8),
         plt.subplot(5,3,9),
         plt.subplot(5,3,10),
         plt.subplot(5,3,11),
         plt.subplot(5,3,12),
         plt.subplot(5,3,13),
         plt.subplot(5,3,14),
         plt.subplot(5,3,15) ]

nv = 0
for var in var_list:
    for omip in range(2):
        ndf = 5*omip + nv
        nax = 3*nv + omip
        nm=nummodel[ndf]
        print("nax =", nax)
        if (omip == 0):
            axes[nax].set_ylabel('ZJ',fontsize=8)

        if (omip == 0):
            if (nv == 0):
                thetaoga_mean_df[ndf].plot(y=thetaoga_mean_df[ndf].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
                axes[nax].set_title(title[nax],{'fontsize':8, 'verticalalignment':'top'})
                axes[nax].tick_params(labelsize=6)
                ax2=axes[nax].twinx()
                ax2.tick_params(labelright=False,labelsize=6)
                for ii in range(nummodel[ndf]):
                    linecol=lincol[ndf][ii]
                    linesty=linsty[ndf][ii]
                    inst=modnam[omip][ii]
                    if (linesty == 'dashed'):
                        lwidth=1.2
                    else:
                        lwidth=1.0
                    #print(ylimvat[nv,0],ylimvat[nv,1])
                    thetaoga_model_df[ndf].plot(y=thetaoga_model_df[ndf].columns[ii],ax=ax2,ylim=ylimvat[nv],xlim=[1948,2018],color=linecol,linewidth=lwidth,linestyle=linesty,label=inst)

                leg = ax2.legend(bbox_to_anchor=(3.5,0.10),loc='upper left',fontsize=8)
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(1.0)
            else:
                thetaoga_mean_df[ndf].plot(y=thetaoga_mean_df[ndf].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
                axes[nax].set_title(title[nax],{'fontsize':8, 'verticalalignment':'top'})
                axes[nax].tick_params(labelsize=6)
                ax2=axes[nax].twinx()
                ax2.tick_params(labelright=False,labelsize=6)
                for ii in range(nummodel[ndf]):
                    linecol=lincol[ndf][ii]
                    linesty=linsty[ndf][ii]
                    inst=modnam[omip][ii]
                    if (linesty == 'dashed'):
                        lwidth=1.2
                    else:
                        lwidth=1.0
                    thetaoga_model_df[ndf].plot(y=thetaoga_model_df[ndf].columns[ii],ax=ax2,ylim=ylimvat[nv],xlim=[1948,2018],color=linecol,linewidth=lwidth,linestyle=linesty,legend=False)

                
        if (omip == 1):
            thetaoga_mean_df[ndf].plot(y=thetaoga_mean_df[ndf].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax],legend=False)
            axes[nax].set_title(title[nax],{'fontsize':8, 'verticalalignment':'top'})
            axes[nax].tick_params(labelleft=False,labelsize=6)
            ax2=axes[nax].twinx()
            ax2.tick_params(labelright=False,labelsize=6)
            for ii in range(nummodel[ndf]):
                linecol=lincol[ndf][ii]
                linesty=linsty[ndf][ii]
                inst=modnam[omip][ii]
                if (linesty == 'dashed'):
                    lwidth=1.2
                else:
                    lwidth=1.0
                thetaoga_model_df[ndf].plot(y=thetaoga_model_df[ndf].columns[ii],ax=ax2,ylim=ylimvat[nv],xlim=[1948,2018],color=linecol,linewidth=lwidth,linestyle=linesty,legend=False)


        if (nv < 4):
            axes[nax].set_xlabel('')

    ndf1 = nv
    ndf2 = nv + 5
    nax = 3*nv + 2
    print("nax =",nax)
    if (nv == 0):
        thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018])
        axes[nax].tick_params(labelleft=False,labelsize=6)
        ohc700m_all.plot(y=ohc700m_all.columns[0],color='deepskyblue',linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018])
        ohc700m_all.plot(y=ohc700m_all.columns[1],color='magenta', linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018])
        ohc700m_all.plot(y=ohc700m_all.columns[2],color='green',linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018])
        ax2=axes[nax].twinx()
        thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[1],color='darkred',linewidth=2,ax=ax2,ylim=ylimvat[nv],xlim=[1948,2018])
        thetaoga_mean_df[ndf2].plot(y=thetaoga_mean_df[ndf2].columns[1],color='darkblue',linewidth=2,ax=ax2,ylim=ylimvat[nv],xlim=[1948,2018],title=title[nax])
        ax2.set_title(title[nax],{'fontsize':8, 'verticalalignment':'top'})
        ax2.tick_params(labelsize=6)
        leg = axes[nax].legend(bbox_to_anchor=(1.3,1.0),loc='upper left',fontsize=8)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1.0)
        leg = ax2.legend(bbox_to_anchor=(1.3,0.43),loc='upper left',fontsize=8)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1.0)
    else:
        thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
        axes[nax].tick_params(labelleft=False,labelsize=6)
        if (nv == 1):
            ohc2000m_all.plot(y=ohc2000m_all.columns[0],color='deepskyblue',linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
            ohc2000m_all.plot(y=ohc2000m_all.columns[1],color='magenta', linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
            ohc2000m_all.plot(y=ohc2000m_all.columns[2],color='green',linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
        ax2=axes[nax].twinx()
        thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[1],color='darkred',linewidth=2,ax=ax2,ylim=ylimvat[nv],xlim=[1948,2018],legend=False)
        thetaoga_mean_df[ndf2].plot(y=thetaoga_mean_df[ndf2].columns[1],color='darkblue',linewidth=2,ax=ax2,ylim=ylimvat[nv],xlim=[1948,2018],title=title[nax],legend=False)
        ax2.set_title(title[nax],{'fontsize':8, 'verticalalignment':'top'})
        ax2.tick_params(labelsize=6)
    
    ax2.set_ylabel('$^{\circ}$C',fontsize=8)
    if (nv < 4):
        axes[nax].set_xlabel('')

    nv += 1

#nv=4
#for omip in range(2):
#    ndf = 5*omip + nv
#    nax = 3*nv + omip
#    print("nax =",nax)
#    print("ndf = ",ndf)
#    nm=nummodel[ndf]
#    if (omip == 0):
#        axes[nax].set_ylabel('ZJ',fontsize=12)
#
#    thetaoga_mean_df[ndf].plot(y=thetaoga_mean_df[ndf].columns[0],color='darkgrey',linewidth=8,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax],legend=False)
#    for ii in range(nummodel[ndf]):
#        linecol=lincol[ndf][ii]
#        linesty=linsty[ndf][ii]
#        inst=modnam[omip][ii]
#        thetaoga_model_df[ndf].plot(y=thetaoga_model_df[ndf].columns[ii],ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],color=linecol,linewidth=1,linestyle=linesty,legend=False)
#
#ndf1 = nv - 1
#ndf2 = nv + 4
#nax = 3*nv + 2
#print("nax = ",nax)
#print("ndf1 = ",ndf1)
#print("ndf2 = ",ndf2)
#thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[0],color='darkgrey',linewidth=4,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
#thetaoga_mean_df[ndf1].plot(y=thetaoga_mean_df[ndf1].columns[1],color='darkred', linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],legend=False)
#thetaoga_mean_df[ndf2].plot(y=thetaoga_mean_df[ndf2].columns[1],color='darkblue',linewidth=2,ax=axes[nax],ylim=ylim[nv],xlim=[1948,2018],title=title[nax],legend=False)

plt.subplots_adjust(left=0.08,right=0.75,hspace=0.15,wspace=0.1,top=0.92,bottom=0.08)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
