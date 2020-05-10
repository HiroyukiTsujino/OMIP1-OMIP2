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

lineinfo = json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1b_all'
suptitle = 'Drake Passage Transport'

#dtime_all = pd.date_range('1948-01-01','2018-12-31',freq='AS-JAN')
dtime_all = np.array(np.linspace(1948,2018,71))
yr_all = 71

template = 'Institution {0:3d} is {1:s}'

drake_annual_model = []
drake_mean_df = []

lincol = []
linsty = []
modnam = []
nummodel = []

for omip in range(2):

    if (omip == 0):
        #dtime = pd.date_range('1948-01-01','2009-12-31',freq='AS-JAN')
        dtime = np.array(np.linspace(1948,2009,62))
        #print(dtime)
        yr_cyc = 62
    else:
        #dtime = pd.date_range('1958-01-01','2018-12-31',freq='AS-JAN')
        dtime = np.array(np.linspace(1958,2018,61))
        #print(dtime)
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

            drake_tmp_df = pd.DataFrame()

            if ( omip == 0 ):
                for loop in range(6):
                    loopfile=metainfo[omip][inst]['path'] + str(loop+1) + '/' + 'mfo_Oyr_FESOM1.4_historical_loop' + str(loop+1) + '_gr_194801-200912.nc'
                    nc = netCDF4.Dataset(loopfile,'r')
                    drake_tmp= nc.variables[vname][:,:]
                    drake   = drake_tmp[nth_line-1,:]
                    time_var = nc.variables['time']
                    cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
                    year     = np.array([tmp.year for tmp in cftime]) - 62 * (5-loop)
                    nc.close()

                    #print(year)
                    df_tmp = pd.DataFrame(drake*fac,index=year,columns=col)
                    drake_tmp_df = pd.concat([drake_tmp_df,df_tmp])

            else:
                for loop in range(6):
                    loopfile=metainfo[omip][inst]['path'] + str(loop+1) + '/' + 'mfo_Oyr_FESOM1.4_historical_loop' + str(loop+1) + '_gr_195801-201812.nc'
                    nc = netCDF4.Dataset(loopfile,'r')
                    drake_tmp= nc.variables[vname][:,:]
                    drake   = drake_tmp[nth_line-1,:]
                    time_var = nc.variables['time']
                    cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
                    year     = np.array([tmp.year for tmp in cftime]) - 61 * (5-loop)
                    nc.close()

                    #print(year)
                    df_tmp = pd.DataFrame(drake*fac,index=year,columns=col)
                    drake_tmp_df = pd.concat([drake_tmp_df,df_tmp])

            drake = drake_tmp_df.iloc[:,0]
            print ('length of data =', len(drake))
            num_data = len(drake)
            drake_lastcyc = np.array(np.zeros(yr_cyc),dtype=np.float64)
            drake_lastcyc[0:yr_cyc] = drake[num_data-yr_cyc:num_data]
            drake_df = pd.DataFrame(drake_lastcyc*fac,index=dtime,columns=col)
            drake_df.index.names = ['year']
            #print(drake_df)

        else:

            nc = netCDF4.Dataset(infile,'r')

            if nth_line > 0:
                drake_tmp = nc.variables[vname][:,:]
                if (inst == 'AWI-FESOM' or inst == 'CMCC-NEMO' or inst == 'GFDL-MOM'):
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

            drake_df = pd.DataFrame(drake_lastcyc*fac,index=dtime,columns=col)

            drake_df.index.names = ['year']

        if i == 0:
            drake_df_all=drake_df
        else:
            drake_df_all=pd.concat([drake_df_all,drake_df],axis=1)
            
        print (drake_df_all)

        i=i+1

    drake_annual_model += [drake_df_all]

    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    drake_mean=drake_df_all.mean(axis=1)
    drake_mean_df_tmp = pd.DataFrame(drake_mean,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    drake_std=drake_df_all.std(axis=1,ddof=0)
    drake_std_df_tmp = pd.DataFrame(drake_std,columns=col)

    drake_mean_df_tmp = pd.concat([drake_mean_df_tmp,drake_std_df_tmp],axis=1)

    drake_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = drake_mean_df_tmp.iloc[:,0] - drake_mean_df_tmp.iloc[:,1]
    drake_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = drake_mean_df_tmp.iloc[:,0] + drake_mean_df_tmp.iloc[:,1]

    drake_mean_df += [drake_mean_df_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    modnam += [namtmp]
    nummodel += [i]


drake_obs1 = np.array(np.zeros(yr_all),dtype=np.float64)
drake_obs1[:] = 134.0
drake_obs1err = np.array(np.zeros(yr_all),dtype=np.float64)
drake_obs1err[:] = 27.0
drake_obs1_df_tmp = pd.DataFrame(pd.Series(drake_obs1,name='Cunningham et al. (2003)',index=dtime_all))
drake_obs1err_df_tmp = pd.DataFrame(pd.Series(drake_obs1err,name='Cunningham et al. (2003) error',index=dtime_all))
drake_obs1_df = pd.concat([drake_obs1_df_tmp, drake_obs1err_df_tmp],axis=1)
drake_obs1_df['OBS1-min'] = drake_obs1_df.iloc[:,0] - drake_obs1_df.iloc[:,1]
drake_obs1_df['OBS1-max'] = drake_obs1_df.iloc[:,0] + drake_obs1_df.iloc[:,1]

drake_obs2 = np.array(np.zeros(yr_all),dtype=np.float64)
drake_obs2[:] = 173.3
drake_obs2err = np.array(np.zeros(yr_all),dtype=np.float64)
drake_obs2err[:] = 10.7
drake_obs2_df_tmp = pd.DataFrame(pd.Series(drake_obs2, name='Donohue et al. (2016)', index=dtime_all))
drake_obs2err_df_tmp = pd.DataFrame(pd.Series(drake_obs2err, name='Donohue et al. (2016) error', index=dtime_all))
drake_obs2_df = pd.concat([drake_obs2_df_tmp, drake_obs2err_df_tmp],axis=1)
drake_obs2_df['OBS2-min'] = drake_obs2_df.iloc[:,0] - drake_obs2_df.iloc[:,1]
drake_obs2_df['OBS2-max'] = drake_obs2_df.iloc[:,0] + drake_obs2_df.iloc[:,1]

print(drake_obs2_df)

drake_annual_all=pd.concat([drake_obs1_df,drake_obs2_df,drake_mean_df[0],drake_mean_df[1],drake_annual_model[0],drake_annual_model[1]],axis=1)

# draw figures

fig  = plt.figure(figsize = (8,11))
fig.suptitle( suptitle, fontsize=18 )


# OMIP1
axes = fig.add_subplot(3,1,1)
drake_annual_all.plot(y=drake_annual_all.columns[0],ax=axes,ylim=[90,220],color='darkgrey',linewidth=2,title='(a) OMIP1')
axes.fill_between(x=drake_annual_all.index,y1=drake_annual_all['OBS1-min'],y2=drake_annual_all['OBS1-max'],alpha=0.5,facecolor='lightgrey')
drake_annual_all.plot(y=drake_annual_all.columns[4],ax=axes,ylim=[90,220],color='black',linewidth=2)
axes.fill_between(x=drake_annual_all.index,y1=drake_annual_all['OBS2-min'],y2=drake_annual_all['OBS2-max'],alpha=0.5,facecolor='grey')
for ii in range(nummodel[0]):
    #print(ii)
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    if (linesty == 'dashed'):
        lwidth = 1.2
    else:
        lwidth = 1
    #print(ii,linecol,linesty)
    drake_annual_model[0].plot(y=drake_annual_model[0].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[90,220],label=inst)
    axes.set_xlabel('')
    axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=12)
    leg = axes.legend(bbox_to_anchor=(1.01,0.3),loc='upper left')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.subplots_adjust(left=0.08,right=0.78,top=0.92)

# OMIP2
axes = fig.add_subplot(3,1,2)
drake_annual_all.plot(y=drake_annual_all.columns[0],ax=axes,ylim=[90,220],color='darkgrey',linewidth=2,title='(b) OMIP2',legend=False)
axes.fill_between(x=drake_annual_all.index,y1=drake_annual_all['OBS1-min'],y2=drake_annual_all['OBS1-max'],alpha=0.5,facecolor='lightgrey')
drake_annual_all.plot(y=drake_annual_all.columns[4],ax=axes,ylim=[90,220],color='black',linewidth=2,legend=False)
axes.fill_between(x=drake_annual_all.index,y1=drake_annual_all['OBS2-min'],y2=drake_annual_all['OBS2-max'],alpha=0.5,facecolor='grey')
for ii in range(nummodel[1]):
    #print(ii)
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    if (linesty == 'dashed'):
        lwidth = 1.2
    else:
        lwidth = 1
        
    #print(ii,linecol,linesty)
    drake_annual_model[1].plot(y=drake_annual_model[1].columns[ii],ax=axes,color=linecol,linewidth=lwidth,linestyle=linesty,ylim=[90,220],legend=False)
    axes.set_xlabel('')
    axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=12)
    #axes.legend(bbox_to_anchor=(0.0,0.0),loc='lower left')
    plt.subplots_adjust(left=0.08,right=0.78)

# MMM

axes = fig.add_subplot(3,1,3)
drake_annual_all.plot(y=drake_annual_all.columns[0],color='darkgrey',linewidth=2,ax=axes,ylim=[90,220],title='(c) MMM')
drake_annual_all.plot(y=drake_annual_all.columns[4],color='black',linewidth=2,ax=axes,ylim=[90,220])
axes.fill_between(x=drake_annual_all.index,y1=drake_annual_all['OMIP1-min'],y2=drake_annual_all['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=drake_annual_all.index,y1=drake_annual_all['OMIP2-min'],y2=drake_annual_all['OMIP2-max'],alpha=0.5,facecolor='lightblue')
drake_annual_all.plot(y=drake_annual_all.columns[8],color='darkred' ,linewidth=2,ax=axes,ylim=[90,220])
drake_annual_all.plot(y=drake_annual_all.columns[12],color='darkblue',linewidth=2,ax=axes,ylim=[90,220])
axes.set_xlabel('year',fontsize=10)
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=12)
leg = axes.legend(bbox_to_anchor=(1.01,1.0),loc='upper left')
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

plt.subplots_adjust(left=0.09,right=0.70, bottom=0.08,top=0.92, hspace=0.22)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
