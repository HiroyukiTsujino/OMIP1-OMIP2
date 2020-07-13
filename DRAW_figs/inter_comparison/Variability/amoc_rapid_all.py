# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../python")
import json
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######

metainfo = [ json.load(open('json/amoc_rapid_omip1.json')),
             json.load(open('json/amoc_rapid_omip2.json'))]
lineinfo = json.load(open('../json/inst_color_style.json'))

outfile = './fig/Fig1a_all'
suptitle = 'AMOC at RAPID section (26.5$^{\circ}$N)'


######## RAPID ########

file_rapid = "../refdata/RAPID/moc_transports.nc"
nco = netCDF4.Dataset(file_rapid,'r')
amoc_rapid_obs = nco.variables['moc_mar_hc10'][:]
time_var_obs = nco.variables['time']
cftime = num2date(time_var_obs[:],time_var_obs.units)
nco.close()

col = pd.Index(['RAPID'],name='institution')
rapid_obs_df = pd.DataFrame(amoc_rapid_obs,index=cftime,columns=col)
print (rapid_obs_df)
rapid_obs_df = rapid_obs_df.set_index([rapid_obs_df.index.year, rapid_obs_df.index])
rapid_obs_df.index.names = ['year','date']
rapid_obs_annual=rapid_obs_df.mean(level='year')

template = 'Institution {0:3d} is {1:s}'

#######################

rapid_annual_model = []
rapid_mean_df = []
#rapid_uncertain_df = []
lincol = []
linsty = []
modnam = []
nummodel = []

for omip in range(2):

    i=0

    if (omip == 0):
        dtime = pd.date_range('1948-01-01','2009-12-01',freq='MS')
        dtimey = np.arange(1948,2010,1)
    else:
        dtime = pd.date_range('1958-01-01','2018-12-01',freq='MS')
        dtimey = np.arange(1958,2019,1)

    coltmp = []
    stytmp = []
    namtmp = []
    
    for inst in metainfo[omip].keys():


        if (inst == 'NorESM-BLOM' or inst == 'GFDL-MOM' ):
            continue

        print (template.format(i,inst))
        
        coltmp +=[lineinfo[inst]["color"]]
        stytmp +=[lineinfo[inst]["style"]]
        namtmp +=[inst]
        fac=float(metainfo[omip][inst]['factor'])
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']

        nc = netCDF4.Dataset(infile,'r')
        if (inst == 'Kiel-NEMO'):
            amoc_rapid = nc.variables['amoc_rapid'][:,0,0]
        else:
            amoc_rapid = nc.variables['amoc_rapid'][:]

        nc.close()

        col = pd.Index([inst + '-OMIP' + str(omip+1)],name='institution')
        rapid_df = pd.DataFrame(amoc_rapid*fac,index=dtime,columns=col)
        rapid_df = rapid_df.set_index([rapid_df.index.year,rapid_df.index])
        rapid_df.index.names = ['year','date']

        if i == 0:
            rapid_df_all=rapid_df
        else:
            rapid_df_all=pd.concat([rapid_df_all,rapid_df],axis=1)
            
        print (rapid_df_all)
        i=i+1

    rapid_annual=rapid_df_all.mean(level='year')

    ###### NorESM-BLOM and GFDL-MOM provided time series of annual mean ########

    inst='NorESM-BLOM'
    if (inst in metainfo[omip]):
        print (template.format(i,inst))

        coltmp +=[lineinfo[inst]["color"]]
        stytmp +=[lineinfo[inst]["style"]]
        namtmp +=[inst]

        fac=float(metainfo[omip][inst]['factor'])
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']

        nc = netCDF4.Dataset(infile,'r')
        amoc_rapid_nor = nc.variables['amoc_rapid'][:]
        nc.close()

        col = pd.Index([inst + '-OMIP' + str(omip+1)],name='institution')
        rapid_nor_df = pd.DataFrame(amoc_rapid_nor*fac,index=dtimey,columns=col)
        rapid_nor_df.index.names = ['year']
        rapid_annual_model_tmp=pd.concat([rapid_annual,rapid_nor_df],axis=1)
        i=i+1

    #####

    inst='GFDL-MOM'
    if (inst in metainfo[omip]):
        print (template.format(i,inst))

        coltmp +=[lineinfo[inst]["color"]]
        stytmp +=[lineinfo[inst]["style"]]
        namtmp +=[inst]

        fac=float(metainfo[omip][inst]['factor'])
        infile = metainfo[omip][inst]['path'] + '/' + metainfo[omip][inst]['fname']

        nc = netCDF4.Dataset(infile,'r')
        amoc_rapid_gfdl = nc.variables['amoc_rapid'][:]
        nc.close()

        col = pd.Index([inst + '-OMIP' + str(omip+1)],name='institution')
        rapid_gfdl_df = pd.DataFrame(amoc_rapid_gfdl*fac,index=dtimey,columns=col)
        rapid_gfdl_df.index.names = ['year']
        rapid_annual_model_tmp=pd.concat([rapid_annual_model_tmp,rapid_gfdl_df],axis=1)
        i=i+1

    ###### multi model mean ######

    #n_bootstraps = 10000
    #n_size = dtimey.size
    #model_ensemble_tmp = rapid_annual_model_tmp.to_numpy()
    #model_ensemble = model_ensemble_tmp.transpose()
    #dout_tmp = uncertain( model_ensemble, 'OMIP'+str(omip+1), i, n_size, n_bootstraps )
    #print(dout_tmp)
    #col = pd.Index(['OMIP' + str(omip+1) + '-mean','OMIP' + str(omip+1) + '-std','OMIP' + str(omip+1) + '-model','OMIP' + str(omip+1) + '-internal','OMIP' + str(omip+1) + '-bootstrap'],name='institution')
    #rapid_uncertain_df_tmp = pd.DataFrame(dout_tmp,index=dtimeyw,columns=col)
    #print(rapid_uncertain_df_tmp)
    #rapid_uncertain_df += [rapid_uncertain_df_tmp]


    rapid_annual_model += [rapid_annual_model_tmp]

    col = pd.Index(['OMIP' + str(omip+1) + '-mean'],name='institution')
    rapid_mean=rapid_annual_model_tmp.mean(axis=1)
    rapid_mean_df_tmp = pd.DataFrame(rapid_mean,columns=col)

    col = pd.Index(['OMIP' + str(omip+1) + '-std'],name='institution')
    rapid_std=rapid_annual_model_tmp.std(axis=1,ddof=0)
    rapid_std_df_tmp = pd.DataFrame(rapid_std,columns=col)

    rapid_mean_df_tmp = pd.concat([rapid_mean_df_tmp,rapid_std_df_tmp],axis=1)

    rapid_mean_df_tmp['OMIP' + str(omip+1) + '-min'] = rapid_mean_df_tmp.iloc[:,0] - rapid_mean_df_tmp.iloc[:,1]
    rapid_mean_df_tmp['OMIP' + str(omip+1) + '-max'] = rapid_mean_df_tmp.iloc[:,0] + rapid_mean_df_tmp.iloc[:,1]

    rapid_mean_df += [rapid_mean_df_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    modnam += [namtmp]
    nummodel += [i]
    

    #############################


print(rapid_mean_df[0])

rapid_annual_all=pd.concat([rapid_obs_annual,rapid_mean_df[0],rapid_mean_df[1],rapid_annual_model[0],rapid_annual_model[1]],axis=1)


#print(rapid_annual_model[0])
#print(rapid_annual_model[1])
print(rapid_annual_all)

# draw figures
fig  = plt.figure(figsize = (8,11))
fig.suptitle( suptitle, fontsize=18 )

# OMIP1
axes = fig.add_subplot(3,1,1)
rapid_annual_all.plot(y=rapid_annual_all.columns[0],ax=axes,ylim=[7,23],color='darkgrey',linewidth=4,title='(a) OMIP1')
for ii in range(nummodel[0]):
    #print(ii)
    linecol=lincol[0][ii]
    linesty=linsty[0][ii]
    inst=modnam[0][ii]
    #print(ii,linecol,linesty)
    if (linesty == 'dashed'):
        linewid=1.2
    else:
        linewid=1
    rapid_annual_model[0].plot(y=rapid_annual_model[0].columns[ii],ax=axes,color=linecol,linewidth=linewid,linestyle=linesty,ylim=[7,23],label=inst)

axes.grid()
axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=12)
leg = axes.legend(bbox_to_anchor=(1.01,0.3),loc='upper left')
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)
    
# OMIP2
axes = fig.add_subplot(3,1,2)
rapid_annual_all.plot(y=rapid_annual_all.columns[0],ax=axes,ylim=[7,23],color='darkgrey',linewidth=4,title='(b) OMIP2',legend=False)
for ii in range(nummodel[1]):
    #print(ii)
    linecol=lincol[1][ii]
    linesty=linsty[1][ii]
    inst=modnam[0][ii]
    #print(ii,linecol,linesty)
    if (linesty == 'dashed'):
        linewid=1.2
    else:
        linewid=1
    rapid_annual_model[1].plot(y=rapid_annual_model[1].columns[ii],ax=axes,color=linecol,linewidth=linewid,linestyle=linesty,ylim=[7,23],legend=False)

axes.grid()
axes.set_xlabel('')
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=12)
#leg = axes.legend(bbox_to_anchor=(1.01,1.0))
#for legobj in leg.legendHandles:
#    legobj.set_linewidth(2.0)

# MMM

axes = fig.add_subplot(3,1,3)
rapid_annual_all.plot(y=rapid_annual_all.columns[0],color='darkgreen',linewidth=2,ax=axes,ylim=[7,23],title='(c) MMM')
axes.fill_between(x=rapid_annual_all.index,y1=rapid_annual_all['OMIP1-min'],y2=rapid_annual_all['OMIP1-max'],alpha=0.5,facecolor='lightcoral')
axes.fill_between(x=rapid_annual_all.index,y1=rapid_annual_all['OMIP2-min'],y2=rapid_annual_all['OMIP2-max'],alpha=0.5,facecolor='lightblue')
rapid_annual_all.plot(y=rapid_annual_all.columns[5],color='darkblue',linewidth=2,ax=axes,ylim=[7,23])
rapid_annual_all.plot(y=rapid_annual_all.columns[1],color='darkred' ,linewidth=2,ax=axes,ylim=[7,23])


axes.grid()
axes.set_xlabel('year',fontsize=10)
axes.set_ylabel(r'$\times 10^9 \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=12)
leg = axes.legend(bbox_to_anchor=(1.01,1.0),loc='upper left')
for legobj in leg.legendHandles:
    legobj.set_linewidth(2.0)

plt.subplots_adjust(left=0.08,right=0.78,bottom=0.08,top=0.92, hspace=0.22)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
