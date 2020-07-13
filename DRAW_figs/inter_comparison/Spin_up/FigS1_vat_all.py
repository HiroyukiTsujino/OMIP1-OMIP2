# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../python")
import json
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.util import strtobool
import netCDF4
from netCDF4 import Dataset, num2date
from uncertain_Wakamatsu import uncertain

#if (len(sys.argv) < 2) :
#    print ('Usage: ' + sys.argv[0] + ' 0 (suppress plt.show) or 1 (execute plt.show)')
#    sys.exit()

suptitle = 'Vertically averaged temperature'


metainfo = [ json.load(open("./json/vat_omip1.json")),
             json.load(open("./json/vat_omip2.json")) ]

metainfo_full = [ json.load(open("./json/vat_omip1_full.json")),
                  json.load(open("./json/vat_omip2_full.json")) ]

outfile = './fig/FigS1_vat_ALL'

lineinfo = json.load(open('../json/inst_color_style.json'))

title_list = [ "(a) OMIP1 (0 - 700 m)", "(b) OMIP2 (0 - 700 m)", "(c) MMM (0 - 700 m)", 
               "(d) OMIP1 (0 - 2000 m)","(e) OMIP2 (0 - 2000 m)","(f) MMM (0 - 2000 m)", 
               "(g) OMIP1 (2000 m - bottom)","(h) OMIP2 (2000 m - bottom)","(i) MMM (2000 m - bottom)",
               "(j) OMIP1 (0 m - bottom)", "(k) OMIP2 (0 m - bottom)", "(l) MMM (0 m - bottom)" ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]
model_full_list = [ metainfo_full[0].keys(), metainfo_full[1].keys() ]

var_list = [ "thetaoga_700", "thetaoga_2000", "thetaoga_2000_bottom", "thetaoga_all" ]
volume_list = np.array([ 2.338e17, 6.216e17, 7.593e17, 1.381e18 ])
degC_to_ZJ = volume_list * 3.99e3 * 1.036e3 * 1.0e-21


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.array([np.linspace(1948,2009,62)]*6),
         np.array([np.linspace(1958,2018,61)]*6) ]

for n in range(2):
    for i in range(6):
        time[n][i] = time[n][i] - (5-i)*71

timenan = [ time[1][:,60], time[0][:,0] ]
time[0] = time[0].reshape(6*62)
time[1] = time[1].reshape(6*61)


#J 間をあけるための Dataset 作成
varnan = np.full(6,np.nan)
DS_nan = [
    xr.Dataset({'var':(['time'],varnan),},coords={'time':timenan[0]}), 
    xr.Dataset({'var':(['time'],varnan),},coords={'time':timenan[1]})
]


#J 単一モデル用 dummy DS (json にエントリーがない場合に使用)
d_dummy = [ np.full( len(time[0]), np.nan ),
            np.full( len(time[1]), np.nan ) ]


#J データ読込 (n=0: OMIP1, n=1: OMIP2)
DS = []
DS_full = []

lincol = []
linsty = []
nummodel = []

for n in range(2):

    d = np.full( (len(var_list),len(model_list[n]),len(time[n])), np.nan )
    d_full = np.full( (len(var_list),len(model_full_list[n]),len(time[n])), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    coltmp = []
    stytmp = []

    nvar = 0
    for var in var_list:

        nmodel = 0
        nmodel_full = 0
        for model in model_list[n]:

            add_to_full = 'no'
            for model_full in model_full_list[n]:
                if (model == model_full):
                    add_to_full = 'yes'
                    break

            print(model + ' Added to full list ' + add_to_full)
                
            try:
                multidata = strtobool(metainfo[n][model]['multidata'])
            except:
                #J 単一モデル用エラー処理 (json にエントリーがない場合)
                d[nvar,nmodel] = d_dummy[n]
                continue

            if (nvar == 0):
                coltmp +=[lineinfo[model]["color"]]
                stytmp +=[lineinfo[model]["style"]]

            path = metainfo[n][model][var]['path']
            fname = metainfo[n][model][var]['fname']
            vname = metainfo[n][model][var]['varname']
            factor = float(metainfo[n][model][var]['factor'])
            infile = path + '/' + fname

            print(infile,vname)

            if (n == 1) and (model == "FSU-HYCOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_fsu = np.full( len(time[n]), np.nan )
                for cyc in range(5):
                    isto = 0 + cyc * 61
                    iedo = isto + 57
                    istf = 0 + cyc * 58
                    iedf = istf + 57
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                cyc = 5
                isto = 0 + cyc * 61
                iedo = isto + 61
                istf = 0 + cyc * 58
                iedf = istf + 61
                print (isto, iedo)
                print (istf, iedf)
                d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor
                #print (d_fsu)
                d[nvar,nmodel] = d_fsu

                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu

            elif (n == 1) and (model == "GFDL-MOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_fsu = np.full( len(time[n]), np.nan )
                for cyc in range(0,3):
                    isto = 0 + cyc * 61
                    iedo = isto + 59
                    istf = 0 + cyc * 60
                    iedf = istf + 59
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                for cyc in range(3,6):
                    isto = 0 + cyc * 61
                    iedo = isto + 60
                    istf = 0 + cyc * 61 - 3
                    iedf = istf + 60
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                #print (d_fsu)
                d[nvar,nmodel] = d_fsu
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu
            
            elif (n == 0) and (model == "MIROC-COCO4.9" ):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_coco = np.full( len(time[n]), np.nan )
                d_coco[0:62*5] = d_tmp[0:62*5] * factor
                d[nvar,nmodel] = d_coco
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_coco

            elif (n == 0) and (model == "GFDL-MOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_fsu = np.full( len(time[n]), np.nan )
                for cyc in range(0,5):
                    isto = 0 + cyc * 62
                    iedo = isto + 59
                    istf = 0 + cyc * 60
                    iedf = istf + 59
                    print (cyc, isto, iedo)
                    print (cyc, istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                for cyc in range(5,6):
                    isto = 0 + cyc * 62
                    iedo = isto + 61
                    istf = 0 + cyc * 62 - 10
                    iedf = istf + 61
                    print (cyc, isto, iedo)
                    print (cyc, istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                #print (d_fsu)
                d[nvar,nmodel] = d_fsu
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu

            else:

                if multidata:
                    if ( model == 'Kiel-NEMO' ):
                        DS_read = xr.open_mfdataset(infile,concat_dim='time_counter',decode_times=False)
                    elif ( model == 'EC-Earth3-NEMO' ):
                        DS_read = xr.open_mfdataset(infile,concat_dim='time',decode_times=False)
                    else:
                        DS_read = xr.open_mfdataset(infile,decode_times=False)

                else:
                    DS_read = xr.open_dataset(infile,decode_times=False)

                #d[nvar,nmodel] = DS_read[vname].values * factor * degC_to_ZJ[nvar]
                if ( model == 'Kiel-NEMO' ):
                    d[nvar,nmodel] = DS_read[vname].values[:,0,0] * factor
                    if (add_to_full == 'yes'):
                        d_full[nvar,nmodel_full] = DS_read[vname].values[:,0,0] * factor
                elif ( model == 'EC-Earth3-NEMO' ):
                    if (var == 'thetaoga_all'):
                        d[nvar,nmodel] = DS_read[vname].values * factor
                        if (add_to_full == 'yes'):
                            d_full[nvar,nmodel_full] = DS_read[vname].values * factor
                    else:
                        d[nvar,nmodel] = DS_read[vname].values[:,0,0] * factor
                        if (add_to_full == 'yes'):
                            d_full[nvar,nmodel_full] = DS_read[vname].values[:,0,0] * factor
                else:
                    d[nvar,nmodel] = DS_read[vname].values * factor
                    if (add_to_full == 'yes'):
                        d_full[nvar,nmodel_full] = DS_read[vname].values * factor

                #if (n == 0):
                #    print(d[nvar,nmodel,-6:-1])
                #    d[nvar,nmodel] = d[nvar,nmodel] - (d[nvar,nmodel,-6:-1]).mean()
                #
                #if (n == 1):
                #    print(d[nvar,nmodel,-15:-9])
                #    d[nvar,nmodel] = d[nvar,nmodel] - (d[nvar,nmodel,-15:-9]).mean()
            nmodel += 1
            if (add_to_full == 'yes'):
                nmodel_full += 1

        nvar += 1
            
    #J サイクル間に NaN データ挿入
    d_new = np.concatenate(
        [d, np.tile(varnan,(len(var_list),len(model_list[n]),1))],
        axis = 2 )

    d_full_new = np.concatenate(
        [d_full, np.tile(varnan,(len(var_list),len(model_full_list[n]),1))],
        axis = 2 )

    nst = 0
    for nv in range(nvar):
        for nm in range(nmodel_full):
            d_full_new[nv,nm,:] = d_full_new[nv,nm,:] - d_full_new[nv,nm,nst]

    time_new = np.concatenate( [time[n], timenan[n]] )

    #J xarray Dataset 再作成
    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new } ).sortby('time')

    DS += [DS_tmp]

    ######

    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d_full_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new } ).sortby('time')

    DS_full += [DS_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    nummodel += [nmodel]

##### output text #####

for n in range(2):
    if (n == 0):
        ifirst_yr=0
    else:
        ifirst_yr=1

    nmodel = 0
    dict_circulation={}
    for model in model_list[n]:
        if (model == 'MIROC-COCO4.9'):
            v700m     = DS[n]['thetaoga_700'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values - DS[n]['thetaoga_700'].sel(model=nmodel).isel(time=ifirst_yr).values
            v2000m    = DS[n]['thetaoga_2000'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values - DS[n]['thetaoga_2000'].sel(model=nmodel).isel(time=ifirst_yr).values
            v2000mbot = DS[n]['thetaoga_2000_bottom'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values - DS[n]['thetaoga_2000_bottom'].sel(model=nmodel).isel(time=ifirst_yr).values
            vtopbot   = DS[n]['thetaoga_all'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values - DS[n]['thetaoga_all'].sel(model=nmodel).isel(time=ifirst_yr).values
        else:
            v700m     = DS[n]['thetaoga_700'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values - DS[n]['thetaoga_700'].sel(model=nmodel).isel(time=ifirst_yr).values
            v2000m    = DS[n]['thetaoga_2000'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values - DS[n]['thetaoga_2000'].sel(model=nmodel).isel(time=ifirst_yr).values
            v2000mbot = DS[n]['thetaoga_2000_bottom'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values - DS[n]['thetaoga_2000_bottom'].sel(model=nmodel).isel(time=ifirst_yr).values
            vtopbot   = DS[n]['thetaoga_all'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values - DS[n]['thetaoga_all'].sel(model=nmodel).isel(time=ifirst_yr).values

        dict_circulation[model]=[v700m,v2000m,v2000mbot,vtopbot]
        nmodel += 1

    mipid='OMIP'+str(n+1)
    summary=pd.DataFrame(dict_circulation,index=['v700m-'+ str(mipid),'v2000m-'+ str(mipid),'v2000m-bot-'+ str(mipid),'vtop-bot-'+ str(mipid)])
    tmp1=summary.mean(axis=1)
    tmp2=summary.std(axis=1,ddof=0)
    print(tmp1,tmp2)
    summary.insert(0,'Z-MMM',tmp1)
    summary.insert(0,'Z-STD',tmp2)
    summary_t=summary.T
    print (summary_t)
    summary_t.to_csv('csv/vat_full_omip' + str(n+1) + '.csv')

# omip1 - omip2

print("Compute uncertainty of the difference omip2-omip1")

ystr = 1980
yend = 2009
nyr = yend - ystr + 1
nmod = len(model_list[0])

d_v700m     = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)
d_v2000m    = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)
d_v2000mbot = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)
d_vtopbot   = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)

nmodel = 0
for model in model_list[0]:
    if (model == 'MIROC-COCO4.9'):
        v700m_omip1     = DS[0]['thetaoga_700'].sel(model=nmodel,time=slice(1909,1938)).values - DS[0]['thetaoga_700'].sel(model=nmodel).isel(time=0).values
        v2000m_omip1    = DS[0]['thetaoga_2000'].sel(model=nmodel,time=slice(1909,1938)).values - DS[0]['thetaoga_2000'].sel(model=nmodel).isel(time=0).values
        v2000mbot_omip1 = DS[0]['thetaoga_2000_bottom'].sel(model=nmodel,time=slice(1909,1938)).values - DS[0]['thetaoga_2000_bottom'].sel(model=nmodel).isel(time=0).values
        vtopbot_omip1   = DS[0]['thetaoga_all'].sel(model=nmodel,time=slice(1909,1938)).values - DS[0]['thetaoga_all'].sel(model=nmodel).isel(time=0).values
        v700m_omip2     = DS[1]['thetaoga_700'].sel(model=nmodel,time=slice(1909,1938)).values - DS[1]['thetaoga_700'].sel(model=nmodel).isel(time=1).values
        v2000m_omip2    = DS[1]['thetaoga_2000'].sel(model=nmodel,time=slice(1909,1938)).values - DS[1]['thetaoga_2000'].sel(model=nmodel).isel(time=1).values
        v2000mbot_omip2 = DS[1]['thetaoga_2000_bottom'].sel(model=nmodel,time=slice(1909,1938)).values - DS[1]['thetaoga_2000_bottom'].sel(model=nmodel).isel(time=1).values
        vtopbot_omip2   = DS[1]['thetaoga_all'].sel(model=nmodel,time=slice(1909,1938)).values - DS[1]['thetaoga_all'].sel(model=nmodel).isel(time=1).values
    else:
        v700m_omip1     = DS[0]['thetaoga_700'].sel(model=nmodel,time=slice(1980,2009)).values - DS[0]['thetaoga_700'].sel(model=nmodel).isel(time=0).values
        v2000m_omip1    = DS[0]['thetaoga_2000'].sel(model=nmodel,time=slice(1980,2009)).values - DS[0]['thetaoga_2000'].sel(model=nmodel).isel(time=0).values
        v2000mbot_omip1 = DS[0]['thetaoga_2000_bottom'].sel(model=nmodel,time=slice(1980,2009)).values - DS[0]['thetaoga_2000_bottom'].sel(model=nmodel).isel(time=0).values
        vtopbot_omip1   = DS[0]['thetaoga_all'].sel(model=nmodel,time=slice(1980,2009)).values - DS[0]['thetaoga_all'].sel(model=nmodel).isel(time=0).values
        v700m_omip2     = DS[1]['thetaoga_700'].sel(model=nmodel,time=slice(1980,2009)).values - DS[1]['thetaoga_700'].sel(model=nmodel).isel(time=1).values
        v2000m_omip2    = DS[1]['thetaoga_2000'].sel(model=nmodel,time=slice(1980,2009)).values - DS[1]['thetaoga_2000'].sel(model=nmodel).isel(time=1).values
        v2000mbot_omip2 = DS[1]['thetaoga_2000_bottom'].sel(model=nmodel,time=slice(1980,2009)).values - DS[1]['thetaoga_2000_bottom'].sel(model=nmodel).isel(time=1).values
        vtopbot_omip2   = DS[1]['thetaoga_all'].sel(model=nmodel,time=slice(1980,2009)).values - DS[1]['thetaoga_all'].sel(model=nmodel).isel(time=1).values

    d_v700m[nmodel,:] = v700m_omip2[:] - v700m_omip1[:]
    d_v2000m[nmodel,:] = v2000m_omip2[:] - v2000m_omip1[:]
    d_v2000mbot[nmodel,:] = v2000mbot_omip2[:] - v2000mbot_omip1[:]
    d_vtopbot[nmodel,:] = vtopbot_omip2[:] - vtopbot_omip1[:]

    nmodel += 1

num_bootstraps = 10000
factor_5pct = 1.64  # 5-95%

dout_700m = uncertain(d_v700m, "v700", nmod, nyr, num_bootstraps )
zval = dout_700m["mean"] / dout_700m["std"]
print(dout_700m)
print("z-value = ",zval)

dout_2000m = uncertain(d_v2000m, "v2000", nmod, nyr, num_bootstraps )
zval = dout_2000m["mean"] / dout_2000m["std"]
print(dout_2000m)
print("z-value = ",zval)

dout_2000mbot = uncertain(d_v2000mbot, "v2000bot", nmod, nyr, num_bootstraps )
zval = dout_2000mbot["mean"] / dout_2000mbot["std"]
print(dout_2000mbot)
print("z-value = ",zval)

dout_topbot = uncertain(d_vtopbot, "topbot", nmod, nyr, num_bootstraps )
zval = dout_topbot["mean"] / dout_topbot["std"]
print(dout_topbot)
print("z-value = ",zval)

#J 描画
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )
ax = [ plt.subplot(4,3,1),
       plt.subplot(4,3,2),
       plt.subplot(4,3,3),
       plt.subplot(4,3,4),
       plt.subplot(4,3,5),
       plt.subplot(4,3,6),
       plt.subplot(4,3,7),
       plt.subplot(4,3,8),
       plt.subplot(4,3,9),
       plt.subplot(4,3,10),
       plt.subplot(4,3,11),
       plt.subplot(4,3,12) ]

for n in range(2):
    nv = 0
    for var in var_list:
        nf = nv * 3 + n
        nmodel = 0
        for model in model_list[n]:
            linecol=lincol[n][nmodel]
            linesty=linsty[n][nmodel]
            if (linesty == 'dashed'):
                lwidth = 1.0
            else:
                lwidth = 0.8
            if (n == 0 and nv == 0):
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=lwidth,linestyle=linesty)
            else:
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],color=linecol,linewidth=lwidth,linestyle=linesty)

            nmodel += 1

        if (n == 0 and nv == 0):
            leg = ax[nf].legend(bbox_to_anchor=(3.45,0.6),loc='upper left')
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

        nv += 1

nv = 0
for var in var_list:

    nf = nv * 3 + 2

    if (nv == 0): 
        DS_full[0][var].mean(dim='model').plot.line(x='time',ax=ax[nf],label='OMIP1-MMM',color='darkred')
    else:
        DS_full[0][var].mean(dim='model').plot.line(x='time',ax=ax[nf],color='darkred')

    ax[nf].fill_between(x=DS_full[0]['time'],
                        y1=DS_full[0][var].min(dim='model'),
                        y2=DS_full[0][var].max(dim='model'),
                        alpha=0.5, facecolor='lightcoral')

    if (nv == 0): 
        DS_full[1][var].mean(dim='model').plot.line(x='time',ax=ax[nf],label='OMIP2-MMM',color='darkblue')
    else:
        DS_full[1][var].mean(dim='model').plot.line(x='time',ax=ax[nf],color='darkblue')

    ax[nf].fill_between(x=DS_full[1]['time'],
                        y1=DS_full[1][var].min(dim='model'),
                        y2=DS_full[1][var].max(dim='model'),
                        alpha=0.5, facecolor='lightblue')

    if (nv == 0): 
        ax[nf].legend(bbox_to_anchor=(1.05,1.0),loc='upper left')
    
    nv += 1

#####

ylim = [ [9.6, 11.4], [5.4, 6.8], [0.6, 2.2], [3.0, 4.2] ]
ytick = [ np.linspace(9.6, 11.4, 10),
          np.linspace(5.4,  6.8,  8),
          np.linspace(0.6,  2.2,  9),
          np.linspace(3.0,  4.2,  7), ]

for n in range(2):
    for nv in range(4):
        nf = nv * 3 + n
        ax[nf].set_title(title_list[nf],{'fontsize':10, 'verticalalignment':'top'})
        ax[nf].tick_params(labelsize=9)
        ax[nf].set_xlim(1592,2018)
        ax[nf].set_xticks(np.arange(1592,2018.1,71))
        if ( nv == 3 ):
            ax[nf].set_xlabel('year',fontsize=10)
        else:
            ax[nf].set_xlabel('',fontsize=10)

        ax[nf].grid()
        ax[nf].set_ylim(ylim[nv])
        ax[nf].set_yticks(ytick[nv])
        if (n == 0):
            ax[nf].set_ylabel(r'$^{\circ}$C',fontsize=12)
        else:
            ax[nf].set_ylabel('')


#####

ylim = [ [-0.4, 1.0], [-0.6, 0.6], [-0.8, 0.6], [-0.6, 0.6] ]
ytick = [ np.linspace(-0.4, 1.0, 8),
          np.linspace(-0.6, 0.6, 7),
          np.linspace(-0.8, 0.6, 8),
          np.linspace(-0.6, 0.6, 7), ]

for nv in range(4):
    nf = nv * 3 + 2
    ax[nf].set_title(title_list[nf],{'fontsize':10,'verticalalignment':'top'})
    ax[nf].tick_params(labelsize=9)
    ax[nf].set_xlim(1592,2018)
    ax[nf].set_xticks(np.arange(1592,2018.1,71))
    if ( nv == 3 ):
        ax[nf].set_xlabel('year',fontsize=10)
    else:
        ax[nf].set_xlabel('',fontsize=10)

    ax[nf].grid()
    ax[nf].set_ylim(ylim[nv])
    ax[nf].set_yticks(ytick[nv])
    ax[nf].set_ylabel('')
#    ax[nf].set_ylabel(r'$^{\circ}$C',fontsize=12)

plt.subplots_adjust(left=0.07,right=0.80,bottom=0.07,top=0.92,hspace=0.30)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()
