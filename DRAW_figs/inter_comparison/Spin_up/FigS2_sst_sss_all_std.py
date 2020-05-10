# -*- coding: utf-8 -*-
import sys
import json
import math
import numpy as np
from scipy import interpolate
import xarray as xr
import matplotlib.pyplot as plt
from distutils.util import strtobool
import netCDF4
from netCDF4 import Dataset, num2date


title_list = [ "(a) OMIP1 SST", 
               "(b) OMIP1 SST drift", 
               "(c) OMIP2 SST", 
               "(d) OMIP2 SST drift", 
               "(e) OMIP1 SSS", 
               "(f) OMIP1 SSS drift", 
               "(g) OMIP2 SSS",
               "(h) OMIP2 SSS drift", ]


var_list = [ "thetao", "so" ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

d_lev33 = np.array(lev33)

if (len(sys.argv) < 1) :
    print ('Usage: ' + sys.argv[0] + ' [show (to check using viewer)]')
    sys.exit()

outfile = './fig/FigS2_sst_sss'
suptitle = 'Drift of globally averaged SST and SSS '

metainfo = [ json.load(open("./json/ts_z_omip1.json")),
             json.load(open("./json/ts_z_omip2.json")) ]

metainfo_full = [ json.load(open("./json/ts_z_omip1_full.json")),
                  json.load(open("./json/ts_z_omip2_full.json")) ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]
model_full_list = [ metainfo_full[0].keys(), metainfo_full[1].keys() ]

print("Drawing "+suptitle)

lineinfo = json.load(open('../json/inst_color_style.json'))

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
varnan = np.full((6,33),np.nan)


#J 単一モデル用 dummy DS (json にエントリーがない場合に使用)
d_dummy = [ np.full( (len(time[0]),33), np.nan ),
            np.full( (len(time[1]),33), np.nan ) ]


#J 参照データ読込
d_woa = np.empty( (2,33) )

print( "Loading WOA13v2 data" )

reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_th.1000'
da_ref = xr.open_dataset(reffile,decode_times=False)["thetao"].mean(dim='time')

wgt0 = np.empty((180,360))
for j in range(180):
    wgt0[j] = math.cos(math.radians(da_ref.lat.values[j]))

wgt = np.tile( wgt0, (33,1,1) ) * np.logical_not(np.isnan(da_ref))
d_woa[0] = np.average(da_ref.fillna(0), weights=wgt, axis=(1,2))

reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_s.1000'
da_ref = xr.open_dataset(reffile,decode_times=False)["so"].mean(dim='time')

wgt = np.tile( wgt0, (33,1,1) ) * np.logical_not(np.isnan(da_ref))
d_woa[1] = np.average(da_ref.fillna(0), weights=wgt, axis=(1,2))


#J データ読込 (n=0: OMIP1, n=1: OMIP2)
lincol = []
linsty = []
DS = []
DS_full = []

for n in range(2):
    d = np.full( (len(var_list),len(model_list[n]),len(time[n]),33), np.nan )
    d_full = np.full( (len(var_list),len(model_full_list[n]),len(time[n]),33), np.nan )
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
            vname = metainfo[n][model][var]['vname']
            factor = float(metainfo[n][model][var]['factor'])
            infile = path + '/' + fname

            if (n == 1) and (model == "FSU-HYCOM"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[var][:,:]
                nc.close()

                d_fsu = np.full( (len(time[n]),33), np.nan )
                for cyc in range(5):
                    isto = 0 + cyc * 61
                    iedo = isto + 57
                    istf = 0 + cyc * 58
                    iedf = istf + 57
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1,:] = d_tmp[istf:iedf+1,:] * factor

                cyc = 5
                isto = 0 + cyc * 61
                iedo = isto + 61
                istf = 0 + cyc * 58
                iedf = istf + 61
                print (isto, iedo)
                print (istf, iedf)
                d_fsu[isto:iedo+1,:] = d_tmp[istf:iedf+1,:] * factor
                #print (d_fsu)
                d[nvar,nmodel] = d_fsu
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu

            elif (n == 0) and (model == "MIROC-COCO4.9"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[var][:,:]
                nc.close()

                d_coco = np.full( (len(time[n]),33), np.nan )
                d_coco[0:62*5,:] = d_tmp[0:62*5,:] * factor
                d[nvar,nmodel] = d_coco
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_coco

            elif (model == "CMCC-NEMO"):
                if (n == 0):
                    num_yr = 372
                else:
                    num_yr = 366
                    
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[var][:,:]
                d_levs = nc.variables['depth'][:]
                nc.close()

                d_levs[0] = 0.0
                d_levs[48] = 5500.0
                print("CMCC-NEMO depth", d_levs)
                d_tmp2 = np.full( (len(time[n]),33), np.nan )
                for nt in range(num_yr):
                    d_tmp1d = d_tmp[nt,:]
                    f1 = interpolate.interp1d(d_levs,d_tmp1d)
                    f2 = f1(d_lev33)
                    d_tmp2[nt,:] = f2[:]

                d_cmcc = np.full( (len(time[n]),33), np.nan )
                d_cmcc[0:num_yr,:] = d_tmp2[0:num_yr,:] * factor
                d[nvar,nmodel] = d_cmcc
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_cmcc

            elif (model == "EC-Earth3-NEMO"):
                
                if (n == 0):
                    num_yr = 62
                else:
                    num_yr = 61
                    
                d_barca = np.full( (len(time[n]),33), np.nan )
                for i in range(6):
                    infile=path + str(i+1) + '/' + fname
                    #print(infile,vname)
                    nc = netCDF4.Dataset(infile,'r')
                    d_tmp = nc.variables[vname][:,:]
                    d_levs = nc.variables['lev'][:]
                    nc.close()
                    d_levs[0] = 0.0
                    if (i == 0):
                        print("EC-Earth3-NEMO depth", d_levs)
                    for nt in range(num_yr):
                        d_tmp1d = d_tmp[nt,:]
                        f1 = interpolate.interp1d(d_levs,d_tmp1d)
                        f2 = f1(d_lev33)
                        d_barca[num_yr*i+nt,:] = f2[:] * factor

                d[nvar,nmodel] = d_barca
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_barca

            elif (model == "GFDL-MOM"):
                if (n == 0):
                    num_yr = 362
                else:
                    num_yr = 363

                d_tmp2 = np.full( (len(time[n]),33), np.nan )
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:,:]
                d_levs = nc.variables['z_l'][:]
                nc.close()
                d_levs[0] = 0.0
                for nt in range(num_yr):
                    d_tmp1d = d_tmp[nt,:]
                    f1 = interpolate.interp1d(d_levs,d_tmp1d)
                    f2 = f1(d_lev33)
                    d_tmp2[nt,:] = f2[:]

                d_gfdl = np.full( (len(time[n]),33), np.nan )

                if ( n == 0 ):
                    for cyc in range(0,5):
                        isto = 0 + cyc * 62
                        iedo = isto + 59
                        istf = 0 + cyc * 60
                        iedf = istf + 59
                        print (cyc, isto, iedo)
                        print (cyc, istf, iedf)
                        d_gfdl[isto:iedo+1,:] = d_tmp2[istf:iedf+1,:] * factor

                    for cyc in range(5,6):
                        isto = 0 + cyc * 62
                        iedo = isto + 61
                        istf = 0 + cyc * 62 - 10
                        iedf = istf + 61
                        print (cyc, isto, iedo)
                        print (cyc, istf, iedf)
                        d_gfdl[isto:iedo+1,:] = d_tmp2[istf:iedf+1,:] * factor

                else:
                    for cyc in range(0,3):
                        isto = 0 + cyc * 61
                        iedo = isto + 59
                        istf = 0 + cyc * 60
                        iedf = istf + 59
                        print (isto, iedo)
                        print (istf, iedf)
                        d_gfdl[isto:iedo+1,:] = d_tmp2[istf:iedf+1,:] * factor

                    for cyc in range(3,6):
                        isto = 0 + cyc * 61
                        iedo = isto + 60
                        istf = 0 + cyc * 61 - 3
                        iedf = istf + 60
                        print (isto, iedo)
                        print (istf, iedf)
                        d_gfdl[isto:iedo+1,:] = d_tmp2[istf:iedf+1,:] * factor

                        
                d[nvar,nmodel] = d_gfdl
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_gfdl

            else:

                if multidata:
#                    DS_read = xr.open_mfdataset(infile,decode_times=False,concat_dim='time')
                    DS_read = xr.open_mfdataset(infile,decode_times=False,concat_dim='time',combine='nested')
                else:
                    DS_read = xr.open_dataset(infile,decode_times=False)
                    
                if model == "AWI-FESOM":
                    DS_read = DS_read.transpose()
                if (model == "NorESM-BLOM"):
                    DS_read = DS_read.interp(depth=lev33)
                if (model == "EC-Earth3-NEMO"):
                    print(DS_read[vname].values)
                    DS_read = DS_read.rename({vname:var})
                    DS_read = DS_read.interp(lev=lev33)
                    print(DS_read[var].values)
                    
                d[nvar,nmodel] = DS_read[var].values * factor
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = DS_read[var].values * factor

            nmodel += 1
            if (add_to_full == 'yes'):
                nmodel_full += 1

        nvar += 1

    #J サイクル間に NaN データ挿入
    d_new = np.concatenate(
        [d, np.tile(varnan,(len(var_list),len(model_list[n]),1,1))],
        axis = 2 )

    d_full_new = np.concatenate(
        [d_full, np.tile(varnan,(len(var_list),len(model_full_list[n]),1,1))],
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
        var_dict[var] = (['model','time','depth'], d_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new,
                                              'depth': lev33, } ).sortby('time')

    DS += [DS_tmp]

    ######

    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time','depth'], d_full_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new,
                                              'depth': lev33, } ).sortby('time')

    DS_full += [DS_tmp]
    
    lincol += [coltmp]
    linsty += [stytmp]

#print( DS )

#J 描画
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )
ax = [ plt.subplot(4,2,1), 
       plt.subplot(4,2,2), 
       plt.subplot(4,2,3),
       plt.subplot(4,2,4),
       plt.subplot(4,2,5), 
       plt.subplot(4,2,6), 
       plt.subplot(4,2,7),
       plt.subplot(4,2,8) ]
clabels = ['$^{\circ}$C', 'psu']

ylim = [ [17.7, 18.9], [34.3, 35.1], [-0.2, 0.7], [-0.3, 0.3] ]
ytick = [ np.linspace(17.7, 18.9, 13),
          np.linspace(34.3, 35.1, 9),
          np.linspace(-0.2,  0.7, 10),
          np.linspace(-0.3,  0.3, 7), ]

col_mmm = [ 'darkred', 'darkblue' ]
sha_mmm = [ 'lightcoral', 'lightblue' ]

for n in range(2):
    for nvar in range(2):
        nax = n*2 + nvar*4
        print(nax+1,title_list[nax])
        var = var_list[nvar]
        nmodel = 0
        for model in model_list[n]:
            linecol=lincol[n][nmodel]
            linesty=linsty[n][nmodel]
            if (linesty == 'dashed'):
                lwidth = 1.0
            else:
                lwidth = 0.8
            DS[n][var].sel(model=nmodel,depth=0.0).plot.line(x='time',ax=ax[nax],label=model,color=linecol,linewidth=lwidth,linestyle=linesty)
            nmodel += 1

        if (nax == 0): 
            ax[nax].legend(bbox_to_anchor=(2.25,0.4),loc='upper left')

        ax[nax].set_title(title_list[nax],{'fontsize':10, 'verticalalignment':'top'})
        ax[nax].tick_params(labelsize=9)
        ax[nax].set_xlim(1592,2018)
        ax[nax].set_xticks(np.arange(1592,2018.1,71))
        if ( n == 1 and nvar == 1 ):
            ax[nax].set_xlabel('year',fontsize=10)
        else:
            ax[nax].set_xlabel('',fontsize=10)

        ax[nax].grid()
        ax[nax].set_ylim(ylim[nvar])
        ax[nax].set_yticks(ytick[nvar])
        ax[nax].set_ylabel(clabels[nvar],fontsize=10)

for n in range(2):
    for nvar in range(2):
        nax = n*2 + nvar*4 + 1
        print(nax+1,title_list[nax])

        var = var_list[nvar]
        label_mmm = 'OMIP'+str(n+1)+'-MMM'
        if (nvar == 0):
            DS_full[n][var].sel(depth=0.0).mean(dim='model').plot.line(x='time',ax=ax[nax],label=label_mmm,color=col_mmm[n])
        else:
            DS_full[n][var].sel(depth=0.0).mean(dim='model').plot.line(x='time',ax=ax[nax],color=col_mmm[n])

        ax[nax].fill_between(x=DS_full[n]['time'],
                             y1=DS_full[n][var].sel(depth=0.0).min(dim='model'),
                             y2=DS_full[n][var].sel(depth=0.0).max(dim='model'),
                             alpha=0.5, facecolor=sha_mmm[n])

        if (nvar == 0): 
            if (n == 0):
                ax[nax].legend(bbox_to_anchor=(1.05,1.0),loc='upper left')
            else:
                ax[nax].legend(bbox_to_anchor=(1.05,2.0),loc='upper left')

        ax[nax].set_title(title_list[nax],{'fontsize':10, 'verticalalignment':'top'})
        ax[nax].tick_params(labelsize=9)
        ax[nax].set_xlim(1592,2018)
        ax[nax].set_xticks(np.arange(1592,2018.1,71))
        if ( n == 1 and nvar == 1 ):
            ax[nax].set_xlabel('year',fontsize=10)
        else:
            ax[nax].set_xlabel('',fontsize=10)

        ax[nax].grid()
        ax[nax].set_ylim(ylim[nvar+2])
        ax[nax].set_yticks(ytick[nvar+2])
        ax[nax].set_ylabel(clabels[nvar],fontsize=10)


plt.subplots_adjust(left=0.08,right=0.80,bottom=0.08,top=0.92,hspace=0.28)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
