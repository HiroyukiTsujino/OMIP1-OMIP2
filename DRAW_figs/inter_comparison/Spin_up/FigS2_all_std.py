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


title_list = [ "(a) OMIP1 temperature anomaly from WOA13v2", 
               "(b) OMIP1 temperature ensemble std", 
               "(c) OMIP2 temperature anomaly from WOA13v2", 
               "(d) OMIP2 temperature ensemble std", 
               "(e) OMIP1 salinity anomaly from WOA13v2", 
               "(f) OMIP1 salinity ensemble std", 
               "(g) OMIP2 salinity anomaly from WOA13v2",
               "(h) OMIP2 salinity ensemble std", ]


var_list = [ "thetao", "so" ]

lev33 = [ 0.,    10.,   20.,   30.,   50.,   75.,   100.,  125.,  150.,  200., 
          250.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,  1000., 1100.,
          1200., 1300., 1400., 1500., 1750., 2000., 2500., 3000., 3500., 4000.,
          4500., 5000., 5500. ]

d_lev33 = np.array(lev33)

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()

if (sys.argv[1] == 'MMM'):
    outfile = './fig/FigS2_MMM'
    suptitle = 'Horizontally Averaged Temperature and Salinity Drift of Model Ensemble '
    metainfo = [ json.load(open("./json/ts_z_omip1_full.json")),
                 json.load(open("./json/ts_z_omip2_full.json")) ]
    model_list = [ metainfo[0].keys(), metainfo[1].keys() ]
    print("Drawing "+suptitle)
    
else:
    outfile = './fig/FigS2_' + sys.argv[1]
    suptitle = sys.argv[1]
    metainfo = [ json.load(open("./json/ts_z_omip1.json")),
                 json.load(open("./json/ts_z_omip2.json")) ]
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    print("Drawing "+sys.argv[1])


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
DS = []
for n in range(2):
    d = np.full( (len(var_list),len(model_list[n]),len(time[n]),33), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    nvar = 0
    for var in var_list:

        nmodel = 0
        for model in model_list[n]:

            try:
                multidata = strtobool(metainfo[n][model]['multidata'])
            except:
                #J 単一モデル用エラー処理 (json にエントリーがない場合)
                d[nvar,nmodel] = d_dummy[n]
                continue

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
                d[nvar,nmodel] = d_fsu - d_woa[nvar]

            elif (n == 0) and (model == "MIROC-COCO4.9"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[var][:,:]
                nc.close()

                d_coco = np.full( (len(time[n]),33), np.nan )
                d_coco[0:62*5,:] = d_tmp[0:62*5,:] * factor
                d[nvar,nmodel] = d_coco - d_woa[nvar]

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
                d[nvar,nmodel] = d_cmcc - d_woa[nvar]

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

                d[nvar,nmodel] = d_barca - d_woa[nvar]

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

                        
                d[nvar,nmodel] = d_gfdl - d_woa[nvar]

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
                    
                d[nvar,nmodel] = DS_read[var].values * factor - d_woa[nvar]

            nmodel += 1

        nvar += 1

    #J サイクル間に NaN データ挿入
    d_new = np.concatenate(
        [d, np.tile(varnan,(len(var_list),len(model_list[n]),1,1))],
        axis = 2 )

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

if len(sys.argv) == 1:
    bounds = [ [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3,
                0.4, 0.7, 1.0], 
               [-0.2, -0.15, -0.1, -0.07, -0.04, -0.02, 0, 0.02, 0.04,
                0.07, 0.1, 0.15, 0.2] ]
    bounds_std = [ [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5], 
                   [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5] ]
else:
    bounds = [ [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3,
                0.4, 0.7, 1.0], 
               [-0.2, -0.15, -0.1, -0.07, -0.04, -0.02, 0, 0.02, 0.04,
                0.07, 0.1, 0.15, 0.2] ]
    bounds_std = [ [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0], 
                   [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5] ]

for n in range(2):
    for nvar in range(2):
        nax = n*2 + nvar*4
        print(nax+1,title_list[nax])
        var = var_list[nvar]
        if len(model_list[n]) > 1:
            da = DS[n][var].mean(dim='model').transpose()
        else:
            da = DS[n][var].transpose()

        da.plot(ax=ax[nax],cmap='RdBu_r',
                levels=bounds[nvar],
                extend='both',
                cbar_kwargs={'orientation': 'vertical',
                             'label': clabels[nvar],
                             'spacing': 'uniform',
                             'ticks': bounds[nvar], } )

        ax[nax].set_title(title_list[nax],{'fontsize':10, 'verticalalignment':'top'})
        ax[nax].tick_params(labelsize=9)
        ax[nax].set_xlim(1593,2018)
        ax[nax].set_xticks(np.arange(1663,2018.1,71))
        ax[nax].invert_yaxis()
        ax[nax].set_yticks(np.arange(0,6000,1000))
        ax[nax].set_facecolor('lightgray')
        if ( n == 1 and nvar == 1 ):
            ax[nax].set_xlabel('year',fontsize=10)
        else:
            ax[nax].set_xlabel('',fontsize=10)

for n in range(2):
    for nvar in range(2):
        nax = n*2 + nvar*4 + 1
        print(nax+1,title_list[nax])
        var = var_list[nvar]
        if len(model_list[n]) > 1:
            da = DS[n][var].std(dim='model').transpose()
        else:
            da = DS[n][var].transpose()

        da.plot(ax=ax[nax],cmap='viridis',
                levels=bounds_std[nvar],
                extend='both',
                cbar_kwargs={'orientation': 'vertical',
                             'spacing': 'uniform',
                             'label': clabels[nvar],
                             'ticks': bounds_std[nvar], } )

        ax[nax].set_title(title_list[nax],{'fontsize':10, 'verticalalignment':'top'})
        ax[nax].tick_params(labelsize=9)
        ax[nax].set_xlim(1593,2018)
        ax[nax].set_xticks(np.arange(1663,2018.1,71))
        ax[nax].invert_yaxis()
        ax[nax].set_yticks(np.arange(0,6000,1000))
        ax[nax].set_facecolor('lightgray')
        if ( n == 1 and nvar == 1 ):
            ax[nax].set_xlabel('year',fontsize=10)
        else:
            ax[nax].set_xlabel('',fontsize=10)

plt.subplots_adjust(left=0.08,right=0.98,bottom=0.08,top=0.92,hspace=0.28)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
