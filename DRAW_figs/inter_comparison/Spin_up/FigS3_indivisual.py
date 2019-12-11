# -*- coding: utf-8 -*-
import sys
import json
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from distutils.util import strtobool
import netCDF4
from netCDF4 import Dataset, num2date


title_list = [ "(a) AMOC maximum at 26N",
               "(b) Drake Passage transport",
               "(c) Indonesian through flow",
               "(d) GMOC minimum at 30S", ]

metainfo = [ json.load(open("./omip1_nofull.json")),
             json.load(open("./omip2_nofull.json")) ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

var_list = [ "amoc", "drake", "itf", "gmoc" ]


if len(sys.argv) == 1:
    outfile = './fig/FigS3.png'
    suptitle = 'Multi Model Mean'
else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/FigS3_' + sys.argv[1] + '.png'
    suptitle = sys.argv[1]


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


#J 単一モデル用 dummy DS (json にエントリーがない場合に使用)
d_dummy = [ np.full( len(time[0]), np.nan ),
            np.full( len(time[1]), np.nan ) ]


#J データ読込 (n=0: OMIP1, n=1: OMIP2)
DS = []
for n in range(2):
    d = np.full( (len(var_list),len(model_list[n]),len(time[n])), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    for nvar in range(4):
        var = var_list[nvar]

        nmodel = 0
        for model in model_list[n]:

            print(model)

            try:
                multidata = strtobool(metainfo[n][model]['multidata'])
                path = metainfo[n][model][var]['path']
            except:
                #J json にエントリーがない場合のエラー処理
                d[nvar,nmodel] = d_dummy[n]
                continue

            fname= metainfo[n][model][var]['fname']
            name = metainfo[n][model][var]['name']
            exdim_name = metainfo[n][model][var]['exdim_name']
            exdim = int(metainfo[n][model][var]['exdim'])
            factor = float(metainfo[n][model][var]['factor'])
            infile = path + '/' + fname


            if (n == 1) and (model == "FSU-HYCOM"):

                DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    if model == 'FSU-HYCOM':
                        DS_read = DS_read.assign_coords(lat=DS_read["lat"].values)

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    if model == 'NCAR-POP':
                        DS_read["lev"] = DS_read.lev.values * 1e-2
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_fsu = np.full( (len(time[n])), np.nan )

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
                d[nvar,nmodel] = d_fsu

            elif (n == 0) and (model == "MIROC-COCO4.9"):
                DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    if model == 'FSU-HYCOM':
                        DS_read = DS_read.assign_coords(lat=DS_read["lat"].values)

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    if model == 'NCAR-POP':
                        DS_read["lev"] = DS_read.lev.values * 1e-2
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_coco = np.full( (len(time[n])), np.nan )
                d_coco[0:62*5] = d_tmp[0:62*5] * factor
                d[nvar,nmodel] = d_coco

            else:

                if multidata:
                    DS_read = xr.open_mfdataset(infile,decode_times=False,concat_dim='time')
                else:
                    DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    if model == 'FSU-HYCOM':
                        DS_read = DS_read.assign_coords(lat=DS_read["lat"].values)

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    if model == 'NCAR-POP':
                        DS_read["lev"] = DS_read.lev.values * 1e-2
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d[nvar,nmodel] = tmp.values * factor

            nmodel += 1

        nvar += 1

    #J サイクル間に NaN データ挿入
    d_new = np.concatenate(
        [d, np.tile(varnan,(len(var_list),len(model_list[n]),1))],
        axis = 2 )

    time_new = np.concatenate( [time[n], timenan[n]] )

    #J xarray Dataset 再作成
    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new } ).sortby('time')

    DS += [DS_tmp]



#J 描画
fig = plt.figure(figsize=(12,16))
fig.suptitle( suptitle, fontsize=20 )
ax = [ plt.subplot(4,1,1), 
       plt.subplot(4,1,2), 
       plt.subplot(4,1,3),
       plt.subplot(4,1,4) ]

lc = [ "red", "blue" ]

for nvar in range(4):
    var = var_list[nvar]
    for omip in range(2):
        da = DS[omip][var].mean(dim='model')
        da.plot(ax=ax[nvar],color=lc[omip])

if len(sys.argv) == 1:
    ylim = [ [11, 20], [130, 170], [-20, -8], [-22, -4] ]
    ytick = [ np.linspace(11, 20, 10),
              np.linspace(130, 170, 9),
              np.linspace(-20,-8,7),
              np.linspace(-22,-4,10), ]
else:
    ylim = [ [7, 24], [100, 210], [-25, -4], [-35, 0] ]
    ytick = [ np.linspace(6, 24, 10),
              np.linspace(100, 210, 12),
              np.linspace(-25,-4,8),
              np.linspace(-35,0,8), ]

for nvar in range(4):
    ax[nvar].set_title(title_list[nvar])
    ax[nvar].set_xlim(1593,2018)
    ax[nvar].set_xticks(np.arange(1663,2018.1,71))
    ax[nvar].grid()
    ax[nvar].set_ylim(ylim[nvar])
    ax[nvar].set_yticks(ytick[nvar])

#J titleとx軸ラベルが重なるのを防ぐ
#J tight_layout は suptitle を考慮しないので、上側を少しあける
plt.tight_layout(rect=[0,0,1,0.95])

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
