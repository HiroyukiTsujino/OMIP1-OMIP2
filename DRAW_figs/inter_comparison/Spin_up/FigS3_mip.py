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

#####

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' 1 (OMIP1) or 2 (OMIP2) or 3 (MMM)')
    sys.exit()


title_list = [ "(a) AMOC maximum at 26.5$^{\circ}$N",
               "(b) Drake Passage transport",
               "(c) Indonesian through flow",
               "(d) GMOC minimum at 30$^{\circ}$S", ]


if (int(sys.argv[1]) == 3):
    outfile = './fig/FigS3_MMM.png'
    suptitle = 'Multi Model Mean'
    metainfo = [ json.load(open("./json/circ_omip1_full.json")),
                 json.load(open("./json/circ_omip2_full.json")) ]
else:
    #model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/FigS3_omip' + sys.argv[1] + '.png'
    suptitle = 'OMIP-' + sys.argv[1]
    metainfo = [ json.load(open("./json/circ_omip1.json")),
                 json.load(open("./json/circ_omip2.json")) ]



model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

lineinfo = json.load(open('../json/inst_color_style.json'))

var_list = [ "amoc", "drake", "itf", "gmoc" ]

lat_woa=np.array(np.linspace(-89.5,89.5,num=180))
print(lat_woa)
                 

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

lincol = []
linsty = []
nummodel = []

for n in range(2):
    d = np.full( (len(var_list),len(model_list[n]),len(time[n])), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    coltmp = []
    stytmp = []

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

            if (nvar == 0):
                coltmp +=[lineinfo[model]["color"]]
                stytmp +=[lineinfo[model]["style"]]

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

                    DS_read = DS_read.assign_coords(lat=DS_read["lat"].values)

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                    print(tmp)
                elif var == 'gmoc':
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

            elif (n == 1) and (model == "GFDL-MOM"):

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

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_fsu = np.full( (len(time[n])), np.nan )

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

                    print(DS_read[name])

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)

                if var == 'amoc':
                    print(DS_read[name])
                    tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_coco = np.full( (len(time[n])), np.nan )
                d_coco[0:62*5] = d_tmp[0:62*5] * factor
                d[nvar,nmodel] = d_coco

            elif (model == "EC-Earth3-NEMO"):

                if (n == 0):
                    num_yr = 62
                else:
                    num_yr = 61

                d_barca = np.full( num_yr*6, np.nan )

                for i in range(6):

                    infile=path + str(i+1) + '/' + fname
                    print(var,infile)
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

                        DS_read = DS_read.assign_coords(lev=DS_read["lev"].values)
                        #print(DS_read[name])

                    if exdim_name != "None":
                        DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)

                    if var == 'amoc':
                        tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                    elif var == 'gmoc':
                        tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                    else:
                        tmp = DS_read[name]

                    d_tmp = tmp.values
                    #print(d_tmp)

                    for nt in range(num_yr):
                        d_barca[num_yr*i+nt] = d_tmp[nt] * factor

                #print('var',d_barca)
                d[nvar,nmodel] = d_barca

            elif (model == "CMCC-NEMO"):

                num_yr = len(time[n])

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

                    DS_read = DS_read.drop("basin")

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
 
                if var == 'amoc':
                    print(DS_read[name].lat[188])
#                    tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                    tmp = DS_read[name].isel(lat=188).max(dim="lev")
#                    tmp = DS_read[name].sel(lat=26.5,method='nearest').max(dim="lev")
                elif var == 'gmoc':
                    print(DS_read[name].lat[99])
#                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].isel(lat=99).min(dim="lev")
#                    tmp = DS_read.sel(lev=slice(2000,6500))[name].sel(lat=-30,method='nearest').min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_coco = np.full( (len(time[n])), np.nan )
                d_coco[0:num_yr] = d_tmp[0:num_yr] * factor
                d[nvar,nmodel] = d_coco
                
            elif (n == 0) and (model == "GFDL-MOM"):

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

                    print(DS_read[name])

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)

                if var == 'amoc':
                    print(DS_read[name])
                    tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_coco = np.full( (len(time[n])), np.nan )
                for cyc in range(0,5):
                    isto = 0 + cyc * 62
                    iedo = isto + 59
                    istf = 0 + cyc * 60
                    iedf = istf + 59
                    print (cyc, isto, iedo)
                    print (cyc, istf, iedf)
                    d_coco[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                for cyc in range(5,6):
                    isto = 0 + cyc * 62
                    iedo = isto + 61
                    istf = 0 + cyc * 62 - 10
                    iedf = istf + 61
                    print (cyc, isto, iedo)
                    print (cyc, istf, iedf)
                    d_coco[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                print(var,factor)
                print(d_coco)
                d[nvar,nmodel] = d_coco

            else:

                if multidata:
                    if ( model == 'Kiel-NEMO' and (var == 'drake' or var == 'itf') ):
                        DS_read = xr.open_mfdataset(infile,decode_times=False,concat_dim='time_counter')
                        print(DS_read)
                    else:
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
                    if model == 'CESM-POP':
                        DS_read["lev"] = DS_read.lev.values * 1e-2
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    if (model == 'Kiel-NEMO'):
                        tmp = DS_read[name].isel(X=0)
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

    lincol += [coltmp]
    linsty += [stytmp]
    nummodel += [nmodel]

#J 描画
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )
ax = [ plt.subplot(4,1,1), 
       plt.subplot(4,1,2), 
       plt.subplot(4,1,3),
       plt.subplot(4,1,4) ]

lc = [ "red", "blue" ]

n = int(sys.argv[1]) - 1

if ( n < 2 ):
    nf = 0
    for var in var_list:
        nmodel = 0
        for model in model_list[n]:
            print(model)
            linecol=lincol[n][nmodel]
            linesty=linsty[n][nmodel]
            DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=1,linestyle=linesty)
            nmodel += 1

        if (nf == 0): 
            leg = ax[nf].legend(bbox_to_anchor=(1.02,1.0),loc='upper left')
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)
        nf += 1

else:
    for nvar in range(4):
        var = var_list[nvar]
        DS[0][var].mean(dim='model').plot(ax=ax[nvar],color='darkred')
        ax[nvar].fill_between(x=DS[0]['time'],
                              y1=DS[0][var].min(dim='model'),
                              y2=DS[0][var].max(dim='model'),
                              alpha=0.5, facecolor='lightcoral')
        DS[1][var].mean(dim='model').plot(ax=ax[nvar],color='darkblue')
        ax[nvar].fill_between(x=DS[1]['time'],
                              y1=DS[1][var].min(dim='model'),
                              y2=DS[1][var].max(dim='model'),
                              alpha=0.5, facecolor='lightblue')

if len(sys.argv) == 1:
    ylim = [ [11, 20], [130, 170], [-20, -8], [-22, -4] ]
    ytick = [ np.linspace(11, 20, 10),
              np.linspace(130, 170, 9),
              np.linspace(-20,-8,7),
              np.linspace(-22,-4,10), ]
else:
    ylim = [ [7, 24], [100, 210], [-25, -4], [-30, 5] ]
    ytick = [ np.linspace(6, 24, 10),
              np.linspace(100, 210, 12),
              np.linspace(-25,-4,8),
              np.linspace(-30,5,8), ]

for nvar in range(4):
    ax[nvar].set_title(title_list[nvar],{'fontsize':10,'verticalalignment':'top'})
    ax[nvar].set_xlim(1592,2018)
    ax[nvar].set_xticks(np.arange(1592,2018.1,71))
    if ( nvar == 3 ):
        ax[nvar].set_xlabel('year',fontsize=10)
    else:
        ax[nvar].set_xlabel('',fontsize=10)

    ax[nvar].grid()
    ax[nvar].set_ylim(ylim[nvar])
    ax[nvar].set_yticks(ytick[nvar])
    ax[nvar].set_ylabel(r'$10^{9} \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=10)

plt.subplots_adjust(left=0.10,right=0.75,top=0.93,bottom=0.05,hspace=0.2)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
