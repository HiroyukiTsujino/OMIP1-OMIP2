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

#if (len(sys.argv) < 2) :
#    print ('Usage: ' + sys.argv[0] + ' 1 (OMIP1) or 2 (OMIP2)')
#    sys.exit()


title_list = [ "(a) OMIP1  AMOC maximum at 26N",     "(b) OMIP2  AMOC maximum at 26N",
               "(c) OMIP1  Drake Passage transport", "(d) OMIP2  Drake Passage transport",
               "(e) OMIP1  Indonesian through flow", "(f) OMIP2  Indonesian through flow",
               "(g) OMIP1  GMOC minimum at 30S",     "(h) OMIP2  GMOC minimum at 30S" ]


outfile = './fig/FigS3_52yr'
suptitle = 'Ocean Circulation Index'
metainfo = [ json.load(open("./json/circ_omip1_1958-2009.json")),
             json.load(open("./json/circ_omip2_1958-2009.json")) ]


model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

lineinfo = json.load(open('../json/inst_color_style-52yr.json'))

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


cycle_len = [ 62, 61 ]
start_52yr = [ 10, 0 ]

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

            print(model,var)
            
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

            if (model == "MIROC-COCO4.9_52yr" or model == "MRI.COM_52yr"):

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

                d_52yr = np.full( (len(time[n])), np.nan )

                for cyc in range(6):
                    isto = start_52yr[n] + cyc * cycle_len[n]
                    iedo = isto + 52
                    istf = 0 + cyc * 52
                    iedf = istf + 52
                    print (isto, iedo)
                    print (istf, iedf)
                    d_52yr[isto:iedo] = d_tmp[istf:iedf] * factor

                d[nvar,nmodel] = d_52yr

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

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                elif var == 'gmoc':
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

    lincol += [coltmp]
    linsty += [stytmp]
    nummodel += [nmodel]

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

lc = [ "red", "blue" ]

ylim = [ [8, 21], [100, 190], [-22, -4], [-20, 2] ]
ytick = [ np.linspace(8, 21, 14),
          np.linspace(100, 190, 10),
          np.linspace(-22,-4,10),
          np.linspace(-20,2,12), ]

nf = 0
nv = 0
for var in var_list:
    for n in range(2):
        nmodel = 0
        for model in model_list[n]:
            print(model)
            linecol=lincol[n][nmodel]
            linesty=linsty[n][nmodel]
            if (n == 1):
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=1,linestyle=linesty)
            else:
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],color=linecol,linewidth=1,linestyle=linesty)
                
            nmodel += 1

        if (n == 1):
            ax[nf].legend(bbox_to_anchor=(1.05,1.0),loc='upper left',borderaxespad=0,fontsize=8)

        ax[nf].set_title(title_list[nf],{'fontsize':10, 'verticalalignment':'top'})
        ax[nf].tick_params(labelsize=9)
        ax[nf].set_xlim(1593,2018)
        ax[nf].set_xticks(np.arange(1663,2018.1,71))
        if ( nv == 3 ):
            ax[nf].set_xlabel('year',fontsize=10)
        else:
            ax[nf].set_xlabel('',fontsize=10)

        ax[nf].grid()
        ax[nf].set_ylim(ylim[nv])
        ax[nf].set_yticks(ytick[nv])
        if ( n == 0 ):
            ax[nf].set_ylabel(r'$10^{9} \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=10)
        else:
            ax[nf].set_ylabel('')

        nf += 1

    nv +=1

plt.subplots_adjust(left=0.07,right=0.80,bottom=0.07,top=0.92,hspace=0.30)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
