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
#    print ('Usage: ' + sys.argv[0] + ' 1 (OMIP1) or 2 (OMIP2) or 3 (MMM)')
#    sys.exit()


title_list = [ "(a) AMOC maximum at 26N", "(b) Drake Passage transport",
               "(c) Indonesian through flow", "(d) GMOC minimum at 30S" ]

outfile = './fig/FigS3_omip2_alpha'
suptitle = 'OMIP2 sensitivity to alpha'
metainfo = json.load(open("./json/circ_omip2_alpha.json"))

model_list = metainfo.keys()

lineinfo = json.load(open('../json/inst_color_style-alpha.json'))

var_list = [ "amoc", "drake", "itf", "gmoc" ]

lat_woa=np.array(np.linspace(-89.5,89.5,num=180))
print(lat_woa)
                 

#J 時刻情報 (各モデルの時刻情報を上書きする)

time = np.array([np.linspace(1958,2018,61)]*6)
for i in range(6):
    time[i] = time[i] - (5-i)*61

time = time.reshape(6*61)


#J 間をあけるための Dataset 作成
#varnan = np.full(6,np.nan)


#J 単一モデル用 dummy DS (json にエントリーがない場合に使用)
d_dummy = np.full( len(time), np.nan )


DS = []

d = np.full( (len(var_list),len(model_list),len(time)), np.nan )

coltmp = []
stytmp = []

for nvar in range(4):
    var = var_list[nvar]

    nmodel = 0
    for model in model_list:

        print(model)
            
        try:
            multidata = strtobool(metainfo[model]['multidata'])
            path = metainfo[model][var]['path']
        except:
            #J json にエントリーがない場合のエラー処理
            d[nvar,nmodel] = d_dummy
            continue

        if (nvar == 0):
            coltmp +=[lineinfo[model]["color"]]
            stytmp +=[lineinfo[model]["style"]]

        fname= metainfo[model][var]['fname']
        name = metainfo[model][var]['name']
        exdim_name = metainfo[model][var]['exdim_name']
        exdim = int(metainfo[model][var]['exdim'])
        factor = float(metainfo[model][var]['factor'])
        infile = path + '/' + fname

        if multidata:
            DS_read = xr.open_mfdataset(infile,decode_times=False,concat_dim='time')
        else:
            DS_read = xr.open_dataset(infile,decode_times=False)

        if var == 'amoc' or var == 'gmoc':
            zdim = metainfo[model][var]["zdim"]
            zvar = metainfo[model][var]["zvar"]
            if zdim != 'lev':
                DS_read = DS_read.rename({zdim:"lev"})
            if zdim != zvar and zvar != 'lev':
                DS_read = DS_read.rename({zvar:"lev"})
            yvar = metainfo[model]["y"]
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

    #J xarray Dataset 再作成
    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d[nvar])
        nvar += 1

    DS = xr.Dataset( var_dict, coords = { 'time': time } ).sortby('time')

#J 描画
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )
ax = [ plt.subplot(4,1,1), 
       plt.subplot(4,1,2), 
       plt.subplot(4,1,3),
       plt.subplot(4,1,4) ]

lc = [ "red", "blue" ]

nf = 0
for var in var_list:
    nmodel = 0
    for model in model_list:
        print(model)
        linecol=coltmp[nmodel]
        linesty=stytmp[nmodel]
        DS[var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=1,linestyle=linesty)
        nmodel += 1

    ax[nf].legend(bbox_to_anchor=(1.02,1.0),loc='upper left', fontsize=8)
    nf += 1

ylim = [ [8, 21], [110, 190], [-22, -4], [-20, 2] ]
ytick = [ np.linspace(8, 21, 14),
          np.linspace(110, 190, 9),
          np.linspace(-22,-4,10),
          np.linspace(-20,2,12), ]

for nvar in range(4):
    ax[nvar].set_title(title_list[nvar],{'fontsize':10, 'verticalalignment':'top'})
    ax[nvar].tick_params(labelsize=9)
    ax[nvar].set_xlim(1653,2018)
    ax[nvar].set_xticks(np.arange(1653,2018.1,61))
    if ( nvar == 3 ):
        ax[nvar].set_xlabel('year',fontsize=10)
    else:
        ax[nvar].set_xlabel('',fontsize=10)

    ax[nvar].grid()
    ax[nvar].set_ylim(ylim[nvar])
    ax[nvar].set_yticks(ytick[nvar])
    ax[nvar].set_ylabel(r'$10^{9} \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=10)

plt.subplots_adjust(left=0.1,right=0.77,bottom=0.07,top=0.92,hspace=0.30)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
