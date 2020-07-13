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

#

#if (len(sys.argv) < 2) :
#    print ('Usage: ' + sys.argv[0] + ' 1 (OMIP1) or 2 (OMIP2) or 3 (MMM)')
#    sys.exit()

metainfo = json.load(open("./json/vat_omip2_aircore.json"))
suptitle = 'Vertically averaged temperature (formulae of moist air)'
outfile = './fig/FigS1_vat_aircore'

lineinfo = json.load(open('../json/inst_color_style-aircore.json'))

title_list = [ "(a) 0 - 700 m", "(b) 0 - 2000 m", "(c) 2000 m - bottom", "(d) 0 m - bottom" ]

model_list = metainfo.keys()

var_list = [ "thetaoga_700", "thetaoga_2000", "thetaoga_2000_bottom", "thetaoga_all" ]
volume_list = np.array([ 2.338e17, 6.216e17, 7.593e17, 1.381e18 ])
degC_to_ZJ = volume_list * 3.99e3 * 1.036e3 * 1.0e-21


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = np.array([np.linspace(1958,2018,61)]*6)

for i in range(6):
    time[i] = time[i] - (5-i)*61

time = time.reshape(6*61)

d = np.full( (len(var_list),len(model_list),len(time)), np.nan )

d_dummy = np.full( len(time), np.nan )

coltmp = []
stytmp = []

nvar = 0
for var in var_list:

    nmodel = 0
    for model in model_list:

        try:
            multidata = strtobool(metainfo[model]['multidata'])
        except:
            #J 単一モデル用エラー処理 (json にエントリーがない場合)
            d[nvar,nmodel] = d_dummy[n]
            continue

        if (nvar == 0):
            coltmp +=[lineinfo[model]["color"]]
            stytmp +=[lineinfo[model]["style"]]

        path = metainfo[model][var]['path']
        fname = metainfo[model][var]['fname']
        vname = metainfo[model][var]['varname']
        factor = float(metainfo[model][var]['factor'])
        infile = path + '/' + fname

        print(infile,vname)

        if multidata:
            DS_read = xr.open_mfdataset(infile,decode_times=False)
        else:
            DS_read = xr.open_dataset(infile,decode_times=False)

        print(nvar,nmodel)
        d[nvar,nmodel] = DS_read[vname].values * factor

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

if len(model_list) > 1:
    nf = 0
    for var in var_list:
        nmodel = 0
        for model in model_list:
            linecol=coltmp[nmodel]
            linesty=stytmp[nmodel]
            DS[var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=1,linestyle=linesty)
            nmodel += 1

        ax[nf].legend(bbox_to_anchor=(1.01,1.0),loc='upper left',borderaxespad=0,fontsize=8)
        nf += 1

ylim = [ [10.1, 11.3], [5.5, 6.6], [1.1, 2.2], [3.3, 4.3] ]
ytick = [ np.linspace(10.1, 11.3, 13),
          np.linspace(5.5, 6.6, 12),
          np.linspace(1.1, 2.2, 12),
          np.linspace(3.3, 4.3, 11), ]

for m in range(4):
    ax[m].set_title(title_list[m],{'fontsize':10, 'verticalalignment':'top'})
    ax[m].tick_params(labelsize=9)
    ax[m].set_xlim(1653,2018)
    ax[m].set_xticks(np.arange(1653,2018.1,61))
    if ( m == 3 ):
        ax[m].set_xlabel('year',fontsize=10)
    else:
        ax[m].set_xlabel('',fontsize=10)

    ax[m].grid()
    ax[m].set_ylim(ylim[m])
    ax[m].set_yticks(ytick[m])
    ax[m].set_ylabel(r'$^{\circ}$C',fontsize=10)

plt.subplots_adjust(left=0.08,right=0.80,bottom=0.07,top=0.92,hspace=0.30)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()
