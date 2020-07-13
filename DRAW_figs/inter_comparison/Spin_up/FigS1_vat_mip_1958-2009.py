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


metainfo = [ json.load(open("./json/vat_omip1_1958-2009.json")),
             json.load(open("./json/vat_omip2_1958-2009.json")) ]
suptitle = 'Vertically averaged temperature'
outfile = './fig/FigS1_vat_52yr'

lineinfo = json.load(open('../json/inst_color_style-52yr.json'))

title_list = [ "(a) OMIP1  0 - 700 m", "(b) OMIP2  0 - 700 m",
               "(c) OMIP1  0 - 2000 m", "(d) OMIP2  0 - 2000 m",
               "(e) OMIP1  2000 m - bottom", "(f) OMIP2  2000 m - bottom",
               "(f) OMIP1  0 m - bottom" , "(h) OMIP2  0 m - bottom" ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

var_list = [ "thetaoga_700", "thetaoga_2000", "thetaoga_2000_bottom", "thetaoga_all" ]
volume_list = np.array([ 2.338e17, 6.216e17, 7.593e17, 1.381e18 ])
degC_to_ZJ = volume_list * 3.99e3 * 1.036e3 * 1.0e-21


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.array([np.linspace(1948,2009,62)]*6),
         np.array([np.linspace(1958,2018,61)]*6) ]

cycle_len = [ 62, 61 ]
start_52yr = [ 10, 0 ]

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

lincol = []
linsty = []
nummodel = []

for n in range(2):

    d = np.full( (len(var_list),len(model_list[n]),len(time[n])), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    coltmp = []
    stytmp = []

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

            if (nvar == 0):
                coltmp +=[lineinfo[model]["color"]]
                stytmp +=[lineinfo[model]["style"]]

            path = metainfo[n][model][var]['path']
            fname = metainfo[n][model][var]['fname']
            vname = metainfo[n][model][var]['varname']
            factor = float(metainfo[n][model][var]['factor'])
            infile = path + '/' + fname

            print(infile,vname)

            if (model == "MRI.COM_52yr" or model == "MIROC-COCO4.9_52yr"):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_52yr = np.full( len(time[n]), np.nan )
                for cyc in range(6):
                    isto = start_52yr[n] + cyc * cycle_len[n]
                    iedo = isto + 52
                    istf = 0 + cyc * 52
                    iedf = istf + 52
                    print (isto, iedo)
                    print (istf, iedf)
                    d_52yr[isto:iedo] = d_tmp[istf:iedf] * factor

                d[nvar,nmodel] = d_52yr

            elif (n == 0) and (model == "MIROC-COCO4.9" ):
                nc = netCDF4.Dataset(infile,'r')
                d_tmp = nc.variables[vname][:]
                nc.close()

                d_coco = np.full( len(time[n]), np.nan )
                d_coco[0:62*5] = d_tmp[0:62*5] * factor
                d[nvar,nmodel] = d_coco

            else:

                DS_read = xr.open_dataset(infile,decode_times=False)
                d[nvar,nmodel] = DS_read[vname].values * factor

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

ylim = [ [10.1, 11.3], [5.5, 6.6], [1.1, 2.2], [3.3, 4.3] ]
ytick = [ np.linspace(10.1, 11.3, 13),
          np.linspace(5.5, 6.6, 12),
          np.linspace(1.1, 2.2, 12),
          np.linspace(3.3, 4.3, 11), ]

nf = 0
nv = 0
for var in var_list:
    for n in range(2):
        nmodel = 0
        for model in model_list[n]:
            linecol=lincol[n][nmodel]
            linesty=linsty[n][nmodel]
            if (n == 1):
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=1,linestyle=linesty)
            else:
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],color=linecol,linewidth=1,linestyle=linesty)

            nmodel += 1

        if (n == 1):
            print("Writing legend for ",nf)
            ax[nf].legend(bbox_to_anchor=(1.05,1.0),loc='upper left',borderaxespad=0,fontsize=8)

        ax[nf].set_title(title_list[nf],{'fontsize':10, 'verticalalignment':'top'})
        ax[nf].tick_params(labelsize=9)
        ax[nf].set_xlim(1593,2018)
        ax[nf].set_xticks(np.arange(1663,2018.1,71))
        if ( nv == 3 ):
            ax[nf].set_xlabel('year',fontsize=10)
        else:
            ax[nf].set_xlabel('')

        ax[nf].grid()
        ax[nf].set_ylim(ylim[nv])
        ax[nf].set_yticks(ytick[nv])
        if ( n == 0 ):
            ax[nf].set_ylabel(r'$^{\circ}$C',fontsize=10)
        else:
            ax[nf].set_ylabel('')

        nf += 1

    nv += 1

plt.subplots_adjust(left=0.07,right=0.80,bottom=0.07,top=0.92,hspace=0.30)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()
