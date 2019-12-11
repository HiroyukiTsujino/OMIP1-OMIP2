# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ylim = [ [-1.2, 2.4], [0, 1.5], [-1.8, 1.5] ]
yint = [ 0.3, 0.3, 0.3 ]

metainfo = [ json.load(open("./json/hfbasin_omip1.json")),
             json.load(open("./json/hfbasin_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

title = [ 'Global', 'Atlantic-Arctic', 'Indo-Pacific' ]

if len(sys.argv) == 1:
    suptitle = 'Multi Model Mean' + ' (northward heat transport ave. from 1980 to 2009)'
    outfile = './fig/heat_transport.png'
else:
    suptitle = sys.argv[1] + ' (northward heat transport ave. from 1980 to 2009)'
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/heat_transport_' + sys.argv[1] + '.png'


print ( 'Model list for OMIP1:', model_list[0] )
print ( 'Model list for OMIP2:', model_list[1] )


#J NCAR-POP 補間用情報
y = np.linspace(-89.5,89.5,num=180)


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]


#J データ読込・平均

data = []
for omip in range(2):
    var = np.empty( (len(model_list[omip]),3,180) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path = metainfo[omip][model]['path']
        fname= metainfo[omip][model]['fname']
        infile =  path + '/' + fname
        factor = float(metainfo[omip][model]['factor'])
        DS = xr.open_dataset( infile, decode_times=False )
        if (model == "GFDL-MOM"):
            DS = DS.rename({"year":"time"})

        DS['time'] = time[omip]
        
        tmp = DS.hfbasin.sel(time=slice(1980,2009)).mean(dim='time',skipna=False)*factor

        if model == 'NCAR-POP':
            tmp = tmp.interp(lat=y).isel(basin=[2,0,1])
        if model == 'NorESM-O-CICE':
            tmp = tmp.rename({"region":"basin"})
            tmp = tmp.interp(lat=y).isel(basin=[3,1,2])
        if model == 'AWI-FESOM':
            tmp = tmp.transpose()
        if model == 'CAS-LICOM3':
            tmp = tmp.sel(lat=slice(None, None, -1))
        if model == 'GFDL-MOM':
            tmp = tmp.interp(yq=y).isel(basin=[2,0,1])

        var[nmodel] = np.where( tmp.values == 0, np.nan, tmp.values )
        nmodel += 1

    data += [var]

DS = xr.Dataset({'omip1': (['model','basin','lat'], data[0]), 
                 'omip2': (['model','basin','lat'], data[1]), },
                coords = {'lat': y} )


#J 描画
fig = plt.figure(figsize=(16,12))
fig.suptitle( suptitle, fontsize=20 )

ax = [
    plt.subplot(2,2,1),
    plt.subplot(2,2,2),
    plt.subplot(2,2,3),
]

for n in range(3):
    DS.omip1.mean(dim='model',skipna=False).isel(basin=n).plot(ax=ax[n], label='OMIP1')
    DS.omip2.mean(dim='model',skipna=False).isel(basin=n).plot(ax=ax[n], label='OMIP2')
    ax[n].set_title(title[n])
    ax[n].set_xlabel("Latitude")
    ax[n].set_xlim(-90,90)
    ax[n].set_xticks(np.linspace(-90,90,7))
    ax[n].set_ylabel("Heat transport [PW]")
    ax[n].set_ylim(ylim[n][0],ylim[n][1])
    ax[n].set_yticks(np.arange(ylim[n][0],ylim[n][1]+yint[n],yint[n]))
    ax[n].legend()
    ax[n].grid()


ax[0].text( 47.5, 0.68,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text( 36.0, 1.11,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text( 24.0, 1.62,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(  9.5, 1.50,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(-10.5, 0.55,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(-19.5,-0.43,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[0].text(-31.0,-0.51,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')

ax[0].vlines( 47.5, 0.68 - 0.15, 0.68 + 0.15)
ax[0].vlines( 36.0, 1.11 - 0.37, 1.11 + 0.37)
ax[0].vlines( 24.0, 1.62 - 0.40, 1.62 + 0.40)
ax[0].vlines(  9.5, 1.50 - 1.54, 1.50 + 1.54)
ax[0].vlines(-10.5, 0.50 - 1.45, 0.50 + 1.45)
ax[0].vlines(-19.5,-0.43 - 0.61,-0.43 + 0.61)
ax[0].vlines(-31.0,-0.51 - 0.39,-0.51 + 0.39)

ax[1].text( 46.0, 0.58,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 37.0, 0.88,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 25.0, 1.20,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 10.5, 1.07,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text(-11.5, 0.56,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text(-32.0, 0.34,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[1].text( 26.5, 1.24,"x",fontsize=17,horizontalalignment='center',verticalalignment='center')

ax[1].vlines( 46.0, 0.58 - 0.24, 0.58 + 0.24)
ax[1].vlines( 37.0, 0.88 - 0.22, 0.88 + 0.22)
ax[1].vlines( 25.0, 1.20 - 0.27, 1.20 + 0.27)
ax[1].vlines( 10.5, 1.07 - 0.33, 1.07 + 0.33)
ax[1].vlines(-11.5, 0.56 - 0.26, 0.55 + 0.26)
ax[1].vlines(-32.0, 0.34 - 0.18, 0.34 + 0.18)
ax[1].vlines( 26.5, 1.24 - 0.33, 1.24 + 0.33)

ax[2].text( 47.5, 0.04,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text( 23.0, 0.64,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text( 10.5, 0.51,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text(-18.5,-1.15,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
ax[2].text(-29.5,-0.91,"o",fontsize=17,horizontalalignment='center',verticalalignment='center')
    
ax[2].vlines( 47.5, 0.04 - 0.16, 0.04 + 0.16)
ax[2].vlines( 23.0, 0.64 - 0.29, 0.64 + 0.29)
ax[2].vlines( 10.5, 0.51 - 1.22, 0.51 + 1.22)
ax[2].vlines(-18.5,-1.15 - 0.61,-1.15 + 0.61)
ax[2].vlines(-29.5,-0.91 - 0.36,-0.91 + 0.36)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
