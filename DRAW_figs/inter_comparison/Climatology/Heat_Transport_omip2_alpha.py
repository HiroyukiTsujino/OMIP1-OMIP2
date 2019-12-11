# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4
import datetime

ystr = 1980
yend = 2009

ylim = [ [-1.4, 2.6], [0, 1.5], [-1.8, 1.5] ]
yint = [ 0.2, 0.2, 0.2 ]

suptitle = ' Northward heat transport (ave. from 1980 to 2009)'

metainfo = json.load(open("./json/hfbasin_omip2_alpha.json"))
model_list = metainfo.keys()

title = [ '(a) Global', '(b) Atlantic-Arctic', '(c) Indo-Pacific' ]

outfile = './fig/heat_transport_omip2_alpha'
print ( 'Model list: ', model_list )

lineinfo = json.load(open('../json/inst_color_style-alpha.json'))

#J NCAR-POP 補間用情報
y = np.linspace(-89.5,89.5,num=180)


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = np.linspace(1958,2018,61)

timem = np.empty((2019-1958)*12,dtype='object')
for yr in range(1958,2019):
    for mon in range(1,13):
        timem[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)


#J データ読込・平均

data = []

coltmp = []
stytmp = []
nmodel = 0

var = np.empty( (len(model_list),3,180) )
for model in model_list:
    path = metainfo[model]['path']
    fname= metainfo[model]['fname']
    infile =  path + '/' + fname
    factor = float(metainfo[model]['factor'])

    coltmp +=[lineinfo[model]["color"]]
    stytmp +=[lineinfo[model]["style"]]

    DS = xr.open_dataset( infile, decode_times=False )
    DS['time'] = time
        
    tmp = DS.hfbasin.sel(time=slice(ystr,yend)).mean(dim='time',skipna=False)*factor

    if ((model == 'CAS-LICOM3_alpha1_0') or (model == 'CAS-LICOM3_alpha0_7')):
            tmp = tmp.sel(lat=slice(None, None, -1))

    var[nmodel] = np.where( tmp.values == 0, np.nan, tmp.values )
    nmodel += 1

var_dict = {'omip2':(['model','basin','lat'], var) }
DS = xr.Dataset(var_dict, coords = {'lat': y } )


#J 描画
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

ax = [
    plt.subplot(3,1,1),
    plt.subplot(3,1,2),
    plt.subplot(3,1,3),
]

for n in range(3):

    nmodel = 0
    for model in model_list:
        linecol=coltmp[nmodel]
        linesty=stytmp[nmodel]
        if (n == 1):
            DS.omip2.sel(model=nmodel).isel(basin=n).where(DS.lat>-32.0).plot(ax=ax[n],label=model,color=linecol,linewidth=1,linestyle=linesty)
        elif (n == 2):
            DS.omip2.sel(model=nmodel).isel(basin=n).where(DS.lat>-31.0).plot(ax=ax[n],label=model,color=linecol,linewidth=1,linestyle=linesty)
        else:
            DS.omip2.sel(model=nmodel).isel(basin=n).plot(ax=ax[n],label=model,color=linecol,linewidth=1,linestyle=linesty)

        nmodel += 1

    ax[n].set_title(title[n])
    if (n==3):
        ax[n].set_xlabel("Latitude")
    else:
        ax[n].set_xlabel("")
        
    ax[n].set_xlim(-90,90)
    ax[n].set_xticks(np.linspace(-90,90,7))
    ax[n].set_ylabel("Heat transport [PW]")
    ax[n].set_ylim(ylim[n][0],ylim[n][1])
    ax[n].set_yticks(np.arange(ylim[n][0],ylim[n][1]+yint[n],yint[n]))
    ax[n].legend(bbox_to_anchor=(0.02,1.0),loc='upper left',borderaxespad=0,fontsize=8)
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

plt.subplots_adjust(left=0.12,right=0.98,top=0.92,bottom=0.08)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
