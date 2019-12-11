# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math


title = [ '(a) OMIP1 - CMEMS', '(b) OMIP2 - CMEMS', '(c) OMIP2 - OMIP1', '(d) CMEMS' ]

metainfo = [ json.load(open("./json/zos_omip1.json")),
             json.load(open("./json/zos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

if len(sys.argv) == 1:
    suptitle = 'Multi Model Mean' + ' (SSH ave. from 1993 to 2009)'
    outfile = './fig/SSH_bias.png'
else:
    suptitle = sys.argv[1] + ' (SSH ave. from 1993 to 2009)'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/SSH_bias_' + sys.argv[1] + '.png'


#J 時刻情報 (各モデルの時刻情報を上書きする)
time1 = np.empty((2010-1948)*12,dtype='object')
for yr in range(1948,2010):
    for mon in range(1,13):
        time1[(yr-1948)*12+mon-1] = datetime.datetime(yr,mon,1)
time2 = np.empty((2019-1958)*12,dtype='object')
for yr in range(1958,2019):
    for mon in range(1,13):
        time2[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)
time = [ time1, time2 ]


#J データ読込・平均

print( "Loading CMEMS data" )
reffile = '../refdata/CMEMS/zos_adt_CMEMS_1x1_monthly_199301-201812.nc'
DS0 = xr.open_dataset( reffile )
da0 = DS0.zos.sel(time=slice('1993','2009'))

##J mask0 = 50S以北,50N以南で True となる2次元配列
mask0 = np.array(abs(DS0.lat)<50).reshape(len(DS0.lat),1)*np.array(~np.isnan(DS0.lon))
##J wgt0 = 緯度に応じた重み (2次元配列, mask0 = False の場所は0に)
wgt0 = np.empty(mask0.shape)
for i in range(len(DS0.zos[0][0][:])):
    for j in range(len(DS0.zos[0][:])):
        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * mask0[j,i]

##J wgt = 平均に使う重み(時間方向も含めた3次元配列)
##J       未定義の格子では重みを 0 にする
wgt = np.tile(wgt0,(len(da0),1,1)) * np.logical_not(np.isnan(da0))
##J 重み付き平均を計算、オフセットとして元データから差し引く
data_ave = np.average(da0.fillna(0),weights=wgt,axis=(1,2))
for n in range(len(data_ave)):
    da0[n] = da0[n] - data_ave[n]

da0 = da0.mean(dim='time',skipna=False)


data = []
for omip in range(2):
    d = np.empty( (len(model_list[omip]),180,360) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path  = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        infile = path + '/' + fname

        DS = xr.open_dataset( infile, decode_times=False )
        if (model == "Kiel-NEMO"):
            DS = DS.where(DS['zos'] != 0.0)
            if (omip == 0):
                DS = DS.rename({"time_counter":"time"})

        DS['time'] = time[omip]

        tmp = DS.zos.sel(time=slice('1993','2009'))

        ##J 50S-50N 平均値計算の前に格子をあわせる
        if model == "NorESM-O-CICE":
            tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
            tmp = tmp.roll(x=-180, roll_coords=True)
        if model == "MIROC-COCO4-9":
            tmp = tmp.sel(lat=slice(None, None, -1))

        ##J 重み付き平均を計算、オフセットとして元データから差し引く
        wgt = np.tile(wgt0,(len(tmp),1,1)) * np.logical_not(np.isnan(tmp))
        data_ave = np.average(tmp.fillna(0),weights=wgt,axis=(1,2))
        for n in range(len(data_ave)):
            tmp[n] = tmp[n] - data_ave[n]

        d[nmodel] = tmp.mean(dim='time',skipna=False)
        nmodel += 1

    data += [d]


DS = xr.Dataset( {'omip1bias': (['model','lat','lon'], data[0] - da0.values),
                  'omip2bias': (['model','lat','lon'], data[1] - da0.values),
                  'omip2-1': (['model','lat','lon'], data[1] - data[0]),
                  'obs': (['lat','lon'], da0.values), },
                 coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                            'lon': np.linspace(0.5,359.5,num=360), } )


#J 描画
fig = plt.figure(figsize=(16,12))
fig.suptitle( suptitle, fontsize=20 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(2,2,1,projection=proj),
    plt.subplot(2,2,2,projection=proj),
    plt.subplot(2,2,3,projection=proj),
    plt.subplot(2,2,4,projection=proj),
]

bounds1 = [-1.0, -0.7, -0.5, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 ]
bounds2 = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
bounds3 = np.arange(-1.8,1.201,0.1)
ticks_bounds3 = np.arange(-1.8,1.201,0.3)

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]

for panel in range(4):
    if item[panel] == 'omip1bias' or item[panel] == 'omip2bias':
        bounds = bounds1
        ticks_bounds = bounds1
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
    if item[panel] == 'obs':
        da = DS[item[panel]]
    else:
        da = DS[item[panel]].mean(dim='model',skipna=False)
    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs = { 'orientation': 'horizontal',
#                            'spacing': 'proportional',
                            'spacing': 'uniform',
                            'label': '[m]',
                            'ticks': ticks_bounds, },
            transform=ccrs.PlateCarree())
    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_title(title[panel])
    ax[panel].background_patch.set_facecolor('lightgray')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
