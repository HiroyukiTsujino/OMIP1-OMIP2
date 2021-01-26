# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import netCDF4
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math


#if (len(sys.argv) < 2):
#    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
#    sys.exit()

#omip_out = int(sys.argv[1])
#nv_out = int(sys.argv[1]) - 1

#head_title = [ 'OMIP1 - CMEMS', 'OMIP2 - CMEMS', 'OMIP2 - OMIP1' ]

title = [ '(a)', '(b)', '(c)', '(d)', '(e)', '(f)' ]
suptitle = ' SSH (average from 1993 to 2009)'
outfile = './fig/SSH_omip2_alpha.png'

metainfo = json.load(open("./json/zos_omip2_alpha.json"))
model_list = metainfo.keys()

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = np.empty((2019-1958)*12,dtype='object')
for yr in range(1958,2019):
    for mon in range(1,13):
        time[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)


#J データ読込・平均

print( "Loading CMEMS data" )
reffile = '../refdata/CMEMS/zos_adt_CMEMS_1x1_monthly_199301-201812.nc'
DS0 = xr.open_dataset( reffile )
da0 = DS0.zos.sel(time=slice('1993','2009'))

##J mask0 = 50S以北,50N以南で True となる2次元配列
mask0 = np.array(abs(DS0.lat)<50).reshape(len(DS0.lat),1)*np.array(~np.isnan(DS0.lon))

# mask based on CMEMS
cmemsmskf = '../refdata/CMEMS/zos_mask_gn_199301-200912.nc'
ncmskcmems = netCDF4.Dataset(cmemsmskf,'r')
maskcmems = ncmskcmems.variables['zosmask'][:,:]
ncmskcmems.close()
################################################
# Ad hoc modification for Mediterranean (mask out entirely)
maskcmems[120:140,0:40] = 0
maskcmems[120:130,355:360] = 0
################################################

##J wgt0 = 緯度に応じた重み (2次元配列, mask0 = False の場所は0に)
wgt0 = np.empty(mask0.shape)
for i in range(len(DS0.zos[0][0][:])):
    for j in range(len(DS0.zos[0][:])):
        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * mask0[j,i] * maskcmems[j,i]

##J wgt = 平均に使う重み(時間方向も含めた3次元配列)
##J       未定義の格子では重みを 0 にする
wgt = np.tile(wgt0,(len(da0),1,1)) * np.logical_not(np.isnan(da0))
##J 重み付き平均を計算、オフセットとして元データから差し引く
data_ave = np.average(da0.fillna(0),weights=wgt,axis=(1,2))
for n in range(len(data_ave)):
    da0[n] = da0[n] - data_ave[n]

da0 = da0.mean(dim='time',skipna=False)

arefile = '../AMIP/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

d = np.empty( (len(model_list),180,360) )

nmodel = 0
for model in model_list:

    path  = metainfo[model]['path']
    fname = metainfo[model]['fname']
    infile = path + '/' + fname

    DS = xr.open_dataset( infile, decode_times=False )
    DS['time'] = time

    tmp = DS.zos.sel(time=slice('1993','2009'))

    ##J 重み付き平均を計算、オフセットとして元データから差し引く
    wgt = np.tile(wgt0,(len(tmp),1,1)) * np.logical_not(np.isnan(tmp))
    data_ave = np.average(tmp.fillna(0),weights=wgt,axis=(1,2))

    for n in range(len(data_ave)):
        tmp[n] = tmp[n] - data_ave[n]

    d[nmodel] = tmp.mean(dim='time',skipna=False)
    nmodel += 1

DS = xr.Dataset( {'omip2bias': (['model','lat','lon'], d - da0.values),
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
    plt.subplot(3,2,1,projection=proj),
    plt.subplot(3,2,2,projection=proj),
    plt.subplot(3,2,3,projection=proj),
    plt.subplot(3,2,4,projection=proj),
    plt.subplot(3,2,5,projection=proj),
    plt.subplot(3,2,6,projection=proj),
]

# [left, bottom, width, height]
ax_cbar = plt.axes([0.15,0.04,0.7,0.02])

bounds1 = [-1.0, -0.7, -0.5, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 ]
bounds2 = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
bounds3 = np.arange(-1.8,1.201,0.1)
ticks_bounds3 = np.arange(-1.8,1.201,0.3)

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]

#####

nv_out=1
nmodel = 0
for model in model_list:
    if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS[item[nv_out]].isel(model=nmodel)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp * maskcmems).sum()
        tmp2 = (area * msktmp * maskcmems).sum()
        rmse = np.sqrt(tmp1/tmp2)
        title[nmodel] = title[nmodel] + model + ' rmse = ' + '{:.2f}'.format(rmse*100) + ' cm'
    elif item[nv_out] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3

    da = DS[item[nv_out]].isel(model=nmodel)

    if (nmodel == 0):
        da.plot(ax=ax[nmodel],cmap=cmap[nv_out],
                levels=bounds,
                extend='both',
                cbar_kwargs = { 'orientation': 'horizontal',
                                'spacing': 'uniform',
                                'label': '[m]',
                                'ticks': ticks_bounds, },
                cbar_ax = ax_cbar,
                transform=ccrs.PlateCarree())
    else:
        da.plot(ax=ax[nmodel],cmap=cmap[nv_out],
                levels=bounds,
                extend='both',
                add_colorbar=False,
                transform=ccrs.PlateCarree())

    ax[nmodel].coastlines()
    ax[nmodel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[nmodel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[nmodel].xaxis.set_major_formatter(lon_formatter)
    ax[nmodel].yaxis.set_major_formatter(lat_formatter)
    ax[nmodel].set_xlabel('')
    ax[nmodel].set_ylabel('')
    ax[nmodel].set_title(title[nmodel])
    ax[nmodel].background_patch.set_facecolor('lightgray')
    nmodel += 1

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
