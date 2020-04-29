# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import netCDF4
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import datetime
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math


if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1 - CMEMS', 'OMIP2 - CMEMS', 'OMIP2 - OMIP1' ]

suptitle = head_title[nv_out]  + ' (SSH ave. from 1993 to 2009)'

metainfo = [ json.load(open("./json/zos_omip1.json")),
             json.load(open("./json/zos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

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
#mask0 = np.array(abs(DS0.lat)<50).reshape(len(DS0.lat),1)*np.array(~np.isnan(DS0.lon))

# mask based on CMEMS
cmemsmskf = '../refdata/CMEMS/zos_mask_gn_199301-200912.nc'
ncmskcmems = netCDF4.Dataset(cmemsmskf,'r')
maskcmems = ncmskcmems.variables['zosmask'][:,:]
ncmskcmems.close()
################################################
# Ad hoc modification for Mediterranean (mask out entirely)
maskcmems[120:140,0:40] = 0
maskcmems[120:130,355:359] = 0

maskmed = np.array(np.empty((180,360)),dtype=np.int64)
maskmed[:,:] = 1
maskmed[120:140,0:40] = 0
maskmed[120:130,355:359] = 0
################################################


##J wgt0 = 緯度に応じた重み (2次元配列, mask0 = False の場所は0に)
#wgt0 = np.empty(mask0.shape)
wgt0 = np.empty(maskcmems.shape)
for i in range(len(DS0.zos[0][0][:])):
    for j in range(len(DS0.zos[0][:])):
#        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * mask0[j,i] * maskcmems[j,i]
        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * maskcmems[j,i]

##J wgt = 平均に使う重み(時間方向も含めた3次元配列)
##J       未定義の格子では重みを 0 にする
wgt = np.tile(wgt0,(len(da0),1,1)) * np.logical_not(np.isnan(da0))
##J 重み付き平均を計算、オフセットとして元データから差し引く
data_ave = np.average(da0.fillna(0),weights=wgt,axis=(1,2))
for n in range(len(data_ave)):
    da0[n] = da0[n] - data_ave[n]

da0 = da0.mean(dim='time',skipna=False)

arefile = '../refdata/PCMDI-SST/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

d_tmp0 = np.empty((180,360))
d_tmp1 = np.empty((180,360))
d_tmp2 = np.empty((180,360))

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
        if model == "NorESM-BLOM":
            tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
            tmp = tmp.roll(x=-180, roll_coords=True)
        if model == "MIROC-COCO4.9":
            tmp = tmp.sel(lat=slice(None, None, -1))

        ##J 重み付き平均を計算、オフセットとして元データから差し引く
        wgt = np.tile(wgt0,(len(tmp),1,1)) * np.logical_not(np.isnan(tmp))
        data_ave = np.average(tmp.fillna(0),weights=wgt,axis=(1,2))
        for n in range(len(data_ave)):
            tmp[n] = tmp[n] - data_ave[n]

        #d[nmodel] = np.where(maskcmems==0, np.NaN, tmp.mean(dim='time',skipna=False).values)
        if model == "MIROC-COCO4.9":
            d[nmodel] = np.where(maskmed == 0, np.NaN, tmp.mean(dim='time',skipna=False).values)
        else:
            d[nmodel] = tmp.mean(dim='time',skipna=False).values
        nmodel += 1

    data += [d]

d_tmp0=np.where(maskcmems==0, np.NaN, da0.values)
d_tmp1=np.where(maskcmems==0, np.NaN, data[0])
d_tmp2=np.where(maskcmems==0, np.NaN, data[1])

DS = xr.Dataset( {'omip1bias': (['model','lat','lon'], d_tmp1 - d_tmp0),
                  'omip2bias': (['model','lat','lon'], d_tmp2 - d_tmp0),
                  'omip2-1': (['model','lat','lon'], data[1] - data[0]),
                  'obs': (['lat','lon'], da0.values), },
                 coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                            'lon': np.linspace(0.5,359.5,num=360), } )


#J 描画
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(4,3,1,projection=proj),
    plt.subplot(4,3,2,projection=proj),
    plt.subplot(4,3,3,projection=proj),
    plt.subplot(4,3,4,projection=proj),
    plt.subplot(4,3,5,projection=proj),
    plt.subplot(4,3,6,projection=proj),
    plt.subplot(4,3,7,projection=proj),
    plt.subplot(4,3,8,projection=proj),
    plt.subplot(4,3,9,projection=proj),
    plt.subplot(4,3,10,projection=proj),
    plt.subplot(4,3,11,projection=proj),
    plt.subplot(4,3,12,projection=proj),
]

# [left, bottom, width, height]
ax_cbar = plt.axes([0.15,0.06,0.7,0.02])

bounds1 = [-1.0, -0.7, -0.5, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 ]
bounds2 = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
bounds3 = np.arange(-1.8,1.201,0.1)
ticks_bounds3 = np.arange(-1.8,1.201,0.3)

cmap = [ 'RdBu_r', 'RdBu_r', 'bwr', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]
outfile = './fig/SSH_bias_allmodels_'+item[nv_out]+'.png'

dict_rmse={}

# MMM

nax = 11
if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
    bounds = bounds1
    ticks_bounds = bounds1
    da = DS[item[nv_out]].mean(dim='model',skipna=False)
    msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
    datmp = np.where( np.isnan(da.values), 0.0, da.values )
    tmp1 = (datmp * datmp * area * msktmp * maskcmems).sum()
    tmp2 = (area * msktmp * maskcmems).sum()
    tmp3 = (datmp * area * msktmp * maskcmems).sum()
    rmse = np.sqrt(tmp1/tmp2)
    bias = tmp3/tmp2
    title_append = ' rmse = ' + '{:.2f}'.format(rmse*100) + ' cm'
    dict_rmse['MMM']=[rmse*100,bias*100]
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    ticks_bounds = bounds2
    title_append = ''
else:
    bounds = bounds3
    ticks_bounds = ticks_bounds3
    title_append = ''

da = DS[item[nv_out]].mean(dim='model',skipna=False)

da.plot(ax=ax[nax],cmap=cmap[nv_out],
        levels=bounds,
        extend='both',
        cbar_kwargs = { 'orientation': 'horizontal',
                        'spacing': 'uniform',
                        'label': '[m]',
                        'ticks': ticks_bounds, },
        cbar_ax = ax_cbar,
        transform=ccrs.PlateCarree())

ax[nax].coastlines()
ax[nax].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[nax].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[nax].xaxis.set_major_formatter(lon_formatter)
ax[nax].yaxis.set_major_formatter(lat_formatter)
ax[nax].set_xlabel('')
ax[nax].set_ylabel('')
ax[nax].set_title('MMM '+title_append,{'fontsize':9,'verticalalignment':'top'})
ax[nax].tick_params(labelsize=8)
ax[nax].background_patch.set_facecolor('lightgray')

#####

nmodel = 0
for model in model_list[0]:
    if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS[item[nv_out]].isel(model=nmodel)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp * maskcmems).sum()
        tmp2 = (area * msktmp * maskcmems).sum()
        tmp3 = (datmp * area * msktmp * maskcmems).sum()
        rmse = np.sqrt(tmp1/tmp2)
        bias = tmp3/tmp2
        title_append = ' rmse = ' + '{:.2f}'.format(rmse*100) + ' cm'
        dict_rmse[model]=[rmse*100,bias*100]
    elif item[nv_out] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
        title_append = ''
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
        title_append = ''

    da = DS[item[nv_out]].isel(model=nmodel)

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
    ax[nmodel].set_title(model+title_append,{'fontsize':9,'verticalalignment':'top'})
    ax[nmodel].tick_params(labelsize=8)
    ax[nmodel].background_patch.set_facecolor('lightgray')
    nmodel += 1

plt.subplots_adjust(left=0.05,right=0.98,bottom=0.12,top=0.93,hspace=0.26,wspace=0.15)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

summary=pd.DataFrame(dict_rmse,index=['OMIP'+str(omip_out)+'_rmse','OMIP'+str(omip_out)+'_mean'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/SSH_bias_OMIP' + str(omip_out) + '.csv')

if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
