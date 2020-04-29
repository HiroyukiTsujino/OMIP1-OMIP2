# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime
import netCDF4


title = [ '(a) OMIP1 - WOA13v2', '(b) OMIP2 - WOA13v2', '(c) OMIP2 - OMIP1', '(d) WOA13v2' ]

metainfo = [ json.load(open("./json/sos_omip1.json")), 
             json.load(open("./json/sos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


if len(sys.argv) == 1:
    suptitle = 'Multi Model Mean' + ' (SSS ave. from 1980 to 2009)'
    outfile = './fig/SSS_bias.png'
else:
    suptitle = sys.argv[1] + ' (SSS ave. from 1980 to 2009)'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/SSS_bias_' + sys.argv[1] + '.png'


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

print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_s.1000'
DS0 = xr.open_dataset( reffile, decode_times=False )
da0 = DS0.so.sel(depth=0).isel(time=0)

arefile = '../refdata/PCMDI-SST/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

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
        if (model == 'Kiel-NEMO'):
            DS = DS.where(DS['sos'] != 0.0)

        DS['time'] = time[omip]

        tmp = DS.sos.sel(time=slice('1980','2009')).mean(dim='time',skipna=False)

        if model == "NorESM-O-CICE":
            tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
            tmp = tmp.roll(x=-180, roll_coords=True)
        if model == "MIROC-COCO4-9":
            tmp = tmp.sel(lat=slice(None, None, -1))

        d[nmodel] = tmp.values
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

bounds1 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
bounds2 = [-0.4, -0.3, -0.2, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4]
bounds3 = [30, 31, 32, 33.0, 33.6, 34.0, 34.3, 34.6, 34.9, 35.2, 35.5, 35.8, 36.1, 36.5, 36.9, 37.3 ]

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]

for panel in range(4):
    if item[panel] == 'omip1bias' or item[panel] == 'omip2bias':
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS[item[panel]].mean(dim='model',skipna=False)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        tmp3 = (datmp * area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        bias = tmp3/tmp2
        title[panel] = title[panel] + '\n' \
            + ' mean bias = ' + '{:.3f}'.format(bias) + 'psu,' + '    bias rmse = ' + '{:.3f}'.format(rmse) + 'psu'
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
    else:
        bounds = bounds3
        ticks_bounds = bounds3
    if item[panel] == 'obs':
        da = DS[item[panel]]
    else:
        da = DS[item[panel]].mean(dim='model',skipna=False)
    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs={'orientation': 'horizontal',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'label': "",
                         'ticks': ticks_bounds,},
            transform=ccrs.PlateCarree())
for panel in range(4):
    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_title(title[panel])
    ax[panel].background_patch.set_facecolor('lightgray')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
