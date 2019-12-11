# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime


title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1', 'deBoyer' ]

metainfo = [ json.load(open("./json/mld_omip1.json")), 
             json.load(open("./json/mld_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


if len(sys.argv) == 1:
    suptitle = 'Multi Model Mean' + ' (MLD September ave. from 1980 to 2009)'
    outfile = './fig/MLD_SEP.png'
else:
    suptitle = sys.argv[1] + ' (MLD September ave. from 1980 to 2009)'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/MLD_SEP_' + sys.argv[1] + '.png'


#J 時刻情報 (各モデルの時刻情報を上書きする)
#time1 = np.empty((2010-1948)*12,dtype='object')
#for yr in range(1948,2010):
#    for mon in range(1,13):
#        time1[(yr-1948)*12+mon-1] = datetime.datetime(yr,mon,1)
#time2 = np.empty((2019-1958)*12,dtype='object')
#for yr in range(1958,2019):
#    for mon in range(1,13):
#        time2[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)
#time = [ time1, time2 ]


#J データ読込・平均

print( "Loading IFREMER data" )
reffile = '../refdata/MLD_deBoyer-Montegut/mld_DR003_monclim.nc'
mskfile = '../refdata/MLD_deBoyer-Montegut/mld_DR003_mask.nc'
DS0 = xr.open_dataset( reffile )
print(DS0)
da0 = DS0.mlotst.sel(time='1850-09-01')
DS1 = xr.open_dataset( mskfile )
da1 = DS1.mldmask

data = []
for omip in range(2):
    d = np.empty( (len(model_list[omip]),180,360) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path  = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        infile = path + '/' + fname
        DS = xr.open_dataset( infile )
        tmp = DS.mlotst.sel(time='1850-09-01')
        d[nmodel] = tmp.values
        nmodel += 1

    data += [d]


DS = xr.Dataset( {'omip1mean': (['model','lat','lon'], data[0]),
                  'omip2mean': (['model','lat','lon'], data[1]),
                  'omip2-1': (['model','lat','lon'], data[1] - data[0]), },
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

bounds1 = np.arange(0,2050,50)
ticks_bounds1 = np.arange(0,2050,500)
bounds2 = np.arange(-400,420,20)
ticks_bounds2 = np.arange(-400,420,50)

cmap = [ 'RdYlBu_r', 'RdYlBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1mean', 'omip2mean', 'omip2-1', 'deBoyer' ]

for panel in range(4):
    if item[panel] == 'omip1mean' or item[panel] == 'omip2mean':
        bounds = bounds1
        ticks_bounds = ticks_bounds1
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = ticks_bounds2
    else:
        bounds = bounds1
        ticks_bounds = ticks_bounds1
    if item[panel] == 'deBoyer':
        da = da0
    else:
        da = DS[item[panel]].mean(dim='model',skipna=True)
    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs={'orientation': 'horizontal',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'label': '[m]',
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
#plt.show()
