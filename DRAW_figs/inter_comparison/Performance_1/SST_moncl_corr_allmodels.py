# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../python")
import json
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime
from taylorDiagram import TaylorDiagram

if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1' ]

suptitle = head_title[nv_out] + ' Correlation of monthly climatology of SST from 1980 to 2009'

metainfo = [ json.load(open("./json/toscor_monclim_omip1.json")), 
             json.load(open("./json/toscor_monclim_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


#J データ読込・平均

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

        tmp = DS['toscor']

        d[nmodel] = tmp.values
        nmodel += 1

    data += [d]


DS = xr.Dataset( {'omip1': (['model','lat','lon'], data[0]),
                  'omip2': (['model','lat','lon'], data[1]),
                  'omip2-1': (['model','lat','lon'], data[1] - data[0]), },
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


bounds1 = [-1.0, -0.95, -0.9, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
bounds2 = [-0.2, -0.15, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r' ]

item = [ 'omip1', 'omip2', 'omip2-1' ]
outfile = './fig/toscor_monclim_'+item[nv_out]+'.png'

# MMM

if item[nv_out] == 'omip1' or item[nv_out] == 'omip2':
    bounds = bounds1
    ticks_bounds = bounds1
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    ticks_bounds = bounds2

da = DS[item[nv_out]].mean(dim='model',skipna=False)
da.plot(ax=ax[11],cmap=cmap[nv_out],
        levels=bounds,
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': "",
                     'ticks': ticks_bounds,},
        cbar_ax = ax_cbar,
        transform=ccrs.PlateCarree())

ax[11].coastlines()
ax[11].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[11].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[11].xaxis.set_major_formatter(lon_formatter)
ax[11].yaxis.set_major_formatter(lat_formatter)
ax[11].set_xlabel('')
ax[11].set_ylabel('')
ax[11].set_title('MMM',{'fontsize':9, 'verticalalignment':'top'})
ax[11].tick_params(labelsize=8)
ax[11].background_patch.set_facecolor('lightgray')

nmodel = 0
for model in model_list[0]:
    if item[nv_out] == 'omip1' or item[nv_out] == 'omip2':
        bounds = bounds1
        ticks_bounds = bounds1
    elif item[nv_out] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2

    da = DS[item[nv_out]].isel(model=nmodel)
    da.plot(ax=ax[nmodel],cmap=cmap[nv_out],
            levels=bounds,
            add_colorbar=False,
            transform=ccrs.PlateCarree())
    ax[nmodel].coastlines()
    ax[nmodel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[nmodel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[nmodel].xaxis.set_major_formatter(lon_formatter)
    ax[nmodel].yaxis.set_major_formatter(lat_formatter)
    ax[nmodel].set_xlabel('')
    ax[nmodel].set_ylabel('')
    ax[nmodel].set_title(model,{'fontsize':9, 'verticalalignment':'top'})
    ax[nmodel].tick_params(labelsize=8)
    ax[nmodel].background_patch.set_facecolor('lightgray')
    nmodel += 1

plt.subplots_adjust(left=0.05,right=0.98,bottom=0.12,top=0.93,hspace=0.26,wspace=0.1)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
