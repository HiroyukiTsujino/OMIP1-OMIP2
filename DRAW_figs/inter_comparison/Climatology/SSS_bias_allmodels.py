# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import netCDF4
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime


if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP1 - WOA13v2', 'OMIP2 - WOA13v2', 'OMIP2 - OMIP1' ]

suptitle = head_title[nv_out]  + ' (SSS ave. from 1980 to 2009)'

metainfo = [ json.load(open("./json/sos_omip1.json")), 
             json.load(open("./json/sos_omip2.json")) ]
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

        if model == "NorESM-BLOM":
            tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
            tmp = tmp.roll(x=-180, roll_coords=True)
        if model == "MIROC-COCO4.9":
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

bounds1 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
bounds2 = [-1.0, -0.4, -0.3, -0.2, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4, 1.0]
bounds3 = [30, 31, 32, 33.0, 33.6, 34.0, 34.3, 34.6, 34.9, 35.2, 35.5, 35.8, 36.1, 36.5, 36.9, 37.3 ]

cmap = [ 'RdBu_r', 'RdBu_r', 'bwr', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'obs' ]
outfile = './fig/SSS_bias_allmodels_'+item[nv_out]+'.png'

dict_rmse={}

# MMM

nax = 11
model = 'MMM'
if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
    bounds = bounds1
    ticks_bounds = bounds1
    da = DS[item[nv_out]].mean(dim='model',skipna=False)
    msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
    datmp = np.where( np.isnan(da.values), 0.0, da.values )
    tmp1 = (datmp * datmp * area * msktmp).sum()
    tmp2 = (area * msktmp).sum()
    tmp3 = (datmp * area * msktmp).sum()
    rmse = np.sqrt(tmp1/tmp2)
    bias = tmp3/tmp2
    title_panel = model + '\n' \
            + ' mean bias = ' + '{:.3f}'.format(bias) + 'psu,' + '    bias rmse = ' + '{:.3f}'.format(rmse) + 'psu'
    #title_append = ' rmse = ' + '{:.3f}'.format(rmse) + ' psu'
    dict_rmse['MMM']=[rmse,bias]
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    ticks_bounds = bounds2
    title_panel = model
else:
    bounds = bounds3
    ticks_bounds = bounds3
    title_panel = 'OBS'

da = DS[item[nv_out]].mean(dim='model',skipna=False)

da.plot(ax=ax[nax],cmap=cmap[nv_out],
        levels=bounds,
        extend='both',
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': "[psu]",
                     'ticks': ticks_bounds,},
        cbar_ax = ax_cbar,
        transform=ccrs.PlateCarree())

ax[nax].coastlines()
ax[nax].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[nax].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[nax].xaxis.set_major_formatter(lon_formatter)
ax[nax].yaxis.set_major_formatter(lat_formatter)
ax[nax].set_xlabel('')
ax[nax].set_ylabel('')
ax[nax].set_title(title_panel,{'fontsize':8, 'verticalalignment':'top', 'linespacing':0.8})
#ax[nax].set_title('MMM'+ title_append,{'fontsize':9,'verticalalignment':'top'})
ax[nax].tick_params(labelsize=8)
ax[nax].background_patch.set_facecolor('lightgray')


nmodel = 0
for model in model_list[0]:
    if item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias':
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS[item[nv_out]].isel(model=nmodel)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        tmp3 = (datmp * area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        bias = tmp3/tmp2
        title_panel = model + '\n' \
            + ' mean bias = ' + '{:.3f}'.format(bias) + 'psu,' + '    bias rmse = ' + '{:.3f}'.format(rmse) + 'psu'
        #title_append = ' rmse = ' + '{:.3f}'.format(rmse) + ' psu'
        dict_rmse[model]=[rmse,bias]
    elif item[nv_out] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
        title_panel = model
    else:
        bounds = bounds3
        ticks_bounds = bounds3
        title_panel = 'OBS'

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
    ax[nmodel].set_title(title_panel,{'fontsize':8, 'verticalalignment':'top', 'linespacing':0.8})
    #ax[nmodel].set_title(model+title_append,{'fontsize':9, 'verticalalignment':'top'})
    ax[nmodel].tick_params(labelsize=8)
    ax[nmodel].background_patch.set_facecolor('lightgray')
    nmodel += 1

plt.subplots_adjust(left=0.05,right=0.98,bottom=0.12,top=0.92,hspace=0.32,wspace=0.15)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

summary=pd.DataFrame(dict_rmse,index=['OMIP'+str(omip_out)+'_rmse','OMIP'+str(omip_out)+'_mean'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/SSS_bias_OMIP' + str(omip_out) + '.csv')

if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
