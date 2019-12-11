# -*- coding: utf-8 -*-
import sys
import json
import math
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' OMIP1 (1) or OMIP2 (2)')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1

head_title = [ 'OMIP-1', 'OMIP-2' ]

suptitle = head_title[nv_out]  + ' VAT700 trend (1993-2009)'

period = [ 1993, 2009 ]
nrec = period[1] - period[0] + 1

metainfo = [ json.load(open("./json/vat700_omip1.json")),
             json.load(open("./json/vat700_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

#J 時刻情報 (各モデルの時刻情報を上書きする)
time_ref = np.linspace(1955,2018,64)
time_bsc = np.linspace(1958,2014,57)
time = [ np.linspace(1948,2009,62), np.linspace(1958,2018,61) ]


#J データ読込・平均

data = []
for omip in range(2):
    d = np.full( (len(model_list[omip]),180,360), np.nan )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        var = metainfo[omip][model]['name']
        infile = path + '/' + fname

        DS_read = xr.open_dataset(infile,decode_times=False)
        if (model == 'Kiel-NEMO'):
            DS_read = DS_read.where(DS_read['thetao'] != 0.0, other=np.nan)
            #DS_read = DS_read.drop_dims("depth")
            
        if model == "CMCC-NEMO":
            DS_read = DS_read.rename({"time_counter":"time"})

        if (omip == 1 and model == "EC-Earth3-NEMO"):
            DS_read['time'] = time_bsc
        else:
            DS_read['time'] = time[omip]

        if model == "NorESM-BLOM":
            DS_read = DS_read.assign_coords(lon=('x', np.where( DS_read.lon < 0, DS_read.lon + 360, DS_read.lon )))
            DS_read = DS_read.roll(x=-180, roll_coords=True)
            DS_read = DS_read.rename({"x":"lon","y":"lat"})
        if model == "MIROC-COCO4.9":
            DS_read = DS_read.sel(lat=slice(None, None, -1))

        #J numpy.polyfit で線形回帰
        #J  陸地(np.nan)は抜く (収束せず落ちるため)
        tmp = DS_read[var].sel(time=slice(period[0],period[1]))
        y = tmp.values[~np.isnan(tmp.values)]
        y = y.reshape(nrec,math.ceil(len(y)/nrec))
        p_org = np.polyfit(np.arange(period[0],period[1]+1),y,1)
        #print(p_org)
        #J 180x360 配列に焼直し
        if (model == "Kiel-NEMO"):
            idx = np.where(~np.isnan(tmp.values[0,0]))
        else:
            idx = np.where(~np.isnan(tmp.values[0]))

        #print(idx)
        for i in range(len(idx[0])):
            d[nmodel,idx[0][i],idx[1][i]] = p_org[0,i]

        #print(d[nmodel,:,:])
        nmodel += 1

        
    data += [d]


print( "Loading Ishii_v7.2 data" )
reffile = '../refdata/Ishii_v7.2/temp/vat700_1955-2018.nc'
DS_read = xr.open_dataset( reffile, decode_times=False )
DS_read['time'] = time_ref

#J numpy.polyfit で線形回帰
#J  陸地(np.nan)は抜く (収束せず落ちるため)
tmp = DS_read["thetao"].sel(time=slice(period[0],period[1]))
y = tmp.values[~np.isnan(tmp.values)]
print( 'Valid points:', len(y) )
print( 'Record Num.:', nrec )
y = y.reshape(nrec,math.ceil(len(y)/nrec))
p_org = np.polyfit(np.arange(period[0],period[1]+1),y,1)
#J 180x360 配列に焼直し
idx = np.where(~np.isnan(tmp.values[0]))
data_ref = np.full((180,360),np.nan)
for i in range(len(idx[0])):
    data_ref[idx[0][i],idx[1][i]] = p_org[0,i]


DS = xr.Dataset( {'omip1': (['model','lat','lon'], data[0]),
                  'omip2': (['model','lat','lon'], data[1]),
                  'obs'  : (['lat','lon'], data_ref), },
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


bounds = [ -0.2, -0.15, -0.1, -0.07, -0.04, -0.02, 0, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2 ]

item = [ 'omip1', 'omip2', 'obs' ]

outfile = './fig/VAT700_trend_'+ item[nv_out] + '.png'

nax=11
daobs = DS['obs']
daobs.plot(ax=ax[nax],cmap='RdBu_r',
           levels=bounds,
           extend='both',
           cbar_kwargs={'orientation':'horizontal',
                        'spacing':'uniform',
                        'label': '[$^\circ\mathrm{C}\,\mathrm{year}^{-1}$]',
                        'ticks': bounds,},
           cbar_ax = ax_cbar,
           transform=ccrs.PlateCarree())
ax[nax].coastlines()
ax[nax].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[nax].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[nax].xaxis.set_major_formatter(lon_formatter)
ax[nax].yaxis.set_major_formatter(lat_formatter)
ax[nax].set_xlabel('')
ax[nax].set_ylabel('')
ax[nax].set_title("Ishii v7.2",{'fontsize':9, 'verticalalignment':'top'})
ax[nax].tick_params(labelsize=8)
ax[nax].background_patch.set_facecolor('lightgray')


nmodel = 0
for model in model_list[0]:
    #da = DS[item[nv_out]].mean(dim='model',skipna=False)
    da = DS[item[nv_out]].isel(model=nmodel)
    da.plot(ax=ax[nmodel],cmap='RdBu_r',
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
    ax[nmodel].set_title(model,{'fontsize':9, 'verticalalignment':'top'})
    ax[nmodel].tick_params(labelsize=8)
    ax[nmodel].background_patch.set_facecolor('lightgray')
    nmodel += 1

plt.subplots_adjust(left=0.05,right=0.98,bottom=0.12,top=0.93,hspace=0.26,wspace=0.15)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
if (len(sys.argv) == 3 and sys.argv[2] == 'show'):
    plt.show()
