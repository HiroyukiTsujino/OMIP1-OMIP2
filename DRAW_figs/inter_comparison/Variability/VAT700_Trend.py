# -*- coding: utf-8 -*-
import sys
import json
import math
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()

period = [ 1993, 2009 ]
nrec = period[1] - period[0] + 1

title = [ '(a) OMIP-1', '(b) OMIP-2', '(c) Ishii et al. (2017) v7.2' ]

metainfo = [ json.load(open("./json/vat700_omip1.json")),
             json.load(open("./json/vat700_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


if (sys.argv[1] == 'MMM'):
    outfile = './fig/VAT700_trend_MMM'
    suptitle = 'Multi Model Mean' + ' VAT700 trend (1993-2009) '
else:
    model_list = [ [sys.argv[1]], [sys.argv[1]] ]
    outfile = './fig/VAT700_trend_' + sys.argv[1]
    suptitle = sys.argv[1] + ' VAT700 trend (1993-2009) '


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

        print(idx)
        for i in range(len(idx[0])):
            d[nmodel,idx[0][i],idx[1][i]] = p_org[0,i]

        print(d[nmodel,:,:])
        nmodel += 1

    data += [d]

print( "Loading Ishii_v7.3 data" )
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

fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(3,1,1,projection=proj),
    plt.subplot(3,1,2,projection=proj),
    plt.subplot(3,1,3,projection=proj),
]

# [left, bottom, width, height]
ax_cbar = plt.axes([0.88,0.20,0.02,0.6])

bounds = [ -0.2, -0.15, -0.1, -0.07, -0.04, -0.02, 0, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2 ]

item = [ 'omip1', 'omip2', 'obs' ]

for panel in range(3):
    if item[panel] == 'obs':
        da = DS[item[panel]]
    else:
        da = DS[item[panel]].mean(dim='model',skipna=False)
    if (panel == 0):
        da.plot(ax=ax[panel],cmap='RdBu_r',
                levels=bounds,
                extend='both',
                cbar_kwargs={'orientation':'vertical',
                             'spacing':'uniform',
                             'label': '[$^\circ\mathrm{C}\,\mathrm{year}^{-1}$]',
                             'ticks': bounds,},
                cbar_ax=ax_cbar,
                add_labels=False,add_colorbar=True,
                transform=ccrs.PlateCarree())
    else:
        da.plot(ax=ax[panel],cmap='RdBu_r',
                add_colorbar=False,
                levels=bounds,
                transform=ccrs.PlateCarree())

    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_title(title[panel],{'fontsize':12, 'verticalalignment':'top'})
    ax[panel].tick_params(labelsize=9)
    ax[panel].background_patch.set_facecolor('lightgray')
    ax[panel].set_ylabel('latitude')
    if (panel == 2):
        ax[panel].set_xlabel('longitude')
    else:
        ax[panel].set_xlabel('')
    
plt.subplots_adjust(left=0.05,right=0.98,top=0.92)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
