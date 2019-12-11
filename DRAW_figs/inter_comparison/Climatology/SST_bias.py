# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../python")
import json
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime
from uncertain_Wakamatsu import uncertain_2d


ystr = 1980
yend = 2009
nyr = yend - ystr + 1
factor = 1.64  # 5-95%


title = [ '(a) OMIP1 - AMIP', '(b) OMIP2 - AMIP', '(c) OMIP2 - OMIP1', '(d) AMIP' ]

metainfo = [ json.load(open("./json/tos_omip1.json")), 
             json.load(open("./json/tos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


if len(sys.argv) == 1:
    suptitle = 'Multi Model Mean' + ' (SST ave. from '+str(ystr)+' to '+str(yend)+')'
    outfile = './fig/SST_bias.png'

else:
    suptitle = sys.argv[1] + ' (SST ave. from '+str(ystr)+' to '+str(yend)+')'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/SST_bias_' + sys.argv[1] + '.png'


#J 時刻情報 (各モデルの時刻情報を上書きする)
time0 = np.empty(nyr,dtype='object')
for yr in range(ystr,yend+1):
    time0[yr-ystr] = datetime.datetime(yr,1,1)
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

print( "Loading AMIP data" )
reffile = '../AMIP/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
mskfile = '../AMIP/sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
#J 年平均計算も行う。ただし日数の重みがつかないので不正確
DS0 = xr.open_dataset( reffile ).resample(time='1YS').mean()
da0 = DS0.tos.sel(time=slice(str(ystr),str(yend)))
DS1 = xr.open_dataset( mskfile )
da1 = DS1.sftof

data = []
for omip in range(2):
    d = np.empty( (len(model_list[omip]),nyr,180,360) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path  = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        infile = path + '/' + fname

        DS = xr.open_dataset( infile, decode_times=False )
        if (model == 'Kiel-NEMO'):
            DS = DS.where(DS['tos'] != 0.0)

        DS['time'] = time[omip]

        #J 年平均計算。ただし日数の重みがつかないので不正確
        DS = DS.resample(time='1YS').mean()

        tmp = DS.tos.sel(time=slice(str(ystr),str(yend)))

        if model == "NorESM-BLOM":
            tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
            tmp = tmp.roll(x=-180, roll_coords=True)
        if model == "MIROC-COCO4.9":
            tmp = tmp.sel(lat=slice(None, None, -1))

        d[nmodel] = tmp.values.reshape(nyr,180,360)
        nmodel += 1

    data += [d]


DS = xr.Dataset( {'omip1bias': (['model','time','lat','lon'], data[0] - da0.values),
                  'omip2bias': (['model','time','lat','lon'], data[1] - da0.values),
                  'omip2-1': (['model','time','lat','lon'], data[1] - data[0]),
                  'amip': (['time','lat','lon'], 
                           np.where(da1.values==0, np.nan, da0.values)), },
                 coords = { 'time': time0,
                            'lat': np.linspace(-89.5,89.5,num=180), 
                            'lon': np.linspace(0.5,359.5,num=360), } )

DS_out_list = []

num_bootstraps = 10000
print( 'Calculating OMIP1 - AMIP' )
dout = uncertain_2d( DS['omip1bias'].values, num_bootstraps )
DS_out_list += [ xr.Dataset( { 'mean': (['lat','lon'], dout[0]),
                               'std':  (['lat','lon'], dout[1]),
                               'M':    (['lat','lon'], dout[2]),
                               'V':    (['lat','lon'], dout[3]),
                               'B':    (['lat','lon'], dout[4]),},
                             coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                                        'lon': np.linspace(0.5,359.5,num=360), }, ) ]
print( 'Calculating OMIP2 - AMIP' )
dout = uncertain_2d( DS['omip2bias'].values, num_bootstraps )
DS_out_list += [ xr.Dataset( { 'mean': (['lat','lon'], dout[0]),
                               'std':  (['lat','lon'], dout[1]),
                               'M':    (['lat','lon'], dout[2]),
                               'V':    (['lat','lon'], dout[3]),
                               'B':    (['lat','lon'], dout[4]),},
                             coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                                        'lon': np.linspace(0.5,359.5,num=360), }, ) ]
print( 'Calculating OMIP2 - OMIP1' )
dout = uncertain_2d( DS['omip2-1'].values, num_bootstraps )
DS_out_list += [ xr.Dataset( { 'mean': (['lat','lon'], dout[0]),
                               'std':  (['lat','lon'], dout[1]),
                               'M':    (['lat','lon'], dout[2]),
                               'V':    (['lat','lon'], dout[3]),
                               'B':    (['lat','lon'], dout[4]),},
                             coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                                        'lon': np.linspace(0.5,359.5,num=360), }, ) ]
DS_out_list += [ DS["amip"].mean(dim='time',skipna=False) ]


#J 描画
print( 'Start drawing' )
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

bounds1 = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0]
bounds2 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
bounds3 = np.arange(-1,30.1,1)
ticks_bounds3 = [0, 5, 10, 15, 20, 25, 30] 

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'amip' ]

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
    if item[panel] == 'amip':
        da = DS_out_list[panel]
    else:
        da = DS_out_list[panel]["mean"]
    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs={'orientation': 'horizontal',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'label': '[$^\circ$C]',
                         'ticks': ticks_bounds,},
            transform=ccrs.PlateCarree())

mpl.rcParams['hatch.color'] = 'limegreen'
mpl.rcParams['hatch.linewidth'] = 0.5
for panel in range(3):
    x = DS_out_list[panel]["lon"].values
    y = DS_out_list[panel]["lat"].values
    z = np.abs(DS_out_list[panel]["mean"]) - factor * DS_out_list[panel]["std"]
    z = np.where( z > 0, 1, np.nan )
    ax[panel].contourf(x,y,z,hatches=['xxx'],colors='none',transform=ccrs.PlateCarree())

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
