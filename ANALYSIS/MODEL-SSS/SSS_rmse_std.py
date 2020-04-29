# -*- coding: utf-8 -*-
import sys
sys.path.append("../../python")
import json
import numpy as np
import xarray as xr
import netCDF4
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime

#------------------------
if (len(sys.argv) < 1):
    print ('Usage: ' + sys.argv[0] + ' mip_id' )
    sys.exit()

mip = int(sys.argv[1])
#------------------------

ystr = 1980
yend = 2009
nyr = yend - ystr + 1

suptitle = 'Multi Model Mean OMIP-' + str(mip) + ' (SSS ave. from '+str(ystr)+' to '+str(yend)+')'
outfile = './fig/SSS_bias_std_OMIP-'+str(mip)+'.png'

# reference data

print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_s.1000'
DS0 = xr.open_dataset( reffile, decode_times=False )
da0 = DS0.so.sel(depth=0).isel(time=0)

# model data

metainfo = [ json.load(open("./json/sos_omip1.json")), 
             json.load(open("./json/sos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

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

omip = mip - 1

d = np.empty( (len(model_list[omip]),nyr,180,360) )
print( "Loading OMIP" + str(omip+1) + " data" )

nmodel = 0
for model in model_list[omip]:

    print(model)
        
    path  = metainfo[omip][model]['path']
    fname = metainfo[omip][model]['fname']
    infile = path + '/' + fname

    DS = xr.open_dataset( infile, decode_times=False )
    if (model == 'Kiel-NEMO'):
        DS = DS.where(DS['sos'] != 0.0)

    DS['time'] = time[omip]

    #J 年平均計算。ただし日数の重みがつかないので不正確
    DS = DS.resample(time='1YS').mean()

    tmp = DS.sos.sel(time=slice(str(ystr),str(yend)))

    if model == "NorESM-BLOM":
        tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
        tmp = tmp.roll(x=-180, roll_coords=True)

    if model == "MIROC-COCO4.9":
        tmp = tmp.sel(lat=slice(None, None, -1))

    d[nmodel] = tmp.values.reshape(nyr,180,360)
    #d[nmodel] = tmp.values.reshape(nyr,180,360) - da0.values.reshape(180,360)
    nmodel += 1

print( 'Calculating bias and model std' )

num_exp, num_t, ny, nx = d.shape

# absolute difference
mmm = np.mean(d,axis=(0,1)) # multi model mean
mmm_bias = np.abs(mmm - da0.values)
# root mean square error
#mmm = np.mean(d,axis=0) # multi model mean
#mmm_power = np.mean(mmm*mmm,axis=0)
#mmm_bias = np.sqrt(mmm_power)

tmm = np.mean(d,axis=1) # time mean
tmm_std = np.std(tmm,axis=0,ddof=1) # model std of time mean


DS_stats = xr.Dataset( { 'bias': (['lat','lon'], mmm_bias),
                         'uncertainty':  (['lat','lon'], tmm_std*2),},
                       coords = { 'lat': np.linspace(-89.5,89.5,num=180),
                                  'lon': np.linspace(0.5,359.5,num=360), }, )

print( 'Output netCDF4' )
path_out='../analysis/STDs/'
outstatsfile= path_out + 'SSS_omip-'+str(omip+1)+'-STDs.nc'
DS_stats.to_netcdf(path=outstatsfile,mode='w',format='NETCDF4')


#J 描画
print( 'Start drawing' )
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(2,1,1,projection=proj),
    plt.subplot(2,1,2,projection=proj),
]

bounds = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
ticks_bounds = bounds

cmap = 'YlOrBr'

DS_stats["bias"].plot(ax=ax[0],
        cmap=cmap,
        levels=bounds,
        extend='both',
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': '[$^\circ$C]',
                     'ticks': ticks_bounds,},
        transform=ccrs.PlateCarree())

mpl.rcParams['hatch.color'] = 'navy'
mpl.rcParams['hatch.linewidth'] = 0.5

x = DS_stats["lon"].values
y = DS_stats["lat"].values
z = np.abs(DS_stats["bias"]) - DS_stats["uncertainty"]
z = np.where( z > 0, 1, np.nan )
ax[0].contourf(x,y,z,hatches=['xxxxx'],colors='none',transform=ccrs.PlateCarree())

ax[0].coastlines()
ax[0].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[0].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[0].xaxis.set_major_formatter(lon_formatter)
ax[0].yaxis.set_major_formatter(lat_formatter)
ax[0].set_title('Bias of MMM relative to WOA13v2')
ax[0].background_patch.set_facecolor('lightgray')

DS_stats["uncertainty"].plot(ax=ax[1],
        cmap=cmap,
        levels=bounds,
        extend='both',
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': '[$^\circ$C]',
                     'ticks': ticks_bounds,},
        transform=ccrs.PlateCarree())

ax[1].coastlines()
ax[1].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[1].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[1].xaxis.set_major_formatter(lon_formatter)
ax[1].yaxis.set_major_formatter(lat_formatter)
ax[1].set_title('Model uncertainty range (2 x Model STD)')
ax[1].background_patch.set_facecolor('lightgray')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
