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
import math

#------------------------
if (len(sys.argv) < 1):
    print ('Usage: ' + sys.argv[0] + ' mip_id' )
    sys.exit()

mip = int(sys.argv[1])
#------------------------

ystr = 1993
yend = 2009
nyr = yend - ystr + 1

metainfo = [ json.load(open("./json/zos_omip1.json")), 
             json.load(open("./json/zos_omip2.json")) ]
#metainfo = [ json.load(open("./json/zos_omip1_wo_coco.json")), 
#             json.load(open("./json/zos_omip2_wo_coco.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

suptitle = 'Multi Model Mean OMIP-' + str(mip) + ' (SSH ave. from '+str(ystr)+' to '+str(yend)+')'
outfile = './fig/SSH_rmse_std_OMIP-'+str(mip)+'.png'

#J データ読込・平均

print( "Loading CMEMS data" )
reffile = '../analysis/SSH/CMEMS/zos_filter_annual_gn_1993-2018.nc'
DS0 = xr.open_dataset( reffile )
da0 = DS0.zos.sel(time=slice(str(ystr),str(yend)))

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
maskcmems[120:130,355:360] = 0
################################################

##J wgt0 = 緯度に応じた重み (2次元配列, mask0 = False の場所は0に)
#wgt0 = np.empty(mask0.shape)
wgt0 = np.empty(maskcmems.shape)
for i in range(len(DS0.zos[0][0][:])):
    for j in range(len(DS0.zos[0][:])):
#        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * mask0[j,i]
        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * maskcmems[j,i]

print("length of da0 ", len(da0))
wgt = np.tile(wgt0,(len(da0),1,1)) * np.logical_not(np.isnan(da0))
print("wgt shape ",wgt.shape)
data_ave = np.average(da0.fillna(0),weights=wgt,axis=(1,2))
for n in range(len(data_ave)):
    print("cmems quasi-global ave =", data_ave[n])
    da0[n] = da0[n] - data_ave[n]
        
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

d_tmp1 = np.empty( (nyr,180,360) )
d_tmp2 = np.empty( (nyr,180,360) )

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
        DS = DS.where(DS['zos'] != 0.0)
        if (omip == 0):
            DS = DS.rename({"time_counter":"time"})

    DS['time'] = time[omip]

    #J 年平均計算。ただし日数の重みがつかないので不正確
    DS = DS.resample(time='1YS').mean()

    tmp = DS.zos.sel(time=slice(str(ystr),str(yend)))

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

    d_tmp1 = tmp.values.reshape(nyr,180,360)
    for n in range(nyr):
        d_tmp2[n] = np.where(maskcmems == 0, np.NaN, d_tmp1[n])

    d[nmodel] = d_tmp2 - da0.values.reshape(nyr,180,360)
    nmodel += 1

print( 'Calculating RMSE bias and model std' )

num_exp, num_t, ny, nx = d.shape

mmm = np.mean(d,axis=0) # multi model mean
mmm_power = np.mean(mmm*mmm,axis=0)
mmm_rmse = np.sqrt(mmm_power)

tmm = np.mean(d,axis=1) # time mean
tmm_std = np.std(tmm,axis=0,ddof=1) # model std of time mean

DS_stats = xr.Dataset( { 'rmse': (['lat','lon'], mmm_rmse),
                         'uncertainty':  (['lat','lon'], tmm_std*2),},
                       coords = { 'lat': np.linspace(-89.5,89.5,num=180),
                                  'lon': np.linspace(0.5,359.5,num=360), }, )

print( 'Output netCDF4' )
path_out='../analysis/STDs/'
outstatsfile=path_out + 'SSH_omip-'+str(omip+1)+'-STDs.nc'
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

bounds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
ticks_bounds = bounds

cmap = 'YlOrBr'

DS_stats["rmse"].plot(ax=ax[0],
        cmap=cmap,
        levels=bounds,
        extend='both',
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': '[m]',
                     'ticks': ticks_bounds,},
        transform=ccrs.PlateCarree())

mpl.rcParams['hatch.color'] = 'green'
mpl.rcParams['hatch.linewidth'] = 0.5

x = DS_stats["lon"].values
y = DS_stats["lat"].values
z = np.abs(DS_stats["rmse"]) - DS_stats["uncertainty"]
z = np.where( z > 0, 1, np.nan )
ax[0].contourf(x,y,z,hatches=['xxxxx'],colors='none',transform=ccrs.PlateCarree())

ax[0].coastlines(resolution='50m')
ax[0].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[0].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[0].xaxis.set_major_formatter(lon_formatter)
ax[0].yaxis.set_major_formatter(lat_formatter)
ax[0].set_title('RMSE of MMM relative to CMEMS')
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
