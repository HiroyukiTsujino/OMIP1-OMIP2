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
from uncertain_Wakamatsu import uncertain_2d


ystr = 1980
yend = 2009
nyr = yend - ystr + 1
factor_5ptail = 1.64  # 5-95%
num_bootstraps = 10000

metainfo = [ json.load(open("./json/sos_omip1.json")), 
             json.load(open("./json/sos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


suptitle = 'Multi Model Mean' + ' (SSS ave. from '+str(ystr)+' to '+str(yend)+')'
outfile = './fig/SSS_diff_bias.png'

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
data = []
for omip in range(2):
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
        nmodel += 1

    data += [d]

DS = xr.Dataset( {'omip2-1': (['model','time','lat','lon'], data[1] - data[0]), },
                 coords = { 'time': time0,
                            'lat': np.linspace(-89.5,89.5,num=180)}) 

print( 'Calculating OMIP2 - OMIP1' )
dout = uncertain_2d( DS['omip2-1'].values, num_bootstraps )
DS_stats = xr.Dataset( { 'mean': (['lat','lon'], dout[0]),
                       'std':  (['lat','lon'], dout[1]),
                       'M':    (['lat','lon'], dout[2]),
                       'V':    (['lat','lon'], dout[3]),
                       'B':    (['lat','lon'], dout[4]),},
                     coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                                'lon': np.linspace(0.5,359.5,num=360), }, )

print( 'Output netCDF4' )
path_out='../analysis/STDs/'
outstatsfile= path_out + 'SSS_omip1-omip2_stats.nc'
DS_stats.to_netcdf(path=outstatsfile,mode='w',format='NETCDF4')


#J 描画
print( 'Start drawing' )
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

bounds = [-0.4, -0.3, -0.2, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4]
ticks_bounds = bounds

cmap = 'RdBu_r'
item = 'omip2-1'

ax1=plt.axes(projection=proj)

da = DS_stats["mean"]

da.plot(ax=ax1,
        cmap=cmap,
        levels=bounds,
        extend='both',
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': '[$^\circ$C]',
                     'ticks': ticks_bounds,},
        transform=ccrs.PlateCarree())

mpl.rcParams['hatch.color'] = 'limegreen'
mpl.rcParams['hatch.linewidth'] = 0.5

x = DS_stats["lon"].values
y = DS_stats["lat"].values
z = np.abs(DS_stats["mean"]) - factor_5ptail * DS_stats["std"]
z = np.where( z > 0, 1, np.nan )
ax1.contourf(x,y,z,hatches=['xxxxx'],colors='none',transform=ccrs.PlateCarree())

ax1.coastlines()
ax1.set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.set_title('OMIP2 - OMIP1')
ax1.background_patch.set_facecolor('lightgray')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
