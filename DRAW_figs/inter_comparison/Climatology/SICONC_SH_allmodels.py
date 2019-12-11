# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime

if (len(sys.argv) < 2):
    print ('Usage: ' + sys.argv[0] + ' month(1-12) ' + ' OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)')
    sys.exit()

mon_cal = int(sys.argv[1])
omip_out = int(sys.argv[2])
nv_out = int(sys.argv[2]) - 1

head_title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1' ]
ystr = 1980
yend = 2009
nyr = yend - ystr + 1

month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

suptitle = head_title[nv_out]  + ' SICONC SH ' + month[mon_cal-1] + ' (ave. from '+str(ystr)+' to '+str(yend)+')'

metainfo = [ json.load(open("./json/siconc_omip1.json")), 
             json.load(open("./json/siconc_omip2.json")) ]
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

print( "Loading AMIP data" )
reffile = '../refdata/PCMDI-SST/siconc_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
mskfile = '../refdata/PCMDI-SST/sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

caltmpl='{0:4d}-{1:02d}-{2:02d}'

DS0 = xr.open_dataset( reffile )
da0 = DS0.siconc.sel(time=slice(caltmpl.format(ystr,mon_cal,1),caltmpl.format(yend,12,1),12))
print(da0['time'])

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
        fac = float(metainfo[omip][model]['factor'])
        infile = path + '/' + fname

        print(model, infile)
        
        DS = xr.open_dataset( infile, decode_times=False )
        DS['time'] = time[omip]

        tmp = DS.siconc.sel(time=slice(caltmpl.format(ystr,mon_cal,1),caltmpl.format(yend,12,1),12))

        if model == "NorESM-BLOM":
            tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
            tmp = tmp.roll(x=-180, roll_coords=True)
        if model == "MIROC-COCO4.9":
            tmp = tmp.sel(lat=slice(None, None, -1))

        d[nmodel] = tmp.values.reshape(nyr,180,360) * fac - 0.1

        if (model == 'Kiel-NEMO'):
            d[nmodel] = np.where(da1.values==0, np.nan, d[nmodel])

        if (model == 'BSC-NEMO'):
            d[nmodel] = np.where(d[nmodel]<-10.0, np.nan, d[nmodel])

        nmodel += 1
        
    data += [d]

DS = xr.Dataset( {'omip1': (['model','time','lat','lon'], data[0]),
                  'omip2': (['model','time','lat','lon'], data[1]),
                  'omip2-1': (['model','time','lat','lon'], data[1] - data[0]),
                  'amip': (['time','lat','lon'], 
                           np.where(da1.values==0, np.nan, da0.values)), },
                 coords = { 'time': time0,
                            'lat': np.linspace(-89.5,89.5,num=180), 
                            'lon': np.linspace(0.5,359.5,num=360), } )

#J 描画
print( 'Start drawing' )
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.Orthographic(central_longitude=180, central_latitude=-90)
#proj = ccrs.NorthPolarStereo()
#proj = ccrs.PlateCarree()
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


bounds1 = np.linspace(0,100,num=11)
bounds2 = np.linspace(-100,100,num=21)
bounds3 = np.linspace(-50,50,num=11)

cmap = [ 'Reds', 'Reds', 'RdBu_r', 'Reds' ]

item = [ 'omip1', 'omip2', 'omip2-1', 'amip' ]

outfile = './fig/SICONC_SH_allmodels_'+item[nv_out]+'_'+month[mon_cal-1]+'.png'

#theta = np.linspace(0, 2*np.pi, 100)
#center, radius = [0.5, 0.5], 0.5
#verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#circle = mpl.path.Path(verts * radius + center)

daamip = DS['amip'].sel(lat=slice(-89.5,-50.5)).mean(dim='time')

nax = 11

if (item[nv_out] == 'omip1' or item[nv_out] == 'omip2'):
    bounds = bounds1
    ticks_bounds = bounds1
elif item[nv_out] == 'omip2-1':
    bounds = bounds2
    ticks_bounds = bounds2
else:
    bounds = bounds1
    ticks_bounds = bounds1

da = DS[item[nv_out]].sel(lat=slice(-89.5,-50.5)).mean(dim='time').mean(dim='model',skipna=False)

cs = da.plot(ax=ax[nax],cmap=cmap[nv_out],
        levels=bounds,
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': '[%]',
                     'ticks': ticks_bounds,},
        cbar_ax = ax_cbar,
        transform=ccrs.PlateCarree())
cs.cmap.set_under('azure')

daamip.plot.contour(ax=ax[nax],
                    levels=[15.0],
                    colors=['blue'],
                    linewidths=2,
                    transform=ccrs.PlateCarree())

da.plot.contour(ax=ax[nax],
                levels=[15.0],
                colors=['red'],
                linewidths=2,
                transform=ccrs.PlateCarree())

ax[nax].set_extent([0,360.1,-90,-50], ccrs.PlateCarree())
ax[nax].add_feature(cfeature.LAND)
ax[nax].coastlines()
ax[nax].gridlines(linestyle='-',color='gray')
#ax[nax].set_boundary(circle,transform=ax[nax].transAxes)
ax[nax].set_title('MMM',{'fontsize':10, 'verticalalignment':'top'})
ax[nax].background_patch.set_facecolor('lightgray')

nmodel = 0
for model in model_list[0]:
    if (item[nv_out] == 'omip1' or item[nv_out] == 'omip2'):
        bounds = bounds1
        ticks_bounds = bounds1
    elif item[nv_out] == 'omip2-1':
        bounds = bounds3
        ticks_bounds = bounds3
    else:
        bounds = bounds1
        ticks_bounds = bounds1

    da = DS[item[nv_out]].isel(model=nmodel).sel(lat=slice(-89.5,-50.5)).mean(dim='time')

    cs = da.plot(ax=ax[nmodel],cmap=cmap[nv_out],
            levels=bounds,
            add_colorbar=False,
            transform=ccrs.PlateCarree())
    cs.cmap.set_under('azure')

    daamip.plot.contour(ax=ax[nmodel],
                        levels=[15.0],
                        colors=['blue'],
                        linewidths=2,
                        transform=ccrs.PlateCarree())

    da.plot.contour(ax=ax[nmodel],
                    levels=[15.0],
                    colors=['red'],
                    linewidths=2,
                    transform=ccrs.PlateCarree())

    ax[nmodel].set_extent([0,360.1,-90,-50], ccrs.PlateCarree())
    ax[nmodel].add_feature(cfeature.LAND)
    ax[nmodel].coastlines()
    ax[nmodel].gridlines(linestyle='-',color='gray')
    #ax[nmodel].set_boundary(circle,transform=ax[nmodel].transAxes)
    ax[nmodel].set_title(model,{'fontsize':10, 'verticalalignment':'top'})
    ax[nmodel].background_patch.set_facecolor('lightgray')
    nmodel += 1
        
plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10, wspace=0.05, hspace=0.1)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
if (len(sys.argv) == 4 and sys.argv[3] == 'show'):
    plt.show()
