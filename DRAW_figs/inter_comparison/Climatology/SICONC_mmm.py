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

head_title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1', 'PCMDI' ]

ystr = 1980
yend = 2009
nyr = yend - ystr + 1

month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

mon_ana = [3, 9, 9, 3]

suptitle = ' SICONC Multi-Model Mean (ave. from '+str(ystr)+' to '+str(yend)+')'

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
damip = []
for mon in mon_ana:
    da0 = DS0.siconc.sel(time=slice(caltmpl.format(ystr,mon,1),caltmpl.format(yend+1,1,1),12))
    damip += [da0]

DS1 = xr.open_dataset( mskfile )
da1 = DS1.sftof

DSM = []

for omip in range(2):

    d = np.empty((4,len(model_list[omip]),nyr,180,360))
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

        nm = 0
        for mon in mon_ana:

            tmp = DS.siconc.sel(time=slice(caltmpl.format(ystr,mon,1),caltmpl.format(yend+1,1,1),12))

            if model == "NorESM-BLOM":
                tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
                tmp = tmp.roll(x=-180, roll_coords=True)
            if model == "MIROC-COCO4.9":
                tmp = tmp.sel(lat=slice(None, None, -1))

            d[nm,nmodel] = tmp.values.reshape(nyr,180,360) * fac - 0.1

            if (model == 'Kiel-NEMO'):
                d[nm,nmodel] = np.where(da1.values==0, np.nan, d[nm,nmodel])

            if (model == 'BSC-NEMO'):
                d[nm,nmodel] = np.where(d[nm,nmodel]<-10.0, np.nan, d[nm,nmodel])

            nm += 1
                
        nmodel += 1
        #if (nmodel == 2):
        #    break
        
    if (omip == 0):
        data1 = d
    else:
        data2 = d

nm = 0
for mon in mon_ana:
    DS = xr.Dataset( {'omip1': (['model','time','lat','lon'], data1[nm]),
                      'omip2': (['model','time','lat','lon'], data2[nm]),
                      'omip2-1': (['model','time','lat','lon'], data2[nm] - data1[nm]),
                      'amip': (['time','lat','lon'], 
                               np.where(da1.values==0, np.nan, damip[nm].values)), },
                     coords = { 'time': time0,
                                'lat': np.linspace(-89.5,89.5,num=180), 
                                'lon': np.linspace(0.5,359.5,num=360), } )
    DSM += [DS]
    nm +=1

    
#J 描画
print( 'Start drawing' )
fig = plt.figure(figsize=(8,11))
fig.suptitle( suptitle, fontsize=18 )

projn = ccrs.Orthographic(central_longitude=0, central_latitude=90)
projs = ccrs.Orthographic(central_longitude=180, central_latitude=-90)
#proj = ccrs.NorthPolarStereo()
#proj = ccrs.PlateCarree()
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(4,4,1,projection=projn),
    plt.subplot(4,4,2,projection=projn),
    plt.subplot(4,4,3,projection=projn),
    plt.subplot(4,4,4,projection=projn),
    plt.subplot(4,4,5,projection=projn),
    plt.subplot(4,4,6,projection=projn),
    plt.subplot(4,4,7,projection=projn),
    plt.subplot(4,4,8,projection=projn),
    plt.subplot(4,4,9,projection=projs),
    plt.subplot(4,4,10,projection=projs),
    plt.subplot(4,4,11,projection=projs),
    plt.subplot(4,4,12,projection=projs),
    plt.subplot(4,4,13,projection=projs),
    plt.subplot(4,4,14,projection=projs),
    plt.subplot(4,4,15,projection=projs),
    plt.subplot(4,4,16,projection=projs),
]

# [left, bottom, width, height]
ax_cbar1 = plt.axes([0.15,0.10,0.7,0.015])
ax_cbar2 = plt.axes([0.15,0.04,0.7,0.015])

bounds1 = np.linspace(0,100,num=11)
bounds2 = np.linspace(-100,100,num=21)
bounds3 = np.linspace(-25,25,num=11)

cmap = [ 'Reds', 'Reds', 'bwr', 'Reds' ]

item = [ 'omip1', 'omip2', 'omip2-1', 'amip' ]

outfile = './fig/SICONC_MMM'

nm=0
for mon in mon_ana:

    if (nm < 2):
        damip = DSM[nm]['amip'].sel(lat=slice(43.5,89.5)).mean(dim='time') - 0.001
    else:
        damip = DSM[nm]['amip'].sel(lat=slice(-89.5,-50.5)).mean(dim='time') - 0.001

    for nv_out in range(4):
        nax = 4 * nm + nv_out
        if (item[nv_out] == 'omip1' or item[nv_out] == 'omip2'):
            bounds = bounds1
            ticks_bounds = bounds1
            if (nm < 2):
                da = DSM[nm][item[nv_out]].sel(lat=slice(43.5,89.5)).mean(dim='time').mean(dim='model',skipna=False)
            else:
                da = DSM[nm][item[nv_out]].sel(lat=slice(-89.5,-50.5)).mean(dim='time').mean(dim='model',skipna=False)
        elif item[nv_out] == 'omip2-1':
            bounds = bounds3
            ticks_bounds = bounds3
            if (nm < 2):
                da = DSM[nm][item[nv_out]].sel(lat=slice(43.5,89.5)).mean(dim='time').mean(dim='model',skipna=False)
            else:
                da = DSM[nm][item[nv_out]].sel(lat=slice(-89.5,-50.5)).mean(dim='time').mean(dim='model',skipna=False)
        else:
            bounds = bounds1
            ticks_bounds = bounds1


        if (nv_out < 2):
            cs = da.plot(ax=ax[nax],cmap=cmap[nv_out],
                         levels=bounds,
                         add_colorbar=False,
                         transform=ccrs.PlateCarree())
            cs.cmap.set_under('azure')
            damip.plot.contour(ax=ax[nax],
                                levels=[15.0],
                                colors=['blue'],
                                linewidths=2,
                                transform=ccrs.PlateCarree())
            da.plot.contour(ax=ax[nax],
                            levels=[15.0],
                            colors=['red'],
                            linewidths=2,
                            transform=ccrs.PlateCarree())
        elif (nv_out == 2):
            if (nm == 0):
                da.plot(ax=ax[nax],cmap=cmap[nv_out],
                        levels=bounds,
                        extend='both',
                        cbar_kwargs={'orientation': 'horizontal',
                                     'spacing':'uniform',
                                     'label': 'difference in [%]',
                                     'ticks': ticks_bounds,},
                        cbar_ax = ax_cbar2,
                        transform=ccrs.PlateCarree())
            else:
                da.plot(ax=ax[nax],cmap=cmap[nv_out],
                        levels=bounds,
                        add_colorbar=False,
                        transform=ccrs.PlateCarree())

        else:
            if (nm == 0):
                cs = damip.plot(ax=ax[nax],cmap=cmap[nv_out],
                                levels=bounds,
                                cbar_kwargs={'orientation': 'horizontal',
                                             'spacing':'uniform',
                                             'label': 'area fraction in [%]',
                                             'ticks': ticks_bounds,},
                                cbar_ax = ax_cbar1,
                                transform=ccrs.PlateCarree())
                cs.cmap.set_under('azure')
            else:
                cs = damip.plot(ax=ax[nax],cmap=cmap[nv_out],
                    levels=bounds,
                    add_colorbar=False,
                    transform=ccrs.PlateCarree())
                cs.cmap.set_under('azure')

        if (nm < 2):
            ax[nax].set_extent([-180,180.1,43,90], ccrs.PlateCarree())
        else:
            ax[nax].set_extent([0,360.1,-90,-50], ccrs.PlateCarree())

        ax[nax].add_feature(cfeature.LAND)
        ax[nax].coastlines()
        ax[nax].gridlines(linestyle='-',color='gray')
        #ax[nmodel].set_boundary(circle,transform=ax[nmodel].transAxes)
        ax[nax].set_title(month[mon-1]+' '+head_title[nv_out])
        ax[nax].background_patch.set_facecolor('lightgray')

    nm+=1    

plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.12, wspace=0.05, hspace=0.08)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
