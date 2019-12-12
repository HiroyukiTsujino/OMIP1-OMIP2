# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()

title = [ '(a) Ensemble mean OMIP1', '(b) Ensemble mean OMIP2',
          '(c) Ensemble std OMIP1', '(d) Ensemble std OMIP2',
          '(e) OMIP2 - OMIP1', '(f) deBoyer' ]

metainfo = [ json.load(open("./json/mld_omip1.json")), 
             json.load(open("./json/mld_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


if (sys.argv[1] == 'MMM'):
    suptitle = 'Multi Model Mean' + ' Winter MLD, JFM (NH), JAS (SH) (ave. from 1980 to 2009)'
    outfile = './fig/MLD_Winter_MMM'
else:
    suptitle = sys.argv[1] + ' (Winter MLD, JFM (NH), JAS (SH) ave. from 1980 to 2009)'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/MLD_Winter_' + sys.argv[1]


#J 時刻情報 (各モデルの時刻情報を上書きする)
#time1 = np.empty((2010-1948)*12,dtype='object')
#for yr in range(1948,2010):
#    for mon in range(1,13):
#        time1[(yr-1948)*12+mon-1] = datetime.datetime(yr,mon,1)
#time2 = np.empty((2019-1958)*12,dtype='object')
#for yr in range(1958,2019):
#    for mon in range(1,13):
#        time2[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)
#time = [ time1, time2 ]


#J データ読込・平均

print( "Loading IFREMER data" )
reffile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_monclim.nc'
mskfile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_mask.nc'
DS0 = xr.open_dataset( reffile )
print(DS0)
danh = DS0.mlotst.sel(time=slice('1850-01-01','1850-03-01'),lat=slice(0.0,90.0)).mean(dim='time')
dash = DS0.mlotst.sel(time=slice('1850-07-01','1850-09-01'),lat=slice(-90.0,-0.0)).mean(dim='time')

lattmp=danh['lat'].values
lontmp=danh['lon'].values
danhob = xr.DataArray(danh.values, dims = ('lat', 'lon'), coords = {'lat': lattmp, 'lon': lontmp} )

lattmp=dash['lat'].values
lontmp=dash['lon'].values
dashob = xr.DataArray(dash.values, dims = ('lat', 'lon'), coords = {'lat': lattmp, 'lon': lontmp} )

DS1 = xr.open_dataset( mskfile )
da1 = DS1.mldmask

datanh = []
datash = []
for omip in range(2):
    dnh = np.empty( (len(model_list[omip]),90,360) )
    dsh = np.empty( (len(model_list[omip]),90,360) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path  = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        infile = path + '/' + fname
        DS = xr.open_dataset( infile )
        tmp = DS.mlotst.sel(time=slice('1850-01-01','1850-03-01'),lat=slice(0.5,89.5)).mean(dim='time')
        dnh[nmodel] = tmp.values
        tmp = DS.mlotst.sel(time=slice('1850-07-01','1850-09-01'),lat=slice(-89.5,-0.5)).mean(dim='time')
        dsh[nmodel] = tmp.values
        nmodel += 1

    datanh += [dnh]
    datash += [dsh]


DSNH = xr.Dataset( {'omip1mean': (['model','lat','lon'], datanh[0]),
                  'omip2mean': (['model','lat','lon'], datanh[1]),
                  'omip2-1': (['model','lat','lon'], datanh[1] - datanh[0]), },
                 coords = { 'lat': np.linspace(0.5,89.5,num=90), 
                            'lon': np.linspace(0.5,359.5,num=360), } )

DSSH = xr.Dataset( {'omip1mean': (['model','lat','lon'], datash[0]),
                  'omip2mean': (['model','lat','lon'], datash[1]),
                  'omip2-1': (['model','lat','lon'], datash[1] - datash[0]), },
                 coords = { 'lat': np.linspace(-89.5,-0.5,num=90), 
                            'lon': np.linspace(0.5,359.5,num=360), } )

#J 描画
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

# [left, bottom, width, height]
axes0 = np.array( [ [0.045, 0.80, 0.375, 0.13],
                    [0.045, 0.67, 0.375, 0.13],])

ax = [ [ plt.axes(axes0[0],projection=proj),
         plt.axes(axes0[1],projection=proj), ],
       [ plt.axes(axes0[0]+np.array([0.50,0,0,0]),projection=proj),
         plt.axes(axes0[1]+np.array([0.50,0,0,0]),projection=proj), ],
       [ plt.axes(axes0[0]+np.array([0,-0.31,0,0]),projection=proj),
         plt.axes(axes0[1]+np.array([0,-0.31,0,0]),projection=proj),],
       [ plt.axes(axes0[0]+np.array([0.50,-0.31,0,0]),projection=proj),
         plt.axes(axes0[1]+np.array([0.50,-0.31,0,0]),projection=proj), ],
       [ plt.axes(axes0[0]+np.array([0,-0.62,0,0]),projection=proj),
         plt.axes(axes0[1]+np.array([0,-0.62,0,0]),projection=proj), ],
       [ plt.axes(axes0[0]+np.array([0.50,-0.6215,0,0]),projection=proj),
         plt.axes(axes0[1]+np.array([0.50,-0.62,0,0]),projection=proj), ] ]

# [left, bottom, width, height]
ax_cbar = [
    plt.axes([0.43,0.68,0.012,0.23]),
    plt.axes([0.93,0.68,0.012,0.23]),
    plt.axes([0.43,0.37,0.012,0.23]),
    plt.axes([0.93,0.37,0.012,0.23]),
    plt.axes([0.43,0.06,0.012,0.23]),
    plt.axes([0.93,0.06,0.012,0.23]),
]

bounds1 = np.arange(0,1550,50)
ticks_bounds1 = np.arange(0,1550,500)
bounds2 = np.arange(-200,220,10)
ticks_bounds2 = np.arange(-200,220,50)
bounds3 = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
ticks_bounds3 =  [10, 20, 30, 50, 100, 200, 300, 500, 1000]

cmap = [ 'RdYlBu_r', 'RdYlBu_r', 'viridis', 'viridis', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1mean', 'omip2mean', 'omip1std', 'omip2std', 'omip2-1', 'deBoyer' ]

boxdic = {"facecolor" : "white",
          "edgecolor" : "black",
          "linewidth" : 1
          }

for panel in range(6):
    for ns in range(2):
        axn = panel * 2 + ns
        q, mod = divmod(panel,2)
        
        if item[panel] == 'omip1mean' or item[panel] == 'omip2mean':
            bounds = bounds1
            ticks_bounds = ticks_bounds1
            if ns == 0:
                da = DSNH[item[panel]].mean(dim='model',skipna=True)
            else:
                da = DSSH[item[panel]].mean(dim='model',skipna=True)
        elif item[panel] == 'omip1std':
            bounds = bounds3
            ticks_bounds = ticks_bounds3
            if ns == 0:
                da = DSNH['omip1mean'].std(dim='model',skipna=True)
            else:
                da = DSSH['omip1mean'].std(dim='model',skipna=True)
        elif item[panel] == 'omip2std':
            bounds = bounds3
            ticks_bounds = ticks_bounds3
            if ns == 0:
                da = DSNH['omip2mean'].std(dim='model',skipna=True)
            else:
                da = DSSH['omip2mean'].std(dim='model',skipna=True)
        elif item[panel] == 'omip2-1':
            bounds = bounds2
            ticks_bounds = ticks_bounds2
            if ns == 0:
                da = DSNH[item[panel]].mean(dim='model',skipna=True)
            else:
                da = DSSH[item[panel]].mean(dim='model',skipna=True)
        else:
            bounds = bounds1
            ticks_bounds = ticks_bounds1
            if ns == 0:
                da = danhob
            else:
                da = dashob

#        if (mod == 1 and ns == 0):
        da.plot(ax=ax[panel][ns],cmap=cmap[panel],
                levels=bounds,
                extend='both',
                cbar_kwargs={'orientation': 'vertical',
                             'spacing':'uniform',
                             'label': '[m]',
                             'ticks': ticks_bounds,},
                cbar_ax=ax_cbar[panel],
                transform=ccrs.PlateCarree())
#        else:
#            if (panel == 4 and ns ==0):
#                da.plot(ax=ax[panel][ns],cmap=cmap[panel],
#                        levels=bounds,
#                        extend='both',
#                        cbar_kwargs={'orientation': 'vertical',
#                                     'spacing':'uniform',
#                                     'label': '[m]',
#                                     'ticks': ticks_bounds,},
#                        cbar_ax=ax_cbar[panel-1],
#                        transform=ccrs.PlateCarree())
#            else:
#                da.plot(ax=ax[panel][ns],cmap=cmap[panel],
#                        levels=bounds,
#                        extend='both',
#                        add_colorbar=False,
#                        transform=ccrs.PlateCarree())
            
for panel in range(6):
    for ns in range(2):
        axn = panel * 2 + ns
        ax[panel][ns].coastlines()
        if ns == 0:
            ax[panel][ns].set_xlabel('')
            ax[panel][ns].set_yticks(np.arange(0,90.1,30),crs=ccrs.PlateCarree())
            ax[panel][ns].set_ylabel('')
            ax[panel][ns].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top'})
            #l, b, w, h = ax[panel][ns].get_position().bounds
            #print(l,b,w,h)
            #b = b - 0.01
            #h = h + 0.03
            #ax[panel][ns].set_position([l,b,w,h])
            ax[panel][ns].text(155,  6,"JFM", size = 10, bbox = boxdic)
        else:
            ax[panel][ns].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
            ax[panel][ns].set_xlabel('')
            ax[panel][ns].set_yticks(np.arange(-90,0.1,30),crs=ccrs.PlateCarree())
            ax[panel][ns].set_ylabel('')
            #l, b, w, h = ax[panel][ns].get_position().bounds
            #print(l,b,w,h)
            #b = b + 0.01
            #h = h + 0.03
            #ax[panel][ns].set_position([l,b,w,h])
            ax[panel][ns].text(155,-84,"JAS", size = 10, bbox = boxdic)

        ax[panel][ns].background_patch.set_facecolor('lightgray')
        ax[panel][ns].xaxis.set_major_formatter(lon_formatter)
        ax[panel][ns].yaxis.set_major_formatter(lat_formatter)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
