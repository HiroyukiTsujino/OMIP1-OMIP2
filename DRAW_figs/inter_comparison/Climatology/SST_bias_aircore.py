# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import netCDF4
import xarray as xr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime

ystr = 1980
yend = 2009
nyr = yend - ystr + 1

title = [ '(a) Bias (Gill 1982)', '(b) Bias (Large and Yeager 2009)',
          '(c) Gill - Large&Yeager', '(d) PCMDI' ]

metainfo = json.load(open("./json/tos_omip2_aircore.json"))
model_list = metainfo.keys()

suptitle = 'SST (ave. from '+str(ystr)+' to '+str(yend)+')'
outfile = './fig/SST_bias_aircore'


#J 時刻情報 (各モデルの時刻情報を上書きする)
time = np.empty((2019-1958)*12,dtype='object')
for yr in range(1958,2019):
    for mon in range(1,13):
        time[(yr-1958)*12+mon-1] = datetime.datetime(yr,mon,1)

#J データ読込・平均

print( "Loading AMIP data" )
reffile = '../refdata/PCMDI-SST/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
mskfile = '../refdata/PCMDI-SST/sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
#J 年平均計算も行う。ただし日数の重みがつかないので不正確
DS0 = xr.open_dataset( reffile ).resample(time='1YS').mean()
da0 = DS0.tos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)
DS1 = xr.open_dataset( mskfile )
da1 = DS1.sftof

arefile = '../refdata/PCMDI-SST/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

d = np.empty( (len(model_list),180,360) )

nmodel = 0
for model in model_list:

    print("processing ",model)

    path  = metainfo[model]['path']
    fname = metainfo[model]['fname']
    infile = path + '/' + fname

    DS = xr.open_dataset( infile, decode_times=False )
    DS['time'] = time

    tmp = DS.tos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)

    d[nmodel] = tmp.values
    nmodel += 1

bias = d - da0.values

DS = xr.Dataset( {'Gbias': (['lat','lon'], bias[0]),
                  'LYbias': (['lat','lon'], bias[1]),
                  'G-LY': (['lat','lon'], d[0] - d[1]),
                  'amip': (['lat','lon'], np.where(da1.values==0, np.nan, da0.values)), },
                 coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                            'lon': np.linspace(0.5,359.5,num=360), } )

#J 描画
print( 'Start drawing' )
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
#ax_cbar = [
#    plt.axes([0.93,0.64,0.012,0.23]),
#    plt.axes([0.93,0.37,0.012,0.23]),
#    plt.axes([0.93,0.10,0.012,0.23]),
#]

bounds1 = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0]
#bounds2 = [-0.7, -0.4, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.4, 0.7]
bounds2 = [-0.4, -0.3, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
bounds3 = np.arange(-1,30.1,1)
ticks_bounds3 = [0, 5, 10, 15, 20, 25, 30] 
bounds4 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5]
ticks_bounds4 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5] 

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'Gbias', 'LYbias', 'G-LY', 'amip' ]

for panel in range(3):
    if (item[panel] == 'Gbias'):
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS['Gbias']
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        tmp3 = (area).sum()
        print(tmp1,tmp2,tmp3)
        rmse = np.sqrt(tmp1/tmp2)
        title[panel] = title[panel]+' rmse = ' + '{:.3f}'.format(rmse) + '$^\circ$C'
    elif (item[panel] == 'LYbias'):
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS['LYbias']
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        title[panel] = title[panel]+' rmse = ' + '{:.3f}'.format(rmse) + '$^\circ$C'
        #print(tmp1,tmp2,tmp1/tmp2)
    elif item[panel] == 'G-LY':
        bounds = bounds2
        ticks_bounds = bounds2
        da = DS[item[panel]]
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
        da = DS[item[panel]]


#    if (panel == 0 or panel == 2 or panel == 4):
#        ii = int(panel / 2)
#        print(ii)
    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
#            cbar_ax = ax_cbar[ii],
            cbar_kwargs={'orientation': 'horizontal',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'shrink':0.65,
                         'label': '[$^\circ$C]',
                         'ticks': ticks_bounds,},
            transform=ccrs.PlateCarree())
#    else:
#        da.plot(ax=ax[panel],cmap=cmap[panel],
#            levels=bounds,
#            extend='both',
#            add_colorbar=False,
#            transform=ccrs.PlateCarree())

        
    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_xlabel('')
    ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top'})
    ax[panel].tick_params(labelsize=9)
    ax[panel].background_patch.set_facecolor('lightgray')

plt.subplots_adjust(left=0.07,right=0.98,bottom=0.05,top=0.92,wspace=0.16,hspace=0.15)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()
