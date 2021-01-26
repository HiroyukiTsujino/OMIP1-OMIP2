# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import netCDF4
import xarray as xr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import math

ystr = 1993
yend = 2009
nyr = yend - ystr + 1
factor_5ptail = 1.64  # 5-95%

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()


title = [ '(a) Ensemble bias (OMIP1 - CMEMS)', '(b) Ensemble bias (OMIP2 - CMEMS)',
          '(c) Ensemble std (OMIP1 bias)', '(d) Ensemble std (OMIP2 bias)',
          '(e) OMIP2 - OMIP1', '(f) CMEMS' ]

#metainfo = [ json.load(open("./json/zos_omip1_wo_coco.json")),
#             json.load(open("./json/zos_omip2_wo_coco.json")) ]
metainfo = [ json.load(open("./json/zos_omip1.json")),
             json.load(open("./json/zos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

if (sys.argv[1] == 'MMM'):
    suptitle = 'Multi Model Mean' + ' SSH (ave. from '+str(ystr)+' to '+str(yend)+')'
    outfile = './fig/SSH_bias_MMM_midlat'
else:
    suptitle = sys.argv[1] + ' SSH (ave. from '+str(ystr)+' to '+str(yend)+')'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/SSH_bias_' + sys.argv[1]


#J 時刻情報 (各モデルの時刻情報を上書きする)
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

print( "Loading CMEMS data" )
reffile = '../refdata/CMEMS/zos_adt_CMEMS_1x1_monthly_199301-201812.nc'
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

maskmed = np.array(np.empty((180,360)),dtype=np.int64)
maskmed[:,:] = 1
maskmed[120:140,0:40] = 0
maskmed[120:130,355:360] = 0
################################################

##J wgt0 = 緯度に応じた重み (2次元配列, mask0 = False の場所は0に)
#wgt0 = np.empty(mask0.shape)
wgt0 = np.empty(maskcmems.shape)
for i in range(len(DS0.zos[0][0][:])):
    for j in range(len(DS0.zos[0][:])):
#        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * mask0[j,i] * maskcmems[j,i]
        wgt0[j,i] = math.cos(math.radians(DS0.lat.values[j])) * maskcmems[j,i]

##J wgt = 平均に使う重み(時間方向も含めた3次元配列)
##J       未定義の格子では重みを 0 にする
wgt = np.tile(wgt0,(len(da0),1,1)) * np.logical_not(np.isnan(da0))
##J 重み付き平均を計算、オフセットとして元データから差し引く
data_ave = np.average(da0.fillna(0),weights=wgt,axis=(1,2))
for n in range(len(data_ave)):
    da0[n] = da0[n] - data_ave[n]

da0 = da0.mean(dim='time',skipna=False)


arefile = '../refdata/PCMDI-SST/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

# uncertainty of difference between omip-1 and omip-2

stdfile = '../analysis/STDs/SSH_omip1-omip2_stats.nc'
DS_stats = xr.open_dataset( stdfile )

d_tmp0 = np.empty((180,360))
d_tmp1 = np.empty((180,360))
d_tmp2 = np.empty((180,360))
d_tmp3 = np.empty((180,360))
d_tmp4 = np.empty((180,360))

data = []
for omip in range(2):
    d = np.empty( (len(model_list[omip]),180,360) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path  = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        infile = path + '/' + fname

        DS = xr.open_dataset( infile, decode_times=False )
        if (model == "Kiel-NEMO"):
            DS = DS.where(DS['zos'] != 0.0)
            if (omip == 0):
                DS = DS.rename({"time_counter":"time"})

        DS['time'] = time[omip]

        tmp = DS.zos.sel(time=slice(str(ystr),str(yend)))

        ##J 50S-50N 平均値計算の前に格子をあわせる
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

        d[nmodel] = tmp.mean(dim='time',skipna=False).values
        nmodel += 1

    data += [d]

d_tmp0=np.where(maskcmems==0, np.NaN, da0.values)
d_tmp1=np.where(maskcmems==0, np.NaN, data[0])
d_tmp2=np.where(maskcmems==0, np.NaN, data[1])
d_tmp3=np.where(maskmed==0, np.NaN, data[0])
d_tmp4=np.where(maskmed==0, np.NaN, data[1])

DS = xr.Dataset( {'omip1bias': (['model','lat','lon'], d_tmp1 - d_tmp0),
                  'omip2bias': (['model','lat','lon'], d_tmp2 - d_tmp0),
                  'omip1mean': (['model','lat','lon'], d_tmp3),
                  'omip2mean': (['model','lat','lon'], d_tmp4),
                  'omip2-1': (['model','lat','lon'], d_tmp4 - d_tmp3),
                  'obs': (['lat','lon'], da0.values), },
                 coords = { 'lat': np.linspace(-89.5,89.5,num=180), 
                            'lon': np.linspace(0.5,359.5,num=360), } )


#J 描画
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(3,2,1,projection=proj),
    plt.subplot(3,2,2,projection=proj),
    plt.subplot(3,2,3,projection=proj),
    plt.subplot(3,2,4,projection=proj),
    plt.subplot(3,2,5,projection=proj),
    plt.subplot(3,2,6),
]

bounds1 = [-1.0, -0.7, -0.5, -0.3, -0.2, -0.1, -0.05, 0., 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
bounds2 = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, -0.02, 0., 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
bounds3 = np.arange(-1.8,1.201,0.1)
ticks_bounds3 = np.arange(-1.8,1.201,0.3)
bounds4 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
ticks_bounds4 = [0.0, 0.5, 1.0] 

cmap = [ 'RdBu_r', 'RdBu_r', 'terrain', 'terrain', 'RdBu_r', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip1std', 'omip2std', 'omip2-1', 'obs' ]

ddof_dic={'ddof' : 0}

for panel in range(6):
    if (item[panel] == 'omip1bias'):
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS['omip1bias'].mean(dim='model',skipna=False)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        tmp3 = (area).sum()
        print(tmp2,tmp3)
        rmse = np.sqrt(tmp1/tmp2)
        title[panel] = title[panel]+' rmse = ' + '{:.2f}'.format(rmse*100) + ' cm'
        print(title[panel])
    elif (item[panel] == 'omip2bias'):
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS['omip2bias'].mean(dim='model',skipna=False)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        tmp3 = (area).sum()
        print(tmp2,tmp3)
        rmse = np.sqrt(tmp1/tmp2)
        title[panel] = title[panel]+' rmse = ' + '{:.2f}'.format(rmse*100) + ' cm'
        print(title[panel])
    elif (item[panel] == 'omip1std'):
        bounds = bounds4
        ticks_bounds = bounds4
        da = DS['omip1bias'].std(dim='model',skipna=False, **ddof_dic)
        tmp = DS['omip1bias'].var(dim='model', skipna=False, **ddof_dic)
        msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
        tmp1 = (datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        z = np.abs(DS['omip1bias'].mean(dim='model',skipna=False)) - 2.0 * da
        z = np.where( z > 0, 1, 0 )
        tmp3 = (z * area * msktmp).sum()
        failcapt=tmp3/tmp2*100
        title[panel] = title[panel] + r' 2$\bar{\sigma}$=' + '{:.2f}'.format(200*rmse) + ' cm ' + '\n' \
            + 'observation uncaptured by model spread = ' + '{:.1f}'.format(failcapt) + '%'
        print(title[panel])
    elif (item[panel] == 'omip2std'):
        bounds = bounds4
        ticks_bounds = bounds4
        da = DS['omip2bias'].std(dim='model',skipna=False, **ddof_dic)
        tmp = DS['omip2bias'].var(dim='model', skipna=False, **ddof_dic)
        msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
        tmp1 = (datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        z = np.abs(DS['omip2bias'].mean(dim='model',skipna=False)) - 2.0 * da
        z = np.where( z > 0, 1, 0 )
        tmp3 = (z * area * msktmp).sum()
        failcapt=tmp3/tmp2*100
        title[panel] = title[panel] + r' 2$\bar{\sigma}$=' + '{:.2f}'.format(200*rmse) + ' cm ' + '\n' \
            + 'observation uncaptured by model spread = ' + '{:.1f}'.format(failcapt) + '%'
        print(title[panel])
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = bounds2
        da = DS[item[panel]].mean(dim='model',skipna=False)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        rmsd = np.sqrt(tmp1/tmp2)
        title[panel] = title[panel] + ' rmsd= ' + '{:.2f}'.format(100*rmsd) + ' cm '
        print(title[panel])
    else:
        bounds = bounds3
        ticks_bounds = ticks_bounds3
        daobs = DS[item[panel]]
        damod = DS['omip1mean'].mean(dim='model',skipna=False)

    if (panel < 5):
        da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs = { 'orientation': 'vertical',
#                            'spacing': 'proportional',
                            'spacing': 'uniform',
                            'label': '[m]',
                            'ticks': ticks_bounds, },
            transform=ccrs.PlateCarree())
        
        if (panel == 2):
            mpl.rcParams['hatch.color'] = 'red'
            mpl.rcParams['hatch.linewidth'] = 0.5
            x = DS["lon"].values
            y = DS["lat"].values
            z = np.abs(DS["omip1bias"]).mean(dim='model',skipna=False) - 2.0 * DS['omip1bias'].std(dim='model',skipna=False, **ddof_dic)
            z = np.where( z > 0, 1, np.nan )
            ax[panel].contourf(x,y,z,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
        if (panel == 3):
            mpl.rcParams['hatch.color'] = 'red'
            mpl.rcParams['hatch.linewidth'] = 0.5
            x = DS["lon"].values
            y = DS["lat"].values
            z = np.abs(DS["omip2bias"]).mean(dim='model',skipna=False) - 2.0 * DS['omip2bias'].std(dim='model',skipna=False, **ddof_dic)
            z = np.where( z > 0, 1, np.nan )
            ax[panel].contourf(x,y,z,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
        
        if (panel == 4):
            mpl.rcParams['hatch.color'] = 'limegreen'
            mpl.rcParams['hatch.linewidth'] = 0.5
            x = DS_stats["lon"].values
            y = DS_stats["lat"].values
            z = np.abs(DS_stats["mean"]) - factor_5ptail * DS_stats["std"]
            z = np.where( z > 0, 1, np.nan )
            ax[panel].contourf(x,y,z,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
        
        ax[panel].coastlines()
        ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
        ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
        ax[panel].xaxis.set_major_formatter(lon_formatter)
        ax[panel].yaxis.set_major_formatter(lat_formatter)
        ax[panel].set_xlabel('')
        ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top'})
        ax[panel].tick_params(labelsize=9)
        ax[panel].background_patch.set_facecolor('lightgray')

    else:
        daobs.sel(lon=150.5).plot(ax=ax[panel], color='darkorange', label='CMEMS')
        damod.sel(lon=150.5).plot(ax=ax[panel], color='darkblue', label='MMM')
        ax[panel].set_title("(f) Longitude = $150.5^{\circ}$E")
        ax[panel].set_xlabel("Latitude")
        ax[panel].set_xlim(25,55)
        ax[panel].set_xticks(np.linspace(25,55,7))
        ax[panel].set_ylabel("SSH mean [m]")
        ax[panel].set_ylim(-0.6,1.2)
        ax[panel].set_yticks(np.linspace(-0.6,1.2,10))
        ax[panel].legend()
        ax[panel].grid()


plt.subplots_adjust(left=0.07,right=0.98,bottom=0.05,top=0.92,wspace=0.16,hspace=0.15)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
