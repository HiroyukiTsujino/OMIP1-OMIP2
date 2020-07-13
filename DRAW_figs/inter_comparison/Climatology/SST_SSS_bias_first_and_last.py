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

if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [1 or 2 (omip1 or omip2)] [show (to check using viewer)]')
    sys.exit()

mip_id = int(sys.argv[1])
mip = mip_id - 1


if (mip_id == 1):
    title = [ '(a) SST bias of the initial 5-year (1948-1952) (ref. PCMDI)', '(b) SST bias of the last cycle 1980-2009 (ref. PCMDI)',
              '(c) SSS bias of the initial 5-year (1948-1952) (ref. WOA13v2)', '(d) SSS bias of the last cycle 1980-2009 (ref. WOA13v2)' ]
    suptitle = 'SST and SSS bias (OMIP1 of MRI.COM)'
    outfile = './fig/SST_SSS_bias_first_last_omip1'
else:
    title = [ '(a) SST bias of the initial 5 year (1958-1962) (ref. PCMDI)', '(b) SST bias of the last cycle 1980-2009 (ref. PCMDI)',
              '(c) SSS bias of the initial 5 year (1958-1962) (ref. WOA13v2)', '(d) SSS bias of the last cycle 1980-2009 (ref. WOA13v2)' ]
    suptitle = 'SST and SSS bias (OMIP2 of MRI.COM)'
    outfile = './fig/SST_SSS_bias_first_last_omip2'


#metainfo = [ json.load(open("./json/tos_omip1_mricom.json")), 
#             json.load(open("./json/tos_omip2_mricom.json")) ]


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

print( "Loading PCMDI-SST data" )
reffile = '../refdata/PCMDI-SST/tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
#J 年平均計算も行う。ただし日数の重みがつかないので不正確
DST0 = xr.open_dataset( reffile ).resample(time='1YS').mean()
#print(DS0)
#da0 = DS0.tos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)

mskfile = '../refdata/PCMDI-SST/sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
DS1 = xr.open_dataset( mskfile )
da1 = DS1.sftof

arefile = '../refdata/PCMDI-SST/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_th.1000'
DST0_woa = xr.open_dataset( reffile, decode_times=False )
dat0_woa = DST0_woa.thetao.sel(depth=0).isel(time=0)

# Salinity

print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_s.1000'
DSS0 = xr.open_dataset( reffile, decode_times=False )
das0 = DSS0.so.sel(depth=0).isel(time=0)

# model

if (mip_id == 1):
    path_in = '../model/MRI.COM/20200515/omip1'
    yr_start_1st = 1638
    yr_start_last = 1948
    yr_per_cycle = 62
else:
    path_in = '../model/MRI.COM/20200515/omip2'
    yr_start_1st = 1653
    yr_start_last = 1958
    yr_per_cycle = 61

# SST

basename_sst = 'hs_sst_woa1x1'

yr_st = yr_start_1st
yr_ed = yr_st + yr_per_cycle
sst_1st = np.array(np.empty((yr_per_cycle*12,ny,nx)),dtype=np.float64)
for yr in range(yr_st,yr_ed):
    suffix = '.' + '{:04}'.format(yr)
    sst_file = path_in + '/' + basename_sst + suffix
    ncsst = netCDF4.Dataset(sst_file,'r')
    lon = ncsst.variables['lon'][:]
    lat = ncsst.variables['lat'][:]
    sst_tmp = ncsst.variables['tos'][:,:,:]
    smon = (yr - yr_st)*12
    emon = (yr - yr_st + 1)*12
    sst_1st[smon:emon,:,:] = sst_tmp[0:12,:,:]

sst_1st = np.where(sst_1st < -9.0e33, np.NaN, sst_1st)
DS_sst1 = xr.Dataset( {'tos': (['time','lat','lon'], sst_1st),},
                      coords = { 'time': time[mip],'lat': lat,'lon': lon, } )
DS_sst1_ann = DS_sst1.resample(time='1YS').mean()

yr_st = yr_start_last
yr_ed = yr_st + yr_per_cycle
sst_last = np.array(np.empty((yr_per_cycle*12,ny,nx)),dtype=np.float64)
for yr in range(yr_st,yr_ed):
    suffix = '.' + '{:04}'.format(yr)
    sst_file = path_in + '/' + basename_sst + suffix
    ncsst = netCDF4.Dataset(sst_file,'r')
    lon = ncsst.variables['lon'][:]
    lat = ncsst.variables['lat'][:]
    sst_tmp = ncsst.variables['tos'][:,:,:]
    smon = (yr - yr_st)*12
    emon = (yr - yr_st + 1)*12
    sst_last[smon:emon,:,:] = sst_tmp[0:12,:,:]

sst_last = np.where(sst_last < -9.0e33, np.NaN, sst_last)
DS_sst2 = xr.Dataset( {'tos': (['time','lat','lon'], sst_last),},
                      coords = { 'time': time[mip],'lat': lat,'lon': lon, } )
DS_sst2_ann = DS_sst2.resample(time='1YS').mean()

ystr=yr_start_last
yend=yr_start_last+4
first_5yr_sst = DS_sst1_ann.tos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)
dat0_5yr = DST0.tos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)

ystr=1980
yend=2009
last_clim_sst = DS_sst2_ann.tos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)
dat0_last = DST0.tos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)

#------------------------------------------------------------------------------------
# SSS

basename_sss = 'hs_sss_woa1x1'

yr_st = yr_start_1st
yr_ed = yr_st + yr_per_cycle

sss_1st = np.array(np.empty((yr_per_cycle*12,ny,nx)),dtype=np.float64)
for yr in range(yr_st,yr_ed):
    suffix = '.' + '{:04}'.format(yr)
    sss_file = path_in + '/' + basename_sss + suffix
    ncsss = netCDF4.Dataset(sss_file,'r')
    lon = ncsss.variables['lon'][:]
    lat = ncsss.variables['lat'][:]
    sss_tmp = ncsss.variables['sos'][:,:,:]
    smon = (yr - yr_st)*12
    emon = (yr - yr_st + 1)*12
    sss_1st[smon:emon,:,:] = sss_tmp[0:12,:,:]

sss_1st = np.where(sss_1st < -9.0e33, np.NaN, sss_1st)
DS_sss1 = xr.Dataset( {'sos': (['time','lat','lon'], sss_1st),},
                      coords = { 'time': time[mip],'lat': lat,'lon': lon, } )
DS_sss1_ann = DS_sss1.resample(time='1YS').mean()

yr_st = yr_start_last
yr_ed = yr_st + yr_per_cycle

sss_last = np.array(np.empty((yr_per_cycle*12,ny,nx)),dtype=np.float64)
for yr in range(yr_st,yr_ed):
    suffix = '.' + '{:04}'.format(yr)
    sss_file = path_in + '/' + basename_sss + suffix
    ncsss = netCDF4.Dataset(sss_file,'r')
    lon = ncsss.variables['lon'][:]
    lat = ncsss.variables['lat'][:]
    sss_tmp = ncsss.variables['sos'][:,:,:]
    smon = (yr - yr_st)*12
    emon = (yr - yr_st + 1)*12
    sss_last[smon:emon,:,:] = sss_tmp[0:12,:,:]

sss_last = np.where(sss_last < -9.0e33, np.NaN, sss_last)
DS_sss2 = xr.Dataset( {'sos': (['time','lat','lon'], sss_last),},
                      coords = { 'time': time[mip],'lat': lat,'lon': lon, } )
DS_sss2_ann = DS_sss2.resample(time='1YS').mean()


ystr=yr_start_last
yend=yr_start_last+4
first_5yr_sss = DS_sss1_ann.sos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)

ystr=1980
yend=2009
last_clim_sss = DS_sss2_ann.sos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)

#DS = xr.Dataset( {'first_5yrs_sst': (['lat','lon'], first_5yr_sst - dat0_woa),
#                  'last_cycle_sst': (['lat','lon'], last_clim_sst - dat0_woa),
#                  'first_5yrs_sss': (['lat','lon'], first_5yr_sss - das0),
#                  'last_cycle_sss': (['lat','lon'], last_clim_sss - das0), },
#                 coords = { 'lat': lat, 'lon': lon, } )

DS = xr.Dataset( {'first_5yrs_sst': (['lat','lon'], first_5yr_sst - dat0_5yr),
                  'last_cycle_sst': (['lat','lon'], last_clim_sst - dat0_last),
                  'first_5yrs_sss': (['lat','lon'], first_5yr_sss - das0),
                  'last_cycle_sss': (['lat','lon'], last_clim_sss - das0), },
                 coords = { 'lat': lat, 'lon': lon, } )

#------------------

da1 = DS['first_5yrs_sst']
da2 = DS['last_cycle_sst']

msktmp = np.where( np.isnan(da1.values), 0.0, 1.0 )
da1tmp = np.where( np.isnan(da1.values), 0.0, da1.values )
da2tmp = np.where( np.isnan(da2.values), 0.0, da2.values )

wgt = msktmp * area

wgt_sum = (wgt).sum()

da1_sum = (da1tmp * wgt).sum()
da1_mean = da1_sum / wgt_sum

da2_sum = (da2tmp*wgt).sum()
da2_mean = da2_sum/wgt_sum
print('Mean (SST)', da1_mean, da2_mean)

var1_sum = ((da1tmp - da1_mean)**2 * wgt).sum()
var2_sum = ((da2tmp - da2_mean)**2 * wgt).sum()

var1 = var1_sum / wgt_sum
var2 = var2_sum / wgt_sum

std1 = np.sqrt(var1)
std2 = np.sqrt(var2)
print('Standard deviation (SST) ', std1, std2)

corr_sum = ((da1tmp-da1_mean)*(da2tmp-da2_mean) * wgt).sum()
corr = corr_sum / wgt_sum / std1 / std2
print('Correlation (SST)', corr)

#------------------

da1 = DS['first_5yrs_sss']
da2 = DS['last_cycle_sss']

msktmp = np.where( np.isnan(da1.values), 0.0, 1.0 )
da1tmp = np.where( np.isnan(da1.values), 0.0, da1.values )
da2tmp = np.where( np.isnan(da2.values), 0.0, da2.values )

wgt = msktmp * area

wgt_sum = (wgt).sum()

da1_sum = (da1tmp * wgt).sum()
da1_mean = da1_sum / wgt_sum

da2_sum = (da2tmp * wgt).sum()
da2_mean = da2_sum / wgt_sum
print('Mean (SSS)', da1_mean, da2_mean)

var1_sum = ((da1tmp - da1_mean)**2 * wgt).sum()
var2_sum = ((da2tmp - da2_mean)**2 * wgt).sum()

var1 = var1_sum / wgt_sum
var2 = var2_sum / wgt_sum

std1 = np.sqrt(var1)
std2 = np.sqrt(var2)
print('Standard deviation (SSS) ', std1, std2)

corr_sum = ((da1tmp-da1_mean)*(da2tmp-da2_mean) * wgt).sum()
corr = corr_sum / wgt_sum / std1 / std2
print('Correlation (SSS)', corr)

#------------------

print( 'Start drawing' )
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(2,2,1,projection=proj),
    plt.subplot(2,2,2,projection=proj),
    plt.subplot(2,2,3,projection=proj),
    plt.subplot(2,2,4,projection=proj),
]

# [left, bottom, width, height]
#ax_cbar = [
#    plt.axes([0.93,0.64,0.012,0.23]),
#    plt.axes([0.93,0.37,0.012,0.23]),
#    plt.axes([0.93,0.10,0.012,0.23]),
#]

bounds1 = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0, 1.5, 2.0]
bounds2 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]

cmap = [ 'RdBu_r', 'RdBu_r', 'bwr', 'bwr' ]

item = [ 'first_5yrs_sst', 'last_cycle_sst', 'first_5yrs_sss', 'last_cycle_sss' ]

ddof_dic={'ddof' : 0}

for panel in range(4):
    print("processing panel ", panel)
    if (item[panel] == 'first_5yrs_sst' or item[panel] == 'last_cycle_sst'):
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS[item[panel]]
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp4 = (datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        tmp3 = (area).sum()
        print(tmp1,tmp2,tmp3)
        rmse = np.sqrt(tmp1/tmp2)
        mean_bias = tmp4/tmp2
        title[panel] = title[panel] + '\n' \
            + ' mean bias = ' + '{:.3f}'.format(mean_bias) + '$^\circ$C,' + '    bias rmse = ' + '{:.3f}'.format(rmse) + '$^\circ$C'
        cblab='[$^\circ$C]'
        print(title[panel])
    else:
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS[item[panel]]
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp4 = (datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        mean_bias = tmp4/tmp2
        cblab='[psu]'
        title[panel] = title[panel] + '\n' \
            + ' mean bias = ' + '{:.3f}'.format(mean_bias) + 'psu,' + '    bias rmse = ' + '{:.3f}'.format(rmse) + 'psu'
        print(title[panel])


    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs={'orientation': 'horizontal',
                         'spacing':'uniform',
                         'label': cblab,
                         'ticks': ticks_bounds,
                         'shrink': 0.9, },
            transform=ccrs.PlateCarree())

    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_xlabel('')
    ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top'})
    ax[panel].tick_params(labelsize=9)
    ax[panel].background_patch.set_facecolor('lightgray')

print("...Done drawing")
plt.subplots_adjust(left=0.07,right=0.95,bottom=0.02,top=0.90,wspace=0.16,hspace=0.15)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05, dpi=200)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05, dpi=200)
#plt.savefig(outpng, bbox_inches='tight', pad_inches=0.1)
#plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.1)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()
