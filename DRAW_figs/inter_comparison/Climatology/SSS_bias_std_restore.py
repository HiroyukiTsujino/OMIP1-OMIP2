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
from numba import jit


#@jit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],i4,i4,i4,i4)', nopython=True)



if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()

title = [ '(a) Ensemble bias (OMIP1 - WOA13v2)', '(b) Ensemble bias (OMIP2 - WOA13v2)',
          '(c) Ensemble STD (OMIP1 - WOA13v2)', '(d) Ensemble STD (OMIP2 - WOA13v2)',
          '(e) OMIP2 - OMIP1', '(f) OMIP2 - OMIP1 (Salinity used for SSS restoring)' ]

metainfo = [ json.load(open("./json/sos_omip1.json")), 
             json.load(open("./json/sos_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


ystr = 1980
yend = 2009
nyr = yend - ystr + 1
factor_5ptail = 1.64  # 5-95%

if (sys.argv[1] == 'MMM'):
    suptitle = 'Multi Model Mean' + ' SSS (ave. from '+str(ystr)+' to '+str(yend)+')'
    outfile = './fig/SSS_bias_MMM'
else:
    suptitle = sys.argv[1] + ' SSS (ave. from '+str(ystr)+' to '+str(yend)+')'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/SSS_bias_' + sys.argv[1]

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

print( "Loading WOA13v2 data" )
reffile = '../refdata/WOA13v2/1deg_L33/annual/woa13_decav_s.1000'
DS0 = xr.open_dataset( reffile, decode_times=False )
da0 = DS0.so.sel(depth=0).isel(time=0)

arefile = '../refdata/PCMDI-SST/areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
area = ncare.variables['areacello'][:,:]
ncare.close()

# OMIP1-SSS

sss1file = '../refdata/OMIP1-SSS/PHC2_salx.2004_08_03.nc'
print("OMIP1-SSS from "+sss1file)
ncsss1 = netCDF4.Dataset(sss1file,'r')
nxs1 = len(ncsss1.dimensions['lon'])
nys1 = len(ncsss1.dimensions['lat'])
sss1 = ncsss1.variables['SALT'][:,:,:]
ncsss1.close()
sss1[:,:,:] = np.where(sss1[:,:,:] < -98.0, np.NaN, sss1[:,:,:])

sss1a = sss1.mean(axis=0)

# OMIP2-SSS

sss2file = '../refdata/OMIP2-SSS/sos_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-4-0_gr_195501-201212-clim.nc'
print("OMIP2-SSS from "+sss2file)
ncsss2 = netCDF4.Dataset(sss2file,'r')
nxs2 = len(ncsss2.dimensions['lon'])
nys2 = len(ncsss2.dimensions['lat'])
latsss2 = ncsss2.variables['lat'][:]
sss2 = ncsss2.variables['sos'][:,:,:]
ncsss2.close()

wgt2 = np.array(np.empty((nys2,nxs2)),dtype=np.float64)
for j in range(0,nys2):
    for i in range(0,nxs2):
        wgt2[j,i] = np.cos(np.radians(latsss2[j]))
    #print(j,wgt2[j,0])

sss_o2 = np.array(np.empty((12,nys1,nxs1)),dtype=np.float64)
are_o2 = np.array(np.empty((nys1,nxs1)),dtype=np.float64)
sss_o2[:,:,:] = 0.0

for mn in range(0,12):
    are_o2[:,:] = 0.0
    for j in range(0,nys1):
        for i in range(0,nxs1):
            for jj in range(0,4):
                jjj = j * 4 + jj
                for ii in range(0,4):
                    iii = i * 4 + ii
                    sss_o2[mn,j,i] = sss_o2[mn,j,i] + wgt2[jjj,iii] * sss2[mn,jjj,iii]
                    are_o2[j,i] = are_o2[j,i] + wgt2[jjj,iii]

    for j in range(0,nys1):
        for i in range(0,nxs1):
            sss_o2[mn,j,i] = sss_o2[mn,j,i] / are_o2[j,i]

sss2a = sss_o2.mean(axis=0)

print(".....done")

# uncertainty of difference between omip-1 and omip-2

stdfile = '../analysis/STDs/SSS_omip1-omip2_stats.nc'
DS_stats = xr.open_dataset( stdfile )

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
        if (model == 'Kiel-NEMO'):
            DS = DS.where(DS['sos'] != 0.0)

        DS['time'] = time[omip]

        tmp = DS.sos.sel(time=slice(str(ystr),str(yend))).mean(dim='time',skipna=False)

        if model == "NorESM-BLOM":
            tmp = tmp.assign_coords(lon=('x', np.where( tmp.lon < 0, tmp.lon + 360, tmp.lon )))
            tmp = tmp.roll(x=-180, roll_coords=True)
        if model == "MIROC-COCO4.9":
            tmp = tmp.sel(lat=slice(None, None, -1))

        d[nmodel] = tmp.values
        nmodel += 1

    data += [d]


DS = xr.Dataset( {'omip1bias': (['model','lat','lon'], data[0] - da0.values),
                  'omip2bias': (['model','lat','lon'], data[1] - da0.values),
                  'omip2-1': (['model','lat','lon'], data[1] - data[0]),
                  'omip2-1_restore': (['lat','lon'], sss2a - sss1a),
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
    plt.subplot(3,2,6,projection=proj),
]

bounds1 = [-1.0, -0.7, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.7, 1.0]
bounds2 = [-1.0, -0.4, -0.3, -0.2, -0.1, -0.06, -0.02, 0.02, 0.06, 0.1, 0.2, 0.3, 0.4, 1.0]
bounds3 = [30, 31, 32, 33.0, 33.6, 34.0, 34.3, 34.6, 34.9, 35.2, 35.5, 35.8, 36.1, 36.5, 36.9, 37.3 ]
bounds4 = [0.05, 0.10, 0.15, 0.2, 0.4, 0.7, 1.0, 1.5]
ticks_bounds4 = [0.0, 0.4, 1.0, 1.5] 

#cmap = [ 'RdBu_r', 'RdBu_r', 'YlGnBu_r', 'YlGnBu_r', 'RdBu_r', 'RdBu_r' ]
cmap = [ 'RdBu_r', 'RdBu_r', 'terrain', 'terrain', 'bwr', 'bwr' ]

item = [ 'omip1bias', 'omip2bias', 'omip1std', 'omip2std', 'omip2-1', 'omip2-1_restore' ]

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
        tmp4 = (datmp * area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        mean_bias = tmp4/tmp2
        title[panel] = title[panel] + '\n' \
            + ' mean bias = ' + '{:.3f}'.format(mean_bias) + 'psu,' + '    bias rmse = ' + '{:.3f}'.format(rmse) + 'psu'
    elif (item[panel] == 'omip2bias'):
        bounds = bounds1
        ticks_bounds = bounds1
        da = DS['omip2bias'].mean(dim='model',skipna=False)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        tmp4 = (datmp * area * msktmp).sum()
        rmse = np.sqrt(tmp1/tmp2)
        mean_bias = tmp4/tmp2
        title[panel] = title[panel] + '\n' \
            + ' mean bias = ' + '{:.3f}'.format(mean_bias) + 'psu,' + '    bias rmse = ' + '{:.3f}'.format(rmse) + 'psu'
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
        title[panel] = title[panel] + r' 2$\bar{\sigma}$=' + '{:.2f}'.format(2*rmse) + ' psu ' + '\n' \
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
        title[panel] = title[panel] + r' 2$\bar{\sigma}$=' + '{:.2f}'.format(2*rmse) + ' psu ' + '\n' \
            + 'observation uncaptured by model spread = ' + '{:.1f}'.format(failcapt) + '%'
        print(title[panel])
    elif (item[panel] == 'omip2-1'):
        bounds = bounds2
        ticks_bounds = bounds2
        da = DS[item[panel]].mean(dim='model',skipna=False)
        msktmp = np.where( np.isnan(da.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(da.values), 0.0, da.values )
        tmp1 = (datmp * datmp * area * msktmp).sum()
        tmp2 = (area * msktmp).sum()
        rmsd = np.sqrt(tmp1/tmp2)
        title[panel] = title[panel] + '     rmsd= ' + '{:.2f}'.format(rmsd) + ' psu '
        print(title[panel])
    else:
        bounds = bounds2
        ticks_bounds = bounds2
        da = DS[item[panel]]

    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend='both',
            cbar_kwargs={'orientation': 'vertical',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'label': "",
                         'ticks': ticks_bounds,},
            transform=ccrs.PlateCarree())

    if (panel == 2):
        mpl.rcParams['hatch.color'] = 'red'
        mpl.rcParams['hatch.linewidth'] = 0.5
        x = DS["lon"].values
        y = DS["lat"].values
        z = np.abs(DS["omip1bias"].mean(dim='model',skipna=False)) - 2.0 * DS['omip1bias'].std(dim='model',skipna=False, **ddof_dic)
        z = np.where( z > 0, 1, np.nan )
        ax[panel].contourf(x,y,z,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
    if (panel == 3):
        mpl.rcParams['hatch.color'] = 'red'
        mpl.rcParams['hatch.linewidth'] = 0.5
        x = DS["lon"].values
        y = DS["lat"].values
        z = np.abs(DS["omip2bias"].mean(dim='model',skipna=False)) - 2.0 * DS['omip2bias'].std(dim='model',skipna=False, **ddof_dic)
        z = np.where( z > 0, 1, np.nan )
        ax[panel].contourf(x,y,z,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
        
    if (panel == 4):
        mpl.rcParams['hatch.color'] = 'lime'
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
    ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top', 'linespacing':0.9})
    ax[panel].tick_params(labelsize=9)
    ax[panel].background_patch.set_facecolor('lightgray')

plt.subplots_adjust(left=0.07,right=0.98,bottom=0.05,top=0.92,wspace=0.16,hspace=0.17)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05, dpi=200)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05, dpi=200)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
