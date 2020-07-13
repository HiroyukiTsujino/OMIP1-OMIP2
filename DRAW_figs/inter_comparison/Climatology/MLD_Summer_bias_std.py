# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime
import netCDF4
from numba import jit


@jit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],i4,i4,i4,i4)', nopython=True)
def onedeg2twodeg(d_two, d_one, am_one, nx, ny, nxm, nym):

    d_two[ny-1,0] = am_one[2*ny-1,0] * d_one[2*ny-1,0] \
                  + am_one[2*ny-1,nxm-1] * d_one[2*ny-1,nxm-1]
    for i in range(1,nx):
        d_two[ny-1,i] = am_one[2*ny-1,2*i-1] * d_one[2*ny-1,2*i-1] \
                      + am_one[2*ny-1,2*i] * d_one[2*ny-1,2*i]
    for j in range(0,ny-1):
        d_two[j,0] = am_one[2*j+1,0]     * d_one[2*j+1,0] \
                   + am_one[2*j+1,nxm-1] * d_one[2*j+1,nxm-1] \
                   + am_one[2*j+2,0]     * d_one[2*j+2,0] \
                   + am_one[2*j+2,nxm-1] * d_one[2*j+2,nxm-1] 
        for i in range(1,nx):
            d_two[j,i] = am_one[2*j+1,2*i-1] * d_one[2*j+1,2*i-1] \
                       + am_one[2*j+2,2*i-1] * d_one[2*j+2,2*i-1] \
                       + am_one[2*j+1,2*i] * d_one[2*j+1,2*i] \
                       + am_one[2*j+2,2*i] * d_one[2*j+2,2*i] 
    return d_two
            
#--------------------

if (len(sys.argv) < 5):
    print ('Usage: '+ sys.argv[0] + ' [MMM or modelname] start_year end_year exLab(0 or 1) [show (to check using viewer)]')
    sys.exit()

#if (len(sys.argv) < 2) :
#    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
#    sys.exit()

stclyr = int(sys.argv[2])
edclyr = int(sys.argv[3])
exlab = int(sys.argv[4])

nyrcl = edclyr - stclyr + 1

#----------------------------------------------------------------

title = [ r'(a) Ensemble bias (OMIP1 - de Boyer Mont$\mathrm{\acute{e}}$gut)', r'(b) Ensemble bias (OMIP2 - de Boyer Mont$\mathrm{\acute{e}}$gut)',
          '(c) Ensemble STD OMIP1', '(d) Ensemble STD OMIP2',
          '(e) OMIP2 - OMIP1', r'(f) de Boyer Mont$\mathrm{\acute{e}}$gut' ]

metainfo = [ json.load(open("./json/mld_season_omip1.json")), 
             json.load(open("./json/mld_season_omip2.json")) ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

if (sys.argv[1] == 'MMM'):
    suptitle = 'Multi Model Mean' + ' Summer MLD, JAS (NH), JFM (SH) (ave. from 1980 to 2009)'
    outfile = './fig/MLD_Summer_bias_MMM'
else:
    suptitle = sys.argv[1] + ' (Summer MLD, JAS (NH), JFM (SH) ave. from 1980 to 2009)'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/MLD_Summer_bias_' + sys.argv[1]


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
reffile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_sumclim.nc'
mskfile = '../analysis/MLD/MLD_deBoyer_Montegut/mld_DR003_summask.nc'

# Obs
ncobsann = netCDF4.Dataset(reffile,'r')
nx = len(ncobsann.dimensions['lon'])
ny = len(ncobsann.dimensions['lat'])
mldobs_ann = ncobsann.variables['mlotst'][:,:]
miss_val_obsann = ncobsann.variables['mlotst'].missing_value
lon_ = ncobsann.variables['lon'][:]
lat_ = ncobsann.variables['lat'][:]

print (nx, ny, miss_val_obsann)
dtwoout = np.array(np.empty((ny,nx)),dtype=np.float64)

# mask
ncmskobs = netCDF4.Dataset(mskfile,'r')
maskobs = ncmskobs.variables['mldmask'][:,:]
ncmskobs.close()

# for ad hoc exclusion of the Weddell Sea

maskdeep = np.array(np.ones((ny,nx)),dtype=np.float64)

print("Warning: Weddell Sea and Labrador Sea will not be masked for summer")
#for j in range(0,ny):
#    for i in range (0,nx):
#        if (lat_[j] < -60.0 and lon_[i] > 300.0):
#            maskdeep[j,i] = 0.0
#if (exlab == 1):
#    for j in range(0,ny):
#        for i in range (0,nx):
#            if (lat_[j] > 45.0 and lat_[j] < 80.0):
#                if (lon_[i] > 280.0 or lon_[i] < 30.0):
#                    maskdeep[j,i] = 0.0
#
#----------------------------------------------
# cell area (1x1)

path_amip = '../refdata/PCMDI-SST'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nxm = len(ncare.dimensions['lon'])
nym = len(ncare.dimensions['lat'])
aream = ncare.variables['areacello'][:,:]
ncare.close()
        

# cell area (2x2)

area = np.array(np.empty((ny,nx)),dtype=np.float64)

area[ny-1,0] = aream[2*ny-1,0] + aream[2*ny-1,nxm-1]
for i in range(1,nx):
    area[ny-1,i] = aream[2*ny-1,2*i-1] + aream[2*ny-1,2*i]

for j in range(0,ny-1):
    area[j,0] = aream[2*j+1,0] + aream[2*j+1,nxm-1] \
              + aream[2*j+2,0] + aream[2*j+2,nxm-1]
    for i in range(1,nx):
        area[j,i] = aream[2*j+1,2*i-1] + aream[2*j+2,2*i-1] \
                  + aream[2*j+1,2*i]   + aream[2*j+2,2*i]

#----------------------------------------------

data = []
for omip in range(2):
    if (omip == 0):
        styr=1948
        edyr=2009
    else:
        styr=1958
        edyr=2018

    nyr = edyr - styr + 1
    print( "Loading OMIP" + str(omip+1) + " data" )

    d = np.empty( (len(model_list[omip]),ny,nx) )

    nmodel = 0
    for model in model_list[omip]:

        path = metainfo[omip][model]['path']
        fdname = metainfo[omip][model]['fsumname']
        fmname = metainfo[omip][model]['fmskname']
        datafile = path + '/' + fdname
        maskfile = path + '/' + fmname
        #DS = xr.open_dataset( datafile )

        print (' ')
        print ('Processing ', model)

        print(datafile)
        ncomipann = netCDF4.Dataset(datafile,'r')
        mldomip_ann = ncomipann.variables['mlotst'][:,:,:]
        miss_val_omipann = ncomipann.variables['mlotst'].missing_value
        mldomip_ann = np.where(np.isnan(mldomip_ann),0.0,mldomip_ann)

        ncmskomip = netCDF4.Dataset(maskfile,'r')
        maskomip = ncmskomip.variables['mldmask'][:,:]
        ncmskomip.close()

        areamask_tmp = maskomip * aream
        areamask = areamask_tmp.astype(np.float64)

        amsk_all = np.array(np.empty((ny,nx)),dtype=np.float64)
        mask_all = np.array(np.empty((ny,nx)),dtype=np.float64)
        mldannm= np.array(np.empty((nyrcl,ny,nx)),dtype=np.float64)

        amsk_all[ny-1,0] = areamask[2*ny-1,0] + areamask[2*ny-1,nxm-1]
        for j in range(0,ny-1):
            amsk_all[j,0] = areamask[2*j+1,0] + areamask[2*j+1,nxm-1] \
                          + areamask[2*j+2,0] + areamask[2*j+2,nxm-1]
            for i in range(1,nx):
                amsk_all[j,i] = areamask[2*j+1,2*i-1] + areamask[2*j+2,2*i-1] \
                              + areamask[2*j+1,2*i] + areamask[2*j+2,2*i]
        for i in range(1,nx):
            amsk_all[ny-1,i] = areamask[2*ny-1,2*i-1] + areamask[2*ny-1,2*i]

        mask_all = np.where(amsk_all > 0.0, 1.0, 0.0)

        nn = 0
        nm = 0
        for nyear in range(styr,edyr+1):
            if (nyear >= stclyr and nyear <= edclyr):
                donein = mldomip_ann[nn,:,:]
                dtwoout[:,:] = 0.0
                onedeg2twodeg(dtwoout, donein, areamask, nx, ny, nxm, nym)
                mldannm[nm,:,:] = dtwoout
                nm += 1
            nn += 1

        mldannm[0:nyrcl,:,:] = mldannm[0:nyrcl,:,:] / (1.0 - mask_all[:,:] + amsk_all[:,:]) 
        mldannm[0:nyrcl,:,:] = np.where(mask_all[:,:] == 0.0, np.NaN, mldannm[0:nyrcl,:,:])
        
        d[nmodel] = mldannm.mean(axis=0)
        nmodel += 1
    
        del amsk_all
        del mask_all
        del mldannm

    data += [d]
    del d

#sys.exit()

DS = xr.Dataset( {'omip1mean': (['model','lat','lon'], data[0]),
                  'omip2mean': (['model','lat','lon'], data[1]),
                  'omip2-1': (['model','lat','lon'], data[1] - data[0]),
                  'obs': (['lat','lon'], mldobs_ann),},
                 coords = { 'lat': lat_ , 'lon': lon_ , } )

#J 描画
print( 'Start drawing' )
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

# [left, bottom, width, height]
#ax_cbar = [
#    plt.axes([0.93,0.64,0.012,0.23]),
#    plt.axes([0.93,0.37,0.012,0.23]),
#    plt.axes([0.93,0.10,0.012,0.23]),
#]

bounds1 = np.arange(0,160,10)
ticks_bounds1 = np.arange(0,160,50)
bounds2 = np.arange(-30,35,5)
ticks_bounds2 = np.arange(-30,35,10)
bounds3 = np.arange(0,55,5)
ticks_bounds3 = np.arange(0,55,5)

cmap = [ 'RdBu_r', 'RdBu_r', 'terrain', 'terrain', 'bwr', 'RdYlBu_r' ]
extflg = [ 'both', 'both', 'both', 'both', 'both', 'max' ]

item = [ 'omip1bias', 'omip2bias', 'omip1std', 'omip2std', 'omip2-1', 'obs' ]

boxdic = {"facecolor" : "white",
          "edgecolor" : "black",
          "linewidth" : 1
          }

ddof_dic={'ddof' : 0}

for panel in range(6):
    if (item[panel] == 'omip1bias'):
        bounds = bounds2
        ticks_bounds = ticks_bounds2
        da = DS['omip1mean'].mean(dim='model',skipna=True)
        daobs = DS['obs']
        dadraw = da - daobs
        datmp = da.values - daobs.values
        msktmp = np.where( np.isnan(datmp), 0.0, 1.0 )
        datmp = np.where( np.isnan(datmp), 0.0, datmp )
        tmp1 = (datmp * datmp * area * msktmp * maskdeep).sum()
        tmp4 = (datmp * area * msktmp * maskdeep).sum()
        tmp2 = (area * msktmp * maskdeep).sum()
        tmp3 = (area).sum()
        print('OMIP-1',tmp1,tmp2,tmp3)
        rmse = np.sqrt(tmp1/tmp2)
        mean_bias = tmp4/tmp2
        title[panel] = title[panel] + '\n' \
            + ' mean bias = ' + '{:.1f}'.format(mean_bias) + ' m,' + '    bias rmse = ' + '{:.1f}'.format(rmse) + ' m'
        print(title[panel])
    elif (item[panel] == 'omip2bias'):
        bounds = bounds2
        ticks_bounds = ticks_bounds2
        da = DS['omip2mean'].mean(dim='model',skipna=True)
        daobs = DS['obs']
        dadraw = da - daobs
        datmp = da.values - daobs.values
        msktmp = np.where( np.isnan(datmp), 0.0, 1.0 )
        datmp = np.where( np.isnan(datmp), 0.0, datmp )
        tmp1 = (datmp * datmp * area * msktmp * maskdeep).sum()
        tmp4 = (datmp * area * msktmp * maskdeep).sum()
        tmp2 = (area * msktmp * maskdeep).sum()
        print('OMIP-2',tmp1,tmp2)
        rmse = np.sqrt(tmp1/tmp2)
        mean_bias = tmp4/tmp2
        title[panel] = title[panel] + '\n' \
            + ' mean bias = ' + '{:.1f}'.format(mean_bias) + ' m,' + '    bias rmse = ' + '{:.1f}'.format(rmse) + ' m'
        print(title[panel])
    elif (item[panel] == 'omip1std'):
        bounds = bounds3
        ticks_bounds = bounds3
        dadraw = DS['omip1mean'].std(dim='model',skipna=True, **ddof_dic)
        tmp = DS['omip1mean'].var(dim='model', skipna=True, **ddof_dic)
        msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
        tmp1 = (datmp * area * msktmp * maskdeep).sum()
        tmp2 = (area * msktmp * maskdeep).sum()
        rmse = np.sqrt(tmp1/tmp2)
        z = np.abs(DS['omip1mean'].mean(dim='model',skipna=False) - DS['obs']) - 2.0 * dadraw
        z = np.where( z > 0, 1, 0 )
        tmp3 = (z * area * msktmp).sum()
        failcapt=tmp3/tmp2*100
        title[panel] = title[panel] + r' 2$\bar{\sigma}$=' + '{:.1f}'.format(2*rmse) + ' m ' + '\n' \
            + 'observation uncaptured by model spread = ' + '{:.1f}'.format(failcapt) + '%'
        print(title[panel])
    elif (item[panel] == 'omip2std'):
        bounds = bounds3
        ticks_bounds = bounds3
        dadraw = DS['omip2mean'].std(dim='model',skipna=True, **ddof_dic)
        tmp = DS['omip2mean'].var(dim='model', skipna=True, **ddof_dic)
        msktmp = np.where( np.isnan(tmp.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(tmp.values), 0.0, tmp.values )
        tmp1 = (datmp * area * msktmp * maskdeep).sum()
        tmp2 = (area * msktmp * maskdeep).sum()
        rmse = np.sqrt(tmp1/tmp2)
        z = np.abs(DS['omip2mean'].mean(dim='model',skipna=False) - DS['obs']) - 2.0 * dadraw
        z = np.where( z > 0, 1, 0 )
        tmp3 = (z * area * msktmp).sum()
        failcapt=tmp3/tmp2*100
        title[panel] = title[panel] + r' 2$\bar{\sigma}$=' + '{:.1f}'.format(2*rmse) + ' m ' + '\n' \
            + 'observation uncaptured by model spread = ' + '{:.1f}'.format(failcapt) + '%'
        print(title[panel])
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = ticks_bounds2
        dadraw = DS[item[panel]].mean(dim='model',skipna=True)
        msktmp = np.where( np.isnan(dadraw.values), 0.0, 1.0 )
        datmp = np.where( np.isnan(dadraw.values), 0.0, dadraw.values )
        tmp1 = (datmp * datmp * area * msktmp * maskdeep).sum()
        tmp2 = (area * msktmp * maskdeep).sum()
        rmsd = np.sqrt(tmp1/tmp2)
        title[panel] = title[panel] + '     rmsd= ' + '{:.1f}'.format(rmsd) + ' m'
        print(title[panel])
    else:
        bounds = bounds1
        ticks_bounds = ticks_bounds1
        dadraw = DS['obs']


    dadraw.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            extend=extflg[panel],
            cbar_kwargs={'orientation': 'vertical',
                         'spacing':'uniform',
                         'label': '[m]',
                         'ticks': ticks_bounds,},
            transform=ccrs.PlateCarree())

    if (panel == 2):
        mpl.rcParams['hatch.color'] = 'red'
        mpl.rcParams['hatch.linewidth'] = 0.5
        x = DS["lon"].values
        y = DS["lat"].values
        z = np.abs(DS["omip1mean"].mean(dim='model',skipna=False) - DS["obs"]) - 2.0 * DS['omip1mean'].std(dim='model',skipna=False, **ddof_dic)
        z = np.where( z > 0, 1, np.nan )
        ax[panel].contourf(x,y,z*maskdeep,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
    if (panel == 3):
        mpl.rcParams['hatch.color'] = 'red'
        mpl.rcParams['hatch.linewidth'] = 0.5
        x = DS["lon"].values
        y = DS["lat"].values
        z = np.abs(DS["omip2mean"].mean(dim='model',skipna=False) - DS["obs"]) - 2.0 * DS['omip2mean'].std(dim='model',skipna=False, **ddof_dic)
        z = np.where( z > 0, 1, np.nan )
        ax[panel].contourf(x,y,z*maskdeep,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
        
    #if (panel == 4):
    #    mpl.rcParams['hatch.color'] = 'limegreen'
    #    mpl.rcParams['hatch.linewidth'] = 0.5
    #    x = DS_stats["lon"].values
    #    y = DS_stats["lat"].values
    #    z = np.abs(DS_stats["mean"]) - factor_5ptail * DS_stats["std"]
    #    z = np.where( z > 0, 1, np.nan )
    #    ax[panel].contourf(x,y,z,hatches=['xxxxxxx'],colors='none',transform=ccrs.PlateCarree())
        
    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_xlabel('')
    ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top', 'linespacing':0.9})
    ax[panel].tick_params(labelsize=9)
    ax[panel].background_patch.set_facecolor('lightgray')

plt.subplots_adjust(left=0.07,right=0.98,bottom=0.05,top=0.92,wspace=0.16,hspace=0.15)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.05, dpi=200)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.05, dpi=200)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 6 and sys.argv[5] == 'show') :
    plt.show()
