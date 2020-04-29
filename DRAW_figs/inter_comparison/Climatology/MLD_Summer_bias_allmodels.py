# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
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
            

if (len(sys.argv) < 5):
    print ('Usage: '+ sys.argv[0] + ' [OMIP1 (1) or OMIP2 (2) or OMIP2-OMIP1 (3)] start_year end_year exLab(0 or 1) [show (to check using viewer)]')
    sys.exit()

omip_out = int(sys.argv[1])
nv_out = int(sys.argv[1]) - 1
stclyr = int(sys.argv[2])
edclyr = int(sys.argv[3])
exlab = int(sys.argv[4])

nyrcl = edclyr - stclyr + 1

head_title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1' ]

metainfo = [ json.load(open("./json/mld_season_omip1.json")), 
             json.load(open("./json/mld_season_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

suptitle = head_title[nv_out] + ' Summer MLD, JAS (NH), JFM (SH) (ave. from 1980 to 2009)'

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

dtwoout = np.array(np.empty((ny,nx)),dtype=np.float64)

# mask
ncmskobs = netCDF4.Dataset(mskfile,'r')
maskobs = ncmskobs.variables['mldmask'][:,:]
ncmskobs.close()

# ad hoc exclusion of the Weddell Sea

maskdeep = np.array(np.ones((ny,nx)),dtype=np.float64)

print("Warning: Weddell Sea and Labrador Sea will not be masked for summer")

#for j in range(0,ny):
#    for i in range (0,nx):
#        if (lat_[j] < -60.0 and lon_[i] > 300.0):
#            maskdeep[j,i] = 0.0
#
#if (exlab == 1):
#    for j in range(0,ny):
#        for i in range (0,nx):
#            if (lat_[j] > 45.0 and lat_[j] < 80.0):
#                if (lon_[i] > 280.0 or lon_[i] < 30.0):
#                    maskdeep[j,i] = 0.0

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

#-----------------------------------------------------

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
                mldannm[nm,:,:] = dtwoout.copy()
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
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

# [left, bottom, width, height]
axes0 = np.array( [ [0.05, 0.840, 0.3, 0.085],
                    [0.05, 0.755, 0.3, 0.085],])

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

bounds1 = np.arange(0,160,10)
ticks_bounds1 = np.arange(0,160,50)
#bounds1 = np.arange(-100,110,10)
#ticks_bounds1 = np.arange(-100,110,20)
bounds2 = np.arange(-30,35,5)
ticks_bounds2 = np.arange(-30,35,10)
#bounds2 = np.arange(-50,60,5)
#ticks_bounds2 = np.arange(-50,60,10)

cmap = [ 'RdYlBu_r', 'RdYlBu_r', 'bwr', 'RdYlBu_r' ]

item = [ 'omip1bias', 'omip2bias', 'omip2-1', 'deBoyer' ]

outfile = './fig/MLD_Summer_bias_allmodels_'+item[nv_out]+'.png'

extflg = [ 'max', 'max', 'both', 'max' ]

boxdic = {"facecolor" : "white",
          "edgecolor" : "black",
          "linewidth" : 1
          }

nax = 11
model = 'MMM'
dict_rmse={}
if (item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias'):
    bounds = bounds1
    ticks_bounds = ticks_bounds1
    if (item[nv_out] == 'omip1bias'):
        da = DS['omip1mean'].mean(dim='model',skipna=True)
    else:
        da = DS['omip2mean'].mean(dim='model',skipna=True)
    daobs = DS['obs']
    #dadraw = da - daobs
    dadraw = da
    datmp = da.values - daobs.values
    #damtmp = da.values
    msktmp = np.where( np.isnan(datmp), 0.0, 1.0 )
    datmp = np.where( np.isnan(datmp), 0.0, datmp )
    #damtmp = np.where( np.isnan(damtmp), 0.0, damtmp )
    tmp1 = (datmp * datmp * area * msktmp * maskdeep).sum()
    tmp2 = (area * msktmp * maskdeep).sum()
    tmp3 = (datmp * area * msktmp * maskdeep).sum()
    tmp4 = (area).sum()
    print('MMM',tmp1,tmp2,tmp4)
    rmse = np.sqrt(tmp1/tmp2)
    mldm = tmp3/tmp2
    title_panel = model + '\n' \
            + ' mean bias = ' + '{:.1f}'.format(mldm) + ' m,' + '    bias rmse = ' + '{:.1f}'.format(rmse) + ' m'
    dict_rmse['MMM']=[rmse,mldm]
else:
    dadraw = DS[item[nv_out]].mean(dim='model',skipna=True)
    bounds = bounds2
    ticks_bounds = ticks_bounds2
    title_panel = model

dadraw.plot(ax=ax[nax],cmap=cmap[nv_out],
        levels=bounds,
        extend=extflg[nv_out],
        cbar_kwargs={'orientation': 'horizontal',
                     'spacing':'uniform',
                     'label': '[m]',
                     'ticks': ticks_bounds,},
        cbar_ax = ax_cbar,
        transform=ccrs.PlateCarree())

ax[nax].coastlines()
ax[nax].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
ax[nax].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
ax[nax].xaxis.set_major_formatter(lon_formatter)
ax[nax].yaxis.set_major_formatter(lat_formatter)
ax[nax].set_xlabel('')
ax[nax].set_ylabel('')
ax[nax].set_title(title_panel,{'fontsize':8, 'verticalalignment':'top', 'linespacing':0.8})
ax[nax].tick_params(labelsize=8)
ax[nax].background_patch.set_facecolor('lightgray')


nmodel = 0
for model in model_list[0]:
    if (item[nv_out] == 'omip1bias' or item[nv_out] == 'omip2bias'):
        bounds = bounds1
        ticks_bounds = ticks_bounds1
        if (item[nv_out] == 'omip1bias'):
            da = DS['omip1mean'].isel(model=nmodel)
        else:
            da = DS['omip2mean'].isel(model=nmodel)
        daobs = DS['obs']
        #dadraw = da - daobs
        dadraw = da
        datmp = da.values - daobs.values
        #damtmp = da.values
        msktmp = np.where( np.isnan(datmp), 0.0, 1.0 )
        datmp = np.where( np.isnan(datmp), 0.0, datmp )
        #damtmp = np.where( np.isnan(damtmp), 0.0, damtmp )
        tmp1 = (datmp * datmp * area * msktmp * maskdeep).sum()
        tmp3 = (datmp * area * msktmp * maskdeep).sum()
        tmp2 = (area * msktmp * maskdeep).sum()
        rmse = np.sqrt(tmp1/tmp2)
        mldm = tmp3/tmp2
        title_panel = model + '\n' \
            + ' mean bias = ' + '{:.1f}'.format(mldm) + ' m,' + '    bias rmse = ' + '{:.1f}'.format(rmse) + ' m'
        dict_rmse[model]=[rmse,mldm]
    else:
        dadraw = DS[item[nv_out]].isel(model=nmodel)
        bounds = bounds2
        ticks_bounds = ticks_bounds2
        title_panel = model


    dadraw.plot(ax=ax[nmodel],cmap=cmap[nv_out],
            levels=bounds,
            extend=extflg[nv_out],
            add_colorbar=False,
            transform=ccrs.PlateCarree())

    ax[nmodel].coastlines()
    ax[nmodel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[nmodel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[nmodel].xaxis.set_major_formatter(lon_formatter)
    ax[nmodel].yaxis.set_major_formatter(lat_formatter)
    ax[nmodel].set_xlabel('')
    ax[nmodel].set_ylabel('')
    ax[nmodel].set_title(title_panel,{'fontsize':8, 'verticalalignment':'top', 'linespacing':0.8})
    ax[nmodel].tick_params(labelsize=8)
    ax[nmodel].background_patch.set_facecolor('lightgray')
    nmodel += 1
        
plt.subplots_adjust(left=0.05,right=0.98,bottom=0.12,top=0.92,hspace=0.32,wspace=0.15)
plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

summary=pd.DataFrame(dict_rmse,index=['OMIP'+str(omip_out)+'_rmse','OMIP'+str(omip_out)+'_mean'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/MLD_summer_OMIP' + str(omip_out) + '.csv')

if (len(sys.argv) == 6 and sys.argv[5] == 'show'):
    plt.show()
