# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import netCDF4
import datetime
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

#----------------------------------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

path_mld = '../refdata/MLD_deBoyer_Montegut'

mldfile = path_mld + '/' + 'mld_DR003_c1m_reg2.0.nc'

ncmld = netCDF4.Dataset(mldfile,'r')
#miss_val_mld = ncmld.variables['mld'].missing_value
miss_val_mld = -9000.e0
nx = len(ncmld.dimensions['lon'])
ny = len(ncmld.dimensions['lat'])

print (ny,nx)

lon_ = ncmld.variables['lon'][:]
lat_ = ncmld.variables['lat'][:]

mask_lnd = ncmld.variables['mask'][:,:]

#----------------------------------------------
# read data and store in np.array

mld_winclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_winclim = np.array(np.zeros((ny,nx)),dtype=np.int64)

mld_sumclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_sumclim = np.array(np.zeros((ny,nx)),dtype=np.int64)

mask_win = np.array(np.ones((ny,nx)),dtype=np.int64)
mask_sum = np.array(np.ones((ny,nx)),dtype=np.int64)

mld_jfmclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_jfmclim = np.array(np.zeros((ny,nx)),dtype=np.int64)
mask_jfm = np.array(np.ones((ny,nx)),dtype=np.int64)
mask_jfm = np.where(mask_lnd == 0.0, 0, mask_jfm)

mld_jasclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_jasclim = np.array(np.zeros((ny,nx)),dtype=np.int64)
mask_jas = np.array(np.ones((ny,nx)),dtype=np.int64)
mask_jas = np.where(mask_lnd == 0.0, 0, mask_jas)

for mn in range(1,13):
    mld_tmp = ncmld.variables['mld'][mn-1,:,:]
    mld = mld_tmp.astype(np.float64)
    if (mn >= 1 and mn <= 3):
        mld_jfmclim = mld_jfmclim + mask_lnd * mld * mon_days[mn-1]
        day_jfmclim = day_jfmclim + mask_lnd * mon_days[mn-1]
        mask_jfm = np.where(mld < miss_val_mld, 0, mask_jfm)

    if (mn >= 7 and mn <= 9):
        mld_jasclim = mld_jasclim + mask_lnd * mld * mon_days[mn-1]
        day_jasclim = day_jasclim + mask_lnd * mon_days[mn-1]
        mask_jas = np.where(mld < miss_val_mld, 0, mask_jas)

mld_jfmclim = mld_jfmclim / (1.0 - mask_jfm + day_jfmclim)
mld_jfmclim = np.where(mask_jfm == 0, np.NaN, mld_jfmclim)

mld_jasclim = mld_jasclim / (1.0 - mask_jas + day_jasclim)
mld_jasclim = np.where(mask_jas == 0, np.NaN, mld_jasclim)

soeq = int(ny/2)

print(soeq)

mld_winclim[0:soeq-1,:]  = mld_jasclim[0:soeq-1,:]
mld_winclim[soeq-1,:] = 0.5 * mld_jfmclim[soeq-1,:] + 0.5 * mld_jasclim[soeq-1,:]
mld_winclim[soeq:ny,:] = mld_jfmclim[soeq:ny,:]

for i in range(0,nx):
    print(" ")
    print(mld_winclim[soeq,i],mld_jfmclim[soeq,i],mld_jasclim[soeq,i])
    print(mld_winclim[soeq-1,i],mld_jfmclim[soeq-1,i],mld_jasclim[soeq-1,i])
    print(mld_winclim[soeq-2,i],mld_jfmclim[soeq-2,i],mld_jasclim[soeq-2,i])

mld_sumclim[0:soeq-1,:]  = mld_jfmclim[0:soeq-1,:]
mld_sumclim[soeq-1,:] = 0.5 * mld_jfmclim[soeq-1,:] + 0.5 * mld_jasclim[soeq-1,:]
mld_sumclim[soeq:ny,:] = mld_jasclim[soeq:ny,:]
#print(mld_sumclim[soeq,:])

#-----

mask_win[0:soeq-1,:] = mask_jas[0:soeq-1,:]
mask_win[soeq-1,:] = (mask_jfm[soeq-1,:] + mask_jas[soeq-1,:])/2
mask_win[soeq:ny,:] = mask_jfm[soeq:ny,:]
mask_win = np.where(mask_win < 0.9, 0, 1)

mask_sum[0:soeq-1,:]  = mask_jfm[0:soeq-1,:]
mask_sum[soeq-1,:] = (mask_jfm[soeq-1,:] + mask_jas[soeq-1,:])/2
mask_sum[soeq:ny,:] = mask_jas[soeq:ny,:]
mask_sum = np.where(mask_sum < 0.9, 0, 1)

#############################################
# Output to netCDF4

path_mld_out = '../analysis/MLD/MLD_deBoyer_Montegut'

fwin_out = path_mld_out + '/' + 'mld_DR003_winclim.nc'
fsum_out = path_mld_out + '/' + 'mld_DR003_sumclim.nc'
fmsk_win = path_mld_out + '/' + 'mld_DR003_winmask.nc'
fmsk_sum = path_mld_out + '/' + 'mld_DR003_summask.nc'

lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))

lon_bnds_[:,0] = lon_[:] - 1.0
lon_bnds_[:,1] = lon_[:] + 1.0
lat_bnds_[:,0] = lat_[:] - 1.0
lat_bnds_[:,1] = lat_[:] + 1.0

ncwin = netCDF4.Dataset(fwin_out, 'w', format='NETCDF4')
ncwin.createDimension('lon', nx)
ncwin.createDimension('lat', ny)
ncwin.createDimension('bnds', 2)

print (nx,ny)

mldwin = ncwin.createVariable('mlotst', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
mldwin.long_name = 'mixed layer depth'
mldwin.units = 'm'
mldwin.missing_value = -9.99e33

lat = ncwin.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncwin.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

lon = ncwin.createVariable('lon', np.dtype('float').char, ('lon'))
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'
lon.standard_name = 'latitude'
lon_bnds = ncwin.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

print (np.shape(mld_winclim))
#mldwin[:,:]=np.where(np.isnan(mld_winclim), -9.99e33, mld_winclim)
mldwin[:,:]=mld_winclim[:,:]
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

ncwin.description="Winter mean mixed layer depth of density + 0.03 criterion"

ncwin.close()

#-----

ncsum = netCDF4.Dataset(fsum_out, 'w', format='NETCDF4')

ncsum.createDimension('lon', nx)
ncsum.createDimension('lat', ny)
ncsum.createDimension('bnds', 2)

lat = ncsum.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncsum.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

lon = ncsum.createVariable('lon', np.dtype('float').char, ('lon'))
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'
lon.standard_name = 'latitude'
lon_bnds = ncsum.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

mldsum = ncsum.createVariable('mlotst', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
mldsum.long_name = 'mixed layer depth'
mldsum.units = 'm'
mldsum.missing_value = -9.99e33

#mldsum[:,:]=np.where(np.isnan(mld_sumclim), -9.99e33, mld_sumclim)
mldsum[:,:]=mld_sumclim[:,:]
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

ncwin.description="Summer mean mixed layer depth of density + 0.03 criterion"

ncsum.close()

#####

ncmsk = netCDF4.Dataset(fmsk_win, 'w', format='NETCDF4')
ncmsk.createDimension('lon', nx)
ncmsk.createDimension('lat', ny)
ncmsk.createDimension('bnds', 2)

mld_mask = ncmsk.createVariable('mldmask', np.dtype(np.int32).char, ('lat', 'lon'), zlib=True)
mld_mask.long_name = 'Land Sea Mask'
mld_mask.units = '1'
mld_mask.missing_value = -999

lat = ncmsk.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncmsk.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

lon = ncmsk.createVariable('lon', np.dtype('float').char, ('lon'))
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'
lon.standard_name = 'latitude'
lon_bnds = ncmsk.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

mld_mask[:,:]=mask_win.astype(np.int32)
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

ncmsk.close()

#####

ncmsk = netCDF4.Dataset(fmsk_sum, 'w', format='NETCDF4')
ncmsk.createDimension('lon', nx)
ncmsk.createDimension('lat', ny)
ncmsk.createDimension('bnds', 2)

mld_mask = ncmsk.createVariable('mldmask', np.dtype(np.int32).char, ('lat', 'lon'), zlib=True)
mld_mask.long_name = 'Land Sea Mask'
mld_mask.units = '1'
mld_mask.missing_value = -999

lat = ncmsk.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncmsk.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

lon = ncmsk.createVariable('lon', np.dtype('float').char, ('lon'))
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'
lon.standard_name = 'latitude'
lon_bnds = ncmsk.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

mld_mask[:,:]=mask_sum.astype(np.int32)
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

ncmsk.close()

#--------------------
# Draw Figures

suptitle = 'MLD Climatology'
title = [ 'Winter' , 'Summer']
outfile = 'fig/deBoyer_MLD_seasonal_climatology.png'

ct = np.arange(0,1500,100)

fig = plt.figure(figsize=(9,15))
fig.suptitle( suptitle, fontsize=20 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(2,1,1,projection=proj),
    plt.subplot(2,1,2,projection=proj),
]

for panel in range(2):
    if (panel == 0):
        tmp=mld_winclim
        ct = np.array([0,10,20,50,100,150,200,250,300,400,500,600,1000,1500,2000,2500,3000])
        norm = colors.BoundaryNorm(ct,256)
    else:
        tmp=mld_sumclim
        ct = np.arange(0,180,10)
        norm = colors.BoundaryNorm(ct,256)

    lon_tmp=np.array(lon_)
    tmp, lon_tmp = add_cyclic_point(tmp, coord=lon_tmp)
    ca=ax[panel].contourf(lon_tmp, lat_, tmp, ct, cmap='RdYlBu_r', norm=norm, extend='max', transform=ccrs.PlateCarree())
    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_title(title[panel])
    fig.colorbar(ca,ax=ax[panel],orientation='horizontal',shrink=0.7)
    #fig.colorbar(c,ax=ax[panel],orientation='horizontal')

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
plt.show()
