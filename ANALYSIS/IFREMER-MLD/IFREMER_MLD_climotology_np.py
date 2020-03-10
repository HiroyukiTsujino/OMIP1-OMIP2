# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
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

print (nx,ny)

lon_ = ncmld.variables['lon'][:]
lat_ = ncmld.variables['lat'][:]

mask_lnd = ncmld.variables['mask'][:,:]

#----------------------------------------------
# read data and store in np.array

mld_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_annclim = np.array(np.zeros((ny,nx)),dtype=np.int64)
mld_monclim = np.array(np.zeros((12,ny,nx)),dtype=np.float64)
mask_mld = mask_lnd

for mn in range(1,13):
    mld_tmp = ncmld.variables['mld'][mn-1,:,:]
    mld = mld_tmp.astype(np.float64)
    mask_mld = np.where(mld < miss_val_mld, 0.0, mask_mld)

for mn in range(1,13):
    mld_tmp = ncmld.variables['mld'][mn-1,:,:]
    mld = mld_tmp.astype(np.float64)
    mld_annclim = mld_annclim + mask_mld * mld * mon_days[mn-1]
    day_annclim = day_annclim + mask_mld * mon_days[mn-1]
    mld_monclim[mn-1,:,:] = mask_mld[:,:] * mld[:,:]

mld_annclim = mld_annclim / (1.0 - mask_mld + day_annclim)

mld_annclim = np.where(mask_mld == 0.0, np.NaN, mld_annclim)

for mn in range(1,13):
    mld_monclim[mn-1,:,:] = np.where(mask_mld[:,:] == 0.0, np.NaN, mld_monclim[mn-1,:,:])

#############################################
# Output to netCDF4

path_mld_out = '../analysis/MLD/MLD_deBoyer_Montegut'
fann_out = path_mld_out + '/' + 'mld_DR003_annclim.nc'
fmon_out = path_mld_out + '/' + 'mld_DR003_monclim.nc'
fmsk_out = path_mld_out + '/' + 'mld_DR003_mask.nc'

lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))

lon_bnds_[:,0] = lon_[:] - 1.0
lon_bnds_[:,1] = lon_[:] + 1.0
lat_bnds_[:,0] = lat_[:] - 1.0
lat_bnds_[:,1] = lat_[:] + 1.0


ncann = netCDF4.Dataset(fann_out, 'w', format='NETCDF4')
ncann.createDimension('lon', nx)
ncann.createDimension('lat', ny)
ncann.createDimension('bnds', 2)

print (nx,ny)

mldann = ncann.createVariable('mlotst', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
mldann.long_name = 'mixed layer depth'
mldann.units = 'm'
mldann.missing_value = -9.99e33

lat = ncann.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncann.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

lon = ncann.createVariable('lon', np.dtype('float').char, ('lon'))
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'
lon.standard_name = 'latitude'
lon_bnds = ncann.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

print (np.shape(mld_annclim))
mldann[:,:]=np.where(np.isnan(mld_annclim), -9.99e33, mld_annclim)
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

ncann.description="Annual mean mixed layer depth of density + 0.03 criterion"

ncann.close()

#-----

ncmon = netCDF4.Dataset(fmon_out, 'w', format='NETCDF4')

ncmon.createDimension('lon', nx)
ncmon.createDimension('lat', ny)
ncmon.createDimension('bnds', 2)
ncmon.createDimension('time', None)

lat = ncmon.createVariable('lat', np.dtype('float').char, ('lat'))
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'
lat.standard_name = 'latitude'
lat_bnds = ncmon.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

lon = ncmon.createVariable('lon', np.dtype('float').char, ('lon'))
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'
lon.standard_name = 'latitude'
lon_bnds = ncmon.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

dtime = pd.date_range('1850-01-01','1850-12-01',freq='MS')
dtime_start = datetime.date(1850, 1, 1)
td=pd.to_datetime(dtime[:]).date - dtime_start
time = ncmon.createVariable('time', np.dtype('int32').char, ('time',))
time.units = 'days since 1850-01-01 00:00:00'
time.axis = 'T'

mldmon = ncmon.createVariable('mlotst', np.dtype(np.float64).char, ('time', 'lat', 'lon'), zlib=True)
mldmon.long_name = 'mixed layer depth'
mldmon.units = 'm'
mldmon.missing_value = -9.99e33

mldmon[:,:,:]=np.where(np.isnan(mld_monclim), -9.99e33, mld_monclim)
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

time_vars = np.array(np.zeros((len(td))))
for i in range(len(td)):
    time_vars[i] = td[i].days

time[:]=time_vars

ncmon.description=" Monthly climatology "

ncmon.close()

#####

ncmsk = netCDF4.Dataset(fmsk_out, 'w', format='NETCDF4')
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

mld_mask[:,:]=mask_mld.astype(np.int32)
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

ncmsk.close()

#--------------------
# Draw Figures

suptitle = 'MLD Climatology'
title = [ 'March' , 'September']
outfile = 'fig/deBoyer_MLD_climatology.png'

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
    tmp=np.array(mld_monclim[panel*6+2])
    lon_tmp=np.array(lon_)
    tmp, lon_tmp = add_cyclic_point(tmp, coord=lon_tmp)
    ca=ax[panel].contourf(lon_tmp, lat_, tmp, ct, transform=ccrs.PlateCarree())
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
