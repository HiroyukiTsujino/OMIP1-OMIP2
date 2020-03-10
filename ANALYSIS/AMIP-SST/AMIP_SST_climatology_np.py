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

#--------------------

if len(sys.argv) == 1:
    print ('Usage: AMIP_SST_climatology_np.py start_year end_year')
    sys.exit()

styr = int(sys.argv[1])
edyr = int(sys.argv[2])

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

print( "Loading AMIP data" )

path_amip = '../refdata/PCMDI-SST'
path_out = '../analysis/SST/PCMDI-SST'

sstfile = path_amip + '/' + 'tos_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn_187001-201712.nc'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
mskfile = path_amip + '/' + 'sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncsst = netCDF4.Dataset(sstfile,'r')
miss_val_sst = ncsst.variables['tos'].missing_value
nx = len(ncsst.dimensions['lon'])
ny = len(ncsst.dimensions['lat'])

print (nx,ny)

lon_ = ncsst.variables['lon'][:]
lat_ = ncsst.variables['lat'][:]
#xx, yy = np.meshgrid(lon_,lat_)

ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]

ncmsk = netCDF4.Dataset(mskfile,'r')
mask = ncmsk.variables['sftof'][:,:]

# Ad hoc modification of the Kaspian Sea
for j in range(ny):
    for i in range(nx):
        if (45 < lon_[i]) & (lon_[i] < 60):
             if (34 < lat_[j]) & (lat_[j] < 50):
                  mask[j,i] = 0

        if mask[j,i] < 100:
            mask[j,i] = 0

mask = mask / 100

#----------------------------------------------
# read data and store in np.array

sst_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
day_annclim = np.array(np.zeros((ny,nx)),dtype=np.int64)
sst_monclim = np.array(np.zeros((12,ny,nx)),dtype=np.float64)
day_monclim = np.array(np.zeros((12,ny,nx)),dtype=np.int64)

start_yr = 1870

for yr in range(styr,edyr+1):

    rec_base = (yr-start_yr)*12

    for mn in range(1,13):

        recn = rec_base + mn - 1
        print (yr,mn,recn)
        sst_tmp = ncsst.variables['tos'][recn,:,:]
        sst = sst_tmp.astype(np.float64)

        undef_flags = (sst == miss_val_sst)
        sst[undef_flags] = np.NaN

        sst_annclim = sst_annclim + mask * sst * mon_days[mn-1]
        day_annclim = day_annclim + mask * mon_days[mn-1]
        sst_monclim[mn-1] = sst_monclim[mn-1] + mask * sst * mon_days[mn-1]
        day_monclim[mn-1] = day_monclim[mn-1] + mask * mon_days[mn-1]

sst_annclim = np.where(day_annclim == 0, np.NaN, sst_annclim / day_annclim)

#for mn in range(12):
#    print(mn+1, mon_days[mn])
#    for j in range(ny):
#        for i in range(nx):
#            if (day_monclim[mn,j,i] == 0):
#                sst_monclim[mn,j,i] = np.NaN
#            else:
#                sst_monclim[mn,j,i] = sst_monclim[mn,j,i]/day_monclim[mn,j,i]

#for mn in range(1,13):
#    sst_monclim[mn-1,:,:] = np.where(day_monclim[mn-1,:,:] == 0, np.NaN, sst_monclim[mn-1,:,:] / day_monclim[mn-1,:,:])

for mn in range(1,13):
    sst_monclim[mn-1,:,:] = sst_monclim[mn-1,:,:] / (1.0 - mask[:,:] + day_monclim[mn-1,:,:])
    sst_monclim[mn-1,:,:] = np.where(mask == 0.0, np.NaN, sst_monclim[mn-1,:,:])


#for j in range(ny):
#    print (sst_annclim[j,nx-1],sst_annclim[j,0])


#############################################
# Output to netCDF4

fann_out = path_out + '/' + 'tos_annclim_gn_' + str(styr) + '01-' + str(edyr) + '12.nc'
fmon_out = path_out + '/' + 'tos_monclim_gn_' + str(styr) + '01-' + str(edyr) + '12.nc'

lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))

lon_bnds_[:,0] = lon_[:] - 0.5
lon_bnds_[:,1] = lon_[:] + 0.5
lat_bnds_[:,0] = lat_[:] - 0.5
lat_bnds_[:,1] = lat_[:] + 0.5

ncann = netCDF4.Dataset(fann_out, 'w', format='NETCDF4')
ncann.createDimension('lon', nx)
ncann.createDimension('lat', ny)
ncann.createDimension('bnds', 2)

tosann = ncann.createVariable('tos', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
tosann.long_name = 'sea surface temperature'
tosann.units = '1'
tosann.missing_value = -9.99e33

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


tosann[:,:]=np.where(np.isnan(sst_annclim), -9.99e33, sst_annclim)
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

ncann.description="Annual mean " + str(styr) + " through " + str(edyr)

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

tosmon = ncmon.createVariable('tos', np.dtype(np.float64).char, ('time', 'lat', 'lon'), zlib=True)
tosmon.long_name = 'sea surface temperature'
tosmon.units = 'degC'
tosmon.missing_value = -9.99e33

tosmon[:,:,:]=np.where(np.isnan(sst_monclim), -9.99e33, sst_monclim)
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

time_vars = np.array(np.zeros((len(td))))
for i in range(len(td)):
  time_vars[i] = td[i].days

time[:]=time_vars

ncmon.description="Monthly climatology " + str(styr) + " through" + str(edyr)

ncmon.close()

#--------------------
# Draw Figures

suptitle = 'AMIP SST Climatology ' + str(styr) + ' to ' + str(edyr)
#title = [ 'Annual mean' ]
title = [ 'January' , 'July']
outfile = 'fig/AMIP_SST_climatology_np.png'

ct = np.arange(-2,31,1)

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
    tmp=np.array(sst_monclim[panel*6])
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
