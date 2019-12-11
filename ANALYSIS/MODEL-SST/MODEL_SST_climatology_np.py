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

if (len(sys.argv) < 5):
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year end_year [modelname or all]')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])
modelname = sys.argv[4]

#--------------------

if (mip == 'omip1'):
    start_yr = 1948
elif (mip == 'omip2'):
    start_yr = 1958
else:
    print(' Invalid mip id :', mip)
    sys.exit()

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

metainfo = json.load(open("./json/tos_" + mip + ".json"))
model_list = metainfo.keys()

#----------------------------------------------

path_out = '../analysis/SST/MODEL'

for model in metainfo.keys():

    if ( modelname != 'all' ):
        if ( model != modelname ):
            continue

    sstfile = metainfo[model]['path'] + '/' + metainfo[model]['fname']
    undef_value = float(metainfo[model]['undef'])
    undef_nan = metainfo[model]['undefnan']
    gtorlt = metainfo[model]['gtorlt']
    sstvname = metainfo[model]['name']
    lonname = metainfo[model]['lonname']
    latname = metainfo[model]['latname']
    londim = metainfo[model]['londim']
    latdim = metainfo[model]['latdim']

    ncsst = netCDF4.Dataset(sstfile,'r')
    sst_vars = ncsst.variables[sstvname].ncattrs()

    miss_val_sst = np.NaN
    if ('_FillValue' in sst_vars):
        miss_val_sst = ncsst.variables[sstvname]._FillValue
    elif ('missing_value' in sst_vars):
        miss_val_sst = ncsst.variables[sstvname].missing_value

    nx = len(ncsst.dimensions[londim])
    ny = len(ncsst.dimensions[latdim])

    print (model, nx, ny, gtorlt, miss_val_sst)

    lon_ = ncsst.variables[lonname][:]
    lat_ = ncsst.variables[latname][:]

    if ( model == 'NorESM-BLOM' ):
        lon_rot = np.roll(lon_,180)
        lon_ = lon_rot
        lon_ = np.where(lon_ < 0.0, lon_+360, lon_)
        print(' NorESM-BLOM longitude rolled ')
        print(lon_)
    
    if ( model == 'MIROC-COCO4.9' ):
        lat_flip = np.flip(lat_)
        lat_ = lat_flip
        print(' MIROC-COCO4.9 latitude flipped ')
        print(lat_)
    
    #----------------------------------------------
    # read data and store in np.array

    sst_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
    day_annclim = np.array(np.zeros((ny,nx)),dtype=np.int64)
    sst_monclim = np.array(np.zeros((12,ny,nx)),dtype=np.float64)
    day_monclim = np.array(np.zeros((12,ny,nx)),dtype=np.int64)

    mask_model = np.array(np.ones((ny,nx)),dtype=np.int64)

    for yr in range(styr,edyr+1):

        rec_base = (yr-start_yr)*12

        for mn in range(1,13):

            recn = rec_base + mn - 1
            if (yr == styr or yr == edyr):
                if (mn == 12):
                    print (yr,mn,recn)

            sst = ncsst.variables[sstvname][recn,:,:]

            if ( model == 'NorESM-BLOM' ):
                sst_rot = np.roll(sst, 180, axis=1)
                sst = sst_rot

            if ( model == 'MIROC-COCO4.9' ):
                sst_flip = np.flip(sst, axis=0)
                sst = sst_flip

            if ( model == 'Kiel-NEMO' ):
                sst = np.where(sst==0.0, np.NaN, sst)

            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flags = (sst > undef_value)
                else:
                    undef_flags = (sst < undef_value)
            else:
                undef_flags = np.isnan(sst)

            sst[undef_flags] = np.NaN

            mask_model = np.where(np.isnan(sst), 0, mask_model)

            sst_annclim = sst_annclim + mask_model * sst * mon_days[mn-1]
            day_annclim = day_annclim + mask_model * mon_days[mn-1]
            sst_monclim[mn-1] = sst_monclim[mn-1] + mask_model * sst * mon_days[mn-1]
            day_monclim[mn-1] = day_monclim[mn-1] + mask_model * mon_days[mn-1]

    print(mask_model)

    sst_annclim = sst_annclim / (1 - mask_model + day_annclim)

    for mn in range(1,13):
        sst_monclim[mn-1,:,:] = sst_monclim[mn-1,:,:] / (1 - mask_model[:,:] + day_monclim[mn-1,:,:])
        sst_monclim[mn-1,:,:] = np.where(mask_model == 0, np.NaN, sst_monclim[mn-1,:,:])

    ncsst.close()

    #############################################
    # Output to netCDF4

    fann_out = path_out + '/' + 'tos_annclim_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'
    fmon_out = path_out + '/' + 'tos_monclim_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'
    fmsk_out = path_out + '/' + 'tos_mask_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'

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
    tosann.units = 'degC'
    tosann.missing_value = -9.99e33

    lat = ncann.createVariable('lat', np.dtype('float').char, ('lat'))
    lat.latg_name = 'latitude'
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
    lat.latg_name = 'latitude'
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
    ncmon.description="Monthly climatology " + str(styr) + " through " + str(edyr)

    ncmon.close()

    #####

    ncmsk = netCDF4.Dataset(fmsk_out, 'w', format='NETCDF4')
    ncmsk.createDimension('lon', nx)
    ncmsk.createDimension('lat', ny)
    ncmsk.createDimension('bnds', 2)

    tos_mask = ncmsk.createVariable('tosmask', np.dtype(np.int32).char, ('lat', 'lon'), zlib=True)
    tos_mask.long_name = 'Land Sea Mask'
    tos_mask.units = '1'
    tos_mask.missing_value = -999

    lat = ncmsk.createVariable('lat', np.dtype('float').char, ('lat'))
    lat.latg_name = 'latitude'
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

    tos_mask[:,:]=mask_model.astype(np.int32)
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_
    
    ncmsk.close()

    #--------------------
    # Draw Figures
    
    suptitle = model + '-SST Climatology ' + str(styr) + ' to ' + str(edyr)
    #title = [ 'Annual mean' ]
    title = [ 'January' , 'July']
    outfile = 'fig/' + model + '-' + mip + '-SST_climatology_np.png'

    ct = np.arange(-2,33,1)

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

    del time_vars
    del lon_bnds_
    del lat_bnds_
    del sst_annclim
    del day_annclim
    del sst_monclim
    del day_monclim

    del mask_model
