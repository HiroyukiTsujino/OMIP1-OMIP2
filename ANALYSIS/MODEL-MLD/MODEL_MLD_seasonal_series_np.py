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

#--------------------

if (len(sys.argv) < 5) :
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

metainfo = json.load(open("./json/mlotst_" + mip + ".json"))
model_list = metainfo.keys()

#----------------------------------------------

path_out = '../analysis/MLD/MODEL'

nyr = edyr - styr + 1

for model in metainfo.keys():

    if ( modelname != 'all' ):
        if ( model != modelname ):
            continue

    print (' ')
    print ('Processing ', model)

    mldfile = metainfo[model]['path'] + '/' + metainfo[model]['fname']
    undef_value = float(metainfo[model]['undef'])
    undef_nan = metainfo[model]['undefnan']
    gtorlt = metainfo[model]['gtorlt']
    mldvname = metainfo[model]['name']
    lonname = metainfo[model]['lonname']
    latname = metainfo[model]['latname']
    londim = metainfo[model]['londim']
    latdim = metainfo[model]['latdim']

    ncmld = netCDF4.Dataset(mldfile,'r')
    mld_vars = ncmld.variables[mldvname].ncattrs()

    miss_val_mld = np.NaN
    if ('_FillValue' in mld_vars):
        miss_val_mld = ncmld.variables[mldvname]._FillValue
    elif ('missing_value' in mld_vars):
        miss_val_mld = ncmld.variables[mldvname].missing_value

    nx = len(ncmld.dimensions[londim])
    ny = len(ncmld.dimensions[latdim])
    soeq = int(ny/2)

    print (model, nx, ny, gtorlt, miss_val_mld)

    lon_ = ncmld.variables[lonname][:]
    lat_ = ncmld.variables[latname][:]

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
    
    if ( model == 'Kiel-NEMO' ):
        mskfile = metainfo[model]['path'] + '/' + 'WOA_mask.nc'
        nckmsk = netCDF4.Dataset(mskfile,'r')
        maskkiel = nckmsk.variables['mask'][:,:,:]
        nckmsk.close()

    #----------------------------------------------
    # read data and store in np.array

    mld_jfmtmp = np.array(np.empty((ny,nx)),dtype=np.float64)
    day_jfmtmp = np.array(np.empty((ny,nx)),dtype=np.int64)
    mld_jastmp = np.array(np.empty((ny,nx)),dtype=np.float64)
    day_jastmp = np.array(np.empty((ny,nx)),dtype=np.int64)

    mask_model = np.array(np.ones((ny,nx)),dtype=np.int64)

    mld_winter = np.array(np.zeros((nyr,ny,nx)),dtype=np.float64)
    mld_summer = np.array(np.zeros((nyr,ny,nx)),dtype=np.float64)

    nn = 0
    for yr in range(styr,edyr+1):

        rec_base = (yr-start_yr)*12

        mld_jfmtmp[:,:] = 0.0
        day_jfmtmp[:,:] = 0.0
        mld_jastmp[:,:] = 0.0
        day_jastmp[:,:] = 0.0


        for mn in range(1,13):

            recn = rec_base + mn - 1
            if (yr == styr or yr == edyr):
                if (mn == 12):
                    print (yr,mn,recn)

            mld = ncmld.variables[mldvname][recn,:,:]

            if ( model == 'NorESM-BLOM' ):
                mld_rot = np.roll(mld, 180, axis=1)
                mld = mld_rot

            if ( model == 'MIROC-COCO4.9' ):
                mld_flip = np.flip(mld, axis=0)
                mld = mld_flip

            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flags = (mld > undef_value)
                else:
                    undef_flags = (mld < undef_value)
            else:
                if ( model == 'Kiel-NEMO'):
                    undef_flags = (maskkiel[0,:,:] == 0)
                else:
                    undef_flags = np.isnan(mld)

            mld[undef_flags] = np.NaN

            mask_model = np.where(np.isnan(mld), 0, mask_model)

            if (mn >= 1 and mn <= 3):
                mld_jfmtmp = mld_jfmtmp + mask_model * mld * mon_days[mn-1]
                day_jfmtmp = day_jfmtmp + mask_model * mon_days[mn-1]

            if (mn >= 7 and mn <= 9):
                mld_jastmp = mld_jastmp + mask_model * mld * mon_days[mn-1]
                day_jastmp = day_jastmp + mask_model * mon_days[mn-1]

        mld_jfmtmp[:,:] = mld_jfmtmp[:,:] / (1 - mask_model[:,:] + day_jfmtmp[:,:])
        mld_jfmtmp[:,:] = np.where(mask_model[:,:] == 0, np.NaN, mld_jfmtmp[:,:])

        mld_jastmp[:,:] = mld_jastmp[:,:] / (1 - mask_model[:,:] + day_jastmp[:,:])
        mld_jastmp[:,:] = np.where(mask_model[:,:] == 0, np.NaN, mld_jastmp[:,:])

        mld_winter[nn,0:soeq,:]  = mld_jastmp[0:soeq,:]
        mld_winter[nn,soeq:ny,:] = mld_jfmtmp[soeq:ny,:]

        mld_summer[nn,0:soeq,:]  = mld_jfmtmp[0:soeq,:]
        mld_summer[nn,soeq:ny,:] = mld_jastmp[soeq:ny,:]

        #for j in range(0,ny):
        #    for i in range(0,nx):
        #        if (mask_model[j,i] == 1 and np.isnan(mld_winter[nn,j,i])):
        #            print("Inconsistent mask for ", model, j, i, mld_winter[nn,j,i])

        nn += 1

    ncmld.close()

    #############################################
    # Output to netCDF4

    lon_bnds_ = np.array(np.empty((nx,2)))
    lat_bnds_ = np.array(np.empty((ny,2)))

    lon_bnds_[:,0] = lon_[:] - 0.5
    lon_bnds_[:,1] = lon_[:] + 0.5
    lat_bnds_[:,0] = lat_[:] - 0.5
    lat_bnds_[:,1] = lat_[:] + 0.5

    fwin_out = path_out + '/' + 'mlotst_winter_' + model + '_' + mip + '_' + str(styr) + '-' + str(edyr) + '.nc'

    ncwin = netCDF4.Dataset(fwin_out, 'w', format='NETCDF4')
    ncwin.createDimension('lon', nx)
    ncwin.createDimension('lat', ny)
    ncwin.createDimension('bnds', 2)
    ncwin.createDimension('time', None)

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

    stdate=str(styr) + '-01-01'
    eddate=str(edyr) + '-12-31'

    dtime = pd.date_range(stdate,eddate,freq='AS-JAN')
    dtime_start = datetime.date(1850, 1, 1)
    td=pd.to_datetime(dtime[:]).date - dtime_start

    time = ncwin.createVariable('time', np.dtype('int32').char, ('time',))
    time.units = 'days since 1850-01-01 00:00:00'
    time.axis = 'T'

    mldwin = ncwin.createVariable('mlotst', np.dtype(np.float64).char, ('time', 'lat', 'lon'), zlib=True)
    mldwin.long_name = 'Ocean mixed layer thickness defined by sigma-t'
    mldwin.units = 'm'
    mldwin.missing_value = -9.99e33

    #mldwin[:,:,:]=np.where(np.isnan(mld_winter), -9.99e33, mld_winter)
    mldwin[:,:,:]=mld_winter
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    time_vars = np.array(np.zeros((len(td))))
    for i in range(len(td)):
        time_vars[i] = td[i].days

    time[:]=time_vars

    ncwin.description="Winter (JFM) mean MLD " + str(styr) + " through " + str(edyr)

    ncwin.close()

    #-----

    fsum_out = path_out + '/' + 'mlotst_summer_' + model + '_' + mip + '_' + str(styr) + '-' + str(edyr) + '.nc'
    ncsum = netCDF4.Dataset(fsum_out, 'w', format='NETCDF4')

    ncsum.createDimension('lon', nx)
    ncsum.createDimension('lat', ny)
    ncsum.createDimension('bnds', 2)
    ncsum.createDimension('time', None)

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

    stdate=str(styr) + '-01-01'
    eddate=str(edyr) + '-12-31'

    dtime = pd.date_range(stdate,eddate,freq='AS-JAN')
    dtime_start = datetime.date(1850, 1, 1)
    td = pd.to_datetime(dtime[:]).date - dtime_start
    time = ncwin.createVariable('time', np.dtype('int32').char, ('time',))
    time.units = 'days since 1850-01-01 00:00:00'
    time.axis = 'T'

    mldsum = ncsum.createVariable('mlotst', np.dtype(np.float64).char, ('time', 'lat', 'lon'), zlib=True)
    mldsum.long_name = 'Ocean mixed layer thickness defined by sigma-t'
    mldsum.units = 'm'
    mldsum.missing_value = -9.99e33

    #mldsum[:,:,:]=np.where(np.isnan(mld_summer), -9.99e33, mld_summer)
    mldsum[:,:,:]=mld_summer
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    time_vars = np.array(np.zeros((len(td))))
    for i in range(len(td)):
        time_vars[i] = td[i].days

    time[:]=time_vars

    ncsum.description="Summer (JAS) mean MLD " + str(styr) + " through " + str(edyr)

    ncsum.close()

    #####

    fmsk_out = path_out + '/' + 'mlotst_mask_season_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'

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

    mld_mask[:,:]=mask_model.astype(np.int32)
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    ncmsk.description="Land-Sea mask implied by physical variable"
    ncmsk.close()

    #--------------------
    # Draw Figures

    suptitle = model + '-MLD Climatology ' + str(mip) + ' ' + str(styr) + ' to ' + str(edyr)
    title = [ 'Winter' , 'Summer']
    outfile = 'fig/' + model + '-' + mip + '-MLD_seasonal_climatology_np.png'
    

    fig = plt.figure(figsize=(8,11))
    fig.suptitle( suptitle, fontsize=18 )

    proj = ccrs.PlateCarree(central_longitude=-140.)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    ax = [
        plt.subplot(2,1,1,projection=proj),
        plt.subplot(2,1,2,projection=proj),
    ]

    for panel in range(2):
        if (panel == 0):
            tmp=mld_winter.mean(axis=0)
            ct = np.array([0,10,20,50,100,200,300,400,500,600,1000,1500,2000,2500,3000,4000,5000])
            norm = colors.BoundaryNorm(ct,256)
        else:
            tmp=mld_summer.mean(axis=0)
            ct = np.arange(0,150,10)
            norm = colors.BoundaryNorm(ct,256)

        lon_tmp=np.array(lon_)
        tmp, lon_tmp = add_cyclic_point(tmp, coord=lon_tmp)
        ca=ax[panel].contourf(lon_tmp, lat_, tmp, levels=ct, cmap='RdYlBu_r', norm=norm, extend='max', transform=ccrs.PlateCarree())
        ax[panel].coastlines()
        ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
        ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
        ax[panel].xaxis.set_major_formatter(lon_formatter)
        ax[panel].yaxis.set_major_formatter(lat_formatter)
        ax[panel].set_title(title[panel])
        fig.colorbar(ca,ax=ax[panel],orientation='horizontal',shrink=0.7)
        del tmp
        del lon_tmp

    plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
    plt.show()

    del lon_bnds_
    del lat_bnds_
    del mld_winter
    del mld_summer
    del mld_jfmtmp
    del day_jfmtmp
    del mld_jastmp
    del day_jastmp

    del mask_model
