# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4
import datetime
from netCDF4 import Dataset
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point

#--------------------

if (len(sys.argv) < 7):
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year start_month end_year end_month [modelname or all]')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
stmn = int(sys.argv[3])
edyr = int(sys.argv[4])
edmn = int(sys.argv[5])
modelname = sys.argv[6]

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

x,y = np.mgrid[0.5:360.5:1,-89.5:90.5:1]
woa_taux = np.zeros((180,360))
woa_tauy = np.zeros((180,360))
woa_rcos = np.zeros((180,360))
woa_rsin = np.zeros((180,360))
rlon = np.arange(0.5,360.5,1)
rlat = np.arange(-89.5,90.5,1)

#----------------------------------------------

metainfo = json.load(open("./json/tau_" + mip + ".json"))
model_list = metainfo.keys()

#----------------------------------------------

path_out = '../analysis/WIND_STRESS/MODEL'

for model in metainfo.keys():

    if ( modelname != 'all' ):
        if ( model != modelname ):
            continue

    print (' ')
    print ('Processing ', model)

    tauxfile = metainfo[model]['path'] + '/' + metainfo[model]['fnamex']
    tauyfile = metainfo[model]['path'] + '/' + metainfo[model]['fnamey']
    undef_value = float(metainfo[model]['undef'])
    undef_nan = metainfo[model]['undefnan']
    gtorlt = metainfo[model]['gtorlt']
    tauxvname = metainfo[model]['namex']
    tauyvname = metainfo[model]['namey']
    lonname = metainfo[model]['lonname']
    latname = metainfo[model]['latname']
    londim = metainfo[model]['londim']
    latdim = metainfo[model]['latdim']

    nctaux = netCDF4.Dataset(tauxfile,'r')
    nctauy = netCDF4.Dataset(tauyfile,'r')
    taux_vars = nctaux.variables[tauxvname].ncattrs()
    tauy_vars = nctauy.variables[tauyvname].ncattrs()

    miss_val_taux = np.NaN
    if ('_FillValue' in taux_vars):
        miss_val_taux = nctaux.variables[tauxvname]._FillValue
    elif ('missing_value' in taux_vars):
        miss_val_taux = nctaux.variables[tauxvname].missing_value

    nx = len(nctaux.dimensions[londim])
    ny = len(nctaux.dimensions[latdim])

    print (model, nx, ny, gtorlt, miss_val_taux)

    lon = nctaux.variables[lonname][:,:]
    lat = nctaux.variables[latname][:,:]

    lon = np.where(lon < 0.0, lon+360, lon)
    #print(lon)
    
    #----------------------------------------------
    # read data and store in np.array

    taux_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
    tauy_annclim = np.array(np.zeros((ny,nx)),dtype=np.float64)
    tauy_annclim_x = np.array(np.zeros((ny,nx)),dtype=np.float64)
    dayx_annclim = np.array(np.zeros((ny,nx)),dtype=np.int64)
    dayy_annclim = np.array(np.zeros((ny,nx)),dtype=np.int64)
    taux_g = np.array(np.zeros((ny,nx)),dtype=np.float64)
    tauy_g = np.array(np.zeros((ny,nx)),dtype=np.float64)
    rcos_g = np.array(np.zeros((ny,nx)),dtype=np.float64)
    rsin_g = np.array(np.zeros((ny,nx)),dtype=np.float64)

    maskx_model = np.array(np.ones((ny,nx)),dtype=np.int64)
    masky_model = np.array(np.ones((ny,nx)),dtype=np.int64)

    for yr in range(styr,edyr+1):

        rec_base = (yr-start_yr)*12

        if (yr == styr):
            stm=stmn
        else:
            stm=1
        if (yr == edyr):
            edm=edmn+1
        else:
            edm=13

        for mn in range(1,13):

            recn = rec_base + mn - 1

            if (yr == styr or yr == edyr):
                if (mn == 12):
                    print (yr,mn,recn)

            taux = nctaux.variables[tauxvname][recn,:,:]
            tauy = nctauy.variables[tauyvname][recn,:,:]

            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flagsx = (taux > undef_value)
                    undef_flagsy = (tauy > undef_value)
                else:
                    undef_flagsx = (taux < undef_value)
                    undef_flagsy = (tauy < undef_value)
            else:
                undef_flagsx = np.isnan(taux)
                undef_flagsy = np.isnan(tauy)

            taux[undef_flagsx] = np.NaN
            tauy[undef_flagsy] = np.NaN

            maskx_model = np.where(np.isnan(taux), 0, maskx_model)
            masky_model = np.where(np.isnan(tauy), 0, masky_model)

            taux_annclim = taux_annclim + maskx_model * taux * mon_days[mn-1]
            tauy_annclim = tauy_annclim + masky_model * tauy * mon_days[mn-1]
            dayx_annclim = dayx_annclim + maskx_model * mon_days[mn-1]
            dayy_annclim = dayy_annclim + masky_model * mon_days[mn-1]

    nctaux.close()
    nctauy.close()

    taux_annclim = taux_annclim / (1 - maskx_model + dayx_annclim)
    tauy_annclim = tauy_annclim / (1 - masky_model + dayy_annclim)

    tauy_annclim = np.where(np.isnan(tauy_annclim), 0, tauy_annclim)

    for j in range(ny):
        for i in range(0,nx-1):
            tauy_annclim_x[j,i] = masky_model[j,i  ] * tauy_annclim[j,i  ] \
                                + masky_model[j,i+1] * tauy_annclim[j,i+1] \
                                + masky_model[j-1,i] * tauy_annclim[j-1,i] \
                                + masky_model[j-1,i+1] * tauy_annclim[j-1,i+1]
            mask_tmp = masky_model[j,i] + masky_model[j,i+1] \
                     + masky_model[j-1,i] + masky_model[j-1,i+1]
            if (mask_tmp > 0):
                tauy_annclim_x[j,i] = tauy_annclim_x[j,i] / mask_tmp
                                
        tauy_annclim_x[j,nx-1] = masky_model[j,nx-1] * tauy_annclim[j,nx-1] \
                               + masky_model[j,0] * tauy_annclim[j,0] \
                               + masky_model[j-1,nx-1] * tauy_annclim[j-1,nx-1] \
                               + masky_model[j-1,0] * tauy_annclim[j-1,0]
        mask_tmp = masky_model[j,nx-1] + masky_model[j,0] \
                 + masky_model[j-1,nx-1] + masky_model[j-1,0]
        if (mask_tmp > 0):
            tauy_annclim_x[j,nx-1] = tauy_annclim_x[j,nx-1] / mask_tmp

    #for j in range(0,ny):
    #    print(j, lon[j,0], lon[j,nx-1], tauy_annclim_x[j,0], tauy_annclim_x[j,nx-1])

    for j in range(ny):
        for i in range(nx):
            if (maskx_model[j,i] == 1):
                if (i == 0):
                    vecx=lon[j,1]-lon[j,nx-1]
                    vecy=lat[j,1]-lat[j,nx-1]
                elif (i == nx - 1):
                    vecx=lon[j,0]-lon[j,nx-2]
                    vecy=lat[j,0]-lat[j,nx-2]
                else:               
                    vecx=lon[j,i+1]-lon[j,i-1]
                    vecy=lat[j,i+1]-lat[j,i-1]

                if (vecx > 180.0):
                    vecx = vecx - 360.0
                elif (vecx < -180.0):
                    vecx = vecx + 360.0

                rcos = vecx / np.sqrt(vecx ** 2 + vecy ** 2)
                rsin = np.sqrt(1.0 - rcos**2) * np.sin(-vecy)
                
                rcos_g[j,i] = rcos
                rsin_g[j,i] = rsin

                taux_g[j,i] = taux_annclim[j,i] * rcos - tauy_annclim_x[j,i] * rsin
                tauy_g[j,i] = taux_annclim[j,i] * rsin + tauy_annclim_x[j,i] * rcos

            else:
                taux_g[j,i] = np.NaN
                tauy_g[j,i] = np.NaN
                rcos_g[j,i] = np.NaN
                rsin_g[j,i] = np.NaN
                    
    #for j in range(0,ny):
    #    print(j, lon_[j,90], lat_[j,90], taux_annclim[j,90], tauy_annclim[j,90])

    lon_ = lon.reshape(lon.size)
    lat_ = lat.reshape(lat.size)

    lon2 = np.hstack((lon_-360,lon_,lon_+360))
    lat2 = np.hstack((lat_,lat_,lat_))

    taux_ = taux_g.reshape(taux_g.size)
    tauy_ = tauy_g.reshape(tauy_g.size)
    rcos_ = rcos_g.reshape(rcos_g.size)
    rsin_ = rsin_g.reshape(rsin_g.size)

    taux_2 = np.hstack((taux_,taux_,taux_))
    tauy_2 = np.hstack((tauy_,tauy_,tauy_))
 
    rcos_2 = np.hstack((rcos_,rcos_,rcos_))
    rsin_2 = np.hstack((rsin_,rsin_,rsin_))

    #print(taux_2.shape,tauy_2.shape)
    #print(lon2.shape,lat2.shape)
    woa_taux = griddata((lon2,lat2),taux_2,(x,y),method='linear').T
    woa_tauy = griddata((lon2,lat2),tauy_2,(x,y),method='linear').T
    woa_rcos = griddata((lon2,lat2),rcos_2,(x,y),method='linear').T
    woa_rsin = griddata((lon2,lat2),rsin_2,(x,y),method='linear').T
                    
    #for j in range(0,180):
    #    print(j, x[180,j], y[180,j], woa_taux[j,180], woa_tauy[j,180])

    del taux_annclim
    del tauy_annclim
    del tauy_annclim_x
    del dayx_annclim
    del dayy_annclim
    del taux_g
    del tauy_g
    del rcos_g
    del rsin_g
    del maskx_model
    del masky_model

    da_woa_taux = xr.DataArray(woa_taux, dims=['lat','lon'], coords=[('lat',rlat),('lon',rlon)])
    da_woa_tauy = xr.DataArray(woa_tauy, dims=['lat','lon'], coords=[('lat',rlat),('lon',rlon)])
    da_woa_rcos = xr.DataArray(woa_rcos, dims=['lat','lon'], coords=[('lat',rlat),('lon',rlon)])
    da_woa_rsin = xr.DataArray(woa_rsin, dims=['lat','lon'], coords=[('lat',rlat),('lon',rlon)])

    ##############################################################
    # Output to netCDF4

    #da_woa.to_netcdf('hoge.nc')

    fannx_out = path_out + '/' + 'tauuo_annclim_' + model + '_' + mip + '_' + str(styr) + str(stmn).zfill(2) + '_' + str(edyr) + str(edmn).zfill(2) + '.nc'
    fanny_out = path_out + '/' + 'tauvo_annclim_' + model + '_' + mip + '_' + str(styr) + str(stmn).zfill(2) + '_' + str(edyr) + str(edmn).zfill(2) + '.nc'

    lon_bnds_ = np.array(np.empty((360,2)))
    lat_bnds_ = np.array(np.empty((180,2)))

    lon_bnds_[:,0] = rlon[:] - 0.5
    lon_bnds_[:,1] = rlon[:] + 0.5
    lat_bnds_[:,0] = rlat[:] - 0.5
    lat_bnds_[:,1] = rlat[:] + 0.5


    ncannx = netCDF4.Dataset(fannx_out, 'w', format='NETCDF4')
    ncanny = netCDF4.Dataset(fanny_out, 'w', format='NETCDF4')
    ncannx.createDimension('lon', 360)
    ncannx.createDimension('lat', 180)
    ncannx.createDimension('bnds', 2)
    ncanny.createDimension('lon', 360)
    ncanny.createDimension('lat', 180)
    ncanny.createDimension('bnds', 2)

    tauxann = ncannx.createVariable('tauuo', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
    tauxann.long_name = 'Surface Downward X stress'
    tauxann.units = 'N m-2'
    tauxann.missing_value = -9.99e33

    tauyann = ncanny.createVariable('tauvo', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
    tauyann.long_name = 'Surface Downward Y stress'
    tauyann.units = 'N m-2'
    tauyann.missing_value = -9.99e33

    latx = ncannx.createVariable('lat', np.dtype('float').char, ('lat'))
    latx.long_name = 'latitude'
    latx.units = 'degrees_north'
    latx.axis = 'Y'
    latx.standard_name = 'latitude'
    latx_bnds = ncannx.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

    laty = ncanny.createVariable('lat', np.dtype('float').char, ('lat'))
    laty.long_name = 'latitude'
    laty.units = 'degrees_north'
    laty.axis = 'Y'
    laty.standard_name = 'latitude'
    laty_bnds = ncanny.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))
    
    lonx = ncannx.createVariable('lon', np.dtype('float').char, ('lon'))
    lonx.long_name = 'longitude'
    lonx.units = 'degrees_east'
    lonx.axis = 'X'
    lonx.standard_name = 'latitude'
    lonx_bnds = ncannx.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

    lony = ncanny.createVariable('lon', np.dtype('float').char, ('lon'))
    lony.long_name = 'longitude'
    lony.units = 'degrees_east'
    lony.axis = 'X'
    lony.standard_name = 'latitude'
    lony_bnds = ncanny.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

    tauxann[:,:] = np.where(np.isnan(woa_taux), -9.99e33, woa_taux)
    lonx[:] = rlon
    latx[:] = rlat
    latx_bnds[:,:] = lat_bnds_
    lonx_bnds[:,:] = lon_bnds_

    tauyann[:,:] = np.where(np.isnan(woa_tauy), -9.99e33, woa_tauy)
    lony[:] = rlon
    laty[:] = rlat
    laty_bnds[:,:] = lat_bnds_
    lony_bnds[:,:] = lon_bnds_

    ncannx.description="Annual mean " + str(styr) + str(stmn) + " through " + str(edyr) + str(edmn)
    ncanny.description="Annual mean " + str(styr) + str(stmn) + " through " + str(edyr) + str(edmn)

    ncannx.close()
    ncanny.close()

    #############################################################################
    

    fig = plt.figure(figsize=(16,12))

    proj = ccrs.PlateCarree(central_longitude=-140.)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    ax1 = plt.subplot(2,2,1,projection=proj)
    ax2 = plt.subplot(2,2,2,projection=proj)
    ax3 = plt.subplot(2,2,3,projection=proj)
    ax4 = plt.subplot(2,2,4,projection=proj)

    da_woa_taux.plot(ax=ax1, vmin=-0.15, 
                     extend='both',
                     cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.7},
                     add_labels=False,
                     transform=ccrs.PlateCarree())
    da_woa_tauy.plot(ax=ax2, vmin=-0.15, 
                     extend='both',
                     cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.7},
                     add_labels=False,
                     transform=ccrs.PlateCarree())

    da_woa_rcos.plot(ax=ax3, vmin=-1.0, 
                     cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.7},
                     add_labels=False,
                     transform=ccrs.PlateCarree())
    da_woa_rsin.plot(ax=ax4, vmin=-1.0, 
                     cbar_kwargs={'orientation': 'horizontal', 'shrink': 0.7},
                     add_labels=False,
                     transform=ccrs.PlateCarree())

    ax1.coastlines()
    ax1.set_xticks(np.arange(-180,180.1,60), crs=ccrs.PlateCarree())
    ax1.set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_title('TAUX')

    ax2.coastlines()
    ax2.set_xticks(np.arange(-180,180.1,60), crs=ccrs.PlateCarree())
    ax2.set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    ax2.set_title('TAUY')

    ax3.coastlines()
    ax3.set_xticks(np.arange(-180,180.1,60), crs=ccrs.PlateCarree())
    ax3.set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)
    ax3.set_title('RCOS')

    ax4.coastlines()
    ax4.set_xticks(np.arange(-180,180.1,60), crs=ccrs.PlateCarree())
    ax4.set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)
    ax4.set_title('RSIN')

    plt.savefig('tau.png', bbox_inches='tight', pad_inches=0.0)
    plt.savefig('tau.eps', bbox_inches='tight', pad_inches=0.0)

    plt.show()
