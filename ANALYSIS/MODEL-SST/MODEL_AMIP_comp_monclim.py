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

if (len(sys.argv) < 4):
    print ('Usage: ' + sys.argv[0] + ' mip start_year end_year')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])

#--------------------

metainfo = json.load(open("./json/tos_" + mip + ".json"))
model_list = metainfo.keys()

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------
# AMIP data

path_amip_cl = '../analysis/SST/PCMDI-SST'
amipann = path_amip_cl + '/' + 'tos_annclim_gn_198001-200912.nc'

ncamipann = netCDF4.Dataset(amipann,'r')
nx = len(ncamipann.dimensions['lon'])
ny = len(ncamipann.dimensions['lat'])
sstamip_ann = ncamipann.variables['tos'][:,:]
miss_val_amipann = ncamipann.variables['tos'].missing_value
lon_ = ncamipann.variables['lon'][:]
lat_ = ncamipann.variables['lat'][:]

amipmon = path_amip_cl + '/' + 'tos_monclim_gn_198001-200912.nc'
ncamipmon = netCDF4.Dataset(amipmon,'r')
sstamip_mon = ncamipmon.variables['tos'][:,:,:]
miss_val_amipmon = ncamipmon.variables['tos'].missing_value

print (nx,ny, miss_val_amipann, miss_val_amipmon)

# mask

path_amip = '../refdata/PCMDI-SST'
amipmskf= path_amip + '/' + 'sftof_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'

ncmskamip = netCDF4.Dataset(amipmskf,'r')
maskamip = ncmskamip.variables['sftof'][:,:]
ncmskamip.close()

# Ad hoc modification of the Kaspian Sea
for j in range(ny):
    for i in range(nx):
        if (45 < lon_[i]) & (lon_[i] < 60):
             if (34 < lat_[j]) & (lat_[j] < 50):
                  maskamip[j,i] = 0

        if maskamip[j,i] < 100:
            maskamip[j,i] = 0

maskamip = maskamip / 100

# cell area

arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]
ncare.close()

#----------------------------------------------

dict_monclim={}

path_omip_cl= '../analysis/SST/MODEL'
path_out = '../analysis/SST/MODEL'

dcorr = np.empty( (len(model_list),180,360) )
nmodel=0
for model in metainfo.keys():

    print (' ')
    print ('Processing ', model)

    omipann = path_omip_cl + '/' + 'tos_annclim_' + model + '_' + mip + '_198001-200912.nc'

    ncomipann = netCDF4.Dataset(omipann,'r')
    sstomip_ann = ncomipann.variables['tos'][:,:]
    miss_val_omipann = ncomipann.variables['tos'].missing_value

    omipmon = path_omip_cl + '/' + 'tos_monclim_' + model + '_' + mip + '_198001-200912.nc'
    ncomipmon = netCDF4.Dataset(omipmon,'r')
    sstomip_mon = ncomipmon.variables['tos'][:,:]
    miss_val_omipmon = ncomipmon.variables['tos'].missing_value

    print (nx,ny, miss_val_omipann, miss_val_omipmon)

    omipmskf = path_omip_cl + '/' + 'tos_mask_' + model + '_' + mip + '_198001-200912.nc'
    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip_tmp = ncmskomip.variables['tosmask'][:,:]
    ncmskomip.close()

    maskomip = maskomip_tmp.astype(np.float64)

    #----------------------------------------------

    maskboth = maskomip * maskamip
    wgtboth = maskboth * area

    wgt_all = np.array(np.empty((12,ny,nx)),dtype=np.float64)
    amipsst_anom = np.array(np.empty((12,ny,nx)),dtype=np.float64)
    omipsst_anom = np.array(np.empty((12,ny,nx)),dtype=np.float64)

    for mn in range(1,13):
        amipsst_anom[mn-1,:,:] = maskboth[:,:] * (sstamip_mon[mn-1,:,:]-sstamip_ann[:,:])
        omipsst_anom[mn-1,:,:] = maskboth[:,:] * (sstomip_mon[mn-1,:,:]-sstomip_ann[:,:])
        wgt_all[mn-1,:,:] = wgtboth[:,:] * mon_days[mn-1]

    wgt_all_sum = (wgt_all).sum()

    amipsst_sum = (amipsst_anom*wgt_all).sum()
    amipsst_mean = amipsst_sum / wgt_all_sum

    omipsst_sum = (omipsst_anom*wgt_all).sum()
    omipsst_mean = omipsst_sum / wgt_all_sum
    print('mean ', amipsst_mean, omipsst_mean)

    varamip_sum = ((amipsst_anom-amipsst_mean)**2 * wgt_all).sum()
    varomip_sum = ((omipsst_anom-omipsst_mean)**2 * wgt_all).sum()

    varamip = varamip_sum/wgt_all_sum
    varomip = varomip_sum/wgt_all_sum

    stdamip = np.sqrt(varamip)
    stdomip = np.sqrt(varomip)
    print('standard deviation ', stdamip, stdomip)

    corr_sum = ((amipsst_anom-amipsst_mean)*(omipsst_anom-omipsst_mean) * wgt_all).sum()
    corr = corr_sum / wgt_all_sum / stdamip / stdomip
    print('Correlation ', corr)

    dist_sum = (((amipsst_anom-amipsst_mean) - (omipsst_anom-omipsst_mean))**2 * wgt_all).sum()
    dist = dist_sum / wgt_all_sum
    print('Distance (raw)          ', dist)

    dist_tmp = varamip + varomip - 2.0 * stdamip * stdomip * corr
    print('Distance (confirmation) ', dist_tmp)

    dict_monclim[model]=[corr,stdomip]

    del wgt_all

    #----------------------------------------------
    # local temporal correlation

    wgt_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    mask_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    varm_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    varo_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    stdm_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    stdo_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    corr_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    rmsd_local = np.array(np.zeros((ny,nx)),dtype=np.float64)
    ramp_local = np.array(np.zeros((ny,nx)),dtype=np.float64)

    for mn in range(1,13):
        wgt_local [:,:] = wgt_local [:,:] + \
            maskboth[:,:] * mon_days[mn-1]
        varm_local[:,:] = varm_local[:,:] + \
            maskboth[:,:] * mon_days[mn-1] * omipsst_anom[mn-1,:,:] ** 2
        varo_local[:,:] = varo_local[:,:] + \
            maskboth[:,:] * mon_days[mn-1] * amipsst_anom[mn-1,:,:] ** 2
        rmsd_local[:,:] = rmsd_local[:,:] + \
            maskboth[:,:] * mon_days[mn-1] * (omipsst_anom[mn-1,:,:] - amipsst_anom[mn-1,:,:]) ** 2

    mask_local = np.where(wgt_local == 365.0, 1.0, 0.0)
    varm_local = mask_local * varm_local / (1.0 - mask_local + wgt_local)
    varo_local = mask_local * varo_local / (1.0 - mask_local + wgt_local)
    rmsd_local = mask_local * rmsd_local / (1.0 - mask_local + wgt_local)
    stdm_local = np.where(mask_local == 1.0, np.sqrt(varm_local), 0.0)
    stdo_local = np.where(mask_local == 1.0, np.sqrt(varo_local), 0.0)
    rmsd_local = np.where(mask_local == 1.0, np.sqrt(rmsd_local), 0.0)

    for mn in range(1,13):
        corr_local[:,:] = corr_local[:,:] \
            + maskboth[:,:] * mon_days[mn-1] \
            * omipsst_anom[mn-1,:,:] * amipsst_anom[mn-1,:,:]

    corr_local = mask_local * corr_local / (1.0 - mask_local + wgt_local) / (1.0 - mask_local + stdm_local * stdo_local)
    stdm_local = np.where(mask_local == 0.0, np.NaN, stdm_local)
    corr_local = np.where(mask_local == 0.0, np.NaN, corr_local)
    rmsd_local = np.where(mask_local == 0.0, np.NaN, rmsd_local)
    ramp_local = np.where(mask_local == 0.0, np.NaN, stdm_local/stdo_local)

    dcorr[nmodel] = corr_local
    nmodel += 1

    #############################################
    # Output to netCDF4

    fcor_out = path_out + '/' + 'tos_moncl_corr_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'

    lon_bnds_ = np.array(np.empty((nx,2)))
    lat_bnds_ = np.array(np.empty((ny,2)))

    lon_bnds_[:,0] = lon_[:] - 0.5
    lon_bnds_[:,1] = lon_[:] + 0.5
    lat_bnds_[:,0] = lat_[:] - 0.5
    lat_bnds_[:,1] = lat_[:] + 0.5

    nccor = netCDF4.Dataset(fcor_out, 'w', format='NETCDF4')
    nccor.createDimension('lon', nx)
    nccor.createDimension('lat', ny)
    nccor.createDimension('bnds', 2)

    toscor = nccor.createVariable('toscor', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
    toscor.long_name = 'Correlation of monthly climatology of SST'
    toscor.units = '1'
    toscor.missing_value = -9.99e33

    lat = nccor.createVariable('lat', np.dtype('float').char, ('lat'))
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'
    lat.standard_name = 'latitude'
    lat_bnds = nccor.createVariable('lat_bnd', np.dtype('float').char, ('lat', 'bnds'))

    lon = nccor.createVariable('lon', np.dtype('float').char, ('lon'))
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon.standard_name = 'latitude'
    lon_bnds = nccor.createVariable('lon_bnd', np.dtype('float').char, ('lon', 'bnds'))

    toscor[:,:]=np.where(np.isnan(corr_local), -9.99e33, corr_local)
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    nccor.description="Correlation based on " + str(styr) + " through " + str(edyr)

    nccor.close()

    #--------------------------------------------------------
    # Draw Figures

    suptitle = 'SST monthly climatology statistics ' + model + ' ' + ' ' + mip
    #title = [ 'Standard deviation' , 'Correlation']
    title = [ 'Ratio of standard deviation (model/obs)' , 'Correlation']
    cmap = [ 'RdBu_r', 'RdBu_r' ]
    outfile = 'fig/SST_monclim_statistics_' + model + '_' + mip + '.png'

    #ct1 = np.arange(0,1050,50)
    #ct1 = np.arange(-2,2,0.2)
    #ct1 = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
    ct1 = np.arange(0,2.2,0.2)
    ct2 = np.arange(-1.0,1.1,0.1)

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
            #tmp=np.log10(ramp_local)
            tmp=ramp_local
            ct = ct1
        else:
            tmp=corr_local
            ct = ct2

        lon_tmp=np.array(lon_)
        tmp, lon_tmp = add_cyclic_point(tmp, coord=lon_tmp)
        ca=ax[panel].contourf(lon_tmp, lat_, tmp, ct, cmap=cmap[panel], transform=ccrs.PlateCarree())
        ax[panel].coastlines()
        ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
        ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
        ax[panel].xaxis.set_major_formatter(lon_formatter)
        ax[panel].yaxis.set_major_formatter(lat_formatter)
        ax[panel].set_title(title[panel])
        fig.colorbar(ca,ax=ax[panel],orientation='horizontal',shrink=0.7)
        del lon_tmp

    plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)
    #plt.show()

    del wgt_local
    del mask_local
    del varm_local
    del varo_local
    del stdm_local
    del stdo_local
    del corr_local
    del rmsd_local
    del ramp_local

    del amipsst_anom
    del omipsst_anom

dict_monclim['Reference']=[1.0,stdamip]
summary=pd.DataFrame(dict_monclim,index=['Correlation','Standard deviation'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/sst_monclim_' + mip + '.csv')

# ----------

dcorr_mean = dcorr.mean(axis=0)
title_mmm = 'Correlation of monthly climatology of SST (MMM) ' + mip
cmap_mmm = 'RdBu_r'
out_mmm = 'fig/SST_monclim_correlation-mmm-' + mip + '.png'

ct = np.arange(-1.0,1.1,0.1)

fig = plt.figure(figsize=(15,9))

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

axcorr = plt.subplot(1,1,1,projection=proj)

lon_tmp=np.array(lon_)
dcorr_mean, lon_tmp = add_cyclic_point(dcorr_mean, coord=lon_tmp)
ca=axcorr.contourf(lon_tmp, lat_, dcorr_mean, ct, cmap=cmap_mmm, transform=ccrs.PlateCarree())
axcorr.coastlines()
axcorr.set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
axcorr.set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
axcorr.xaxis.set_major_formatter(lon_formatter)
axcorr.yaxis.set_major_formatter(lat_formatter)
axcorr.set_title(title_mmm)
fig.colorbar(ca,ax=axcorr,orientation='horizontal',shrink=0.7)
del lon_tmp

plt.savefig(out_mmm, bbox_inches='tight', pad_inches=0.0)
#plt.show()
