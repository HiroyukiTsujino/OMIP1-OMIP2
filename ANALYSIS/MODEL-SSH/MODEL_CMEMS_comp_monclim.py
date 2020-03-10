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

if (len(sys.argv) < 5) :
    print ('Usage: ' + sys.argv[0] + ' mip start_year end_year filtered(yes or no)')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])
filtered = sys.argv[4]

#--------------------

metainfo = json.load(open('./json/zos_' + mip + '.json'))
model_list = metainfo.keys()

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

path_cmems = '../analysis/SSH/CMEMS'

if (filtered == 'yes'):
    cmemsann = path_cmems + '/' + 'zos_filter_annclim_gn_199301-200912.nc'
    cmemsmon = path_cmems + '/' + 'zos_filter_monclim_gn_199301-200912.nc'
    cmemsmskf= path_cmems + '/' + 'zos_filter_mask_gn_199301-200912.nc'
else:
    cmemsann = path_cmems + '/' + 'zos_annclim_gn_199301-200912.nc'
    cmemsmon = path_cmems + '/' + 'zos_monclim_gn_199301-200912.nc'
    cmemsmskf= path_cmems + '/' + 'zos_mask_gn_199301-200912.nc'
    

nccmemsann = netCDF4.Dataset(cmemsann,'r')
nx = len(nccmemsann.dimensions['lon'])
ny = len(nccmemsann.dimensions['lat'])
sshcmems_ann = nccmemsann.variables['zos'][:,:]
miss_val_cmemsann = nccmemsann.variables['zos'].missing_value
lon_ = nccmemsann.variables['lon'][:]
lat_ = nccmemsann.variables['lat'][:]

nccmemsmon = netCDF4.Dataset(cmemsmon,'r')
sshcmems_mon = nccmemsmon.variables['zos'][:,:,:]
miss_val_cmemsmon = nccmemsmon.variables['zos'].missing_value

print (nx,ny, miss_val_cmemsann, miss_val_cmemsmon)

# mask

ncmskcmems = netCDF4.Dataset(cmemsmskf,'r')
maskcmems = ncmskcmems.variables['zosmask'][:,:]
ncmskcmems.close()

################################################
# Ad hoc modification for Mediterranean (mask out entirely)
maskcmems[120:140,0:40] = 0
maskcmems[120:130,355:359] = 0
################################################

# cell area

path_amip = '../refdata/PCMDI-SST'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
area = ncare.variables['areacello'][:,:]
ncare.close()

#----------------------------------------------

dict_monclim={}

path_omip_cl= '../analysis/SSH/MODEL'
path_out = '../analysis/SSH/MODEL'

dcorr = np.empty( (len(model_list),180,360) )
nmodel=0

for model in metainfo.keys():

    print (' ')
    print ('Processing ', model)

    omipann = path_omip_cl + '/' + 'zos_annclim_' + model + '_' + mip + '_199301-200912.nc'
    ncomipann = netCDF4.Dataset(omipann,'r')
    sshomip_ann = ncomipann.variables['zos'][:,:]
    miss_val_omipann = ncomipann.variables['zos'].missing_value

    omipmon = path_omip_cl + '/' + 'zos_monclim_' + model + '_' + mip + '_199301-200912.nc'
    ncomipmon = netCDF4.Dataset(omipmon,'r')
    sshomip_mon = ncomipmon.variables['zos'][:,:]
    miss_val_omipmon = ncomipmon.variables['zos'].missing_value

    #print (nx,ny, miss_val_omipann, miss_val_omipmon)

    omipmskf = path_omip_cl + '/' + 'zos_mask_' + model + '_' + mip + '_199301-200912.nc'
    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip_tmp = ncmskomip.variables['zosmask'][:,:]
    ncmskomip.close()

    maskomip = maskomip_tmp.astype(np.float64)

    #----------------------------------------------

    maskboth = maskomip * maskcmems
    wgtboth = maskboth * area
    wgtboth_sum = (wgtboth).sum()

    wgt_all = np.array(np.empty((12,ny,nx)),dtype=np.float64)
    cmemsssh_anom = np.array(np.empty((12,ny,nx)),dtype=np.float64)
    omipssh_anom = np.array(np.empty((12,ny,nx)),dtype=np.float64)

    cmemsann_sum = (sshcmems_ann*wgtboth).sum()
    cmemsann_mean = cmemsann_sum/wgtboth_sum
    omipann_sum = (sshomip_ann*wgtboth).sum()
    omipann_mean = omipann_sum/wgtboth_sum
    print ('annual mean ', cmemsann_mean, omipann_mean)

    cmemsmon_sum = np.array(np.empty((12)),dtype=np.float64)
    cmemsmon_mean = np.array(np.empty((12)),dtype=np.float64)
    omipmon_sum = np.array(np.empty((12)),dtype=np.float64)
    omipmon_mean = np.array(np.empty((12)),dtype=np.float64)

    for mn in range(1,13):
        cmemsmon_sum[mn-1] = (sshcmems_mon[mn-1,:,:]*wgtboth[:,:]).sum()
        cmemsmon_mean[mn-1] = cmemsmon_sum[mn-1]/wgtboth_sum
        omipmon_sum[mn-1] = (sshomip_mon[mn-1,:,:]*wgtboth[:,:]).sum()
        omipmon_mean[mn-1] = omipmon_sum[mn-1]/wgtboth_sum
        print (mn, cmemsmon_mean[mn-1], omipmon_mean[mn-1])

    for mn in range(1,13):
        cmemsssh_anom[mn-1,:,:] = maskboth[:,:] * ((sshcmems_mon[mn-1,:,:]-cmemsmon_mean[mn-1] + omipmon_mean[mn-1])-(sshcmems_ann[:,:]-cmemsann_mean + omipann_mean))
        omipssh_anom[mn-1,:,:] = maskboth[:,:] * (sshomip_mon[mn-1,:,:]-sshomip_ann[:,:])
        wgt_all[mn-1,:,:] = wgtboth[:,:] * mon_days[mn-1]


    #-----------------------------------------------

    wgt_all_sum = (wgt_all).sum()

    cmemsssh_sum = (cmemsssh_anom*wgt_all).sum()
    cmemsssh_mean = cmemsssh_sum / wgt_all_sum

    omipssh_sum = (omipssh_anom*wgt_all).sum()
    omipssh_mean = omipssh_sum / wgt_all_sum
    print('mean ', cmemsssh_mean, omipssh_mean)

    varcmems_sum = ((cmemsssh_anom-cmemsssh_mean)**2 * wgt_all).sum()
    varomip_sum = ((omipssh_anom-omipssh_mean)**2 * wgt_all).sum()

    varcmems = varcmems_sum/wgt_all_sum
    varomip = varomip_sum/wgt_all_sum
    print('variance ', varcmems, varomip)

    stdcmems = np.sqrt(varcmems)
    stdomip = np.sqrt(varomip)
    print('standard deviation ', stdcmems, stdomip)

    corr_sum = ((cmemsssh_anom-cmemsssh_mean)*(omipssh_anom-omipssh_mean) * wgt_all).sum()
    corr = corr_sum / wgt_all_sum / stdcmems / stdomip
    print('Correlation ', corr)

    dist_sum = (((cmemsssh_anom-cmemsssh_mean) - (omipssh_anom-omipssh_mean))**2 * wgt_all).sum()
    dist = dist_sum / wgt_all_sum
    print('Distance (raw)          ', dist)

    dist_tmp = varcmems + varomip - 2.0 * stdcmems * stdomip * corr
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
            maskboth[:,:] * mon_days[mn-1] * omipssh_anom[mn-1,:,:] ** 2
        varo_local[:,:] = varo_local[:,:] + \
            maskboth[:,:] * mon_days[mn-1] * cmemsssh_anom[mn-1,:,:] ** 2
        rmsd_local[:,:] = rmsd_local[:,:] + \
            maskboth[:,:] * mon_days[mn-1] * (omipssh_anom[mn-1,:,:] - cmemsssh_anom[mn-1,:,:]) ** 2

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
            * omipssh_anom[mn-1,:,:] * cmemsssh_anom[mn-1,:,:]

    corr_local = mask_local * corr_local / (1.0 - mask_local + wgt_local) / (1.0 - mask_local + stdm_local * stdo_local)
    stdm_local = np.where(mask_local == 0.0, np.NaN, stdm_local)
    corr_local = np.where(mask_local == 0.0, np.NaN, corr_local)
    rmsd_local = np.where(mask_local == 0.0, np.NaN, rmsd_local)
    ramp_local = np.where(mask_local == 0.0, np.NaN, stdm_local/stdo_local)

    dcorr[nmodel] = corr_local
    nmodel += 1

    #############################################
    # Output to netCDF4

    fcor_out = path_out + '/' + 'zos_moncl_corr_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'

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

    zoscor = nccor.createVariable('zoscor', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
    zoscor.long_name = 'Correlation of monthly climatology of SSH'
    zoscor.units = '1'
    zoscor.missing_value = -9.99e33

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

    zoscor[:,:]=np.where(np.isnan(corr_local), -9.99e33, corr_local)
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    nccor.description="Correlation based on " + str(styr) + " through " + str(edyr)

    nccor.close()

    #--------------------------------------------------------
    # Draw Figures

    suptitle = 'SSH monthly climatology statistics ' + model + ' ' + ' ' + mip
    #title = [ 'Standard deviation' , 'Correlation']
    title = [ 'Ratio of standard deviation (model/obs)' , 'Correlation']
    cmap = [ 'RdBu_r', 'RdBu_r' ]
    outfile = 'fig/SSH_monclim_statistics_' + model + '_' + mip + '.png'

    #ct1 = np.arange(0,1050,50)
    #ct1 = np.arange(-2,2,0.2)
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
    del cmemsssh_anom
    del omipssh_anom
    del cmemsmon_sum
    del cmemsmon_mean
    del omipmon_sum
    del omipmon_mean

dict_monclim['Reference']=[1.0,stdcmems]
summary=pd.DataFrame(dict_monclim,index=['Correlation','Standard deviation'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/zos_monclim_' + mip + '.csv')

# ----------
dcorr_mean = dcorr.mean(axis=0)
title_mmm = 'Correlation of monthly climatology of SSH (MMM) ' + mip
cmap_mmm = 'RdBu_r'
out_mmm = 'fig/SSH_monclim_correlation-mmm-' + mip + '.png'

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

