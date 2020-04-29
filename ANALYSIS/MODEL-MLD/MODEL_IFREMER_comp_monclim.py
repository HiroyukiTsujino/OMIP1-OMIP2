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

#---------------------------------------------------------------------

if (len(sys.argv) < 5):
    print ('Usage: '+ sys.argv[0] + ' mip start_year end_year exlab(0 or 1)')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])
exlab = int(sys.argv[4])

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

#----------------------------------------------

metainfo = json.load(open("./json/mlotst_" + mip + ".json"))
model_list = metainfo.keys()

#----------------------------------------------

path_obs = '../analysis/MLD/MLD_deBoyer_Montegut'
fobsann = path_obs + '/' + 'mld_DR003_annclim.nc'

ncobsann = netCDF4.Dataset(fobsann,'r')
nx = len(ncobsann.dimensions['lon'])
ny = len(ncobsann.dimensions['lat'])
mldobs_ann = ncobsann.variables['mlotst'][:,:]
miss_val_obsann = ncobsann.variables['mlotst'].missing_value
lon_ = ncobsann.variables['lon'][:]
lat_ = ncobsann.variables['lat'][:]

fobsmon = path_obs + '/' + 'mld_DR003_monclim.nc'
ncobsmon = netCDF4.Dataset(fobsmon,'r')
mldobs_mon = ncobsmon.variables['mlotst'][:,:,:]
miss_val_obsmon = ncobsmon.variables['mlotst'].missing_value

print (nx, ny, miss_val_obsann, miss_val_obsmon)

dtwoout = np.array(np.empty((ny,nx)),dtype=np.float64)

# mask

fobsmskf= path_obs + '/' + 'mld_DR003_mask.nc'
ncmskobs = netCDF4.Dataset(fobsmskf,'r')
maskobs = ncmskobs.variables['mldmask'][:,:]
ncmskobs.close()

# ad hoc exclusion of the Weddell Sea

for j in range(0,ny):
    for i in range (0,nx):
#        if (lat_[j] < -60.0 and lon_[i] > 300.0):
        if (lat_[j] < -60.0):
            maskobs[j,i] = 0.0

if (exlab == 1):
    for j in range(0,ny):
        for i in range (0,nx):
            if (lat_[j] > 45.0 and lat_[j] < 80.0):
                if (lon_[i] > 280.0 or lon_[i] < 30.0):
                    maskobs[j,i] = 0.0
    

#----------------------------------------------
# cell area

path_amip = '../refdata/PCMDI-SST'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nxm = len(ncare.dimensions['lon'])
nym = len(ncare.dimensions['lat'])
aream = ncare.variables['areacello'][:,:]
ncare.close()
        
#----------------------------------------------

dict_monclim={}

path_omip_cl= '../analysis/MLD/MODEL'
path_out= '../analysis/MLD/MODEL'

dcorr = np.empty( (len(model_list),90,180) )
nmodel=0

for model in metainfo.keys():

    print (' ')
    print ('Processing ', model)

    fomipann = path_omip_cl + '/' + 'mlotst_annclim_' + model + '_' + mip + '_198001-200912.nc'
    ncomipann = netCDF4.Dataset(fomipann,'r')
    mldomip_ann = ncomipann.variables['mlotst'][:,:]
    miss_val_omipann = ncomipann.variables['mlotst'].missing_value
    mldomip_ann = np.where(np.isnan(mldomip_ann),0.0,mldomip_ann)

    fomipmon = path_omip_cl + '/' + 'mlotst_monclim_' + model + '_' + mip + '_198001-200912.nc'
    ncomipmon = netCDF4.Dataset(fomipmon,'r')
    mldomip_mon = ncomipmon.variables['mlotst'][:,:]
    miss_val_omipmon = ncomipmon.variables['mlotst'].missing_value
    mldomip_mon = np.where(np.isnan(mldomip_mon),0.0,mldomip_mon)

    #print (nxm, nym, miss_val_omipann, miss_val_omipmon)

    omipmskf = path_omip_cl + '/' + 'mlotst_mask_' + model + '_' + mip + '_198001-200912.nc'
    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip = ncmskomip.variables['mldmask'][:,:]
    ncmskomip.close()

    areamask_tmp = maskomip * aream
    areamask = areamask_tmp.astype(np.float64)

    amsk_all = np.array(np.empty((ny,nx)),dtype=np.float64)
    mask_all = np.array(np.empty((ny,nx)),dtype=np.float64)
    mldannm= np.array(np.empty((ny,nx)),dtype=np.float64)
    mldmonm = np.array(np.empty((12,ny,nx)),dtype=np.float64)

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

    donein = mldomip_ann.copy()
    dtwoout[:,:] = 0.0
    onedeg2twodeg(dtwoout, donein, areamask, nx, ny, nxm, nym)

    mldannm = dtwoout / (1.0 - mask_all + amsk_all) 

    for m in range(0,12):
        donein = mldomip_mon[m,:,:]
        dtwoout[:,:] = 0.0
        onedeg2twodeg(dtwoout, donein, areamask, nx, ny, nxm, nym)
        mldmonm[m,:,:] = dtwoout / (1.0 - mask_all + amsk_all) 

    #----------------------------------------------
    # total space-time correlation

    wgtboth = amsk_all * maskobs
    wgtboth_sum = (wgtboth).sum()

    obsmld_anom = np.array(np.empty((12,ny,nx)),dtype=np.float64)
    omipmld_anom = np.array(np.empty((12,ny,nx)),dtype=np.float64)
    wgt_all = np.array(np.empty((12,ny,nx)),dtype=np.float64)

    for mn in range(1,13):
        obsmld_anom[mn-1,:,:] = mldobs_mon[mn-1,:,:] - mldobs_ann[:,:]
        omipmld_anom[mn-1,:,:] = mldmonm[mn-1,:,:] - mldannm[:,:]
        wgt_all[mn-1,:,:] = wgtboth[:,:] * mon_days[mn-1]

    wgt_all_sum = (wgt_all).sum()
    obsmld_sum = (obsmld_anom * wgt_all).sum()
    obsmld_mean = obsmld_sum / wgt_all_sum

    omipmld_sum = (omipmld_anom * wgt_all).sum()
    omipmld_mean = omipmld_sum / wgt_all_sum
    print('Mean ', obsmld_mean, omipmld_mean)

    varobs_sum = ((obsmld_anom-obsmld_mean)**2 * wgt_all).sum()
    varomip_sum = ((omipmld_anom-omipmld_mean)**2 * wgt_all).sum()

    varobs = varobs_sum/wgt_all_sum
    varomip = varomip_sum/wgt_all_sum
    print('Variance ', varobs, varomip)

    stdobs = np.sqrt(varobs)
    stdomip = np.sqrt(varomip)
    print('Standard deviation ', stdobs, stdomip)

    corr_sum = ((obsmld_anom-obsmld_mean)*(omipmld_anom-omipmld_mean) * wgt_all).sum()
    corr = corr_sum / wgt_all_sum / stdobs / stdomip
    print('Correlation ', corr)

    dist_sum = (((obsmld_anom-obsmld_mean) - (omipmld_anom-omipmld_mean))**2 * wgt_all).sum()
    dist = dist_sum / wgt_all_sum
    print('Distance (raw)          ', dist)

    dist_tmp = varobs + varomip - 2.0 * stdobs * stdomip * corr
    print('Distance (confirmation) ', dist_tmp)

    dict_monclim[model]=[corr,stdomip]

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
            mask_all[:,:] * maskobs[:,:] * mon_days[mn-1]
        varm_local[:,:] = varm_local[:,:] + \
            mask_all[:,:] * maskobs[:,:] * mon_days[mn-1] * omipmld_anom[mn-1,:,:] ** 2
        varo_local[:,:] = varo_local[:,:] + \
            mask_all[:,:] * maskobs[:,:] * mon_days[mn-1] * obsmld_anom[mn-1,:,:] ** 2
        rmsd_local[:,:] = rmsd_local[:,:] + \
            mask_all[:,:] * maskobs[:,:] * mon_days[mn-1] * (mldmonm[mn-1,:,:] - mldobs_mon[mn-1,:,:]) ** 2

    mask_local = np.where(wgt_local == 365.0, 1.0, 0.0)
    varm_local = mask_local * varm_local / (1.0 - mask_local + wgt_local)
    varo_local = mask_local * varo_local / (1.0 - mask_local + wgt_local)
    rmsd_local = mask_local * rmsd_local / (1.0 - mask_local + wgt_local)
    stdm_local = np.where(mask_local == 1.0, np.sqrt(varm_local), 0.0)
    stdo_local = np.where(mask_local == 1.0, np.sqrt(varo_local), 0.0)
    rmsd_local = np.where(mask_local == 1.0, np.sqrt(rmsd_local), 0.0)

    for mn in range(1,13):
        corr_local[:,:] = corr_local[:,:] \
            + mask_all[:,:] * maskobs[:,:] * mon_days[mn-1] \
            * omipmld_anom[mn-1,:,:] * obsmld_anom[mn-1,:,:]

    corr_local = mask_local * corr_local / (1.0 - mask_local + wgt_local) / (1.0 - mask_local + stdm_local * stdo_local)
    stdm_local = np.where(mask_local == 0.0, np.NaN, stdm_local)
    corr_local = np.where(mask_local == 0.0, np.NaN, corr_local)
    rmsd_local = np.where(mask_local == 0.0, np.NaN, rmsd_local)
    ramp_local = np.where(mask_local == 0.0, np.NaN, stdm_local/stdo_local)

    #print(wgt_local [0:ny-1,90])
    #print(mask_local[0:ny-1,90])
    #print(stdm_local[0:ny-1,90])
    #print(corr_local[0:ny-1,90])

    dcorr[nmodel] = corr_local
    nmodel += 1

    #############################################
    # Output to netCDF4

    if (exlab == 1):
        fcor_out = path_out + '/' + 'mld_moncl_corr_woNAWD_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'
    else:
        fcor_out = path_out + '/' + 'mld_moncl_corr_' + model + '_' + mip + '_' + str(styr) + '01-' + str(edyr) + '12.nc'

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

    mldcor = nccor.createVariable('mldcor', np.dtype(np.float64).char, ('lat', 'lon'), zlib=True)
    mldcor.long_name = 'Correlation of monthly climatology of MLD'
    mldcor.units = '1'
    mldcor.missing_value = -9.99e33

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

    mldcor[:,:]=np.where(np.isnan(corr_local), -9.99e33, corr_local)
    lon[:]=lon_
    lat[:]=lat_
    lat_bnds[:,:]=lat_bnds_
    lon_bnds[:,:]=lon_bnds_

    nccor.description="Correlation based on " + str(styr) + " through " + str(edyr)

    nccor.close()

    #--------------------------------------------------------
    # Draw Figures

    suptitle = 'MLD Statistics ' + model + ' ' + ' ' + mip
    #title = [ 'Standard deviation' , 'Correlation']
    title = [ 'Ratio of standard deviation (model/obs)' , 'Correlation']
    cmap = [ 'RdBu_r', 'RdBu_r' ]
    outfile = 'fig/MLD_statistics_' + model + '_' + mip + '.png'

    #ct1 = np.arange(0,1050,50)
    ct1 = np.arange(-2,2,0.2)
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
            tmp=np.log10(ramp_local)
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

    del amsk_all
    del mask_all
    del mldannm
    del mldmonm
    del obsmld_anom
    del omipmld_anom
    del wgt_all
    del wgt_local
    del mask_local
    del varm_local
    del varo_local
    del stdm_local
    del stdo_local
    del corr_local
    del rmsd_local
    del ramp_local

dict_monclim['Reference']=[1.0,stdobs]
summary=pd.DataFrame(dict_monclim,index=['Correlation','Standard deviation'])
summary_t=summary.T
print (summary_t)

if (exlab == 1):
    summary_t.to_csv('csv/mld_monclim_woNAWD_' + mip + '.csv')
else:
    summary_t.to_csv('csv/mld_monclim_' + mip + '.csv')
    
# ----------

dcorr_mean = dcorr.mean(axis=0)
title_mmm = 'Correlation of monthly climatology of MLD (MMM) ' + mip
cmap_mmm = 'RdBu_r'
out_mmm = 'fig/MLD_monclim_correlation-mmm-' + mip + '.png'

ct = np.arange(-1.0,1.1,0.1)

fig = plt.figure(figsize=(11,8))

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
