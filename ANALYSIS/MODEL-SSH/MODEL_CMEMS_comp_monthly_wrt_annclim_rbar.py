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

if (len(sys.argv) < 4) :
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year end_year filtered(yes or no)')
    sys.exit()

mip = sys.argv[1]
styr = int(sys.argv[2])
edyr = int(sys.argv[3])
filtered = sys.argv[4]

#--------------------

metainfo = json.load(open("./json/zos_" + mip + ".json"))
model_list = metainfo.keys()

#--------------------

mon_days = np.array([31,28,31,30,31,30,31,31,30,31,30,31],dtype=np.int64)
print(mon_days)

if (mip == 'omip1'):
    start_yr_omip = 1948
elif (mip == 'omip2'):
    start_yr_omip = 1958
else:
    print(' Invalid mip id :', mip)
    sys.exit()

#----------------------------------------------

# cell area

path_amip = '../refdata/PCMDI-SST'
arefile = path_amip + '/' + 'areacello_input4MIPs_SSTsAndSeaIce_CMIP_PCMDI-AMIP-1-1-4_gn.nc'
ncare = netCDF4.Dataset(arefile,'r')
nx = len(ncare.dimensions['lon'])
ny = len(ncare.dimensions['lat'])
lon_ = ncare.variables['lon'][:]
lat_ = ncare.variables['lat'][:]
area = ncare.variables['areacello'][:,:]
ncare.close()

# normalize area

area_norm = 2.0 * area / (area[89,0] + area[90,0])
#print(area_norm[0:ny,0])

# mask

path_cmems = '../analysis/SSH/CMEMS'

if (filtered == 'yes'):
    cmemsmskf= path_cmems + '/' + 'zos_filter_mask_gn_199301-200912.nc'
else:
    cmemsmskf= path_cmems + '/' + 'zos_mask_gn_199301-200912.nc'
    
ncmskcmems = netCDF4.Dataset(cmemsmskf,'r')
maskcmems = ncmskcmems.variables['zosmask'][:,:]
ncmskcmems.close()

################################################
# Ad hoc modification for Mediterranean (mask out entirely)

maskcmems[120:140,0:40] = 0
maskcmems[120:130,355:360] = 0
################################################

# annual mean climatology

if (filtered == 'yes'):
    cmemsann = path_cmems + '/' + 'zos_filter_annclim_gn_199301-200912.nc'
else:
    cmemsann = path_cmems + '/' + 'zos_annclim_gn_199301-200912.nc'

nccmemsann = netCDF4.Dataset(cmemsann,'r')
sshcmems_ann = nccmemsann.variables['zos'][:,:]
miss_val_cmemsann = nccmemsann.variables['zos'].missing_value

print (nx,ny, miss_val_cmemsann)

path_cmems_org = '../refdata/CMEMS'
if (filtered == 'yes'):
    cmemsssh = path_cmems_org + '/zos_adt_filter_CMEMS_1x1_monthly_199301-201812.nc'
else:
    cmemsssh = path_cmems_org + '/zos_adt_CMEMS_1x1_monthly_199301-201812.nc'

nccmems = netCDF4.Dataset(cmemsssh,'r')
#undef_val_cmems = nccmems.variables['zos'].missing_value
undef_val_cmems = -9.0e33
start_yr_cmems = 1993

#----------------------------------------------

dict_monthly={}

path_omip = '../analysis/SSH/MODEL'
path_omip_cl= '../analysis/SSH/MODEL'

r_t = np.array(np.empty((edyr-styr+1)*12),dtype=np.float64)

for model in metainfo.keys():

    gtorlt = metainfo[model]['gtorlt']
    undef_nan = metainfo[model]['undefnan']

    print (' ')
    print ('Processing ', model)

    omipann = path_omip_cl + '/' + 'zos_annclim_' + model + '_' + mip + '_199301-200912.nc'
    ncomipann = netCDF4.Dataset(omipann,'r')
    sshomip_ann = ncomipann.variables['zos'][:,:]
    miss_val_omipann = ncomipann.variables['zos'].missing_value

    #print (nx,ny, miss_val_omipann)

    omipmskf = path_omip_cl + '/' + 'zos_mask_' + model + '_' + mip + '_199301-200912.nc'
    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip_tmp = ncmskomip.variables['zosmask'][:,:]
    ncmskomip.close()

    maskomip = maskomip_tmp.astype(np.float64)

    #----------------------------------------------

    maskboth = maskomip * maskcmems
    wgtboth = maskboth * area
    wgtboth_sum = (wgtboth).sum()

    wgt_all = np.array(np.empty(((edyr-styr+1)*12,ny,nx)),dtype=np.float64)
    cmemsssh_anomg = np.array(np.empty(((edyr-styr+1)*12,ny,nx)),dtype=np.float64)
    omipssh_anomg = np.array(np.empty(((edyr-styr+1)*12,ny,nx)),dtype=np.float64)
    cmemsssh_anoml = np.array(np.empty(((edyr-styr+1)*12,ny,nx)),dtype=np.float64)
    omipssh_anoml = np.array(np.empty(((edyr-styr+1)*12,ny,nx)),dtype=np.float64)

    cmemsann_sum = (sshcmems_ann*wgtboth).sum()
    cmemsann_mean = cmemsann_sum / wgtboth_sum
    omipann_sum = (sshomip_ann*wgtboth).sum()
    omipann_mean = omipann_sum / wgtboth_sum
    #print (cmemsann_mean, omipann_mean)

    #------

    omipssh = metainfo[model]['path'] + '/' + metainfo[model]['fname']
    ncomip = netCDF4.Dataset(omipssh,'r')
    undef_val_omip = float(metainfo[model]['undef'])

    for yr in range(styr,edyr+1):

        rec_base_cmems = (yr-start_yr_cmems)*12
        rec_base_omip = (yr-start_yr_omip)*12

        for mn in range(1,13):

            recn_cmems = rec_base_cmems + mn - 1
            recn_omip  = rec_base_omip + mn - 1

            recd = (yr - styr) * 12 + mn - 1
            sshcmems_tmp = nccmems.variables['zos'][recn_cmems,:,:]
            sshcmems = sshcmems_tmp.astype(np.float64)
            cmemsssh_sum = (sshcmems[:,:]*wgtboth[:,:]).sum()
            cmemsssh_mean = cmemsssh_sum/wgtboth_sum
            undef_flags = (sshcmems < undef_val_cmems)
            sshcmems[undef_flags] = np.NaN

            sshomip = ncomip.variables['zos'][recn_omip,:,:]

            if ( model == 'NorESM-BLOM' ):
                ssh_rot = np.roll(sshomip, 180, axis=1)
                sshomip = ssh_rot

            if ( model == 'MIROC-COCO4.9' ):
                ssh_flip = np.flip(sshomip, axis=0)
                sshomip = ssh_flip

            omipssh_sum = (sshomip[:,:]*wgtboth[:,:]).sum()
            omipssh_mean = omipssh_sum/wgtboth_sum

            if (undef_nan == 'False'):
                if (gtorlt == 'gt'):
                    undef_flags = (sshomip > undef_val_omip)
                else:
                    undef_flags = (sshomip < undef_val_omip)
            else:
                undef_flags = np.isnan(sshomip)

            sshomip[undef_flags] = np.NaN
            
            #print (yr,mn,recn_cmems,recn_omip,recd,cmemsssh_mean, omipssh_mean)

            cmemsssh_anomg[recd,:,:] = maskboth[:,:] * \
                ((sshcmems[:,:] - cmemsssh_mean) - (sshcmems_ann[:,:] - cmemsann_mean))
            omipssh_anomg[recd,:,:] = maskboth[:,:] * (sshomip[:,:] - sshomip_ann[:,:] - omipssh_mean + omipann_mean)

            cmemsssh_anoml[recd,:,:] = maskboth[:,:] * \
                ((sshcmems[:,:] - cmemsssh_mean + omipssh_mean) \
                 - (sshcmems_ann[:,:] - cmemsann_mean + omipann_mean))
            omipssh_anoml[recd,:,:] = maskboth[:,:] * (sshomip[:,:] - sshomip_ann[:,:])
            wgt_all[recd,:,:] = wgtboth[:,:] * mon_days[mn-1]

    #----------------
    # Statistics

    for yr in range(styr,edyr+1):
        for mn in range(1,13):

            recd = (yr - styr) * 12 + mn - 1

            varcmems_sum = (cmemsssh_anomg[recd,:,:]**2 * wgtboth[:,:]).sum()
            varomip_sum = (omipssh_anomg[recd,:,:]**2 * wgtboth[:,:]).sum()

            stdcmems = np.sqrt(varcmems_sum / wgtboth_sum)
            stdomip = np.sqrt(varomip_sum / wgtboth_sum)

            corr_sum = ((cmemsssh_anomg[recd,:,:])*(omipssh_anomg[recd,:,:]) * wgtboth[:,:]).sum()
            r_t[recd] = corr_sum / stdcmems / stdomip / wgtboth_sum * mon_days[mn-1]

            print(yr, mn, ' stdcmems stdomip r_t  ', stdcmems, stdomip, r_t[recd])

    rbar = r_t.sum()/(365*(edyr-styr+1))
    print('RBAR = ', rbar)

    wgt_all_sum = (wgt_all).sum()

    varcmems_sum = ((cmemsssh_anoml)**2 * wgt_all).sum()
    varomip_sum = ((omipssh_anoml)**2 * wgt_all).sum()

    stdcmems_sum = np.sqrt(varcmems_sum)
    stdomip_sum = np.sqrt(varomip_sum)
    print('standard deviation ', stdcmems_sum, stdomip_sum)

    sites_sum = ((sshcmems_ann - cmemsann_mean + omipann_mean - sshomip_ann)**2 * wgtboth).sum()
    sites = sites_sum * 365 * (edyr - styr + 1) / stdcmems_sum / stdomip_sum
    print('SITES ', sites)

    dict_monthly[model]=[rbar,sites]

    del wgt_all
    del cmemsssh_anoml
    del omipssh_anoml
    del cmemsssh_anomg
    del omipssh_anomg

summary=pd.DataFrame(dict_monthly,index=['RBAR','SITES'])
summary_t=summary.T
print (summary_t)
if (filtered == 'yes'):
    summary_t.to_csv('csv/zos_filter_rbar_sites_monthly_' + mip + '_corrected.csv')
else:
    summary_t.to_csv('csv/zos_rbar_sites_monthly_' + mip + '_corrected.csv')
