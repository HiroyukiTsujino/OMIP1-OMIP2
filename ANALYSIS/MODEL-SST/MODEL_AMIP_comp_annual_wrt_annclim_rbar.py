
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
    print ('Usage: ' + sys.argv[0] + ' mip_id start_year end_year')
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

#-- READ OMIP SST
if (mip == 'omip1'):
    start_yr_omip = 1948
    yr_range='1948-2009'
elif (mip == 'omip2'):
    start_yr_omip = 1958
    yr_range='1958-2018'
else:
    print(' Invalid mip id :', mip)
    sys.exit()

#----------------------------------------------

path_amip = '../refdata/PCMDI-SST'

# cell area

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

# climatology (long-term mean)

path_amip_cl = '../analysis/SST/PCMDI-SST'
amipann = path_amip_cl + '/' + 'tos_annclim_gn_198001-200912.nc'
ncamipann = netCDF4.Dataset(amipann,'r')
sstamip_ann = ncamipann.variables['tos'][:,:]
miss_val_amipann = ncamipann.variables['tos'].missing_value

print (nx,ny, miss_val_amipann)

#-- read AMIP SST
path_amip_ann =  '../analysis/SST/PCMDI-SST'
amipsst = path_amip_ann + '/AMIP-tos_annual_gn_1870-2017.nc'
ncamip = netCDF4.Dataset(amipsst,'r')
undef_val_amip = ncamip.variables['tos'].missing_value
undef_val_amip = undef_val_amip * 0.9
start_yr_amip = 1870


#----------------------------------------------

dict_annual={}

path_omip_cl= '../analysis/SST/MODEL'
path_omip_ann= '../analysis/SST/MODEL'

r_t = np.array(np.empty(edyr-styr+1),dtype=np.float64)

nm = 0
for model in metainfo.keys():

    print (' ')
    print ('Processing ', model)

    omipann = path_omip_cl + '/' + 'tos_annclim_' + model + '_' + mip + '_198001-200912.nc'
    ncomipann = netCDF4.Dataset(omipann,'r')
    sstomip_ann = ncomipann.variables['tos'][:,:]
    miss_val_omipann = ncomipann.variables['tos'].missing_value

    #print (nx,ny, miss_val_omipann)

    omipmskf = path_omip_cl + '/' + 'tos_mask_' + model + '_' + mip + '_198001-200912.nc'
    ncmskomip = netCDF4.Dataset(omipmskf,'r')
    maskomip_tmp = ncmskomip.variables['tosmask'][:,:]
    ncmskomip.close()

    maskomip = maskomip_tmp.astype(np.float64)

    #----------------------------------------------

    maskboth = maskomip * maskamip
    wgtboth = maskboth * area_norm

    wgt_all = np.array(np.empty(((edyr-styr+1),ny,nx)),dtype=np.float64)
    amipsst_anomg = np.array(np.empty(((edyr-styr+1),ny,nx)),dtype=np.float64)
    omipsst_anomg = np.array(np.empty(((edyr-styr+1),ny,nx)),dtype=np.float64)
    amipsst_anoml = np.array(np.empty(((edyr-styr+1),ny,nx)),dtype=np.float64)
    omipsst_anoml = np.array(np.empty(((edyr-styr+1),ny,nx)),dtype=np.float64)

    omipsst = path_omip_ann + '/' + 'tos_annual_' + model + '_' + mip + '_' + yr_range + '.nc'
    ncomip = netCDF4.Dataset(omipsst,'r')
    undef_val_omip = ncomip.variables['tos'].missing_value
    undef_val_omip = undef_val_omip * 0.9

    wgtboth_sum = wgtboth.sum()

    amip_overall = (wgtboth*sstamip_ann).sum() / wgtboth_sum
    omip_overall = (wgtboth*sstomip_ann).sum() / wgtboth_sum
    print(wgtboth_sum, amip_overall, omip_overall)
    
    for yr in range(styr,edyr+1):

        rec_base = yr - start_yr_amip
        recn = rec_base
        recd = yr - styr
        #print (yr,recn,recd)
        sstamip_tmp = ncamip.variables['tos'][recn,:,:]
        sstamip = sstamip_tmp.astype(np.float64)
        undef_flags = (sstamip < undef_val_amip)
        sstamip[undef_flags] = np.NaN
        amip_global = np.nansum(wgtboth*sstamip) / wgtboth_sum
        #print(amip_global)
        amipsst_anomg[recd,:,:] = maskboth[:,:] * (sstamip[:,:] - sstamip_ann[:,:] - (amip_global - amip_overall))
        amipsst_anoml[recd,:,:] = maskboth[:,:] * (sstamip[:,:] - sstamip_ann[:,:])
        wgt_all[recd,:,:] = wgtboth[:,:] 

    for yr in range(styr,edyr+1):

        rec_base = yr - start_yr_omip
        recn = rec_base
        recd = yr - styr
        #print (yr,recn,recd)
        sstomip = ncomip.variables['tos'][recn,:,:]
        undef_flags = (sstomip < undef_val_omip)
        sstomip[undef_flags] = np.NaN
        omip_global = np.nansum(wgtboth*sstomip) / wgtboth_sum
        #print(omip_global)
        omipsst_anomg[recd,:,:] = maskboth[:,:] * (sstomip[:,:] - sstomip_ann[:,:] - (omip_global - omip_overall))
        omipsst_anoml[recd,:,:] = maskboth[:,:] * (sstomip[:,:] - sstomip_ann[:,:])

    #----------------
    # Statistics

    for yr in range(styr,edyr+1):

        recd = yr - styr
        varamip_sum = (amipsst_anomg[recd,:,:]**2 * wgtboth[:,:]).sum()
        varomip_sum = (omipsst_anomg[recd,:,:]**2 * wgtboth[:,:]).sum()

        stdamip = np.sqrt(varamip_sum / wgtboth_sum)
        stdomip = np.sqrt(varomip_sum / wgtboth_sum)

        corr_sum = ((amipsst_anomg[recd,:,:])*(omipsst_anomg[recd,:,:]) * wgtboth[:,:]).sum()
        r_t[recd] = corr_sum / stdamip / stdomip / wgtboth_sum

        print(yr, ' stdamip stdomip r_t  ', stdamip, stdomip, r_t[recd])

    rbar = r_t.sum()/(edyr-styr+1)

    print('RBAR = ', rbar)
        
    varamip_sum = (amipsst_anoml**2 * wgt_all).sum()
    varomip_sum = (omipsst_anoml**2 * wgt_all).sum()

    stdamip_sum = np.sqrt(varamip_sum)
    stdomip_sum = np.sqrt(varomip_sum)

    sites_sum = ((sstamip_ann - sstomip_ann)**2 * wgtboth).sum()
    sites = sites_sum * (edyr - styr + 1) / stdamip_sum / stdomip_sum
    print('SITES ', sites)

    dict_annual[model]=[rbar,sites]

    del wgt_all
    del amipsst_anomg
    del omipsst_anomg
    del amipsst_anoml
    del omipsst_anoml
    nm += 1
    #if (nm == 1):
    #    break

summary=pd.DataFrame(dict_annual,index=['RBAR','SITES'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/sst_rbar_sites_annual_' + mip + '_corrected.csv')
