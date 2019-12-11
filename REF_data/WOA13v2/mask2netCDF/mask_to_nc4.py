# -*- coding: utf-8 -*-
#import fix_proj_lib
import json
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import netCDF4
from netCDF4 import Dataset, num2date
import datetime

basininfo=json.load(open('mask_woa13.json'))

nx = 360
ny = 180
aa = -999

path_mask = '../MASK'
path_woa13 = '../DATA'

fmsk_in = path_mask + '/' + 'basinmask_01.gd'
fmsk_out = path_woa13 + '/' + 'WOA13_1x1_mask.nc'

mask = np.array(np.empty((ny,nx)),dtype=np.int32)
lon_bnds_ = np.array(np.empty((nx,2)))
lat_bnds_ = np.array(np.empty((ny,2)))
lonW =  0.5; lonE = 359.5
latS = -89.5; latN = 89.5
lon_ = np.linspace(lonW,lonE,nx)
lat_ = np.linspace(latS,latN,ny)

lon_bnds_[:,0] = lon_[:] - 0.5
lon_bnds_[:,1] = lon_[:] + 0.5
lat_bnds_[:,0] = lat_[:] - 0.5
lat_bnds_[:,1] = lat_[:] + 0.5

f1 = open(fmsk_in,'rb')
for j in range(ny):
   print (j, lat_[j])
   mask[j,:] = np.fromfile(f1, dtype = '>i', count = nx)

f1.close()

#num_bas=0
#basin_list=[int(mask[0,0])]
#for i in range(nx):
#    for j in range(ny):
#        found=0
#        for n in range(0,len(basin_list)):
#            if (mask[j,i] == basin_list[n]):
#               found=1
#               exit
#
#        if (found == 0):
#            basin_list.append(int(mask[j,i]))
#            print ("New index ", lon_[i], lat_[j], int(mask[j,i]))

basin_list = np.unique(mask)
#print ()

mask=np.where(mask==0, 3, mask)

print(basin_list)

#####

ncmsk = netCDF4.Dataset(fmsk_out, 'w', format='NETCDF4')
ncmsk.createDimension('lon', nx)
ncmsk.createDimension('lat', ny)
ncmsk.createDimension('bnds', 2)

basin_mask = ncmsk.createVariable('basin_mask', np.dtype(np.int32).char, ('lat', 'lon'), zlib=True)
basin_mask.long_name = 'Basin mask'
basin_mask.units = '1'
basin_mask.missing_value = -999

print (basin_mask.missing_value.dtype)

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

basin_mask[:,:]=mask
lon[:]=lon_
lat[:]=lat_
lat_bnds[:,:]=lat_bnds_
lon_bnds[:,:]=lon_bnds_

stmp=str(basininfo['Basin_Index'])
print(stmp)
ncmsk.Basin_Index=stmp.replace('\'','')

ncmsk.Note1="Global should eXclude " + basininfo['Basin_Index']['53'] + "(53)"
ncmsk.Note2="Atlantic Ocean (1) should eXclude " + basininfo['Basin_Index']['4'] + "(4), " + basininfo['Basin_Index']['5'] + "(5), " + basininfo['Basin_Index']['6'] + "(6), " + basininfo['Basin_Index']['9'] + "(9) "
ncmsk.Note3="Pacific Ocean (2) should INclude " + basininfo['Basin_Index']['12'] + "(12)"
ncmsk.Note4="Indian Ocean (3) should eXclude " + basininfo['Basin_Index']['7'] + "(7), whereas INclude " + basininfo['Basin_Index']['8'] + "(8), " + basininfo['Basin_Index']['56'] + "(56)"
ncmsk.Note5="Zeros in the original data found west of Darwin, Australia has been filled with " + basininfo['Basin_Index']['3'] + "(3)"



ncmsk.close()
