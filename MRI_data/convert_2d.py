# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr


if len(sys.argv) != 2:
    print ('Usage: python3 convert_scalar.py jsonfile')
    sys.exit()

metainfo = json.load(open(sys.argv[1]))


x_bnds = np.stack( [ np.linspace(0,359,360), np.linspace(1,360,360) ], 1 )
y_bnds = np.stack( [ np.linspace(-90,89,180), np.linspace(-89,90,180) ], 1 )



for ofile in metainfo.keys():

    ifile = metainfo[ofile]['ifile']
    ystr  = int(metainfo[ofile]['ystr'])
    yend  = int(metainfo[ofile]['yend'])
    name  = metainfo[ofile]['name']
    outname = metainfo[ofile]['outname']
#    standard_name = metainfo[ofile]['standard_name']
#    units = metainfo[ofile]['units']


    encoding = { outname:{'_FillValue': -9.99e33}, \
                 'lat':{'_FillValue': False}, \
                 'lat_bnds':{'_FillValue': False}, \
                 'lon':{'_FillValue': False}, \
                 'lon_bnds':{'_FillValue': False}, \
                 'time':{'_FillValue': False, 'dtype': 'f8'} }

    files = []
    for yr in range(ystr,yend+1):
        files.append( ifile + str(yr) )


    ds = xr.open_mfdataset( files )

    if name != outname:
        ds = ds.rename({name:outname})

    del ds[outname].attrs["_Fillvalue"]

    ds["lon"] = ds.lon.assign_attrs({"units": "degrees_east", \
                                     "long_name": "longitude", \
                                     "standard_name": "longitude", \
                                     "bounds": "lon_bnds" })
    ds["lat"] = ds.lat.assign_attrs({"units": "degrees_north", \
                                     "long_name": "latitude", \
                                     "standard_name": "latitude", \
                                     "bounds": "lat_bnds" })
    ds["time"] = ds.time.assign_attrs({"long_name": "time"})

    ds_bnds = xr.Dataset({'lon_bnds': (['lon','bnds'], x_bnds), \
                          'lat_bnds': (['lat','bnds'], y_bnds) } )

    ds = xr.merge( [ds[outname], ds_bnds] )

    ds.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
