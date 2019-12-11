# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

exp_list = [ 'omip1', 'omip2' ]
ystr = [ 1948, 1958 ]
yend = [ 2009, 2018 ]

sector = np.array( [ 'global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean' ], dtype=str )

y = np.linspace(-89.5,89.5,num=180)
y_bnds = np.stack( [ np.linspace(-90,89,180), np.linspace(-89,90,180) ], 1 )



for n in range(len(exp_list)):
    exp = exp_list[n]
    files = []
    for yr in range(ystr[n],yend[n]+1):
        files.append( './indir/' + exp + '/hs_nht_tot.' + str(yr) )
        
    DS = xr.open_mfdataset(files)

    time = DS.time.values


    encoding = { "hfbasin":{"_FillValue":-9.99e33,}, \
                 "lat":{"_FillValue": False,}, \
                 "lat_bnds":{"_FillValue": False,}, }

    coords = { 'time': ( 'time', time, { 'standard_name': 'time', \
                                         'axis': 'T', } ), \
               'lat' : ( 'lat', y, { 'units': 'degrees_north', \
                                     'standard_name': 'latitude', \
                                     'long_name': 'latitude', \
                                     'axis': 'Y', } ), }

    hfbasin_attr = { 'units': DS.glb.units, \
                     'standard_name': DS.glb.standard_name, }


    hfbasin = np.empty((len(time),3,len(y)))
    for i in range(len(time)):
        hfbasin[i,0] = DS.glb[i].interp(lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})
        hfbasin[i,1] = DS.atl[i].interp(lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})
        hfbasin[i,2] = DS.inp[i].interp(lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})


    ds_out = xr.Dataset({'hfbasin': (['time','basin','lat'], np.float32(hfbasin), hfbasin_attr), \
                         'sector': (['basin'], sector), \
                         'lat_bnds': (['lat','bnds'], y_bnds), }, \
                        coords=coords, )

    ofile = './outdir/' + exp + '/hfbasin.nc'
    ds_out.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
