# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

exp_list = [ 'omip1', 'omip2' ]
ystr = [ 1948, 1958 ]
yend = [ 2009, 2018 ]
file_base = 'hs_u_woa1x1.'
varname = 'uo'

y_bnds = np.stack( [ np.linspace(-20,19,40), np.linspace(-19,20,40) ], 1 )
z = np.array([0.,10.,20.,30.,50.,75.,100.,125.,150.,200.,250.,300.,400.,500.])

for n in range(len(exp_list)):
    exp = exp_list[n]

    files = []
    for yr in range(ystr[n],yend[n]+1):
        files.append( './indir/' + exp + '/' + file_base + str(yr) )

    DS = xr.open_mfdataset(files)
    DS = xr.concat([DS.isel(depth=0).assign_coords(depth=-DS.depth.values[0]),DS],dim='depth')
    DS = DS.chunk(chunks={'time':1, 'depth':DS.depth.size, 'lat':DS.lat.size, 'lon':DS.lon.size})

    da = DS[varname].sel(lat=slice(-20,20)).interp(lon=220,depth=z,kwargs={'bounds_error':False,'fill_value':-9.99e33}).transpose('time','depth','lat')

    time = DS.time.values
    
    encoding = { "uo":{"_FillValue":-9.99e33,}, \
                 "lat":{"_FillValue": False,}, \
                 "lat_bnds":{"_FillValue": False,}, \
                 "depth":{"_FillValue": False,}, }

    coords = { 'time': ( 'time', time, { 'standard_name': 'time', \
                                         'axis': 'T', } ), \
               'depth': ( 'depth', z, { 'units': 'm', \
                                        'standard_name': 'depth', \
                                        'long_name': 'depth', \
                                        'axis': 'Z', \
                                        'positive': 'down', } ),
               'lat': ( 'lat', da.lat.values, 
                        { 'units': 'degrees_north', \
                          'standard_name': 'latitude', \
                          'long_name': 'latitude', \
                          'axis': 'Y', } ), }
    attr = { 'units': 'm s-1' }

    ds_out = xr.Dataset( { "uo": (['time','depth','lat'],da.values,attr),
                           "lat_bnds": (['lat','bnds'],y_bnds), },
                         coords = coords )

    ofile = './outdir/' + exp + '/' + varname + '.nc'
    ds_out.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
