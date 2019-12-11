# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr


exp_list = [ 'omip1', 'omip2' ]

ystr = [ 1948, 1958 ]
yend = [ 2009, 2018 ]


encoding = { "amoc_rapid":{"_FillValue":-9.99e33,}, \
             "time":{"_FillValue": False, "dtype":"f8" }, }


for n in range(len(exp_list)):
    exp = exp_list[n]

    files = []
    for yr in range(ystr[n],yend[n]+1):
        files.append('./inmdir/' + exp + '/hs_moc.' + str(yr))

    DS = xr.open_mfdataset(files)
    time = DS.time.values
    ntime = len(time)


    coords = { 'time': ( 'time', time, { 'standard_name': 'time', \
                                         'axis': 'T' } ) }

    attrs = { 'units': DS.atl.units, \
              'standard_name': DS.atl.standard_name }

    amoc = np.empty(ntime)
    for i in range(ntime):
        amoc[i] = np.max( DS.atl[i].interp(lat=[26.5]) )

    da_out = xr.DataArray(np.float32(amoc),name='amoc_rapid', attrs=attrs, \
                          coords=coords, dims=['time'])

    ofile = './outdir/' + exp + '/amoc_rapid.nc'
    da_out.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
