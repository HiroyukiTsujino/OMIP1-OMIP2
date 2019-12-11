# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

exp_list = [ 'omip1', 'omip2' ]

sector = np.array( [ 'global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean' ], dtype=str )


y = np.linspace(-89.5,89.5,num=180)
y_bnds = np.stack( [ np.linspace(-90,89,180), np.linspace(-89,90,180) ], 1 )


nsect = len(sector)
ny    = len(y)


for n in range(len(exp_list)):
    exp = exp_list[n]

    DS = xr.open_mfdataset('./indir/' + exp + '/hs_mocip.*', use_cftime = True)

    dep  = DS.depth.values
    time = DS.time.values
    ntime = len(time)
    nz    = len(dep)


    encoding = { "msftmrho":{"_FillValue":-9.99e33,}, \
                 "lat":{"_FillValue": False,}, \
                 "lat_bnds":{"_FillValue": False,}, \
                 "sigma2":{"_FillValue": False,}, }

    coords = { 'time': ( 'time', time, { 'standard_name': 'time', \
                                         'axis': 'T', } ), \
               'sigma2': ( 'sigma2', dep, { 'units': 'kg m-3', \
                                          'standard_name': 'sigma2', \
                                          'long_name': 'sigma2', \
                                          'axis': 'Z', \
                                          'positive': 'down', } ),
               'lat': ( 'lat', y, { 'units': 'degrees_north', \
                                    'standard_name': 'latitude', \
                                    'long_name': 'latitude', \
                                    'axis': 'Y', } ), }

    moc_attr = { 'units': DS.glb.units, \
                 'standard_name': DS.glb.standard_name, }

    moc = np.empty((ntime,nsect,nz,ny))
    for i in range(ntime):
        moc[i,0] = DS.glb[i].interp(lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})
        moc[i,1] = DS.atl[i].interp(lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})
        moc[i,2] = DS.inp[i].interp(lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})

    ds_out = xr.Dataset({"msftmrho": (['time','basin','sigma2','lat'], \
                                      np.float32(moc), moc_attr), \
                         'sector': (['basin'], sector), \
                         'lat_bnds': (['lat','bnds'], y_bnds), }, \
                        coords=coords, )
                   
    ofile = './outdir/' + exp + '/msftmrho.nc'
    ds_out.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
