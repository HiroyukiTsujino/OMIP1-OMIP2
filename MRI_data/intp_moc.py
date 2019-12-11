# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr


exp_list = [ 'omip1', 'omip2' ]

sector = np.array( [ 'global_ocean', 'atlantic_arctic_ocean', 'indian_pacific_ocean' ], dtype=str )

y = np.linspace(-89.5,89.5,num=180)
y_bnds = np.stack( [ np.linspace(-90,89,180), np.linspace(-89,90,180) ], 1 )
z = np.array([0.,10.,20.,30.,50.,75.,100.,125.,150.,200.,250.,300.,400.,500.,600.,700.,800.,900.,1000.,1100.,1200.,1300.,1400.,1500.,1750.,2000.,2500.,3000.,3500.,4000.,4500.,5000.,5500.])


nsect = len(sector)
nz    = len(z)
ny    = len(y)


for n in range(len(exp_list)):
    exp = exp_list[n]

    DS = xr.open_mfdataset('./indir/' + exp + '/hs_moc.*', use_cftime = True)
    time = DS.time.values
    ntime = len(time)

    encoding = { "msftmyz":{"_FillValue":-9.99e33,}, \
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
               'lat': ( 'lat', y, { 'units': 'degrees_north', \
                                    'standard_name': 'latitude', \
                                    'long_name': 'latitude', \
                                    'axis': 'Y', } ), }

    moc_attr = { 'units': DS.glb.units, \
                 'standard_name': DS.glb.standard_name, }


    moc = np.empty((ntime,nsect,nz,ny))
    for i in range(ntime):
        moc[i,0] = DS.glb[i].interp(depth=z,lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})
        moc[i,1] = DS.atl[i].interp(depth=z,lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})
        moc[i,2] = DS.inp[i].interp(depth=z,lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33})

    ds_out = xr.Dataset({"msftmyz": (['time','basin','depth','lat'], \
                                     np.float32(moc), moc_attr), \
                         'sector': (['basin'], sector), \
                         'lat_bnds': (['lat','bnds'], y_bnds), }, \
                        coords=coords, )
                   
    ofile = './outdir/' + exp + '/msftmyz.nc'
    ds_out.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
