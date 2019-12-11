# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr

exp_list = [ 'omip1', 'omip2' ]
ystr = [ 1948, 1958 ]
yend = [ 2009, 2018 ]
file_base_list = [ 'hs_zmt_omip1-omip2.', 'hs_zms_omip1-omip2.' ]
varname_list = [ 'thetao', 'so' ]

sector = np.array( [ 'global_ocean', 'atlantic_arctic_ocean', 'indian_ocean', 'pacific_ocean' ], dtype=str )


y = np.linspace(-89.5,89.5,num=180)
y_bnds = np.stack( [ np.linspace(-90,89,180), np.linspace(-89,90,180) ], 1 )
z = np.array([0.,10.,20.,30.,50.,75.,100.,125.,150.,200.,250.,300.,400.,500.,600.,700.,800.,900.,1000.,1100.,1200.,1300.,1400.,1500.,1750.,2000.,2500.,3000.,3500.,4000.,4500.,5000.,5500.])

nsect = len(sector)
nz    = len(z)
ny    = len(y)


for n in range(len(exp_list)):
    exp = exp_list[n]

    for m in range(len(varname_list)):
        varname = varname_list[m]
        file_base = file_base_list[m]

        files = []
        for yr in range(ystr[n],yend[n]+1):
            files.append( './indir/' + exp + '/' + file_base + str(yr) )

        DS = xr.open_mfdataset(files)
        DS = xr.concat([DS.isel(depth=0).assign_coords(depth=-DS.depth.values[0]),DS], \
                       dim = 'depth')
        DS = DS.chunk(chunks={'time': 1, 'depth': DS.depth.size, 'lat': DS.lat.size})
        DS["glb"] = DS.glb.where( DS.glb != -9.99e33 )
        DS["atl"] = DS.atl.where( DS.atl != -9.99e33 )
        DS["ind"] = DS.ind.where( DS.ind != -9.99e33 )
        DS["pac"] = DS.pac.where( DS.pac != -9.99e33 )


        time = DS.time.values
        ntime = len(time)


        encoding = { varname:{"_FillValue":-9.99e33,}, \
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

        zm_attr = { 'units': DS.glb.units, }
                                          

        zm = np.empty((ntime,nsect,nz,ny))
        for i in range(ntime):
            zm[i,0] = DS.glb[i].interp(depth=z,lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33}).T
            zm[i,1] = DS.atl[i].interp(depth=z,lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33}).T
            zm[i,2] = DS.ind[i].interp(depth=z,lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33}).T
            zm[i,3] = DS.pac[i].interp(depth=z,lat=y,kwargs={'bounds_error':False,'fill_value':-9.99e33}).T


        ds_out = xr.Dataset({varname: (['time','basin','depth','lat'], \
                                       np.float32(zm), zm_attr), \
                             'sector': (['basin'], sector), \
                             'lat_bnds': (['lat','bnds'], y_bnds), }, \
                            coords=coords, )

        ofile = './outdir/' + exp + '/' + varname + '_basin.nc'
        ds_out.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
