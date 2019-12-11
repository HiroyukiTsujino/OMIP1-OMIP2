# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import cftime


exp_list = [ 'omip1', 'omip2' ]
ystr = [ 1638, 1653 ]
yend = [ 2009, 2018 ]
file_base_list = [ 'hs_drake.', 'hs_itf.' ]

sector = np.array( [ 'drake_passage', 'indonesian_throughflow' ] )


encoding = { 'mfo':{'_FillValue':-9.99e33}, \
             'time':{'_FillValue': False}, }

attrs = { 'units': 'kg s-1' }


for n in range(len(exp_list)):

    exp = exp_list[n]
    nrec = yend[n] - ystr[n] + 1


    time = np.empty(nrec,dtype='object')
    for yr in range(ystr[n],yend[n]+1):
        time[yr-ystr[n]] = cftime.datetime(yr,1,1)

    t_units = 'days since ' + str(ystr[n]) + '-01-01 00:00:00.000000'
    t_calendar = 'gregorian'
    t = cftime.date2num(time,t_units,calendar=t_calendar)


    mfo = np.empty((nrec,len(sector)))
    for m in range(len(file_base_list)):
        file_base = file_base_list[m]

        for yr in range(ystr[n],yend[n]+1):
            filein = open( './logdir/' + exp + '/' + file_base + str(yr), mode = 'rb' )
            mfo[yr-ystr[n],m] = np.fromfile(filein, dtype='>f')



    coords = { 'time': ( 'time', t, { 'standard_name': 'time', \
                                      'axis': 'T', \
                                      'units': t_units, \
                                      'calendar': t_calendar, } ) }

    ds_out = xr.Dataset({'mfo': (['time','line'], np.float32(mfo), attrs), \
                         'sector': (['line'], sector) }, \
                        coords=coords, )


    ofile = './outdir/' + exp + '/mfo.nc'
    ds_out.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
