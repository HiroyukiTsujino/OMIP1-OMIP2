# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import xarray as xr
import cftime


if len(sys.argv) != 2:
    print ('Usage: python3 convert_scalar.py jsonfile')
    sys.exit()


metainfo = json.load(open(sys.argv[1]))


for ofile in metainfo.keys():

    ifile = metainfo[ofile]['ifile']
    ystr  = int(metainfo[ofile]['ystr'])
    yend  = int(metainfo[ofile]['yend'])
    month = int(metainfo[ofile]['month'])
    nv    = int(metainfo[ofile]['nv'])
    nvar  = int(metainfo[ofile]['nvar'])
    name  = metainfo[ofile]['name']
    standard_name = metainfo[ofile]['standard_name']
    units = metainfo[ofile]['units']

    nrec = (yend-ystr+1)*month

    t_units = 'days since ' + str(ystr) + '-01-01 00:00:00.000000'
    t_calendar = 'gregorian'


    encoding = { name:{'_FillValue':-9.99e33}, \
                 'time':{'_FillValue':False}, }
    attrs = { 'units': units, \
              'standard_name': standard_name, }


    time = np.empty(nrec,dtype='object')
    var  = np.empty(nrec)
    for yr in range(ystr,yend+1):

        filein = open( ifile + str(yr), mode = 'rb' )
        tmp = np.fromfile(filein, dtype='>f').reshape(month,nvar)

        for mon in range(1,month+1):
            i = (yr-ystr)*month + mon-1
            time[i] = cftime.datetime(yr,mon,1)
            var[i]  = tmp[mon-1,nv-1]


    t = cftime.date2num(time,t_units,calendar=t_calendar)
    coords = { 'time': ( 'time', t, { 'standard_name': 'time', \
                                      'axis': 'T', \
                                      'units': t_units, \
                                      'calendar': t_calendar, } ) }

    da = xr.DataArray(np.float32(var),name=name,attrs=attrs, \
                      coords=coords, dims=['time'])


    da.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')


