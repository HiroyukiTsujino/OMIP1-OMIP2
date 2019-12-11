# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import cftime

exp_list = [ 'omip1', 'omip2' ]
ystr = [ 1638, 1653 ]
yend = [ 2009, 2018 ]
file_base_list = [ 'hs_t_have.', 'hs_s_have.' ]
varname_list = [ 'thetao', 'so' ]
units_list = [ 'degC', '1e-3' ]

dep = np.array([-1.0,1.0,3.5,6.5,10.0,15.0,22.0,30.5,40.0,50.0,60.0,70.0,80.0,90.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,212.5,230.0,250.0,272.5,300.0,330.0,362.5,400.0,440.0,485.0,540.0,600.0,665.0,740.0,820.0,905.0,1000.0,1100.0,1212.5,1350.0,1500.0,1650.0,1812.5,2000.0,2225.0,2475.0,2725.0,3000.0,3300.0,3600.0,3900.0,4200.0,4550.0,4975.0,5500.0,6150.0,6525.0])

z = np.array([0.,10.,20.,30.,50.,75.,100.,125.,150.,200.,250.,300.,400.,500.,600.,700.,800.,900.,1000.,1100.,1200.,1300.,1400.,1500.,1750.,2000.,2500.,3000.,3500.,4000.,4500.,5000.,5500.])


for n in range(len(exp_list)):
    exp = exp_list[n]
    nrec = yend[n] - ystr[n] + 1

    for m in range(len(varname_list)):
        varname = varname_list[m]
        units   = units_list[m]
        file_base = file_base_list[m]

        time = np.empty(nrec,dtype='object')
        tave = np.empty((nrec,len(z)))

        for yr in range(ystr[n],yend[n]+1):
            filein = open( './logdir/' + exp + '/' + file_base + str(yr), mode='rb')
            tmp = np.fromfile(filein, dtype='>f')
            tmp = np.append([tmp[0]],tmp)
            f = interp1d(dep,tmp)
            tave[yr-ystr[n]] = f(z)
            time[yr-ystr[n]] = cftime.datetime(yr,1,1)

        t_units = 'days since ' + str(ystr[n]) + '-01-01 00:00:00.000000'
        t_calendar = 'gregorian'
        t = cftime.date2num(time,t_units,calendar=t_calendar)

        encoding = { varname:{"_FillValue":-9.99e33,}, \
                     "depth":{"_FillValue": False,}, \
                     "time":{"_FillValue": False,}, }

        coords = { 'time': ( 'time', t, { 'standard_name': 'time', \
                                          'axis': 'T', \
                                          'units': t_units, \
                                          'calendar': t_calendar, } ), \
                   'depth': ( 'depth', z, { 'units': 'm', \
                                            'standard_name': 'depth', \
                                            'long_name': 'depth', \
                                            'axis': 'Z', \
                                            'positive': 'down', } ) }

        tave_attr = { 'units': units, }


        da = xr.DataArray(np.float32(tave),name=varname, attrs=tave_attr, \
                          coords=coords, dims=['time','depth'])

        ofile = './outdir/' + exp + '/' + varname + '.nc'
        da.to_netcdf(ofile, encoding=encoding, unlimited_dims='time')
