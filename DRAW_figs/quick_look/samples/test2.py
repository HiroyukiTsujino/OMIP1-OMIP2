# -*- coding: utf-8 -*-
import sys
import glob
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#model_list = ("IPSL-CM6A-LR", "CNRM-ESM2-1")
model_list = ("CNRM-ESM2-1","IPSL-CM6A-LR")

if len(sys.argv) == 1:
    print 'Usage: test2.py sector_name'
    nmax = 0
    for model in model_list:
        for file in glob.glob("./data/mfo*" + model + "*"):
            nc = netCDF4.Dataset(file,'r')
            sector = nc.variables['sector'][:]
            if len(sector) > nmax:
                sector_max = sector
                nmax = len(sector)
            nc.close()
    print 'Sector name list:'
    for i in range(len(sector_max)):
        print ' ' + str(i+1) + ': ' + ''.join(sector_max[i])
    sys.exit()

sector_name = sys.argv[1]

mfo_df = pd.DataFrame()

for model in model_list:

    mfo_model_df = pd.DataFrame()
    for file in glob.glob("./data/mfo*" + model + "*"):
        nc = netCDF4.Dataset(file,'r')
        sector = nc.variables['sector'][:]
        mfo    = nc.variables['mfo'][:]
        time_var = nc.variables['time']
        dtime = num2date(time_var[:],time_var.units)
        nc.close()

        n = -1
        for i in range(len(sector)):
            name = ''.join(sector[i])
            if name == sector_name:
                n = i
                break

        if n == -1:
            df_tmp = pd.DataFrame(index=dtime,columns=[model])
        else:
            df_tmp = pd.DataFrame([row[n]/1e9 for row in mfo],index=dtime,columns=[model])
        df_tmp = df_tmp.set_index([df_tmp.index.year,df_tmp.index])
        df_tmp.index.names = ['year','date']

        mfo_model_df = pd.concat([mfo_model_df,df_tmp.mean(level='year',skipna=False)])

    mfo_df = pd.concat([mfo_df,mfo_model_df],axis=1)


mfo_df.plot()
plt.title(sector_name)

plt.show()

