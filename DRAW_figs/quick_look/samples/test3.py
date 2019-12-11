# -*- coding: utf-8 -*-
import glob
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rows_count = 4
columns_count = 4

model_list = ("CNRM-ESM2-1","IPSL-CM6A-LR")

sector_name_list = []
nmax = 0
for model in model_list:
    for file in glob.glob("./data/mfo*" + model + "*"):
        nc = netCDF4.Dataset(file,'r')
        if nc.dimensions['line'].size > nmax:
            nmax = nc.dimensions['line'].size
            sector = nc.variables['sector'][:]
        nc.close()
for i in range(len(sector)):
    sector_name_list.append(''.join(sector[i]))


mfo_df_all = []
for sector_name in sector_name_list:

    print sector_name
    
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

    mfo_df_all.append(mfo_df)


axes = []

fig = plt.figure(figsize=(16,16))
for i in range(len(sector_name_list)):
    axes.append(fig.add_subplot(rows_count,columns_count,i+1))
    axes[i].plot(mfo_df_all[i])
    axes[i].set_title(sector_name_list[i])

fig.subplots_adjust(wspace=0.3,hspace=0.3)

plt.show()

