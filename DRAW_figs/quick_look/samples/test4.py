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



mfo_df = [ pd.DataFrame() ] * nmax

for model in model_list:

    mfo_model_df = pd.DataFrame()

    for file in glob.glob("./data/mfo*" + model + "*"):
        nc = netCDF4.Dataset(file,'r')
        line_count = nc.dimensions['line'].size
        sector   = nc.variables['sector'][:]
        mfo      = nc.variables['mfo'][:]
        time_var = nc.variables['time']
        cftime = num2date(time_var[:],time_var.units,calendar=time_var.calendar)
        year     = np.array([tmp.year for tmp in cftime])
        nc.close()

        mfo_file_df = pd.DataFrame()
        for i in range(line_count):
            sector_name = ''.join(sector[i])

            df_tmp = pd.DataFrame([row[i]/1e9 for row in mfo],index=year,columns=[sector_name])
            df_tmp.index.names = ['year']
            df_tmp = df_tmp.mean(level='year',skipna=False)

            mfo_file_df = pd.concat([mfo_file_df,df_tmp],axis=1)

        mfo_model_df = pd.concat([mfo_model_df,mfo_file_df])

    for sector_name in sector_name_list:
        if sector_name in mfo_model_df.columns:
            hoge = mfo_model_df.loc[:,[sector_name]].rename(columns={sector_name:model})
        else:
            hoge = pd.DataFrame(index=mfo_model_df.index,columns=[model])

        i = sector_name_list.index(sector_name)
        mfo_df[i] = pd.concat([mfo_df[i],hoge],axis=1)


axes = []

fig = plt.figure(figsize=(16,16))
for i in range(len(sector_name_list)):
    axes.append(fig.add_subplot(rows_count,columns_count,i+1))
    axes[i].plot(mfo_df[i])
    axes[i].set_title(sector_name_list[i])
    axes[i].set_xticks(np.arange(1700,2010+1,62))
    axes[i].grid(which="major",axis="x",color="gray",alpha=0.8,linestyle="--",linewidth=1)

fig.legend(model_list)
fig.subplots_adjust(wspace=0.5,hspace=0.5)

plt.savefig('mfo.png', bbox_inches='tight', pad_inches=0.0)

plt.show()


