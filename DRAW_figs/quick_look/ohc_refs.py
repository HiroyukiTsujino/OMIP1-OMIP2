# -*- coding: utf-8 -*-
import sys
import json
import netCDF4
from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from datetime import datetime 

######

outfile = './fig/ohc_refs.png'

#######
# Zanna

file_zanna = "./Zanna_OHC/OHC_GF_1870_2018.nc"
ncz = netCDF4.Dataset(file_zanna,'r')
ohc700m_zanna_tmp = ncz.variables['OHC_700m'][:]
ohc700m_zanna = ohc700m_zanna_tmp - (ohc700m_zanna_tmp[135:139]).mean()
ohc2000m_zanna_tmp = ncz.variables['OHC_2000m'][:]
ohc2000m_zanna = ohc2000m_zanna_tmp - (ohc2000m_zanna_tmp[135:139]).mean()
time_var_zanna = ncz.variables['time'][:]
ncz.close()

#sys.exit()

#col = pd.Index(['Zanna_700m','Zanna_2000m'],name='reference data')
zanna_df = pd.DataFrame({'Zanna_700m':ohc700m_zanna,'Zanna_2000m':ohc2000m_zanna},index=time_var_zanna)
zanna_df.index.names = ['year']

###########
# Ishii

file_ishii = "./Ishii_v7_2/v7.2/OHC_0-700.txt"
names=['year','Ishii_700m','Ishii_700m_se']
ishii_700m_df=pd.read_table(file_ishii,sep='\s+',names=names,index_col='year')
ishii_700m_df['Ishii_700m'] = ishii_700m_df['Ishii_700m'] - ishii_700m_df.loc[2005:2009,'Ishii_700m'].mean()

file_ishii = "./Ishii_v7_2/v7.2/OHC_0-2000.txt"
names=['year','Ishii_2000m','Ishii_2000m_se']
ishii_2000m_df=pd.read_table(file_ishii,sep='\s+',names=names,index_col='year')
ishii_2000m_df['Ishii_2000m'] = ishii_2000m_df['Ishii_2000m'] - ishii_2000m_df.loc[2005:2009,'Ishii_2000m'].mean()

###########
# Chen

file_chen = "./Chen_OHC/IAP_OHC_estimate_update.txt"
names=['year', 'month', 'Chen_700m','Chen_700m_smth', 'Chen_700m_se', 'Chen_700-2000m','Chen_700-2000m_smth', 'Chen_700-2000m_se']
chen_df=pd.read_table(file_chen,sep='\s+',names=names,skiprows=18)
chen_df['Chen_700m'] = chen_df['Chen_700m'] * 10.0
chen_df['Chen_700-2000m'] = chen_df['Chen_700-2000m'] * 10.0
chen_df['Chen_2000m'] = chen_df['Chen_700m'] + chen_df['Chen_700-2000m']
#time_chen=pd.to_datetime(chen_df.year*10000+chen_df.month*100+1,format='%Y%m%d')
chen_df.index=chen_df.year
chen_df_ann=chen_df.mean(level='year',skipna=False)
chen_df_ann.drop(['year','month','Chen_700m_smth','Chen_700-2000m_smth'], axis='columns', inplace=True)
chen_df_ann.drop(2019, inplace=True)
chen_df_ann['Chen_700m'] = chen_df_ann['Chen_700m'] - chen_df_ann.loc[2005:2009,'Chen_700m'].mean()
chen_df_ann['Chen_2000m'] = chen_df_ann['Chen_2000m'] - chen_df_ann.loc[2005:2009,'Chen_2000m'].mean()
chen_df_ann['Chen_700-2000m'] = chen_df_ann['Chen_700-2000m'] - chen_df_ann.loc[2005:2009,'Chen_700-2000m'].mean()

print(chen_df_ann)

#sys.exit()

###########
# NOAA

file_noaa = "./NOAA_OHC/NOAA_h22-w0-700m.dat.txt"
names=['year', 'NOAA_700m','NOAA_700m_se', 'NOAA_700m_NH', 'NOAA_700m_NH_se', 'NOAA_700m_SH', 'NOAA_700m_SH_se']
noaa_700m_df=pd.read_table(file_noaa,sep='\s+',names=names,skiprows=1)
noaa_700m_df['year'] = np.floor(noaa_700m_df['year'])
#print(noaa_700m_df['year'])
noaa_700m_df['NOAA_700m'] = noaa_700m_df['NOAA_700m'] * 10.0
noaa_700m_df.index=noaa_700m_df.year
noaa_700m_df.drop(['year','NOAA_700m_NH','NOAA_700m_NH_se','NOAA_700m_SH','NOAA_700m_SH_se'], axis='columns', inplace=True)
noaa_700m_df['NOAA_700m'] = noaa_700m_df['NOAA_700m'] - noaa_700m_df.loc[2005:2009,'NOAA_700m'].mean()

file_noaa = "./NOAA_OHC/NOAA_h22-w0-2000m.dat.txt"
names=['year', 'NOAA_2000m','NOAA_2000m_se', 'NOAA_2000m_NH', 'NOAA_2000m_NH_se', 'NOAA_2000m_SH', 'NOAA_2000m_SH_se']
noaa_2000m_df=pd.read_table(file_noaa,sep='\s+',names=names,skiprows=1)
noaa_2000m_df['year'] = np.floor(noaa_2000m_df['year'])
#print(noaa_2000m_df['year'])
noaa_2000m_df['NOAA_2000m'] = noaa_2000m_df['NOAA_2000m'] * 10.0
noaa_2000m_df.index=noaa_2000m_df.year
noaa_2000m_df.drop(['year','NOAA_2000m_NH','NOAA_2000m_NH_se','NOAA_2000m_SH','NOAA_2000m_SH_se'], axis='columns', inplace=True)
noaa_2000m_df['NOAA_2000m'] = noaa_2000m_df['NOAA_2000m'] - noaa_2000m_df.loc[2005:2009,'NOAA_2000m'].mean()

#print(noaa_df)

#sys.exit()


#######
# merge data

ohc700m_all=pd.concat([zanna_df['Zanna_700m'],ishii_700m_df['Ishii_700m'],chen_df_ann['Chen_700m'],noaa_700m_df['NOAA_700m']],axis=1)

print (ohc700m_all)

ohc2000m_all=pd.concat([zanna_df['Zanna_2000m'],ishii_2000m_df['Ishii_2000m'],chen_df_ann['Chen_2000m'],noaa_2000m_df['NOAA_2000m']],axis=1)

print (ohc2000m_all)

# draw figures

title_name=['OHCA 0-700m','OHCA 0-2000m']
fig  = plt.figure(figsize = (15,9))

axes = fig.add_subplot(2,1,1)
ohc700m_all.plot(y=ohc700m_all.columns[0:4],linewidth=2,ax=axes,ylim=[-380,80],title=title_name[0])
axes.set_ylabel('[ZJ]',fontsize=12)
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8)

axes = fig.add_subplot(2,1,2)
ohc2000m_all.plot(y=ohc2000m_all.columns[0:4],linewidth=2,ax=axes,ylim=[-520,120],title=title_name[1])
axes.set_ylabel('[ZJ]',fontsize=12)
axes.legend(bbox_to_anchor=(1.2,1.0))
plt.subplots_adjust(left=0.1,right=0.8,hspace=0.3)

plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

plt.show()
