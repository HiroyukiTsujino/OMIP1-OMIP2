# -*- coding: utf-8 -*-
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime


title = [ 'OMIP1', 'OMIP2', 'OMIP2 - OMIP1', 'deBoyer' ]

metainfo = [ json.load(open("./json/mld_omip1.json")), 
             json.load(open("./json/mld_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]


if len(sys.argv) == 1:
    suptitle = 'Multi Model Mean' + ' (Winter MLD, JFM (NH), JAS (SH) ave. from 1980 to 2009)'
    outfile = './fig/MLD_Winter.png'
else:
    suptitle = sys.argv[1] + ' (Winter MLD, JFM (NH), JAS (SH) ave. from 1980 to 2009)'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/MLD_Winter_' + sys.argv[1] + '.png'


#J データ読込・平均

datanh = []
datash = []
modnam = []

for omip in range(2):
    dnh = np.empty( (len(model_list[omip]),90,360) )
    dsh = np.empty( (len(model_list[omip]),90,360) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    namtmp = []
    for model in model_list[omip]:
        path  = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        infile = path + '/' + fname
        DS = xr.open_dataset( infile )
        tmp = DS.mlotst.sel(time=slice('1850-01-01','1850-03-01'),lat=slice(0.5,89.5)).mean(dim='time')
        dnh[nmodel] = tmp.values
        tmp = DS.mlotst.sel(time=slice('1850-07-01','1850-09-01'),lat=slice(-89.5,-0.5)).mean(dim='time')
        dsh[nmodel] = tmp.values
        namtmp +=[model]
        nmodel += 1

    datanh += [dnh]
    datash += [dsh]
    modnam += [namtmp]

models = list(model_list[0])
print(models)

DSNH = xr.Dataset( {'omip1mean': (['model','lat','lon'], datanh[0]),
                    'omip2mean': (['model','lat','lon'], datanh[1]),},
                   coords = { 'model': np.linspace(1,11,num=11),
                              'lat': np.linspace(0.5,89.5,num=90), 
                              'lon': np.linspace(0.5,359.5,num=360), } )

print(DSNH)

DSSH = xr.Dataset( {'omip1mean': (['model','lat','lon'], datash[0]),
                    'omip2mean': (['model','lat','lon'], datash[1]),},
                 coords = { 'model': np.linspace(1,11,num=11),
                            'lat': np.linspace(-89.5,-0.5,num=90), 
                            'lon': np.linspace(0.5,359.5,num=360), } )

print(' ')
dict_mld={}
nm=0
for model_name in model_list[0]:
    nm+=1
    shmld=DSSH['omip1mean'].sel(model=nm,lat=slice(-89.5,-60.5),lon=slice(0.5,359.5)).max().values
    nhmld1=DSNH['omip1mean'].sel(model=nm,lat=slice(45.5,79.5),lon=slice(280.5,359.5)).max().values
    nhmld2=DSNH['omip1mean'].sel(model=nm,lat=slice(45.5,79.5),lon=slice(0.5,29.5)).max().values
    nhmld = max(nhmld1,nhmld2)
    dict_mld[modnam[0][nm-1]]=[shmld,nhmld]

summary=pd.DataFrame(dict_mld,index=['SH-MLD-OMIP1','NH-MLD-OMIP1'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/winter_mld_omip1.csv')


dict_mld={}
nm=0
for model_name in model_list[1]:
    nm+=1
    shmld=DSSH['omip2mean'].sel(model=nm,lat=slice(-89.5,-60.5),lon=slice(0.5,359.5)).max().values
    nhmld1=DSNH['omip2mean'].sel(model=nm,lat=slice(45.5,79.5),lon=slice(280.5,359.5)).max().values
    nhmld2=DSNH['omip2mean'].sel(model=nm,lat=slice(45.5,79.5),lon=slice(0.5,29.5)).max().values
    nhmld = max(nhmld1,nhmld2)
    dict_mld[modnam[1][nm-1]]=[shmld,nhmld]

summary=pd.DataFrame(dict_mld,index=['SH-MLD-OMIP2','NH-MLD-OMIP2'])
summary_t=summary.T
print (summary_t)
summary_t.to_csv('csv/winter_mld_omip2.csv')
