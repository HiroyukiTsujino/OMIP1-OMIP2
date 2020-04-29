# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../python")
import json
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from distutils.util import strtobool
import netCDF4
from netCDF4 import Dataset, num2date
from uncertain_Wakamatsu import uncertain

#####

suptitle = 'Ocean circulation metrics'
outfile = './fig/FigS3_ALL'


title_list = [ "(a) OMIP1 (AMOC maximum at 26.5$^{\circ}$N)", "(b) OMIP2 (AMOC maximum at 26.5$^{\circ}$N)", "(c) MMM (AMOC maximum at 26.5$^{\circ}$N)",
               "(d) OMIP1 (Drake Passage transport)", "(e) OMIP2 (Drake Passage transport)", "(f) MMM (Drake Passage transport)",
               "(g) OMIP1 (Indonesian through flow)", "(h) OMIP2 (Indonesian through flow)", "(i) MMM (Indonesian through flow)",
               "(j) OMIP1 (GMOC minimum at 30$^{\circ}$S)", "(k) OMIP2 (GMOC minimum at 30$^{\circ}$S)", "(l) MMM (GMOC minimum at 30$^{\circ}$S)" ]

metainfo = [ json.load(open("./json/circ_omip1.json")),
             json.load(open("./json/circ_omip2.json")) ]

metainfo_full = [ json.load(open("./json/circ_omip1_full.json")),
                  json.load(open("./json/circ_omip2_full.json")) ]

model_list = [ metainfo[0].keys(), metainfo[1].keys() ]
model_full_list = [ metainfo_full[0].keys(), metainfo_full[1].keys() ]

lineinfo = json.load(open('../json/inst_color_style.json'))

var_list = [ "amoc", "drake", "itf", "gmoc" ]

lat_woa=np.array(np.linspace(-89.5,89.5,num=180))
#print(lat_woa)

#J 時刻情報 (各モデルの時刻情報を上書きする)
time = [ np.array([np.linspace(1948,2009,62)]*6),
         np.array([np.linspace(1958,2018,61)]*6) ]
for n in range(2):
    for i in range(6):
        time[n][i] = time[n][i] - (5-i)*71
timenan = [ time[1][:,60], time[0][:,0] ]
time[0] = time[0].reshape(6*62)
time[1] = time[1].reshape(6*61)


#J 間をあけるための Dataset 作成
varnan = np.full(6,np.nan)


#J 単一モデル用 dummy DS (json にエントリーがない場合に使用)
d_dummy = [ np.full( len(time[0]), np.nan ),
            np.full( len(time[1]), np.nan ) ]


#J データ読込 (n=0: OMIP1, n=1: OMIP2)
DS = []
DS_full = []

lincol = []
linsty = []
nummodel = []

for n in range(2):
    d = np.full( (len(var_list),len(model_list[n]),len(time[n])), np.nan )
    d_full = np.full( (len(var_list),len(model_full_list[n]),len(time[n])), np.nan )
    print( "Loading OMIP" + str(n+1) + " data" )

    coltmp = []
    stytmp = []

    for nvar in range(4):
        var = var_list[nvar]

        nmodel = 0
        nmodel_full = 0
        for model in model_list[n]:

            add_to_full = 'no'
            for model_full in model_full_list[n]:
                if (model == model_full):
                    add_to_full = 'yes'
                    break

            print(model + ' Added to full list ' + add_to_full)
            
            try:
                multidata = strtobool(metainfo[n][model]['multidata'])
                path = metainfo[n][model][var]['path']
            except:
                #J json にエントリーがない場合のエラー処理
                d[nvar,nmodel] = d_dummy[n]
                continue

            if (nvar == 0):
                coltmp +=[lineinfo[model]["color"]]
                stytmp +=[lineinfo[model]["style"]]

            fname= metainfo[n][model][var]['fname']
            name = metainfo[n][model][var]['name']
            exdim_name = metainfo[n][model][var]['exdim_name']
            exdim = int(metainfo[n][model][var]['exdim'])
            factor = float(metainfo[n][model][var]['factor'])
            infile = path + '/' + fname


            if (n == 1) and (model == "FSU-HYCOM"):

                DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    DS_read = DS_read.assign_coords(lat=DS_read["lat"].values)

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                    print(tmp)
                elif var == 'gmoc':
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values
                
                d_fsu = np.full( (len(time[n])), np.nan )

                for cyc in range(5):
                    isto = 0 + cyc * 61
                    iedo = isto + 57
                    istf = 0 + cyc * 58
                    iedf = istf + 57
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                cyc = 5
                isto = 0 + cyc * 61
                iedo = isto + 61
                istf = 0 + cyc * 58
                iedf = istf + 61
                print (isto, iedo)
                print (istf, iedf)
                d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor
                d[nvar,nmodel] = d_fsu
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu

            elif (n == 1) and (model == "GFDL-MOM"):

                DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_fsu = np.full( (len(time[n])), np.nan )

                for cyc in range(0,3):
                    isto = 0 + cyc * 61
                    iedo = isto + 59
                    istf = 0 + cyc * 60
                    iedf = istf + 59
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                for cyc in range(3,6):
                    isto = 0 + cyc * 61
                    iedo = isto + 60
                    istf = 0 + cyc * 61 - 3
                    iedf = istf + 60
                    print (isto, iedo)
                    print (istf, iedf)
                    d_fsu[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                d[nvar,nmodel] = d_fsu
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_fsu

            elif (n == 0) and (model == "MIROC-COCO4.9"):

                DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    print(DS_read[name])

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)

                if var == 'amoc':
                    print(DS_read[name])
                    tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_coco = np.full( (len(time[n])), np.nan )
                d_coco[0:62*5] = d_tmp[0:62*5] * factor
                d[nvar,nmodel] = d_coco
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_coco


            elif (model == "EC-Earth3-NEMO"):

                if (n == 0):
                    num_yr = 62
                else:
                    num_yr = 61

                d_barca = np.full( num_yr*6, np.nan )

                for i in range(6):

                    infile=path + str(i+1) + '/' + fname
                    print(var,infile)
                    DS_read = xr.open_dataset(infile,decode_times=False)

                    if var == 'amoc' or var == 'gmoc':

                        zdim = metainfo[n][model][var]["zdim"]
                        zvar = metainfo[n][model][var]["zvar"]

                        if zdim != 'lev':
                            DS_read = DS_read.rename({zdim:"lev"})
                        if zdim != zvar and zvar != 'lev':
                            DS_read = DS_read.rename({zvar:"lev"})
                        yvar = metainfo[n][model]["y"]
                        if yvar != 'lat':
                            DS_read = DS_read.rename({yvar:"lat"})

                        DS_read = DS_read.assign_coords(lev=DS_read["lev"].values)
                        #print(DS_read[name])

                    if exdim_name != "None":
                        DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)

                    if var == 'amoc':
                        tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                    elif var == 'gmoc':
                        tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                    else:
                        tmp = DS_read[name]

                    d_tmp = tmp.values
                    #print(d_tmp)

                    for nt in range(num_yr):
                        d_barca[num_yr*i+nt] = d_tmp[nt] * factor

                #print('var',d_barca)
                d[nvar,nmodel] = d_barca
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_barca

            elif (model == "CMCC-NEMO"):

                num_yr = len(time[n])

                DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    DS_read = DS_read.drop("basin")

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
 
                if var == 'amoc':
                    print(DS_read[name].lat[188])
#                    tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                    tmp = DS_read[name].isel(lat=188).max(dim="lev")
#                    tmp = DS_read[name].sel(lat=26.5,method='nearest').max(dim="lev")
                elif var == 'gmoc':
                    print(DS_read[name].lat[99])
#                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].isel(lat=99).min(dim="lev")
#                    tmp = DS_read.sel(lev=slice(2000,6500))[name].sel(lat=-30,method='nearest').min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_coco = np.full( (len(time[n])), np.nan )
                print(var,len(time[n]),len(d_tmp))

                d_coco[0:num_yr] = d_tmp[0:num_yr] * factor
                d[nvar,nmodel] = d_coco
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_coco
                
            elif (n == 0) and (model == "GFDL-MOM"):

                DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    print(DS_read[name])

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)

                if var == 'amoc':
                    print(DS_read[name])
                    tmp = DS_read[name].interp(method='linear',lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    tmp = DS_read[name]

                d_tmp = tmp.values

                d_coco = np.full( (len(time[n])), np.nan )
                for cyc in range(0,5):
                    isto = 0 + cyc * 62
                    iedo = isto + 59
                    istf = 0 + cyc * 60
                    iedf = istf + 59
                    print (cyc, isto, iedo)
                    print (cyc, istf, iedf)
                    d_coco[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                for cyc in range(5,6):
                    isto = 0 + cyc * 62
                    iedo = isto + 61
                    istf = 0 + cyc * 62 - 10
                    iedf = istf + 61
                    print (cyc, isto, iedo)
                    print (cyc, istf, iedf)
                    d_coco[isto:iedo+1] = d_tmp[istf:iedf+1] * factor

                print(var,factor)
                print(d_coco)
                d[nvar,nmodel] = d_coco
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = d_coco

            else:

                if multidata:
                    if ( model == 'Kiel-NEMO' and (var == 'drake' or var == 'itf') ):
                        DS_read = xr.open_mfdataset(infile,decode_times=False,concat_dim='time_counter')
                        #print(DS_read)
                    else:
                        DS_read = xr.open_mfdataset(infile,decode_times=False,concat_dim='time')
                        #print(DS_read)
                else:
                    DS_read = xr.open_dataset(infile,decode_times=False)

                if var == 'amoc' or var == 'gmoc':
                    zdim = metainfo[n][model][var]["zdim"]
                    zvar = metainfo[n][model][var]["zvar"]
                    if zdim != 'lev':
                        DS_read = DS_read.rename({zdim:"lev"})
                    if zdim != zvar and zvar != 'lev':
                        DS_read = DS_read.rename({zvar:"lev"})
                    yvar = metainfo[n][model]["y"]
                    if yvar != 'lat':
                        DS_read = DS_read.rename({yvar:"lat"})

                    if model == 'FSU-HYCOM':
                        DS_read = DS_read.assign_coords(lat=DS_read["lat"].values)

                if exdim_name != "None":
                    DS_read = DS_read.rename({exdim_name:'exdim'}).isel(exdim=exdim)
            
                if var == 'amoc':
                    tmp = DS_read[name].interp(lat=26.5).max(dim="lev")
                elif var == 'gmoc':
                    if model == 'CESM-POP':
                        DS_read["lev"] = DS_read.lev.values * 1e-2
                    tmp = DS_read.sel(lev=slice(2000,6500))[name].interp(lat=-30).min(dim="lev")
                else:
                    if (model == 'Kiel-NEMO'):
                        tmp = DS_read[name].isel(X=0)
                    else:
                        tmp = DS_read[name]

                d[nvar,nmodel] = tmp.values * factor
                if (add_to_full == 'yes'):
                    d_full[nvar,nmodel_full] = tmp.values * factor

            nmodel += 1
            if (add_to_full == 'yes'):
                nmodel_full += 1

        nvar += 1

    #J サイクル間に NaN データ挿入
    d_new = np.concatenate(
        [d, np.tile(varnan,(len(var_list),len(model_list[n]),1))],
        axis = 2 )

    d_full_new = np.concatenate(
        [d_full, np.tile(varnan,(len(var_list),len(model_full_list[n]),1))],
        axis = 2 )

    time_new = np.concatenate( [time[n], timenan[n]] )

    #J xarray Dataset 再作成
    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new } ).sortby('time')

    DS += [DS_tmp]

    ######

    var_dict = {}
    nvar = 0
    for var in var_list:
        var_dict[var] = (['model','time'], d_full_new[nvar])
        nvar += 1

    DS_tmp = xr.Dataset( var_dict, coords = { 'time': time_new } ).sortby('time')

    #nt = len(time_new)
    #for nn in range(nt):
    #    print(nn, DS[n]['time'].isel(time=nn).values, DS[n]['amoc'].isel(model=2,time=nn).values)

    DS_full += [DS_tmp]

    lincol += [coltmp]
    linsty += [stytmp]
    nummodel += [nmodel]


##### output text #####

for n in range(2):
    nmodel = 0
    dict_circulation={}
    for model in model_list[n]:
        if (model == 'MIROC-COCO4.9'):
            amoctmp = DS[n]['amoc'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values
            gmoctmp = DS[n]['gmoc'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values
            draketmp = DS[n]['drake'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values
            itftmp = DS[n]['itf'].sel(model=nmodel,time=slice(1909,1938)).mean(dim='time').values
        else:
            amoctmp = DS[n]['amoc'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values
            gmoctmp = DS[n]['gmoc'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values
            draketmp = DS[n]['drake'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values
            itftmp = DS[n]['itf'].sel(model=nmodel,time=slice(1980,2009)).mean(dim='time').values

        dict_circulation[model]=[amoctmp,gmoctmp,draketmp,itftmp]
        nmodel += 1

    mipid='OMIP'+str(n+1)
    summary=pd.DataFrame(dict_circulation,index=['AMOC-'+ str(mipid),'GMOC-'+ str(mipid),'ACC-'+ str(mipid),'ITF-'+ str(mipid)])
    tmp1=summary.mean(axis=1)
    tmp2=summary.std(axis=1,ddof=0)
    print(tmp1,tmp2)
    summary.insert(0,'Z-MMM',tmp1)
    summary.insert(0,'Z-STD',tmp2)
    summary_t=summary.T
    print (summary_t)
    summary_t.to_csv('csv/circulation_index_omip' + str(n+1) + '.csv')


# omip1 - omip2

print("Compute uncertainty of the difference omip2-omip1")

ystr = 1980
yend = 2009
nyr = yend - ystr + 1
nmod = len(model_list[0])

d_amoc  = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)
d_gmoc  = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)
d_drake = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)
d_itf   = np.array(np.empty( (nmod, nyr) ),dtype=np.float64)

nmodel = 0
for model in model_list[0]:
    if (model == 'MIROC-COCO4.9'):
        amoc_omip1 = DS[0]['amoc'].sel(model=nmodel,time=slice(1909,1938)).values
        gmoc_omip1 = DS[0]['gmoc'].sel(model=nmodel,time=slice(1909,1938)).values
        drake_omip1 = DS[0]['drake'].sel(model=nmodel,time=slice(1909,1938)).values
        itf_omip1 = DS[0]['itf'].sel(model=nmodel,time=slice(1909,1938)).values
        amoc_omip2 = DS[1]['amoc'].sel(model=nmodel,time=slice(1909,1938)).values
        gmoc_omip2 = DS[1]['gmoc'].sel(model=nmodel,time=slice(1909,1938)).values
        drake_omip2 = DS[1]['drake'].sel(model=nmodel,time=slice(1909,1938)).values
        itf_omip2 = DS[1]['itf'].sel(model=nmodel,time=slice(1909,1938)).values
    else:
        amoc_omip1 = DS[0]['amoc'].sel(model=nmodel,time=slice(1980,2009)).values
        gmoc_omip1 = DS[0]['gmoc'].sel(model=nmodel,time=slice(1980,2009)).values
        drake_omip1 = DS[0]['drake'].sel(model=nmodel,time=slice(1980,2009)).values
        itf_omip1 = DS[0]['itf'].sel(model=nmodel,time=slice(1980,2009)).values
        amoc_omip2 = DS[1]['amoc'].sel(model=nmodel,time=slice(1980,2009)).values
        gmoc_omip2 = DS[1]['gmoc'].sel(model=nmodel,time=slice(1980,2009)).values
        drake_omip2 = DS[1]['drake'].sel(model=nmodel,time=slice(1980,2009)).values
        itf_omip2 = DS[1]['itf'].sel(model=nmodel,time=slice(1980,2009)).values

    d_amoc[nmodel,:] = amoc_omip2[:] - amoc_omip1[:]
    d_gmoc[nmodel,:] = gmoc_omip2[:] - gmoc_omip1[:]
    d_drake[nmodel,:] = drake_omip2[:] - drake_omip1[:]
    d_itf[nmodel,:] = itf_omip2[:] - itf_omip1[:]

    nmodel += 1

num_bootstraps = 10000
factor_5pct = 1.64  # 5-95%

dout_amoc = uncertain(d_amoc, "amoc", nmod, nyr, num_bootstraps )
zval = dout_amoc["mean"] / dout_amoc["std"]
print(dout_amoc)
print("z-value = ",zval)

dout_gmoc = uncertain(d_gmoc, "gmoc", nmod, nyr, num_bootstraps )
zval = dout_gmoc["mean"] / dout_gmoc["std"]
print(dout_gmoc)
print("z-value = ",zval)

dout_drake = uncertain(d_drake, "drake", nmod, nyr, num_bootstraps )
zval = dout_drake["mean"] / dout_drake["std"]
print(dout_drake)
print("z-value = ",zval)

dout_itf = uncertain(d_itf, "itf", nmod, nyr, num_bootstraps )
zval = dout_itf["mean"] / dout_itf["std"]
print(dout_itf)
print("z-value = ",zval)

#J 描画
fig = plt.figure(figsize=(11,8.0))
fig.suptitle( suptitle, fontsize=18 )
ax = [ plt.subplot(4,3,1),
       plt.subplot(4,3,2),
       plt.subplot(4,3,3),
       plt.subplot(4,3,4),
       plt.subplot(4,3,5),
       plt.subplot(4,3,6),
       plt.subplot(4,3,7),
       plt.subplot(4,3,8),
       plt.subplot(4,3,9),
       plt.subplot(4,3,10),
       plt.subplot(4,3,11),
       plt.subplot(4,3,12) ]

lc = [ "red", "blue" ]

for n in range(2):
    nv = 0
    for var in var_list:
        nf = nv * 3 + n
        nmodel = 0
        for model in model_list[n]:
            #print(model)
            linecol=lincol[n][nmodel]
            linesty=linsty[n][nmodel]
            if (linesty == 'dashed'):
                lwidth = 0.7
                linesty = 'dotted'
            else:
                lwidth = 0.5
            if (n == 0 and nv == 0):
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],label=model,color=linecol,linewidth=lwidth,linestyle=linesty)
            else:
                DS[n][var].sel(model=nmodel).plot.line(x='time',ax=ax[nf],color=linecol,linewidth=lwidth,linestyle=linesty)
                
            nmodel += 1

        if (n == 0 and nv == 0):
            leg = ax[nf].legend(bbox_to_anchor=(3.45,0.6),loc='upper left')
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

        nv += 1

# MMM
for nvar in range(4):
    var = var_list[nvar]
    nf = nvar * 3 + 2
    if (nvar == 0): 
        DS_full[0][var].mean(dim='model').plot(ax=ax[nf],color='darkred',label='OMIP1-MMM')
    else:
        DS_full[0][var].mean(dim='model').plot(ax=ax[nf],color='darkred')
    
    ax[nf].fill_between(x=DS_full[0]['time'],
                        y1=DS_full[0][var].min(dim='model'),
                        y2=DS_full[0][var].max(dim='model'),
                        alpha=0.5, facecolor='lightcoral')
    if (nvar == 0): 
        DS_full[1][var].mean(dim='model').plot(ax=ax[nf],color='darkblue',label='OMIP2-MMM')
    else:
        DS_full[1][var].mean(dim='model').plot(ax=ax[nf],color='darkblue')
        
    ax[nf].fill_between(x=DS_full[1]['time'],
                        y1=DS_full[1][var].min(dim='model'),
                        y2=DS_full[1][var].max(dim='model'),
                        alpha=0.5, facecolor='lightblue')

    if (nvar == 0): 
        ax[nf].legend(bbox_to_anchor=(1.05,1.0),loc='upper left')

ylim = [ [7, 24], [100, 210], [-25, -4], [-30, 5] ]
ytick = [ np.linspace(6, 24, 10),
          np.linspace(100, 210, 12),
          np.linspace(-25,-4,8),
          np.linspace(-30,5,8), ]

for n in range(3):
    for nvar in range(4):
        nf = nvar * 3 + n
        ax[nf].set_title(title_list[nf],{'fontsize':10,'verticalalignment':'top'})
        ax[nf].tick_params(labelsize=8)
        ax[nf].set_xlim(1592,2018)
        ax[nf].set_xticks(np.arange(1592,2018.1,71))
        if ( nvar == 3 ):
            ax[nf].set_xlabel('year',fontsize=10)
        else:
            ax[nf].set_xlabel('',fontsize=10)

        ax[nf].grid()
        ax[nf].set_ylim(ylim[nvar])
        ax[nf].set_yticks(ytick[nvar])
        if (n == 0):
            ax[nf].set_ylabel(r'$10^{9} \mathrm{kg}\,\mathrm{s}^{-1}$',fontsize=10)
        else:
            ax[nf].set_ylabel('')

#J titleとx軸ラベルが重なるのを防ぐ
#J tight_layout は suptitle を考慮しないので、上側を少しあける
plt.subplots_adjust(left=0.07,right=0.80,bottom=0.07,top=0.92,hspace=0.30)
#plt.tight_layout(rect=[0,0,1.0,0.98])

outpdf = outfile + '.pdf'
outpng = outfile + '.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

print("figure is saved to " + outpng + " and " + outpdf)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    plt.show()

