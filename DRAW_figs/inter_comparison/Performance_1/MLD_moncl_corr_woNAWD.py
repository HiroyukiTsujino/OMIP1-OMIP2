# -*- coding: utf-8 -*-
import sys
sys.path.append("../../../python")
import json
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import datetime
from taylorDiagram import TaylorDiagram


if (len(sys.argv) < 2) :
    print ('Usage: ' + sys.argv[0] + ' [MMM or modelname] [show (to check using viewer)]')
    sys.exit()

if (sys.argv[1] == 'MMM'):
    suptitle = 'Multi Model Mean' + ' Correlation of monthly climatology of MLD from 1980 to 2009'
    outfile = './fig/mldcor_monclim_woNAWD_MMM'

else:
    suptitle = sys.argv[1] + ' (Correlation of monthly climatology of MLD from 1980 to 2009)'
    model_list[0] = [sys.argv[1]]
    model_list[1] = [sys.argv[1]]
    outfile = './fig/mldcor_monclim_woNAWD_' + sys.argv[1]

title = [ '(a) OMIP1', '(b) OMIP2', '(c) OMIP2 - OMIP1', '(d) Taylor diagram' ]

metainfo = [ json.load(open("./json/mldcor_monclim_woNAWD_omip1.json")), 
             json.load(open("./json/mldcor_monclim_woNAWD_omip2.json")) ]
model_list = [ metainfo[0].keys(), metainfo[1].keys() ]

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["red", "darkblue"]
miparcs = ["orange", "lightblue"]
titles = {"interannual":" (d) Taylor diagram for \n spatio-temporal variation", "monclim":" (d) Taylor diagram for \n spatio-temporal variation"}

#J データ読込・平均

data = []
for omip in range(2):
    d = np.empty( (len(model_list[omip]),90,180) )
    print( "Loading OMIP" + str(omip+1) + " data" )

    nmodel = 0
    for model in model_list[omip]:
        path  = metainfo[omip][model]['path']
        fname = metainfo[omip][model]['fname']
        infile = path + '/' + fname

        DS = xr.open_dataset( infile, decode_times=False )

        tmp = DS['mldcor']

        d[nmodel] = tmp.values
        nmodel += 1

    data += [d]


DS = xr.Dataset( {'omip1': (['model','lat','lon'], data[0]),
                  'omip2': (['model','lat','lon'], data[1]),
                  'omip2-1': (['model','lat','lon'], data[1] - data[0]), },
                 coords = { 'lat': np.linspace(-88.0,90.0,num=90), 
                            'lon': np.linspace(0,358,num=180), } )



#J 描画
fig = plt.figure(figsize=(11,8))
fig.suptitle( suptitle, fontsize=18 )

proj = ccrs.PlateCarree(central_longitude=-140.)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

ax = [
    plt.subplot(2,2,1,projection=proj),
    plt.subplot(2,2,2,projection=proj),
    plt.subplot(2,2,3,projection=proj),
]

bounds1 = [-1.0, -0.95, -0.9, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0]
bounds2 = [-0.2, -0.15, -0.1, -0.05, -0.02, -0.01, 0.00, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
tickbounds1 = [-1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0]
tickbounds2 = [-0.2, -0.1, -0.02, 0.00, 0.02, 0.1, 0.2]

cmap = [ 'RdBu_r', 'RdBu_r', 'RdBu_r' ]

item = [ 'omip1', 'omip2', 'omip2-1' ]

for panel in range(3):
    if item[panel] == 'omip1' or item[panel] == 'omip2':
        bounds = bounds1
        ticks_bounds = tickbounds1
    elif item[panel] == 'omip2-1':
        bounds = bounds2
        ticks_bounds = tickbounds2

    da = DS[item[panel]].mean(dim='model',skipna=False)
    da.plot(ax=ax[panel],cmap=cmap[panel],
            levels=bounds,
            cbar_kwargs={'orientation': 'horizontal',
#                         'spacing':'proportional',
                         'spacing':'uniform',
                         'label': "",
                         'ticks': ticks_bounds,},
            transform=ccrs.PlateCarree())
for panel in range(3):
    ax[panel].coastlines()
    ax[panel].set_xticks(np.arange(-180,180.1,60),crs=ccrs.PlateCarree())
    ax[panel].set_yticks(np.arange(-90,90.1,30),crs=ccrs.PlateCarree())
    ax[panel].xaxis.set_major_formatter(lon_formatter)
    ax[panel].yaxis.set_major_formatter(lat_formatter)
    ax[panel].set_title(title[panel],{'fontsize':10, 'verticalalignment':'top'})
    ax[panel].tick_params(labelsize=9)
    ax[panel].set_xlabel('')
    ax[panel].set_ylabel('')
    ax[panel].background_patch.set_facecolor('lightgray')

###################################################
# Draw Taylor Diagram    

mldstats=[]
num_models=[]

intvl='monclim'
for omip in ['omip1', 'omip2']:
    df=pd.read_csv('csv_mld/mld_' + intvl + '_woNAWD_' + omip + '.csv',index_col=0)
    mldstats += [df]
    num_models += [len(df.index) - 1]

print(num_models)

stdrefs={}
for index, row in mldstats[0].iterrows():
    #print(index)
    #print(row[0],row[1])
    if (index == 'Reference'):
        stdrefs['monclim'] = row[1]

print(stdrefs)

rects = dict(interannual=224,
             monclim=224)
legcol = dict(interannual=[0.825,0.05],
              monclim=[0.825,0.05])

id_df=0

dia = TaylorDiagram(stdrefs[intvl],
                    fig=fig,
                    rect=rects[intvl],
                    label='deBoyer et al. (2004)',
                    srange=(0.0, 2.0))

for mip in range(2):
    id_df=id_df+1
    i = 0
    for index, row in mldstats[id_df-1].iterrows():
        print(row[0],row[1])
        if (index != 'Reference'):
            stddev=row[1]
            corrcoef=row[0]
            name=index + '-omip' + str(mip+1)
            dia.add_sample(stddev, corrcoef,
                           marker=markers[i], ms=9, ls='',
                           #mfc='k', mec='k', # B&W
                           mfc=mipcols[mip], mec=miparcs[mip], # Colors
                           label=name)
            i = i + 1
            
# Add RMS contours, and label them
contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
# Tricky: ax is the polar ax (used for plots), _ax is the
# container (used for layout)
#dia._ax.set_title(mip.capitalize())
dia._ax.set_title(titles[intvl],{'fontsize':10, 'verticalalignment':'baseline'})

# Add a figure legend and title. For loc option, place x,y tuple inside [ ].
# Can also use special options here:
# http://matplotlib.sourceforge.net/users/legend_guide.html

tmp = [p.get_label() for p in dia.samplePoints[0:num_models[0]+1]]
print(tmp)

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints[0:num_models[0]+num_models[1]+1] ],
           numpoints=1, fontsize=8, loc=legcol[intvl])

plt.subplots_adjust(left=0.04,right=0.84,hspace=0.2,wspace=0.15,top=0.95,bottom=0.08)

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

plt.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
plt.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 3 and sys.argv[2] == 'show') :
    plt.show()

print("figure is saved to " + outpng + " and " + outpdf)
