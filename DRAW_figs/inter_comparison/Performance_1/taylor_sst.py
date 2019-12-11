#!/usr/bin/env python

__version__ = "Time-stamp: <2018-12-06 11:55:22 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

"""
Example of use of TaylorDiagram. Illustration dataset courtesy of Michael
Rawlins.

Rawlins, M. A., R. S. Bradley, H. F. Diaz, 2012. Assessment of regional climate
model simulation estimates over the Northeast United States, Journal of
Geophysical Research (2012JGRD..11723112R).
"""

import sys
sys.path.append("../../../python")
import numpy as NP
import pandas as PD
import matplotlib.pyplot as PLT

from taylorDiagram import TaylorDiagram
#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["red", "darkblue"]
miparcs = ["orange", "lightblue"]
titles = {"interannual":"interannual variability", "monclim":"monthly climatology"}

sststats=[]
num_models=[]

for intvl in ['interannual','monclim']:
    for omip in ['omip1', 'omip2']:
        df=PD.read_csv('csv_sst/sst_' + intvl + '_' + omip + '.csv',index_col=0)
        sststats += [df]
        num_models += [len(df.index) - 1]

print(num_models)

stdrefs={}

for index, row in sststats[0].iterrows():
    #print(index)
    #print(row[0],row[1])
    if (index == 'Reference'):
        stdrefs['interannual'] = row[1]

for index, row in sststats[2].iterrows():
    #print(index)
    #print(row[0],row[1])
    if (index == 'Reference'):
        stdrefs['monclim'] = row[1]

print(stdrefs)
#sys.exit()

# Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
#colors = PLT.matplotlib.cm.Set1(NP.linspace(0,1,len(samples['winter'])))
#colors = PLT.matplotlib.cm.Set1(NP.linspace(0,1,num_models))

# Here set placement of the points marking 95th and 99th significance
# levels. For more than 102 samples (degrees freedom > 100), critical
# correlation levels are 0.195 and 0.254 for 95th and 99th
# significance levels respectively. Set these by eyeball using the
# standard deviation x and y axis.

#x95 = [0.01, 0.68] # For Tair, this is for 95th level (r = 0.195)
#y95 = [0.0, 3.45]
#x99 = [0.01, 0.95] # For Tair, this is for 99th level (r = 0.254)
#y99 = [0.0, 3.45]

#####x95 = [0.05, 13.9] # For Prcp, this is for 95th level (r = 0.195)
#####y95 = [0.0, 71.0]
#####x99 = [0.05, 19.0] # For Prcp, this is for 99th level (r = 0.254)
#####y99 = [0.0, 70.0]

rects = dict(interannual=121,
             monclim=122)
legcol = dict(interannual=[0.38,0.7],
              monclim=[0.88,0.7])

fig = PLT.figure(figsize=(15,8))
fig.suptitle("Sea Surface Temeperature", size='x-large')

#for season in ['winter','spring','summer','autumn']:
id_df=0
for intvl in ['interannual','monclim']:

    print(rects[intvl],stdrefs[intvl])

    dia = TaylorDiagram(stdrefs[intvl],
                        fig=fig,
                        rect=rects[intvl],
                        label='PCMDI-SST')

    #dia.ax.plot(x95,y95,color='k')
    #dia.ax.plot(x99,y99,color='k')

    # Add samples to Taylor diagram

    for mip in range(2):
        id_df=id_df+1
        i = 0
        for index, row in sststats[id_df-1].iterrows():
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
    dia._ax.set_title(titles[intvl])

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html

    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints[0:num_models[0]+1] ],
               numpoints=1, prop=dict(size='small'), loc=legcol[intvl])

fig.tight_layout()

PLT.savefig('fig/SST_Taylor.png')
PLT.show()
