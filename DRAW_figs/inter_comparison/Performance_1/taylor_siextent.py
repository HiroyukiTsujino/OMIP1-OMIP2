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
import json
import numpy as NP
import pandas as PD
import matplotlib.pyplot as PLT

from taylorDiagram import TaylorDiagram
#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["red", "darkblue"]
miparcs = ["orange", "lightblue"]
titles = {"NH_MAR":"(a) NH Mar", "NH_SEP":"(b) NH Sep","SH_MAR":"(c) SH Mar", "SH_SEP":"(d) SH Sep"}

month = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
         'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

markinfo = json.load(open('../json/inst_color_style.json'))

sststats=[]
num_models=[]

for hemi in ['NH', 'SH']:
    for mon in [3, 9]:
        for omip in ['OMIP1', 'OMIP2']:

            df=PD.read_csv('csv_siextent/siextent_'+ hemi + '_' + month[mon-1] + '_' + omip + '.csv',index_col=0)
            sststats += [df]
            num_models += [len(df.index) - 1]

print("number of models", num_models)

stdrefs={}

for index, row in sststats[0].iterrows():
    if (index == 'Reference'):
        stdrefs['NH_MAR'] = row[1]

for index, row in sststats[2].iterrows():
    if (index == 'Reference'):
        stdrefs['NH_SEP'] = row[1]

for index, row in sststats[4].iterrows():
    if (index == 'Reference'):
        stdrefs['SH_MAR'] = row[1]

for index, row in sststats[6].iterrows():
    if (index == 'Reference'):
        stdrefs['SH_SEP'] = row[1]

print("stdrefs", stdrefs)
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

rects = dict(NH_MAR=221,
             NH_SEP=222,
             SH_MAR=223,
             SH_SEP=224)

legcol = dict(NH_MAR=[0.90,0.1],
              NH_SEP=[0.90,0.1],
              SH_MAR=[0.90,0.1],
              SH_SEP=[0.90,0.1])

stdrange = dict(NH_MAR=(0.0, 1.2),
                NH_SEP=(0.0, 2.0),
                SH_MAR=(0.0, 1.8),
                SH_SEP=(0.0, 1.8))

# [left, bottom, width, height]

fig = PLT.figure(figsize=(11,8))
fig.suptitle("Sea ice extent", size='x-large')
#ax_leg = PLT.axes([0.9,0.1,0.1,0.7])

#for season in ['winter','spring','summer','autumn']:
id_df=0
for intvl in ['NH_MAR','NH_SEP','SH_MAR','SH_SEP']:

    print(rects[intvl],stdrefs[intvl])

    dia = TaylorDiagram(stdrefs[intvl],
                        fig=fig,
                        rect=rects[intvl],
                        label='NSIDC-SII',
                        srange=(stdrange[intvl]))

    #dia.ax.plot(x95,y95,color='k')
    #dia.ax.plot(x99,y99,color='k')

    # Add samples to Taylor diagram

    for mip in range(2):
        id_df=id_df+1
        i = 0
        for index, row in sststats[id_df-1].iterrows():
            print(row[0],row[1])
            if (index != 'Reference'):
                if (index == 'MRICOM'):
                    index = 'MRI.COM'
                if (index == 'FSU-COAPS'):
                    index = 'FSU-HYCOM'
                mi = int(markinfo[index]["marker"])
                stddev=row[1]
                corrcoef=row[0]
                name=index + '-omip' + str(mip+1)
                dia.add_sample(stddev, corrcoef,
                               marker=markers[mi], ms=9, ls='',
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
    dia._ax.set_title(titles[intvl],loc='center')

    # Add a figure legend and title. For loc option, place x,y tuple inside [ ].
    # Can also use special options here:
    # http://matplotlib.sourceforge.net/users/legend_guide.html

dia._ax.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints[0:num_models[0]+num_models[1]+1] ],
               numpoints=1, fontsize=9, loc=legcol['NH_MAR'],bbox_to_anchor=(1.1,0.3))

PLT.subplots_adjust(left=0.1,right=0.82,bottom=0.05,top=0.90,hspace=0.25,wspace=0.1)

outfile='fig/SIextent_Taylor'

outpdf = outfile+'.pdf'
outpng = outfile+'.png'

PLT.savefig(outpng, bbox_inches='tight', pad_inches=0.0)
PLT.savefig(outpdf, bbox_inches='tight', pad_inches=0.0)

if (len(sys.argv) == 2 and sys.argv[1] == 'show') :
    PLT.show()

print("figure is saved to " + outpng + " and " + outpdf)
