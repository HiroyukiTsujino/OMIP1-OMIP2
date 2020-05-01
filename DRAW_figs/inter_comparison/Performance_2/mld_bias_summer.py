# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["darkred", "darkblue"]
miparcs = ["orange", "lightblue"]
titles = {"interannual":"interannual variability", "monclim":"monthly climatology"}

markinfo = json.load(open('../json/inst_color_style.json'))

mldcirc=[]
num_models=[]

infl1 = "./csv_clim/MLD_summer_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/MLD_summer_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
#num_models len(df.index)
#print(num_models)

fig = plt.figure(figsize=(11,6))
fig.suptitle("MLD bias summer", size='x-large')

axes=fig.add_subplot(1,2,1)

for index, row in df.iterrows():
    print(row['OMIP1_rmse'],row['OMIP2_rmse'])
    btm=row['OMIP1_rmse']
    side=row['OMIP2_rmse']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='lightblue',marker=markers[mi],s=50,label=name)

x=np.arange(5,21,1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
axes.set_title('Summer MLD bias rmse')
axes.set_xlabel('OMIP1_rmse [m]')
axes.set_ylabel('OMIP2_rmse [m]')
axes.set_xlim(5,20)
axes.set_ylim(5,20)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(1,2,2)

for index, row in df.iterrows():
    print(row['OMIP1_mean'],row['OMIP2_mean'])
    btm=row['OMIP1_mean']
    side=row['OMIP2_mean']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='red',edgecolors='orange',marker=markers[mi],s=50,label=name)

x=np.arange(-15,16,1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
axes.set_title('Summer MLD bias mean')
axes.set_xlabel('OMIP1_mean [m]')
axes.set_ylabel('OMIP2_mean [m]')
axes.set_xlim(-15,15)
axes.set_ylim(-15,15)
axes.grid(b=True,which='major',axis='both')
axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

plt.subplots_adjust(left=0.07,right=0.82,bottom=0.15,top=0.85)

#fig.tight_layout()

plt.savefig('fig/mld_bias_summer.png')
plt.show()
