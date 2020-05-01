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

markinfo = json.load(open('../json/inst_color_style.json'))

infl1 = "./csv_clim/SSS_bias_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/SSS_bias_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
#num_models len(df.index)
#print(num_models)

fig = plt.figure(figsize=(11,6))
fig.suptitle("SSS bias", size='x-large')

axes=fig.add_subplot(1,2,1)

for index, row in df.iterrows():
    print(row['OMIP1_rmse'],row['OMIP2_rmse'])
    btm=row['OMIP1_rmse']
    side=row['OMIP2_rmse']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='blue',marker=markers[mi],s=50,label=name)

x=np.arange(0.2,0.8,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
axes.set_title('SSS bias rmse')
axes.set_xlabel('OMIP1_rmse [degC]')
axes.set_ylabel('OMIP2_rmse [degC]')
axes.set_xlim(0.2,0.7)
axes.set_ylim(0.2,0.7)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(1,2,2)

for index, row in df.iterrows():
    print(row['OMIP1_mean'],row['OMIP2_mean'])
    btm=row['OMIP1_mean']
    side=row['OMIP2_mean']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='red',edgecolors='red',marker=markers[mi],s=50,label=name)

x=np.arange(-0.2,0.5,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
axes.set_title('SSS bias mean')
axes.set_xlabel('OMIP1_mean [psu]')
axes.set_ylabel('OMIP2_mean [psu]')
axes.set_xlim(-0.2,0.4)
axes.set_ylim(-0.2,0.4)
axes.grid(b=True,which='major',axis='both')
axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

plt.subplots_adjust(left=0.07,right=0.82,bottom=0.15,top=0.85)

#fig.tight_layout()

plt.savefig('fig/SSS_bias.png')
plt.show()
