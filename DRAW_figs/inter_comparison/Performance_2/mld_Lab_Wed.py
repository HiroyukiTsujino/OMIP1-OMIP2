# -*- coding: utf-8 -*-

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#-----------------------------------------------------------------------------

markers = ["d","o","v","^","<",">","p","*","+","s","h","D"]
mipcols = ["darkred", "darkblue"]
miparcs = ["orange", "lightblue"]
titles = {"interannual":"interannual variability", "monclim":"monthly climatology"}

markinfo = json.load(open('../json/inst_color_style.json'))

### Winter

infl1 = "./csv_mld/winter_mld_omip1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_mld/winter_mld_omip2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)

fig = plt.figure(figsize=(11,5.5))
fig.suptitle("MLD in Labrador Sea and Weddell Sea", size='x-large')

axes=fig.add_subplot(1,2,1)

for index, row in df.iterrows():
    print(row['NH-MLD-OMIP1'],row['NH-MLD-OMIP2'])
    btm=row['NH-MLD-OMIP1']
    side=row['NH-MLD-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    markcol='blue'
    edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df.loc[:,['NH-MLD-OMIP1','NH-MLD-OMIP2']].corr()
print(r)
x_np = df[['NH-MLD-OMIP1']].values
y_np = df['NH-MLD-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0,4100,500)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(500,3500,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('Labrador Sea Winter MLD')
axes.set_xlabel('OMIP1 [m]')
axes.set_ylabel('OMIP2 [m]')
axes.set_xlim(0,4000)
axes.set_ylim(0,4000)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(1,2,2)

for index, row in df.iterrows():
    print(row['SH-MLD-OMIP1'],row['SH-MLD-OMIP2'])
    btm=row['SH-MLD-OMIP1']
    side=row['SH-MLD-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    markcol='red'
    edgecol='orange'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df.loc[:,['SH-MLD-OMIP1','SH-MLD-OMIP2']].corr()
print(r)
x_np = df[['SH-MLD-OMIP1']].values
y_np = df['SH-MLD-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0,5100,500)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(500,4500,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('Weddell Sea MLD')
axes.set_xlabel('OMIP1 [m]')
axes.set_ylabel('OMIP2 [m]')
axes.set_xlim(0,5000)
axes.set_ylim(0,5000)
axes.grid(b=True,which='major',axis='both')
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
axes.legend(bbox_to_anchor=(1.01,0.8),loc='upper left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.85,hspace=0.30,wspace=0.25)

#fig.tight_layout()

plt.savefig('fig/mld_LabWed.png')
plt.show()
