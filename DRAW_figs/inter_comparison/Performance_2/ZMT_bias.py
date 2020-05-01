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

markinfo = json.load(open('../json/inst_color_style.json'))

mldcirc=[]
num_models=[]

infl1 = "./csv_clim/ZMT_bias_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/ZMT_bias_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)
dfm=df.drop('MMM',axis=0)
print(dfm)
#num_models len(df.index)
#print(num_models)

fig = plt.figure(figsize=(11,8))
fig.suptitle("ZMT bias", size='x-large')

axes=fig.add_subplot(2,2,1)

for index, row in df.iterrows():
    print(row['OMIP1_rmse_Southern'],row['OMIP2_rmse_Southern'])
    btm=row['OMIP1_rmse_Southern']
    side=row['OMIP2_rmse_Southern']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='lightblue',marker=markers[mi],s=50,label=name)

r = dfm.loc[:,['OMIP1_rmse_Southern','OMIP2_rmse_Southern']].corr()
print(r)
x_np = dfm[['OMIP1_rmse_Southern']].values
y_np = dfm['OMIP2_rmse_Southern'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0.0,2.4,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(0.1,1.0,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('ZMT bias rmse Southern')
axes.set_xlabel('OMIP1_Southern [$^{\circ}$C]')
axes.set_ylabel('OMIP2_Southern [$^{\circ}$C]')
axes.set_xlim(0.0,1.2)
axes.set_ylim(0.0,1.2)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,2)

for index, row in df.iterrows():
    print(row['OMIP1_rmse_Atlantic'],row['OMIP2_rmse_Atlantic'])
    btm=row['OMIP1_rmse_Atlantic']
    side=row['OMIP2_rmse_Atlantic']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='lightblue',marker=markers[mi],s=50,label=name)


r = dfm.loc[:,['OMIP1_rmse_Atlantic','OMIP2_rmse_Atlantic']].corr()
print(r)
x_np = dfm[['OMIP1_rmse_Atlantic']].values
y_np = dfm['OMIP2_rmse_Atlantic'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0.0,2.8,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(0.2,1.2,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('ZMT bias rmse Atlantic')
axes.set_xlabel('OMIP1_Atlantic [$^{\circ}$C]')
axes.set_ylabel('OMIP2_Atlantic [$^{\circ}$C]')
axes.set_xlim(0.0,1.4)
axes.set_ylim(0.0,1.4)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

axes=fig.add_subplot(2,2,3)

for index, row in df.iterrows():
    print(row['OMIP1_rmse_Indian'],row['OMIP2_rmse_Indian'])
    btm=row['OMIP1_rmse_Indian']
    side=row['OMIP2_rmse_Indian']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='lightblue',marker=markers[mi],s=50,label=name)

r = dfm.loc[:,['OMIP1_rmse_Indian','OMIP2_rmse_Indian']].corr()
print(r)
x_np = dfm[['OMIP1_rmse_Indian']].values
y_np = dfm['OMIP2_rmse_Indian'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0.0,2.8,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(0.2,1.2,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('ZMT bias rmse Indian')
axes.set_xlabel('OMIP1_Indian [$^{\circ}$C]')
axes.set_ylabel('OMIP2_Indian [$^{\circ}$C]')
axes.set_xlim(0.0,1.4)
axes.set_ylim(0.0,1.4)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,4)

for index, row in df.iterrows():
    print(row['OMIP1_rmse_Pacific'],row['OMIP2_rmse_Pacific'])
    btm=row['OMIP1_rmse_Pacific']
    side=row['OMIP2_rmse_Pacific']
    mi = int(markinfo[index]["marker"])
    name=index
    axes.scatter(btm,side,c='blue',edgecolors='lightblue',marker=markers[mi],s=50,label=name)

r = dfm.loc[:,['OMIP1_rmse_Pacific','OMIP2_rmse_Pacific']].corr()
print(r)
x_np = dfm[['OMIP1_rmse_Pacific']].values
y_np = dfm['OMIP2_rmse_Pacific'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(0.0,2.0,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(0.1,0.8,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('ZMT bias rmse Pacific')
axes.set_xlabel('OMIP1_Pacific [$^{\circ}$C]')
axes.set_ylabel('OMIP2_Pacific [$^{\circ}$C]')
axes.set_xlim(0.0,1.0)
axes.set_ylim(0.0,1.0)
axes.grid(b=True,which='major',axis='both')
axes.legend(bbox_to_anchor=(1.01,0.8),loc='lower left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.85,hspace=0.30)

#fig.tight_layout()

plt.savefig('fig/ZMT_bias.png')
plt.show()
