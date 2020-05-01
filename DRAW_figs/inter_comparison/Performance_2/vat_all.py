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

infl1 = "./csv_spin/vat_omip1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_spin/vat_omip2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)

fig = plt.figure(figsize=(11,8))
fig.suptitle("VAT (1980-2009) of the last cycle relative to the first year mean", size='x-large')

axes=fig.add_subplot(2,2,1)

for index, row in df.iterrows():
    print(row['v700m-OMIP1'],row['v700m-OMIP2'])
    btm=row['v700m-OMIP1']
    side=row['v700m-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'
        
    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df[['v700m-OMIP1','v700m-OMIP2']].corr()
print(r)
x_np = df[['v700m-OMIP1']].values
y_np = df['v700m-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-0.4,1.0,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.25)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-0.3,0.7,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('VAT(0-700m)')
axes.set_xlabel('OMIP1_v700m [$^{\circ}$C]')
axes.set_ylabel('OMIP2_v700m [$^{\circ}$C]')
axes.set_xlim(-0.4,0.9)
axes.set_ylim(-0.4,0.9)
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,2)

for index, row in df.iterrows():
    print(row['v2000m-OMIP1'],row['v2000m-OMIP2'])
    btm=row['v2000m-OMIP1']
    side=row['v2000m-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'
    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df[['v2000m-OMIP1','v2000m-OMIP2']].corr()
print(r)
x_np = df[['v2000m-OMIP1']].values
y_np = df['v2000m-OMIP2'].values
print(x_np,y_np)
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-0.7,0.8,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-0.6,0.5,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('VAT(0-2000m)')
axes.set_xlabel('OMIP1-v2000m [$^{\circ}$C]')
axes.set_ylabel('OMIP2-v2000m [$^{\circ}$C]')
axes.set_xlim(-0.7,0.7)
axes.set_ylim(-0.7,0.7)
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

axes=fig.add_subplot(2,2,3)

for index, row in df.iterrows():
    print(row['v2000m-bot-OMIP1'],row['v2000m-bot-OMIP2'])
    btm=row['v2000m-bot-OMIP1']
    side=row['v2000m-bot-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'
    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df[['v2000m-bot-OMIP1','v2000m-bot-OMIP2']].corr()
print(r)
x_np = df[['v2000m-bot-OMIP1']].values
y_np = df['v2000m-bot-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-0.8,0.9,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-0.7,0.5,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('VAT(2000m-bottom)')
axes.set_xlabel('OMIP1-v2000m-bot [$^{\circ}$C]')
axes.set_ylabel('OMIP2-v2000m-bot [$^{\circ}$C]')
axes.set_xlim(-0.8,0.8)
axes.set_ylim(-0.8,0.8)
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
axes.grid(b=True,which='major',axis='both')
#axes.legend(bbox_to_anchor=(1.35,0.9))

axes=fig.add_subplot(2,2,4)

for index, row in df.iterrows():
    print(row['vtop-bot-OMIP1'],row['vtop-bot-OMIP2'])
    btm=row['vtop-bot-OMIP1']
    side=row['vtop-bot-OMIP2']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'
    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

r = df[['vtop-bot-OMIP1','vtop-bot-OMIP2']].corr()
print(r)
x_np = df[['vtop-bot-OMIP1']].values
y_np = df['vtop-bot-OMIP2'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(-0.7,0.8,0.1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(-0.6,0.5,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('VAT(0m-bottom)')
axes.set_xlabel('OMIP1-vtop-bot [$^{\circ}$C]')
axes.set_ylabel('OMIP2-vtop-bot [$^{\circ}$C]')
axes.set_xlim(-0.7,0.7)
axes.set_ylim(-0.7,0.7)
axes.axhline(0,color="black",linewidth=1.0)
axes.axvline(0,color="black",linewidth=1.0)
axes.grid(b=True,which='major',axis='both')
axes.legend(bbox_to_anchor=(1.01,0.8),loc='lower left')

plt.subplots_adjust(left=0.07,right=0.75,bottom=0.10,top=0.85,hspace=0.35,wspace=0.3)

#fig.tight_layout()

plt.savefig('fig/vat_all.png')
plt.show()
