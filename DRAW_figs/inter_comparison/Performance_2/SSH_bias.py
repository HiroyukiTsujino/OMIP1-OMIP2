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

mldcirc=[]
num_models=[]

infl1 = "./csv_clim/SSH_bias_OMIP1.csv"
df1=pd.read_csv(infl1,index_col=0)
infl2 = "./csv_clim/SSH_bias_OMIP2.csv"
df2=pd.read_csv(infl2,index_col=0)
df = pd.merge(df1,df2,left_index=True,right_index=True)
print(df)

fig = plt.figure(figsize=(6,5))
#fig.suptitle("SSH bias rmse", size='x-large')

axes=fig.add_subplot(1,1,1)

for index, row in df.iterrows():
    print(row['OMIP1_rmse'],row['OMIP2_rmse'])
    btm=row['OMIP1_rmse']
    side=row['OMIP2_rmse']
    mi = int(markinfo[index]["marker"])
    name=index
    if (index == 'MMM'):
        markcol='red'
        edgecol='red'
    else:
        markcol='blue'
        edgecol='lightblue'

    axes.scatter(btm,side,c=markcol,edgecolors=edgecol,marker=markers[mi],s=50,label=name)

dfm=df.drop('MMM',axis=0)
print(dfm)
r = dfm.loc[:,['OMIP1_rmse','OMIP2_rmse']].corr()
print(r)
x_np = dfm[['OMIP1_rmse']].values
y_np = dfm['OMIP2_rmse'].values
lr = LinearRegression()
lr.fit(x_np, y_np)
print("coefficients", lr.coef_)
print("intercept", lr.intercept_)
print("score", lr.score(x_np, y_np))

x=np.arange(5,21,1)
y=x.copy()
axes.plot(x,y,color="grey",linewidth=0.5)
xx=x.reshape(-1,1)
axes.plot(xx,lr.predict(xx),color='red',linewidth=0.25)
axes.text(6,18.5,"$r^2$="+'{:.3f}'.format(lr.score(x_np,y_np)))
axes.set_title('SSH bias rmse')
axes.set_xlabel('OMIP1_rmse [m]')
axes.set_ylabel('OMIP2_rmse [m]')
axes.set_xlim(5,20)
axes.set_ylim(5,20)
axes.grid(b=True,which='major',axis='both')
axes.legend(bbox_to_anchor=(1.01,0.9),loc='upper left')

plt.subplots_adjust(left=0.10,right=0.70,bottom=0.10,top=0.75)

#fig.tight_layout()

plt.savefig('fig/SSH_bias.png')
plt.show()
