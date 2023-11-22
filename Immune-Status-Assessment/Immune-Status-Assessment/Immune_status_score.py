#!/usr/bin/env python
# encoding: utf-8



import numpy as np
import pandas as pd
# ---- Load Data------
df = pd.read_csv('CBC_log_norm.csv',encoding='gbk')
df=df.iloc[:,1:16]
#print(df)
for i in range(0,16715):
    w = [0.033507783,0.028023357,0.089361353,0.023105702,0.132087605,0.06132467,0.035868842,0.092644057,
         0.028731643,0.073189748,0.047448335,0.057712101,0.065757518,0.124933278,0.106304007]

    score=w[0]*df.iloc[i][0]+w[1]*df.iloc[i][1]+w[2]*df.iloc[i][2]+w[3]*df.iloc[i][3]+w[4]*df.iloc[i][4]+w[5]*df.iloc[i][5]+w[6]*df.iloc[i][6]+w[7]*df.iloc[i][7]+w[8]*df.iloc[i][8]+w[9]*df.iloc[i][9]+w[10]*df.iloc[i][10]+w[11]*df.iloc[i][11]+w[12]*df.iloc[i][12]+w[13]*df.iloc[i][13]+w[14]*df.iloc[i][14]
    print(score)
