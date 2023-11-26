#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec


#***********************************Import data (healthy individuals)******************
#**************************************Mean and Standard Deviation ***************************************
data = pd.read_csv("CBC_log.csv",encoding="gbk")
data=data.iloc[:,1:16]

#print(data)
# ##16715
# **************************Mean values for 15 indicators************************
u=[1.966850638,1.493308144,1.12653678,0.338878581,0.123207044,0.031654829,4.04142695,3.546295432,2.001699131,1.038156654,0.402900656,0.989441321,0.180709781,0.062908538,0.015605378]
# ***********************Standard deviations for 15 indicators*****************
sig=[0.170500598,0.194747393,0.166497665,0.085147587,0.096164967,0.018091575,0.120074724,0.186182107,0.212112492,0.456958673,0.181062008,0.19026847	,0.055985809,0.054762819,0.009427501]
output = []
#********************************Standardization based on the three-platform model*********************
for i in range(0,16715):
    #WBC
    if data.iloc[i][0]<=u[0]:
        y1 =np.exp(-(data.iloc[i][0] - u[0]) ** 2 / (2 * sig[0] ** 2))/2
    else:
        y1 = 1-(np.exp(-(data.iloc[i][0] - u[0]) ** 2 / (2 * sig[0] ** 2)) / 2)
    #NEUT
    if data.iloc[i][1]<=u[1]:
        y2 =np.exp(-(data.iloc[i][1] - u[1]) ** 2 / (2 * sig[1] ** 2)) / 2
    else:
        y2 = 1-np.exp(-(data.iloc[i][1] - u[1]) ** 2 / (2 * sig[1] ** 2)) / 2
    #print("y2：",y2)
    # #LYMPH
    if data.iloc[i][2]<=u[2]:
        y3 =np.exp(-(data.iloc[i][2] - u[2]) ** 2 / (2 * sig[2] ** 2)) / 2
    else:
        y3 = 1-np.exp(-(data.iloc[i][2] - u[2]) ** 2 / (2 * sig[2] ** 2)) / 2
    #print("y3：",y3)
    # #MONO
    if data.iloc[i][3]<=u[3]:
        y4 = 1-np.exp(-(data.iloc[i][3] - u[3]) ** 2 / (2 * sig[3] ** 2)) / 2
    else:
        y4 = np.exp(-(data.iloc[i][3] - u[3]) ** 2 / (2 * sig[3] ** 2)) / 2
    #print("y4：",y4)
    #EO
    if data.iloc[i][4]<=u[4]:
        y5 = 1-np.exp(-(data.iloc[i][4] - u[4]) ** 2 / (2 * sig[4] ** 2)) / 2
    else:
        y5 =np.exp(-(data.iloc[i][4] - u[4]) ** 2 / (2 * sig[4] ** 2)) / 2
    #print("y5：",y5)
    #BASO
    if data.iloc[i][5]<=u[5]:
        y6 = 1-np.exp(-(data.iloc[i][5] - u[5]) ** 2 / (2 * sig[5] ** 2)) /2
    else:
        y6 = np.exp(-(data.iloc[i][5] - u[5]) ** 2 / (2 * sig[5] ** 2)) / 2
    #print("y6：",y6)
    # #NEUT%
    if data.iloc[i][6]<=u[6]:
        y7 = 1-np.exp(-(data.iloc[i][6] - u[6]) ** 2 / (2 * sig[6] ** 2)) / 2
    else:
        y7 = np.exp(-(data.iloc[i][6] - u[6]) ** 2 / (2 * sig[6] ** 2)) / 2
    #print("y7：",y7)
    # #LYMPH%
    if data.iloc[i][7]<=u[7]:
        y8 =np.exp(-(data.iloc[i][7] - u[7]) ** 2 / (2 * sig[7] ** 2)) / 2
    else:
        y8 = 1-np.exp(-(data.iloc[i][7] - u[7]) ** 2 / (2 * sig[7] ** 2)) / 2
    #print("y8：",y8)
    # #MONO%
    if data.iloc[i][8]<=u[8]:
        y9 = 1-np.exp(-(data.iloc[i][8] - u[8]) ** 2 / (2 * sig[8] ** 2)) / 2
    else:
        y9 = np.exp(-(data.iloc[i][8] - u[8]) ** 2 / (2 * sig[8] ** 2)) / 2
    #print("y9：",y9)
    # #EO%
    if data.iloc[i][9]<=u[9]:
        y10 = 1-np.exp(-(data.iloc[i][9] - u[9]) ** 2 / (2 * sig[9] ** 2)) / 2
    else:
        y10 = np.exp(-(data.iloc[i][9] - u[9]) ** 2 / (2 * sig[9] ** 2)) / 2
    #print("y10：",y10)
    # #BASO%
    if data.iloc[i][10]<=u[10]:
        y11 = 1-np.exp(-(data.iloc[i][10] - u[10]) ** 2 / (2 * sig[10] ** 2)) / 2
    else:
        y11 = np.exp(-(data.iloc[i][10] - u[10]) ** 2 / (2 * sig[10] ** 2)) / 2
    #print("y11：",y11)
    #NLR
    if data.iloc[i][11]<=u[11]:
        y12 = 1 - np.exp(-(data.iloc[i][11] - u[11]) ** 2 / (2 * sig[11] ** 2)) / 2
    else:
        y12 = np.exp(-(data.iloc[i][11] - u[11]) ** 2 / (2 * sig[11] ** 2)) / 2
    #print("y12：", y12)
    # # MLR
    if data.iloc[i][12] <= u[12]:
        y13 = 1 - np.exp(-(data.iloc[i][12] - u[12]) ** 2 / (2 * sig[12] ** 2)) / 2
    else:
        y13 = np.exp(-(data.iloc[i][12] - u[12]) ** 2 / (2 * sig[12] ** 2)) / 2
    #print("y13：", y13)
    # # ELR
    if data.iloc[i][13] <= u[13]:
        y14 =  1 - np.exp(-(data.iloc[i][13] - u[13]) ** 2 / (2 * sig[13] ** 2)) / 2
    else:
        y14 = np.exp(-(data.iloc[i][13] - u[13]) ** 2 / (2 * sig[13] ** 2)) / 2
    # # NLR
    if data.iloc[i][14] <= u[14]:
        y15 = 1 - np.exp(-(data.iloc[i][14] - u[14]) ** 2 / (2 * sig[14] ** 2)) / 2
    else:
        y15 = np.exp(-(data.iloc[i][14] - u[14]) ** 2 / (2 * sig[14] ** 2)) / 2
    # #print("y15：", y15)
    output.append([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15])
output_df = pd.DataFrame(output, columns=["WBC", "NEUT", "LYMPH", "MONO", "EO", "BASO", "NEUT%", "LYMPH%", "MONO%", "EO%", "BASO%", "NLR", "MLR", "ELR", "BLR"])
output_df.to_csv("CBC_log_norm.csv", index=False)