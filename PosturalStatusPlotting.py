import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data_DF = pd.read_csv('SkeletonRecording.csv')
Data_DF = pd.read_csv('Input data/TwoMinuteStep_0404-092056.csv')
Data_DF.drop('Unnamed: 76', axis = 1, inplace = True)
Data_NP = np.array(Data_DF)

from PostDraw import PosturalDrawing
Fig, ax = plt.subplots(1,3)
#plt.figure(100)
Margin = 0.1
XRange = [np.min(Data_NP[:,range(1,75,3)])-Margin, np.max(Data_NP[:,range(1,75,3)])+Margin]
YRange = [np.min(Data_NP[:,range(2,75,3)])-Margin, np.max(Data_NP[:,range(2,75,3)])+Margin]
ZRange = [np.min(Data_NP[:,range(3,75,3)])-Margin, np.max(Data_NP[:,range(3,75,3)])+Margin]
for t in range (0, Data_DF.shape[0], 15):
    C1 = 'X'
    C2 = 'Y'
    Labels = ['X(m)', 'Y(m)']
    R1 = XRange
    R2 = YRange
    PosturalDrawing(ax[0], C1, C2, R1, R2, Labels, Data_DF, t)
    
    C1 = 'Z'
    C2 = 'Y'
    Labels = ['Z(m)', 'Y(m)']
    R1 = ZRange
    R2 = YRange
    PosturalDrawing(ax[1], C1, C2, R1, R2, Labels, Data_DF, t)
    
    C1 = 'X'
    C2 = 'Z'
    Labels = ['X(m)', 'Z(m)']
    R1 = XRange
    R2 = ZRange
    PosturalDrawing(ax[2], C1, C2, R1, R2, Labels, Data_DF, t)
    