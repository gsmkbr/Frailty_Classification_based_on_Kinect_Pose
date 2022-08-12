class ImportData:
    def __init__(self, DIR):
        self.DIR = DIR
        
    def DataReading(self, PP, FileName):
        import pandas as pd
        import numpy as np
        if PP.ActiveDataSet == 'Cheng':
            self.Data_DF = pd.read_csv(self.DIR+FileName)
            self.Data_DF.drop('Unnamed: 76', axis = 1, inplace = True)
        elif PP.ActiveDataSet == 'Andersson':
            delta_T = 0.01
            TempDF = pd.read_csv(PP.DataRootDirectory+'\\KinectLabels.csv')
            self.Data_DF = pd.DataFrame(columns = TempDF.columns)
            file = open(self.DIR+FileName, "r")
            LabelEquivalece = {'Head':'Head',
                               'Shoulder-Center':'Neck',
                               'Shoulder-Right':'ShoulderRight',
                               'Shoulder-Left':'ShoulderLeft',
                               'Elbow-Right':'ElbowRight',
                               'Elbow-Left':'ElbowLeft',
                               'Wrist-Right':'WristRight',
                               'Wrist-Left':'WristLeft',
                               'Hand-Right':'HandRight',
                               'Hand-Left':'HandLeft',
                               'Spine':'SpineMid',
                               'Hip-centro':'SpineBase',
                               'Hip-Right':'HipRight',
                               'Hip-Left':'HipLeft',
                               'Knee-Right':'KneeRight',
                               'Knee-Left':'KneeLeft',
                               'Ankle-Right':'AnkleRight',
                               'Ankle-Left':'AnkleLeft',
                               'Foot-Right':'FootRight',
                               'Foot-Left':'FootLeft'}
            String = file.readline()
            counter =0
            #print(self.DIR, FileName)
            while String != '':
                self.Data_DF = self.Data_DF.append(TempDF, ignore_index = True)
                self.Data_DF.iloc[counter]['AbsTime'] = counter*delta_T
                for Joint in range (0,20):
                    #String = file.readline()
                    Split = String.split(';')
                    #print(Split)
                    Split[3] = Split[3][0:-1]
                    Keyword = LabelEquivalece[Split[0]]
                    #print(Keyword)
                    Coord = ['X','Y','Z']
                    for component in range(0,3):
                        Label = Keyword + Coord[component]
                        self.Data_DF.iloc[counter][Label] = float(Split[component+1])
                    String = file.readline()
                counter = counter + 1
                #print(counter)
            
        
    def MissingDataRemoval(self):
        import numpy as np
        NoTimeFrames = self.Data_DF.shape[0]
        for TimeFrame in range (NoTimeFrames-1, -1, -1):
            Vector = np.array(self.Data_DF.iloc[TimeFrame])
            Vector[np.where(Vector!=0)]=1
            NonZeroCount = np.sum(Vector)
            if NonZeroCount < 10:
                self.Data_DF.drop(TimeFrame, axis=0, inplace=True)
        self.Data_DF = self.Data_DF.reset_index(drop=True)
        self.Data_NP = np.array(self.Data_DF)
        self.NoTimeFrames = self.Data_DF.shape[0]
        
    def DataSmoothing(self, PP):
        import pandas as pd
        import numpy as np
        self.Data_DF = self.Data_DF.rolling(PP.SmoothWindowSize, center=True, min_periods = 1, axis = 0, win_type=PP.WinType).mean(std = PP.GaussianStDev)
        self.Data_NP = np.array(self.Data_DF)
    
    def DataSlicing(self, PP, FileInitial):
        from math import floor
        import numpy as np
        StartIndex = floor(PP.SlicePercentage[FileInitial][0]*self.NoTimeFrames)
        EndIndex = floor(PP.SlicePercentage[FileInitial][1]*self.NoTimeFrames)
        
        self.Data_DF = self.Data_DF.iloc[StartIndex:EndIndex,:]
        self.Data_DF = self.Data_DF.reset_index(drop=True)
        self.NoTimeFrames = self.Data_DF.shape[0]
        self.Data_NP = np.array(self.Data_DF)
        
    def IsolateCycles(self, PP, FileInitial, HumanIndex):
        import numpy as np
        from math import fabs, floor
        
        def SliceUsefulInterval(StartPercentage, EndPercentage, DataFrame):
            StartIndex = floor(StartPercentage/100*self.NoTimeFrames)
            EndIndex = floor(EndPercentage/100*self.NoTimeFrames)
            SlicedDataFrame = DataFrame.iloc[StartIndex:EndIndex,:]
            return SlicedDataFrame
        
        StartPercentage = 0
        EndPercentage = 100
        
        DistanceType = {'1': 'Single',
                        '2': 'Pair'}
        
        if FileInitial == 'GaitAnalysis' or FileInitial == '4':
            self.DistanceType = DistanceType['2']
            self.NoCycles = 1
            RefKey1 = 'AnkleRight'
            RefKey2 = 'AnkleLeft'
            self.NoRepititionPerCycle = 2
            YLabel = 'Distance between \''+RefKey1+'\' & \''+RefKey2+'\' (m)'
        elif FileInitial == 'ArmCurlTestMultiple':
            StartPercentage = 50
            EndPercentage = 90
            self.DistanceType = DistanceType['2']
            self.NoCycles = 1
            RefKey1 = 'Head'         #'AnkleLeft'
            RefKey2 = 'WristRight'
            self.NoRepititionPerCycle = 1
            YLabel = 'Distance between \''+RefKey1+'\' & \''+RefKey2+'\' (m)'
        elif FileInitial == 'TwoMinuteStep':
            StartPercentage = 50
            EndPercentage = 70
            self.DistanceType = DistanceType['2']
            self.NoCycles = 1
            RefKey1 = 'AnkleLeft'
            RefKey2 = 'AnkleRight'
            self.NoRepititionPerCycle = 2
            YLabel = 'Distance between \''+RefKey1+'\' & \''+RefKey2+'\' (m)'
        elif FileInitial == 'ThirtySecondsChairStand':
            StartPercentage = 50
            EndPercentage = 100
            self.DistanceType = DistanceType['2']
            self.NoCycles = 1
            RefKey1 = 'KneeLeft'
            RefKey2 = 'SpineShoulder'
            self.NoRepititionPerCycle = 1
            YLabel = 'Distance between \''+RefKey1+'\' & \''+RefKey2+'\' (m)'
        
        SlicedDataFrame = SliceUsefulInterval(StartPercentage, EndPercentage, self.Data_DF)
        self.Data_DF = SlicedDataFrame.reset_index(drop=True)
        self.NoTimeFrames = self.Data_DF.shape[0]
        self.Data_NP = np.array(self.Data_DF)
            
        #FilteredData_DF = self.Data_DF.rolling(1000, min_periods = 1, axis = 0, win_type=PP.WinType).mean(std = PP.GaussianStDev)
        #if FileInitial == 'GaitAnalysis' or '4':
            #RefKey1 = 'AnkleRight'
            #RefKey2 = 'AnkleLeft'
#        RefJoint1 = {'X':np.array(FilteredData_DF[RefKey1+'X']), 
#                      'Y':np.array(FilteredData_DF[RefKey1+'Y']), 
#                      'Z':np.array(FilteredData_DF[RefKey1+'Z'])}
#        RefJoint2 = {'X':np.array(FilteredData_DF[RefKey2+'X']), 
#                      'Y':np.array(FilteredData_DF[RefKey2+'Y']), 
#                      'Z':np.array(FilteredData_DF[RefKey2+'Z'])}
        if self.DistanceType == 'Pair':
            RefJoint1 = {'X':np.array(self.Data_DF[RefKey1+'X']), 
                          'Y':np.array(self.Data_DF[RefKey1+'Y']), 
                          'Z':np.array(self.Data_DF[RefKey1+'Z'])}
            RefJoint2 = {'X':np.array(self.Data_DF[RefKey2+'X']), 
                          'Y':np.array(self.Data_DF[RefKey2+'Y']), 
                          'Z':np.array(self.Data_DF[RefKey2+'Z'])}
            #JointsDistance = np.sqrt(np.power((RefJoint1['X'] - RefJoint2['X']),2)+np.power((RefJoint1['Z'] - RefJoint2['Z']),2))
            JointsDistance = np.sqrt(np.power((RefJoint1['X'] - RefJoint2['X']),2)+np.power((RefJoint1['Y'] - RefJoint2['Y']),2)+np.power((RefJoint1['Z'] - RefJoint2['Z']),2))
        elif self.DistanceType == 'Single':
            SingleRefJoint = {'X':np.array(self.Data_DF[SingleRefKey+'X']), 
                          'Y':np.array(self.Data_DF[SingleRefKey+'Y']), 
                          'Z':np.array(self.Data_DF[SingleRefKey+'Z'])}
            JointsDistance = SingleRefJoint['Y']
            
            
        from scipy.signal import argrelextrema
        #LocalMaxLocations = argrelextrema(JointsDistance, np.greater)[0]
        #LocalMinLocations = argrelextrema(JointsDistance, np.less)[0]
        from pyampd.ampd import find_peaks
        
        LocalMaxLocations = find_peaks(JointsDistance)
        LocalMinLocations = find_peaks(-JointsDistance)
        AllLocalExremumLocations = np.append(LocalMaxLocations, LocalMinLocations)
        SortedLocalExremumLocations = np.sort(AllLocalExremumLocations)
        #MaxVal = JointsDistance[SortedLocalExremumLocations].max()
        #MinVal = JointsDistance[SortedLocalExremumLocations].min()
        MaxVal = np.quantile(JointsDistance[SortedLocalExremumLocations],0.9)
        MinVal = np.quantile(JointsDistance[SortedLocalExremumLocations],0.1)
        OverallCondition = False
        counter = 0
        while not OverallCondition:
            MaxWeight = 0.8
            weight = MaxWeight - 0.05*counter
            Threshold = weight*(MaxVal - MinVal)
            CycleMaxLocations = []
            CycleMinLocations = []
            #self.NoCycles = 1
            #for i in range (0,SortedLocalExremumLocations.size-1):
            #for i in range (0,SortedLocalExremumLocations.size-5):
            for i in range (1,SortedLocalExremumLocations.size-(2*self.NoRepititionPerCycle*self.NoCycles+1)):
                Condition = []
                OverallCondition = True
                for j in range (0,2*self.NoRepititionPerCycle*self.NoCycles+1):
                    Condition.extend([fabs(JointsDistance[SortedLocalExremumLocations[i+j+1]]-JointsDistance[SortedLocalExremumLocations[i+j]]) > Threshold])
                    OverallCondition = OverallCondition and Condition[j]
#                    Condition1 = fabs(JointsDistance[SortedLocalExremumLocations[i+1]]-JointsDistance[SortedLocalExremumLocations[i]]) > Threshold
#                    Condition2 = fabs(JointsDistance[SortedLocalExremumLocations[i+2]]-JointsDistance[SortedLocalExremumLocations[i+1]]) > Threshold
#                    Condition3 = fabs(JointsDistance[SortedLocalExremumLocations[i+3]]-JointsDistance[SortedLocalExremumLocations[i+2]]) > Threshold
#                    Condition4 = fabs(JointsDistance[SortedLocalExremumLocations[i+4]]-JointsDistance[SortedLocalExremumLocations[i+3]]) > Threshold
#                    Condition5 = fabs(JointsDistance[SortedLocalExremumLocations[i+5]]-JointsDistance[SortedLocalExremumLocations[i+4]]) > Threshold
#                    Condition6 = fabs(JointsDistance[SortedLocalExremumLocations[i+6]]-JointsDistance[SortedLocalExremumLocations[i+5]]) > Threshold
#                    Condition7 = fabs(JointsDistance[SortedLocalExremumLocations[i+7]]-JointsDistance[SortedLocalExremumLocations[i+6]]) > Threshold
#                    Condition8 = fabs(JointsDistance[SortedLocalExremumLocations[i+8]]-JointsDistance[SortedLocalExremumLocations[i+7]]) > Threshold
#                    Condition9 = fabs(JointsDistance[SortedLocalExremumLocations[i+9]]-JointsDistance[SortedLocalExremumLocations[i+8]]) > Threshold
                
                #OverallCondition = Condition1 and Condition2 and Condition3 and Condition4 and Condition5
                #OverallCondition = Condition1 and Condition2 and Condition3 and Condition4 and Condition5 and Condition6 and Condition7 and Condition8 and Condition9
                if OverallCondition:
                #if fabs(JointsDistance[SortedLocalExremumLocations[i+1]]-JointsDistance[SortedLocalExremumLocations[i]]) > Threshold:
#                    if JointsDistance[SortedLocalExremumLocations[i+1]] > JointsDistance[SortedLocalExremumLocations[i]]:
#                        CycleMaxLocations.extend([SortedLocalExremumLocations[i+1]])
#                    else:
#                        CycleMinLocations.extend([SortedLocalExremumLocations[i+1]])
                    if JointsDistance[SortedLocalExremumLocations[i+1]] > JointsDistance[SortedLocalExremumLocations[i]]:
                        StartIndex = SortedLocalExremumLocations[i+1]
                        #EndIndex = SortedLocalExremumLocations[i+5]
                        EndIndex = SortedLocalExremumLocations[i+(2*self.NoRepititionPerCycle*self.NoCycles+1)]
                    else:
                        StartIndex = SortedLocalExremumLocations[i]
                        #EndIndex = SortedLocalExremumLocations[i+4]
                        EndIndex = SortedLocalExremumLocations[i+(2*self.NoRepititionPerCycle*self.NoCycles)]
                    break
            
            if weight<0:
                SortedLocationsSize = SortedLocalExremumLocations.size
                StartIndex = SortedLocalExremumLocations[SortedLocationsSize-2*self.NoRepititionPerCycle*self.NoCycles-2]
                EndIndex = SortedLocalExremumLocations[SortedLocationsSize-2]
                break
            
            counter+=1
        
#            AllCycleLocations = np.append(CycleMaxLocations, CycleMinLocations)
#            SortedCycleLocations = np.sort(AllCycleLocations)
#            print(SortedCycleLocations)
#            index=0
#            counter=0
#            NoCycles = 1
#            while index == 0:
#                print(RefJoint1['Z'][SortedCycleLocations[counter]] , RefJoint2['Z'][SortedCycleLocations[counter]])
#                #if RefJoint1['Z'][SortedCycleLocations[counter]] > RefJoint2['Z'][SortedCycleLocations[counter]] and SortedCycleLocations[counter] in CycleMaxLocations:
#                if SortedCycleLocations[counter] in CycleMaxLocations:
#                    index=1
#                    StartIndex = SortedCycleLocations[counter]
#                    print(StartIndex)
#                    if (counter+4*NoCycles) <= SortedCycleLocations.size-1:
#                        EndIndex = SortedCycleLocations[counter+4*NoCycles]
#                    else:
#                        EndIndex = RefJoint1['Z'].size
#                        #EndIndex = min(SortedCycleLocations[-1], StartIndex+20*4*NoCycles)
#                counter += 1
#                if counter == SortedCycleLocations.size:
#                    StartIndex = SortedCycleLocations[counter-4*NoCycles-1]
#                    EndIndex = min(SortedCycleLocations[counter-1], StartIndex+20*4*NoCycles)
#                    break
#            print(StartIndex, EndIndex)
        
        
#            RefKey1 = 'WristRight'
#            RefKey2 = 'WristLeft'
#            RefJoint1 = {'X':np.array(self.Data_DF[RefKey1+'X']), 
#                          'Y':np.array(self.Data_DF[RefKey1+'Y']), 
#                          'Z':np.array(self.Data_DF[RefKey1+'Z'])}
#            RefJoint2 = {'X':np.array(self.Data_DF[RefKey2+'X']), 
#                          'Y':np.array(self.Data_DF[RefKey2+'Y']), 
#                          'Z':np.array(self.Data_DF[RefKey2+'Z'])}
#            JointsDistance = np.sqrt(np.power((RefJoint1['X'] - RefJoint2['X']),2)+np.power((RefJoint1['Y'] - RefJoint2['Y']),2)+np.power((RefJoint1['Z'] - RefJoint2['Z']),2))
        #plt.plot()
        if PP.PlotCycleIsolation == True:
            import matplotlib.pyplot as plt
    #        plt.figure()
    #        
    #        plt.plot(JointsDistance[StartIndex:EndIndex])
    ##            
            plt.figure()
            plt.plot(JointsDistance[:], linewidth=3)
            for Index in LocalMaxLocations:
                plt.plot(Index, JointsDistance[Index], 'ro', markersize=10, alpha=0.75)
            
            for Index in LocalMinLocations:
                plt.plot(Index, JointsDistance[Index], 'go', markersize=10, alpha=0.75)
            plt.title('HumanIndex: {}, Weight: {}\nStartIndex: {}, EndIndex: {}'.format(HumanIndex, weight, StartIndex, EndIndex))
            plt.xlabel('Time Frame')
            plt.ylabel(YLabel)
            import matplotlib     
            font = {'family' : 'normal', 'size' : 12}	
            matplotlib.rc('font', **font)
    #        matplotlib.rc('xtick', labelsize=20)     
    #        matplotlib.rc('ytick', labelsize=20)
    #        matplotlib.rc('xlabel', labelsize=20)     
    #        matplotlib.rc('ylabel', labelsize=20)
            
        
        #x = input('Enter a key to continue...')
        
        
        
        print('StartIndex, EndIndex, Weight: ({}, {}, {})'.format(StartIndex, EndIndex, weight))
        
        if StartIndex == EndIndex:
            EndIndex = self.NoTimeFrames-1
        
        TargetTimeFrameRange = list(np.arange(StartIndex, EndIndex))
        self.Data_DF = self.Data_DF.iloc[TargetTimeFrameRange]
        self.Data_DF = self.Data_DF.reset_index(drop=True)
        self.NoTimeFrames = self.Data_DF.shape[0]
        self.Data_NP = np.array(self.Data_DF)
        
        X = self.Data_DF
        self.AveragedFeatures = np.array(X).mean()
        self.Weight = weight
    
    def DataResampling(self):
        import pandas as pd
        import numpy as np
        #Normalizing the time intervals
        Times = (self.Data_DF['AbsTime']-self.Data_DF['AbsTime'][0])/(self.Data_DF['AbsTime'][self.NoTimeFrames-1]-self.Data_DF['AbsTime'][0])
        Times = pd.to_datetime(Times, unit='s', origin=pd.Timestamp('2020-12-21'))
        
#        import matplotlib.pyplot as plt
#        plt.figure()
#        ts = pd.Series(np.array(self.Data_DF['AnkleLeftZ']), index = list(Times))
#        plt.plot(ts)
#        ts = ts.resample('10L').mean()
#        ts = ts.interpolate(method='cubic')
#        plt.plot(ts)
        NIntervalsPerCycle = 25
        NTotalIntervals = NIntervalsPerCycle * self.NoCycles
        from math import floor
        delta = floor(1000 / NTotalIntervals)
        ResamplingUnit = str(delta)+'L'
        DataDF = pd.DataFrame(columns = self.Data_DF.columns)
        for Col in DataDF.columns:
            ts = pd.Series(np.array(self.Data_DF[Col]), index = list(Times))
            ts = ts.resample(ResamplingUnit).mean()
            ts = ts.interpolate(method='cubic')
            DataDF[Col] = ts
        DataDF.index = ts.index
        self.Data_DF = DataDF.copy()
        
        self.NoTimeFrames = self.Data_DF.shape[0]
        self.Data_NP = np.array(self.Data_DF)
        
    
    def OrientationCalc(self, PP):
        def AngleCalculator(StartJoint, EndJoint):
            from math import atan, acos, sqrt, pi, fabs
            DeltaX = EndJoint[0]-StartJoint[0]
            DeltaY = EndJoint[1]-StartJoint[1]
            DeltaZ = EndJoint[2]-StartJoint[2]
            #l_h = sqrt(DeltaX**2+DeltaZ**2)
            l_r = sqrt(DeltaX**2+DeltaY**2+DeltaZ**2)
            if DeltaX == 0:
                if DeltaZ>0:
                    phiR = pi/2
                elif DeltaZ<0:
                    phiR = 3*pi/2
                elif DeltaZ==0:
                    phiR = 0
            elif DeltaX>0:
                if DeltaZ>=0:
                    phiR = atan(DeltaZ/DeltaX)
                elif DeltaZ<0:
                    phiR = 2*pi-atan(fabs(DeltaZ)/DeltaX)
            elif DeltaX<0:
                if DeltaZ>=0:
                    phiR = pi - atan(DeltaZ/fabs(DeltaX))
                elif DeltaZ<0:
                    phiR = pi+atan(DeltaZ/DeltaX)
            thetaR = acos(DeltaY/l_r)
            phiD = phiR/pi*180
            thetaD = thetaR/pi*180
            return [phiD, thetaD]
        
        import numpy as np
        self.Alpha = np.zeros((PP.NoMainJoints, 2, self.NoTimeFrames))
        
        self.AlphaResampledConcat = np.array([])
        for limb in range (0, PP.NoMainJoints):
            StartJointName = PP.MainJoints[limb][0]
            EndJointName = PP.MainJoints[limb][1]
            
            #NoTimeFrames = self.Data_DF[StartJointName+'X'].size
            for TimeFrame in range (0, self.NoTimeFrames):
                X1 = self.Data_DF[StartJointName+'X'][TimeFrame]
                Y1 = self.Data_DF[StartJointName+'Y'][TimeFrame]
                Z1 = self.Data_DF[StartJointName+'Z'][TimeFrame]
                X2 = self.Data_DF[EndJointName+'X'][TimeFrame]
                Y2 = self.Data_DF[EndJointName+'Y'][TimeFrame]
                Z2 = self.Data_DF[EndJointName+'Z'][TimeFrame]
                StartJoint = [X1, Y1, Z1]
                EndJoint = [X2, Y2, Z2]
                self.Alpha[limb,:,TimeFrame] = AngleCalculator(StartJoint, EndJoint)
            for angle in range (0,2):
                self.AlphaResampledConcat = np.append(self.AlphaResampledConcat, 
                                                     self.Alpha[limb,angle,:])
        
            
    def AngularVelocity(self, PP):
        import numpy as np
        self.omega = np.zeros((PP.NoMainJoints, 2, self.NoTimeFrames-PP.AngVelTimeStep))
        for limb in range (0, PP.NoMainJoints):
            for angle in range (0,PP.NoAngles):
                for TimeFrame in range (0, self.NoTimeFrames-PP.AngVelTimeStep):
                    if self.Data_DF['AbsTime'][TimeFrame+PP.AngVelTimeStep] != self.Data_DF['AbsTime'][TimeFrame]:
                        self.omega[limb, angle, TimeFrame] = (self.Alpha[limb, angle, TimeFrame+PP.AngVelTimeStep] - self.Alpha[limb, angle, TimeFrame]) / (self.Data_DF['AbsTime'][TimeFrame+PP.AngVelTimeStep] - self.Data_DF['AbsTime'][TimeFrame])
    
    def LinearVelocity(self, PP, FileInitial):
        import numpy as np
        from math import sqrt
        
        if PP.ActiveDataSet == 'Cheng':
            FilesIndex = {'ArmCurlTestMultiple':'1', 
                               'EightFootUpAndGo':'2', 
                               'FunctionalReaching':'3',
                               'GaitAnalysis':'4',
                               'OneLegStance':'5',
                               'ThirtySecondsChairStand':'6',
                               'TwoMinuteStep':'7'}
        elif PP.ActiveDataSet == 'Andersson':
            FilesIndex = {'1':'1', 
                           '2':'2', 
                           '3':'3',
                           '4':'4',
                           '5':'5',
                           '6':'6',
                           '7':'7'}
        
        self.LinVel = {}
        ActiveIndex = FilesIndex[FileInitial]
        for Joint in PP.TargetJoints[ActiveIndex]:
        #for Joint in range (0, 25):
            self.LinVel[PP.KW[str(Joint)]] = np.zeros((1, self.NoTimeFrames-PP.LinVelTimeStep))
            for TimeFrame in range (0, self.NoTimeFrames-PP.LinVelTimeStep):
                DeltaT = self.Data_DF['AbsTime'][TimeFrame+PP.LinVelTimeStep] - self.Data_DF['AbsTime'][TimeFrame]
                if DeltaT != 0:
                    Velocity = {}
                    for component in ['X', 'Y', 'Z']:
                        StartLocation = self.Data_DF[PP.KW[str(Joint)]+component][TimeFrame]
                        EndLocation = self.Data_DF[PP.KW[str(Joint)]+component][TimeFrame+PP.LinVelTimeStep]
                        Velocity[component] = (EndLocation - StartLocation)/DeltaT
                    VelocityMagnitude = sqrt(Velocity['X']**2.0 + Velocity['Y']**2.0 + Velocity['Z']**2.0)
                    self.LinVel[PP.KW[str(Joint)]][0, TimeFrame] = VelocityMagnitude
    
    def DistanceCalculations(self, PP, FileInitial):
        
        def DistanceCalculator(Data, RefKey1, RefKey2):
            import numpy as np
            RefJoint1 = {'X':np.array(Data[RefKey1+'X']), 
                          'Y':np.array(Data[RefKey1+'Y']), 
                          'Z':np.array(Data[RefKey1+'Z'])}
            RefJoint2 = {'X':np.array(Data[RefKey2+'X']), 
                          'Y':np.array(Data[RefKey2+'Y']), 
                          'Z':np.array(Data[RefKey2+'Z'])}
            JointsDistance = np.sqrt(np.power((RefJoint1['X'] - RefJoint2['X']),2)+np.power((RefJoint1['Y'] - RefJoint2['Y']),2)+np.power((RefJoint1['Z'] - RefJoint2['Z']),2))
            return JointsDistance
        
        if PP.ActiveDataSet == 'Cheng':
            FilesIndex = {'ArmCurlTestMultiple':'1', 
                           'EightFootUpAndGo':'2', 
                           'FunctionalReaching':'3',
                           'GaitAnalysis':'4',
                           'OneLegStance':'5',
                           'ThirtySecondsChairStand':'6',
                           'TwoMinuteStep':'7'}
        elif PP.ActiveDataSet == 'Andersson':
            FilesIndex = {'1':'1', 
                           '2':'2', 
                           '3':'3',
                           '4':'4',
                           '5':'5',
                           '6':'6',
                           '7':'7'}
        
        self.ActiveFileIndex = FilesIndex[FileInitial]
        
        self.DistanceVectors = {}
        import numpy as np
        self.DistResampledConcat = np.array([])
        if PP.JointsCombinationStatus == 'CustomizedJoints(SelectedCombinations)' or PP.JointsCombinationStatus == 'AllJointsCombinations':
            NCombinations = len(PP.JointsCombs[self.ActiveFileIndex])
            for CombIndex in range (0, NCombinations):
                FirstKey = PP.JointsCombs[self.ActiveFileIndex][CombIndex][0]
                SecondKey = PP.JointsCombs[self.ActiveFileIndex][CombIndex][1]
                JointsDistanceVector = DistanceCalculator(self.Data_DF, RefKey1 = FirstKey, RefKey2 = SecondKey)
                JointPair = FirstKey + ' & ' + SecondKey
                self.DistanceVectors[JointPair] = JointsDistanceVector
                self.DistResampledConcat = np.append(self.DistResampledConcat, 
                                                     self.DistanceVectors[JointPair])
    
    def JointCosDissimilarity(self, PP, FileInitial):
        def JointCosDissimCalculator(JointCoord1, JointCoord2, RefCoord):
            from math import sqrt
            import numpy as np
            VectorSeries1 = {'X': JointCoord1['X'] - RefCoord['X'],
                             'Y': JointCoord1['Y'] - RefCoord['Y'],
                             'Z': JointCoord1['Z'] - RefCoord['Z']}
            VectorSeries2 = {'X': JointCoord2['X'] - RefCoord['X'],
                             'Y': JointCoord2['Y'] - RefCoord['Y'],
                             'Z': JointCoord2['Z'] - RefCoord['Z']}
            
            VectorLength = VectorSeries1['X'].size
            DissimilarityVector = np.array([])
            for comp in range(0, VectorLength):
                V1 = [VectorSeries1['X'][comp],
                      VectorSeries1['Y'][comp],
                      VectorSeries1['Z'][comp]]
                V2 = [VectorSeries2['X'][comp],
                      VectorSeries2['Y'][comp],
                      VectorSeries2['Z'][comp]]
                V1Mag = sqrt(V1[0]**2.0+V1[1]**2.0+V1[2]**2.0)
                V2Mag = sqrt(V2[0]**2.0+V2[1]**2.0+V2[2]**2.0)
                DotProduct = V1[0]*V2[0] + V1[1]*V2[1] + V1[2]*V2[2]
                if PP.JointSimilarityStatus == 'DisSimilarity':
                    DeltaCosDissim = 1 - DotProduct/(V1Mag*V2Mag)
                elif PP.JointSimilarityStatus == 'Similarity':
                    DeltaCosDissim = DotProduct/(V1Mag*V2Mag)
                DissimilarityVector = np.append(DissimilarityVector, DeltaCosDissim)
            return DissimilarityVector
        
        if PP.ActiveDataSet == 'Cheng':
            FilesIndex = {'ArmCurlTestMultiple':'1', 
                           'EightFootUpAndGo':'2', 
                           'FunctionalReaching':'3',
                           'GaitAnalysis':'4',
                           'OneLegStance':'5',
                           'ThirtySecondsChairStand':'6',
                           'TwoMinuteStep':'7'}
        elif PP.ActiveDataSet == 'Andersson':
            FilesIndex = {'1':'1', 
                           '2':'2', 
                           '3':'3',
                           '4':'4',
                           '5':'5',
                           '6':'6',
                           '7':'7'}
        
        self.ActiveFileIndex = FilesIndex[FileInitial]
        
        #self.DistanceVectors = {}
        import numpy as np
        #self.JointDissimilarityConcat = np.array([])
        #Specifying the reference coordinate type:
        # 1. 'Joint'
        # 2. 'FixedCoordinate'
        ReferenceCoordType = 'Joint'
        if ReferenceCoordType == 'Joint':
            ReferenceJoint = 'SpineMid'
            RefCoord = {'X':self.Data_DF[ReferenceJoint+'X'],
                        'Y':self.Data_DF[ReferenceJoint+'Y'],
                        'Z':self.Data_DF[ReferenceJoint+'Z']}
        elif ReferenceCoordType == 'FixedCoordinate':
            ReferenceJoint = 'FixedCoordinate'
            ReferenceCoordinate = (0,0,0)
            ConstantVector = self.Data_DF['SpineMid'+'X'].copy()
            RefCoord = {}
            Coords = ['X', 'Y', 'Z']
            for comp in range(0,3):
                ConstantVector[:] = ReferenceCoordinate[comp]
                RefCoord[Coords[comp]] = ConstantVector
        
        self.JointDissimConcat = np.array([])        
        for Combination in PP.JointsCombs[self.ActiveFileIndex]:
            if ReferenceJoint not in Combination:
                JointCoord1 = {'X':self.Data_DF[Combination[0]+'X'],
                               'Y':self.Data_DF[Combination[0]+'Y'],
                               'Z':self.Data_DF[Combination[0]+'Z']}
                JointCoord2 = {'X':self.Data_DF[Combination[1]+'X'],
                               'Y':self.Data_DF[Combination[1]+'Y'],
                               'Z':self.Data_DF[Combination[1]+'Z']}
                    
                DissimilarityVector = JointCosDissimCalculator(JointCoord1, JointCoord2, RefCoord)
                self.JointDissimConcat = np.append(self.JointDissimConcat, np.array(DissimilarityVector))

    def JointTriangleArea(self, PP, FileInitial):
        def JointTriangleAreaCalculator(JointCoord1, JointCoord2, RefCoord):
            from math import sqrt
            import numpy as np
            VectorSeries1 = {'X': RefCoord['X'] - JointCoord1['X'],
                             'Y': RefCoord['Y'] - JointCoord1['Y'],
                             'Z': RefCoord['Z'] - JointCoord1['Z']}
            VectorSeries2 = {'X': JointCoord2['X'] - RefCoord['X'],
                             'Y': JointCoord2['Y'] - RefCoord['Y'],
                             'Z': JointCoord2['Z'] - RefCoord['Z']}
            
            VectorLength = VectorSeries1['X'].size
            TriangleAreaVector = np.array([])
            for comp in range(0, VectorLength):
                V1 = [VectorSeries1['X'][comp],
                      VectorSeries1['Y'][comp],
                      VectorSeries1['Z'][comp]]
                V2 = [VectorSeries2['X'][comp],
                      VectorSeries2['Y'][comp],
                      VectorSeries2['Z'][comp]]
                [a1,a2,a3] = V1
                [b1,b2,b3] = V2
                CrossProduct = [a2*b3 - a3*b2,
                                a3*b1 - a1*b3,
                                a1*b2 - a2*b1]
                
                CrossProductMag = sqrt(CrossProduct[0]**2.0+CrossProduct[1]**2.0+CrossProduct[2]**2.0)
                TriangleArea = CrossProductMag/2
                TriangleAreaVector = np.append(TriangleAreaVector, TriangleArea)
            return TriangleAreaVector
        
        if PP.ActiveDataSet == 'Cheng':
            FilesIndex = {'ArmCurlTestMultiple':'1', 
                           'EightFootUpAndGo':'2', 
                           'FunctionalReaching':'3',
                           'GaitAnalysis':'4',
                           'OneLegStance':'5',
                           'ThirtySecondsChairStand':'6',
                           'TwoMinuteStep':'7'}
        elif PP.ActiveDataSet == 'Andersson':
            FilesIndex = {'1':'1', 
                           '2':'2', 
                           '3':'3',
                           '4':'4',
                           '5':'5',
                           '6':'6',
                           '7':'7'}
        
        self.ActiveFileIndex = FilesIndex[FileInitial]
        
        #self.DistanceVectors = {}
        import numpy as np
        ReferenceJoint = 'SpineMid'
        RefCoord = {'X':self.Data_DF[ReferenceJoint+'X'],
                    'Y':self.Data_DF[ReferenceJoint+'Y'],
                    'Z':self.Data_DF[ReferenceJoint+'Z']}
        
        self.JointTriangleAreaConcat = np.array([])        
        for Combination in PP.JointsCombs[self.ActiveFileIndex]:
            if ReferenceJoint not in Combination:
                JointCoord1 = {'X':self.Data_DF[Combination[0]+'X'],
                               'Y':self.Data_DF[Combination[0]+'Y'],
                               'Z':self.Data_DF[Combination[0]+'Z']}
                JointCoord2 = {'X':self.Data_DF[Combination[1]+'X'],
                               'Y':self.Data_DF[Combination[1]+'Y'],
                               'Z':self.Data_DF[Combination[1]+'Z']}
                    
                TriangleAreaVector = JointTriangleAreaCalculator(JointCoord1, JointCoord2, RefCoord)
                self.JointTriangleAreaConcat = np.append(self.JointTriangleAreaConcat, np.array(TriangleAreaVector))                

    def HistBasedFeatures(self, PP, counter, FileInitial):
        import numpy as np
        if counter == 0:
            PP.PhiHistRange = []
            PP.ThetaHistRange = []
            for limb in range (0, PP.NoMainJoints):
                OutlierThreshold = (0.05,0.95)
                Border = 10
                PP.PhiHistRange.append((np.quantile(self.Alpha[limb, 0, :], OutlierThreshold[0])-Border
                                    ,np.quantile(self.Alpha[limb, 0, :], OutlierThreshold[1])+Border))
                PP.ThetaHistRange.append((np.quantile(self.Alpha[limb, 1, :], OutlierThreshold[0])-Border
                                    ,np.quantile(self.Alpha[limb, 1, :], OutlierThreshold[1])+Border))
        
        #Calculation of histogram-based features from limbs angles
        PhiHist = np.zeros((PP.NoMainJoints, PP.NBins))
        ThetaHist = np.zeros((PP.NoMainJoints, PP.NBins))
        self.PhiHistNormalized = np.zeros((PP.NoMainJoints, PP.NBins))
        self.ThetaHistNormalized = np.zeros((PP.NoMainJoints, PP.NBins))
        for limb in range (0, PP.NoMainJoints):
            PhiHist[limb, :] = np.histogram(self.Alpha[limb, 0, :], bins = PP.NBins, range = PP.PhiHistRange[limb], density = False)[0]
            ThetaHist[limb, :] = np.histogram(self.Alpha[limb, 1, :], bins = PP.NBins, range = PP.ThetaHistRange[limb], density = False)[0]
#            PhiHist[limb, :] = np.histogram(self.Alpha[limb, 0, :], bins = PP.NBins, range = PP.HistRangeForPhi, density = True)[0]
#            ThetaHist[limb, :] = np.histogram(self.Alpha[limb, 1, :], bins = PP.NBins, range = PP.HistRangeForTheta, density = True)[0]
            if PhiHist[limb,:].sum() != 0:
                self.PhiHistNormalized[limb,:] = PhiHist[limb,:]/PhiHist[limb,:].sum()
            if ThetaHist[limb,:].sum()!= 0:
                self.ThetaHistNormalized[limb,:] = ThetaHist[limb,:]/ThetaHist[limb,:].sum()
        
        
#        self.PhiHistNormalized = PhiHist*(PP.HistRangeForPhi[1]-PP.HistRangeForPhi[0])/PP.NBins
#        self.ThetaHistNormalized = ThetaHist*(PP.HistRangeForTheta[1]-PP.HistRangeForTheta[0])/PP.NBins
        
        self.FeaturesConcatenated = np.zeros((1, PP.NoMainJoints*PP.NoAngles*PP.NBins))
        for limb in range (0, PP.NoMainJoints):
            for AngleIndex, AngleType in enumerate(PP.ActiveAngles):
                StartIndex = (limb*PP.NoAngles + AngleIndex)*PP.NBins
                EndIndex = StartIndex + PP.NBins
                if AngleType == 'Phi':
                    self.FeaturesConcatenated[0,StartIndex:EndIndex] = self.PhiHistNormalized[limb, :]
                elif AngleType == 'Theta':
                    self.FeaturesConcatenated[0,StartIndex:EndIndex] = self.ThetaHistNormalized[limb, :]
        
        # Calculation of histogram-based features from limbs angular velocities
        NBins = 50
        PhiAngVelHist = np.zeros((PP.NoMainJoints, NBins))
        ThetaAngVelHist = np.zeros((PP.NoMainJoints, NBins))
        self.PhiAngVelHistNormalized = np.zeros((PP.NoMainJoints, NBins))
        self.ThetaAngVelHistNormalized = np.zeros((PP.NoMainJoints, NBins))
        for limb in range (0, PP.NoMainJoints):
#            PhiHistRange = (np.quantile(self.omega[limb, 0, :], 0.1), np.quantile(self.omega[limb, 0, :], 0.9))
#            PhiAngVelHist[limb, :] = np.histogram(self.omega[limb, 0, :], bins = PP.NBins, range = PhiHistRange, density = True)[0]
#            ThetaHistRange = (np.quantile(self.omega[limb, 1, :], 0.1), np.quantile(self.omega[limb, 1, :], 0.9))
#            ThetaAngVelHist[limb, :] = np.histogram(self.omega[limb, 1, :], bins = PP.NBins, range = ThetaHistRange, density = True)[0]
            PhiHistRange = (-51,51)
            PhiAngVelHist[limb, :] = np.histogram(self.omega[limb, 0, :], bins = NBins, range = PhiHistRange, density = False)[0]
            ThetaHistRange = (-51,51)
            ThetaAngVelHist[limb, :] = np.histogram(self.omega[limb, 1, :], bins = NBins, range = ThetaHistRange, density = False)[0]
            if PhiAngVelHist[limb,:].sum() != 0:
                self.PhiAngVelHistNormalized[limb,:] = PhiAngVelHist[limb,:]/PhiAngVelHist[limb,:].sum()
            if ThetaAngVelHist[limb,:].sum()!= 0:
                self.ThetaAngVelHistNormalized[limb,:] = ThetaAngVelHist[limb,:]/ThetaAngVelHist[limb,:].sum()
        #self.PhiAngVelHistNormalized = PhiAngVelHist*(PhiHistRange[1]-PhiHistRange[0])/PP.NBins
        #self.ThetaAngVelHistNormalized = ThetaAngVelHist*(ThetaHistRange[1]-ThetaHistRange[0])/PP.NBins
        
        self.AngularVelocity = np.zeros((1, PP.NoMainJoints*PP.NoAngles*NBins))
        for limb in range (0, PP.NoMainJoints):
            for AngleIndex, AngleType in enumerate(PP.ActiveAngles):
                StartIndex = (limb*PP.NoAngles + AngleIndex)*NBins
                EndIndex = StartIndex + NBins
                if AngleType == 'Phi':
                    self.AngularVelocity[0,StartIndex:EndIndex] = self.PhiAngVelHistNormalized[limb, :]
                elif AngleType == 'Theta':
                    self.AngularVelocity[0,StartIndex:EndIndex] = self.ThetaAngVelHistNormalized[limb, :]
                    
        #Calculation of histogram-based features from Joints velocity
        if PP.ActiveDataSet == 'Cheng':
            FilesIndex = {'ArmCurlTestMultiple':'1', 
                           'EightFootUpAndGo':'2', 
                           'FunctionalReaching':'3',
                           'GaitAnalysis':'4',
                           'OneLegStance':'5',
                           'ThirtySecondsChairStand':'6',
                           'TwoMinuteStep':'7'}
        elif PP.ActiveDataSet == 'Andersson':
            FilesIndex = {'1':'1', 
                           '2':'2', 
                           '3':'3',
                           '4':'4',
                           '5':'5',
                           '6':'6',
                           '7':'7'}
        
        ActiveIndex = FilesIndex[FileInitial]
        if counter == 0:
            PP.JointsVelHistRange = {}
            for Joint in PP.TargetJoints[ActiveIndex]:
                OutlierThreshold = (0.01,0.95)
                Border = 0
                PP.JointsVelHistRange[PP.KW[str(Joint)]] = (np.quantile(self.LinVel[PP.KW[str(Joint)]][0, :], OutlierThreshold[0])-Border
                                    ,np.quantile(self.LinVel[PP.KW[str(Joint)]][0, :], OutlierThreshold[1])+Border)
        
        NoHists = len(PP.TargetJoints[ActiveIndex])
        LinVelHist = np.zeros((NoHists, PP.NBins))
        self.LinVelHistNormalized = np.zeros((NoHists, PP.NBins))
        for index,Joint in enumerate(PP.TargetJoints[ActiveIndex]):
            LinVelHist[index, :] = np.histogram(self.LinVel[PP.KW[str(Joint)]][0, :], bins = PP.NBins, range = PP.JointsVelHistRange[PP.KW[str(Joint)]], density = False)[0]
            if LinVelHist[index,:].sum() != 0:
                self.LinVelHistNormalized[index,:] = LinVelHist[index,:]/LinVelHist[index,:].sum()
    
        
        self.JointsLinVel = np.zeros((1, NoHists*PP.NBins))
        for index, Joint in enumerate(PP.TargetJoints[ActiveIndex]):
                StartIndex = index*PP.NBins
                EndIndex = StartIndex + PP.NBins
                self.JointsLinVel[0,StartIndex:EndIndex] = self.LinVelHistNormalized[index, :]

        
        
        #Calculation of histogram-based features from joints distances
        NoJointsCombinations = len(self.DistanceVectors)
        
        if counter == 0:
            #PP.DistanceHistRange = {}
            DistanceHistRange = {}
            for CombIndex in range (0, NoJointsCombinations):
                FirstKey = PP.JointsCombs[self.ActiveFileIndex][CombIndex][0]
                SecondKey = PP.JointsCombs[self.ActiveFileIndex][CombIndex][1]
                JointPair = FirstKey + ' & ' + SecondKey
                OutlierThreshold = (0.05,0.95)
                #Border = 0.1
                ExtensionPercent = 5
#                PP.DistanceHistRange[JointPair] = [np.quantile(self.DistanceVectors[JointPair], OutlierThreshold[0])
#                                    ,np.quantile(self.DistanceVectors[JointPair], OutlierThreshold[1])]
#                PP.DistanceHistRange[JointPair][0] = PP.DistanceHistRange[JointPair][0]*(1-ExtensionPercent/100)
#                PP.DistanceHistRange[JointPair][1] = PP.DistanceHistRange[JointPair][1]*(1+ExtensionPercent/100)
                DistanceHistRange[JointPair] = [np.quantile(self.DistanceVectors[JointPair], OutlierThreshold[0])
                                    ,np.quantile(self.DistanceVectors[JointPair], OutlierThreshold[1])]
                DistanceHistRange[JointPair][0] = DistanceHistRange[JointPair][0]*(1-ExtensionPercent/100)
                DistanceHistRange[JointPair][1] = DistanceHistRange[JointPair][1]*(1+ExtensionPercent/100)
            PP.DistanceHistRange[FileInitial] = DistanceHistRange
        
        NBins = 10
        DistanceHist = np.zeros((NoJointsCombinations, NBins))
        DistanceHistNormalized = np.zeros((NoJointsCombinations, NBins))
        for CombIndex in range (0, NoJointsCombinations):
            FirstKey = PP.JointsCombs[self.ActiveFileIndex][CombIndex][0]
            SecondKey = PP.JointsCombs[self.ActiveFileIndex][CombIndex][1]
            JointPair = FirstKey + ' & ' + SecondKey
            #HistRange = (np.min(self.DistanceVectors[JointPair]), np.max(self.DistanceVectors[JointPair]))
            #HistRange = (0,1.2)
#            DistanceHist[CombIndex, :] = np.histogram(self.DistanceVectors[JointPair], bins = NBins, range = HistRange, density = True)[0]
            DistanceHist[CombIndex, :] = np.histogram(self.DistanceVectors[JointPair], bins = NBins, range = PP.DistanceHistRange[FileInitial][JointPair], density = False)[0]
            if DistanceHist[CombIndex, :].sum() != 0:
                DistanceHistNormalized[CombIndex, :] = DistanceHist[CombIndex, :]/DistanceHist[CombIndex, :].sum()
        
        self.DistHistConcatenated = np.zeros((1, NoJointsCombinations*NBins))
        for CombIndex in range (0, NoJointsCombinations):
            StartIndex = (CombIndex)*NBins
            EndIndex = StartIndex + NBins
            self.DistHistConcatenated[0,StartIndex:EndIndex] = DistanceHistNormalized[CombIndex, :]
        
    def FFTBaseFeatures(self, PP):
        import numpy as np
        from scipy.fftpack import fft
        from math import fabs
        NoFFTBins = 40
        #SpecEnergyArray = np.zeros((PP.NoMainJoints, NoFFTBins))
        self.SpecFeature = np.zeros((1, PP.NoMainJoints*PP.NoAngles*NoFFTBins))
        for limb in range (0, PP.NoMainJoints):
            for AngleIndex, AngleType in enumerate(PP.ActiveAngles):
                N = len(self.omega[limb, AngleIndex, :])
                VectorFFT = fft(self.omega[limb, AngleIndex, :])
                VectorFFTMagnitude = np.abs(VectorFFT[0:N//2])
                BinCounts = (N//2)//NoFFTBins
                SpecEnergy = []
                for mode in range (0, NoFFTBins):
                    SpecEnergy.append(VectorFFTMagnitude[mode*BinCounts:(mode+1)*BinCounts].sum())
                
                StartIndex = (limb*PP.NoAngles + AngleIndex)*NoFFTBins
                EndIndex = StartIndex + NoFFTBins
                self.SpecFeature[0,StartIndex:EndIndex] = np.array(SpecEnergy)
                self.SpecFeature[0,StartIndex:EndIndex] = self.SpecFeature[0,StartIndex:EndIndex]/fabs(np.sum(self.SpecFeature[0,StartIndex:EndIndex]))
                #SpecEnergyArray[limb,AngleIndex] = SpecEnergy
        
#        self.SpecFeature = np.zeros((1, PP.NoMainJoints*2*NoFFTBins))
#        for limb in range (0, PP.NoMainJoints):
#            for AngleIndex in range (0,2):
#                StartIndex = (limb*2 + AngleIndex)*NoFFTBins
#                EndIndex = StartIndex + NoFFTBins
#                self.SpecFeature[0,StartIndex:EndIndex] = np.array(SpecEnergyArray[limb,AngleIndex])
#                if AngleIndex == 0:
#                    self.SpecFeature[0,StartIndex:EndIndex] = self.PhiAngVelHistNormalized[limb, :]
#                elif AngleIndex == 1:
#                    self.AngularVelocity[0,StartIndex:EndIndex] = self.ThetaAngVelHistNormalized[limb, :]
        
    def GeneralFeatures(self, HInfo, HumanIndex, PP):
        import numpy as np
        import pandas as pd
        if PP.ActiveDataSet == 'Cheng':
            GenFeatures = HInfo.drop('Flevel', axis = 1, inplace = False)
            GeneralCharacteristics = HInfo[['SEX','AGE','BH','BW','BMI','TPA']]
            #GeneralCharacteristics = HInfo[['SEX','AGE','BH','BW','BMI','CC','HG','TPA']]
            DecissiveCharacteristics = HInfo[['WL','EH','LWS','LST','LPA']]
            ExperciseBasedCharacteristics = HInfo[['OLS/P-1','OLS/P-2','OLS/P-3',
                                                   'FR/P-1','FR/P-2','FR/P-3',
                                                   'STS/P',
                                                   'TUG/P-1','TUG/P-2','TUG/P-3',
                                                   'AC/P',
                                                   'GS/P-1','GS/P-2','GS/P-3',
                                                   'TMST/P']]
            #self.GenFeatures = np.array(GenFeatures.iloc[HumanIndex])
            self.GeneralChar = np.array(GeneralCharacteristics.iloc[HumanIndex])
            self.DecissiveChar = np.array(DecissiveCharacteristics.iloc[HumanIndex])
            self.ExperciseBasedChar = np.array(ExperciseBasedCharacteristics.iloc[HumanIndex])
        elif PP.ActiveDataSet == 'Andersson':
            self.GeneralChar = np.array([])
            self.DecissiveChar = np.array([])
            self.ExperciseBasedChar = np.array([])
        
    def NoFeatures(self):
        self.NFeatures = {'KinectFeatures': self.FeaturesConcatenated.size,
                          'GeneralChar': self.GeneralChar.size,
                          'DecissiveChar': self.DecissiveChar.size,
                          'ExperciseBasedChar': self.ExperciseBasedChar.size,
                          'AngularVelocity': self.AngularVelocity.size, 
                          'SpecFeature': self.SpecFeature.size,
                          'JointsDistance':self.DistHistConcatenated.size,
                          'JointsLinVel':self.JointsLinVel.size,
                          'DistResampledConcat':self.DistResampledConcat.size,
                          'AlphaResampledConcat':self.AlphaResampledConcat.size,
                          'JointDissimConcat':self.JointDissimConcat.size,
                          'JointTriangleAreaConcat':self.JointTriangleAreaConcat.size}
        
    def ActivatingFeatures(self, PP):
        import numpy as np
        FeaturesCollection = {'KinectFeatures':self.FeaturesConcatenated,
                          'GeneralChar':self.GeneralChar,
                          'DecissiveChar':self.DecissiveChar,
                          'ExperciseBasedChar':self.ExperciseBasedChar, 
                          'AngularVelocity': self.AngularVelocity, 
                          'SpecFeature': self.SpecFeature,
                          'JointsDistance': self.DistHistConcatenated,
                          'JointsLinVel':self.JointsLinVel,
                          'DistResampledConcat':self.DistResampledConcat,
                          'AlphaResampledConcat':self.AlphaResampledConcat,
                          'JointDissimConcat':self.JointDissimConcat,
                          'JointTriangleAreaConcat':self.JointTriangleAreaConcat}
        
        #ActiveFeatures = ['KinectFeatures', 'JointsDistance']
        self.ActiveFeatures = ['DistResampledConcat', 'JointDissimConcat', 'JointTriangleAreaConcat', 'AlphaResampledConcat']
        #  
#                    
#                          
        
        
#                          'AlphaResampledConcat',
        
        #Features = np.array([])
        for FeatureType in self.ActiveFeatures:
            #Features = np.append(Features, FeaturesCollection[FeatureType])
            PP.FeaturesVector = np.append(PP.FeaturesVector, FeaturesCollection[FeatureType])
#        N1 = self.NFeatures['KinectFeatures']
#        N2 = self.NFeatures['GeneralChar']
#        N3 = self.NFeatures['DecissiveChar']
#        N4 = self.NFeatures['ExperciseBasedChar']
#        self.NoTotalFeatures = N1 + N2 + N3 + N4
#        Features = np.zeros((1, self.NoTotalFeatures))
#        Features[0, 0:N1] = self.FeaturesConcatenated
#        Features[0, N1:N1+N2] = self.GeneralChar
#        Features[0, N1+N2:N1+N2+N3] = self.DecissiveChar
#        Features[0, N1+N2+N3:N1+N2+N3+N4] = self.ExperciseBasedChar
        
        
#        Features = np.array([])
#        Features = np.append(Features, self.FeaturesConcatenated)
#        Features = np.append(Features, self.GeneralChar)
#        Features = np.append(Features, self.DecissiveChar)
#        Features = np.append(Features, self.ExperciseBasedChar)
        #self.NoTotalFeatures = Features.size
        #self.FeaturesVector = Features