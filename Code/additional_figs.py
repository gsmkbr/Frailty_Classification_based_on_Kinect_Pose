class AdditionalFigures:
    def __init__(self, RootDir):
        import pandas as pd
        SubjectNo = 642
        #Defining exercise type for plotting:
        # 'ArmCurled' or '2MinStep' or 'Gait' or '30SecSitStand'
        self.ExerciseType = 'Gait'
        
        if self.ExerciseType == 'ArmCurled':
            FileName = 'ArmCurlTestMultiple_0703-113432'
        elif self.ExerciseType == '2MinStep':
            FileName = 'TwoMinuteStep_0703-113827'
        elif self.ExerciseType == 'Gait':
            FileName = 'GaitAnalysis_0703-113601'
        elif self.ExerciseType == '30SecSitStand':
            FileName = 'ThirtySecondsChairStand_0703-112448'
        
        LocalAddress = '\\KN'+str(SubjectNo).zfill(3)+'\\'+FileName+'.csv'
        self.DF = pd.read_csv(RootDir+LocalAddress)
        self.DF.drop('Unnamed: 76', axis = 1, inplace = True)

    def FilteringEffect(self):
        import pandas as pd
        import numpy as np
        if self.ExerciseType == 'ArmCurled':
            JointCoord1 = 'WristRightY'
        elif self.ExerciseType == '2MinStep':
            JointCoord1 = 'AnkleLeftY'
            JointCoord2 = 'AnkleRightY'
        elif self.ExerciseType == 'Gait':
            JointCoord1 = 'AnkleLeftZ'
            JointCoord2 = 'AnkleRightZ'
        elif self.ExerciseType == '30SecSitStand':
            JointCoord1 = 'SpineBaseY'
        
        WinSize = 100
        GaussianStdDev = 3
        Kernel = 'gaussian'
        self.DF_Filtered = self.DF.rolling(WinSize, center=True, min_periods = 1, axis = 0, win_type=Kernel).mean(std = GaussianStdDev)
        self.DF_Filtered_NP = np.array(self.DF_Filtered)
        
        import numpy as np
        PlotRange = np.arange(0,self.DF.shape[0])
        import matplotlib.pyplot as plt
        plt.figure()
        
        if self.ExerciseType == 'ArmCurled':
            plt.plot(PlotRange, np.array(self.DF[JointCoord1]), linewidth=3, color='b', alpha=0.6, label='Original Data')
            plt.plot(PlotRange, np.array(self.DF_Filtered[JointCoord1]), linewidth=3, color='g', alpha=0.75, label='Filtered Data')
        elif self.ExerciseType == '2MinStep':
            plt.plot(PlotRange, np.array(self.DF[JointCoord1]), linewidth=3, color='b', alpha=0.6, label='Original Data (AL)')
            plt.plot(PlotRange, np.array(self.DF_Filtered[JointCoord1]), linewidth=3, color='g', alpha=0.75, label='Filtered Data (AL)')
            plt.plot(PlotRange, np.array(self.DF[JointCoord2]), linewidth=3, color='r', alpha=0.6, label='Original Data (AR)')
            plt.plot(PlotRange, np.array(self.DF_Filtered[JointCoord2]), linewidth=3, color='m', alpha=0.7, label='Filtered Data (AR)')
        elif self.ExerciseType == 'Gait':
            plt.plot(PlotRange, np.array(self.DF[JointCoord1]) - np.array(self.DF[JointCoord2]), linewidth=3, color='b', alpha=0.6, label='Original Data (AL)')
            plt.plot(PlotRange, np.array(self.DF_Filtered[JointCoord1]) - np.array(self.DF_Filtered[JointCoord2]), linewidth=3, color='g', alpha=0.75, label='Filtered Data (AL)')
        elif self.ExerciseType == '30SecSitStand':
            plt.plot(PlotRange, np.array(self.DF[JointCoord1]), linewidth=3, color='b', alpha=0.6, label='Original Data')
            plt.plot(PlotRange, np.array(self.DF_Filtered[JointCoord1]), linewidth=3, color='g', alpha=0.75, label='Filtered Data')
        
        plt.xlabel('Time Frame')
        plt.ylabel('Vertical Coordinate (m)')
        plt.legend()
        
    def PostureTrace(self):
        import matplotlib.pyplot as plt
        from PostDraw import PosturalDrawing2
        XYStatus = True
        ZYStatus = True
        XZStatus = False
        Fig, ax = plt.subplots(1,2, figsize=(8,6))
        import numpy as np
        AnimationMargin = 0.05
        XRange = [np.min(self.DF_Filtered_NP[:,range(1,75,3)])-AnimationMargin, np.max(self.DF_Filtered_NP[:,range(1,75,3)])+AnimationMargin]
        YRange = [np.min(self.DF_Filtered_NP[:,range(2,75,3)])-AnimationMargin, np.max(self.DF_Filtered_NP[:,range(2,75,3)])+AnimationMargin]
        ZRange = [np.min(self.DF_Filtered_NP[:,range(3,75,3)])-AnimationMargin, np.max(self.DF_Filtered_NP[:,range(3,75,3)])+AnimationMargin]
        
        COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
        FrameStart = 100
        FrameEnd = 151
        TimeFrameSkip = 10
        TransActivity = False
        Transparency = 1.0
        Counter=0
        
        for t in range (FrameStart, FrameEnd, TimeFrameSkip):
            if TransActivity == True:
                Transparency = Transparency - Counter*0.03
            
            FigAxesCounter = 0
            if XYStatus == True:
                C1 = 'X'
                C2 = 'Y'
                Labels = ['Medial-Lateral\nMovement (m)', 'Cranial-Caudal Movement (m)']
                R1 = XRange
                R2 = YRange
                PosturalDrawing2(ax[FigAxesCounter], C1, C2, R1, R2, Labels, self.DF_Filtered, t, COLORS[Counter], Transparency)
                FigAxesCounter += 1
            
            if ZYStatus == True:
                C1 = 'Z'
                C2 = 'Y'
                Labels = ['Anterior-Posterior\nMovement (m)', 'Cranial-Caudal Movement (m)']
                R1 = ZRange
                R2 = YRange
                PosturalDrawing2(ax[FigAxesCounter], C1, C2, R1, R2, Labels, self.DF_Filtered, t, COLORS[Counter], Transparency)
                FigAxesCounter += 1
            
            if XZStatus == True:
                C1 = 'X'
                C2 = 'Z'
                Labels = ['Medial-Lateral\nMovement (m)', 'Anterior-Posterior Movement (m)']
                R1 = XRange
                R2 = ZRange
                PosturalDrawing2(ax[FigAxesCounter], C1, C2, R1, R2, Labels, self.DF_Filtered, t, COLORS[Counter], Transparency)
                FigAxesCounter +=1
            
            Counter+=1
            
    def RestJointsPositionAndLabels(self):
        import matplotlib.pyplot as plt
        from PostDraw import PosturalDrawing3
        Fig, ax = plt.subplots(1,1, figsize=(4,6))
        import numpy as np
        AnimationMargin = 0.1
        XRange = [np.min(self.DF_Filtered_NP[:,range(1,75,3)])-AnimationMargin, np.max(self.DF_Filtered_NP[:,range(1,75,3)])+AnimationMargin]
        YRange = [np.min(self.DF_Filtered_NP[:,range(2,75,3)])-AnimationMargin, np.max(self.DF_Filtered_NP[:,range(2,75,3)])+AnimationMargin]
        ZRange = [np.min(self.DF_Filtered_NP[:,range(3,75,3)])-AnimationMargin, np.max(self.DF_Filtered_NP[:,range(3,75,3)])+AnimationMargin]
        
        Frame = 55
        Transparency = 0.8
        COLOR = 'b'
        SymbolsProperties = 'go'
        MarkerSize = 8
        C1 = 'X'
        C2 = 'Y'
        Labels = ['X (m)', 'Y (m)']
        R1 = XRange
        R2 = YRange
        PosturalDrawing3(ax, C1, C2, R1, R2, Labels, self.DF_Filtered, Frame, COLOR, Transparency, SymbolsProperties, MarkerSize)
        
#        Counter=0
#        for t in range (FrameStart, FrameEnd, TimeFrameSkip):
#            if TransActivity == True:
#                Transparency = Transparency - Counter*0.03
#            C1 = 'X'
#            C2 = 'Y'
#            Labels = ['X (m)', 'Y (m)']
#            R1 = XRange
#            R2 = YRange
#            PosturalDrawing2(ax[0], C1, C2, R1, R2, Labels, self.DF_Filtered, t, COLORS[Counter], Transparency)
#            