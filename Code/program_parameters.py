class ProgramParameters:
    def __init__(self):
        self.NoHumans = 787
        self.StartHumanCode = 1
        # Selecting the sampling status:
        #                       1. 'UninormlyDistributedTargets'
        #                       2. 'RandomlyDistributedTargets'
        #                       3. 'CustomizedRange'
        self.SamplingStatus = 'CustomizedRange'
        self.NoRequiredSamples = 50
        self.HumanRange = (0, 20)
        self.NoSamplesForAnalysis = 20
        self.ActiveSubjectsPercentage = 100
        #Selecting the target
        # 1. 'SEX' for Andersson dataset
        # 2. 'Flevel' for Frailty level dataset
        self.ClassificationTarget = 'SEX'
        
        self.NSelectedFeatures = 200
        self.TestPercentage = 20
        self.NCVFolds = 5
        
        # Selecting the active dataset for training
        #   1. Cheng
        #   2. Andersson
        self.ActiveDataSet = 'Andersson'
        if self.ActiveDataSet == 'Cheng':
            self.DataRootDirectory = '..\\..\\..\\Kinect data\\787 Input data'
        elif self.ActiveDataSet == 'Andersson':
            self.DataRootDirectory = '..\\Data'
        self.AngVelTimeStep = 5
        self.LinVelTimeStep = 1
        #Specifying joint similarity status:
        # 1. DisSimilarity
        # 2. Similarity
        self.JointSimilarityStatus = 'DisSimilarity'
        
    def GeneralControlParams(self):
        self.ReadingFromFiles = True
        
        self.NoExecutionsWithoutFileReading = 5
        
        #Activate/Deactivate Cross validation Search
        self.CrossValidationSearch = True
        
        #Specifying the data split approach:
        #1. Automatic using train_test_split of sklearn
        #2. Manual
        self.DataSplit = 'Automatic'
        
        
        #Activate/Deactivate random forest based feature selection
        self.FeatureSelection = True
        
        # Specifying the portion of timeseries used in the analysis
        #       'AllData' for using the whole data
        #       'SliceOfData' for using a defined slice of timeseries
        #       'IsolatedCycles' for isolating various motion cycles
        self.DataPortionUsage = 'IsolatedCycles'
        # Activaing the isolation function with two statuses
        #   True or False
        #self.ActivatingDataIsolation = False
        
        # Activating principal component analysis
        # True or False
        self.PCAActivation = False
        
        # Activating Feature Scaling
        # True or False
        self.ScalingStatus = True
        
        # Activating/Deactivating DTW Approach
        self.DTWActivity = False
        
        #Activating/DeActivating the Excel Write/Read
        # 1. 'Active'
        # 2. 'DeActive'
        self.ExcelReadWrite = 'DeActive'
        
        self.AnimationStatus = 'OFF'  #ON or OFF
        
        # Activating some aditional figures
        self.AdditionalFigures = False
        
        # Activating the figures for cycles isolation
        self.PlotCycleIsolation = False
        
        if self.ActiveDataSet == 'Cheng':
            self.InvalidIndices = {'ArmCurlTestMultiple':[10, 115, 133, 142, 362, 409, 587, 629, 634, 665, 675, 677, 713, 735, 763]
                                , 'GaitAnalysis':[1, 11, 12, 20, 26, 30, 32, 33, 34, 41, 42, 48, 61, 69, 71, 73, 74, 76, 85, 86, 89, 95
                                                  , 100, 106, 108, 109, 112, 114, 116, 122, 123, 130, 131, 134, 135, 138, 141, 144, 148, 152, 155, 159, 163, 167, 169, 171, 176, 179, 182, 183, 184, 187, 188, 189, 191, 192, 193, 194, 197
                                                  , 203, 205, 206, 209, 216, 220, 221, 222, 228, 229, 230, 233, 234, 239, 242, 247, 251, 255, 258, 259, 260, 261, 262, 266, 270, 277, 280, 281, 289, 290, 296, 297, 303, 304, 310, 311, 315
                                                  , 316, 317, 320, 321, 323, 325, 326, 327, 328, 330, 334, 337, 341, 342, 343, 344, 346, 352, 355, 356, 360, 366, 368, 370, 378, 381, 382, 387, 392, 395
                                                  , 401, 403, 404, 406, 407, 408, 409, 411, 416, 420, 424, 426, 428, 430, 431, 434, 458, 468, 473, 487
                                                  , 552, 554, 557, 559, 560, 572, 573, 577, 578, 582, 586, 592, 597, 598, 599
                                                  , 602, 608, 614, 617, 623, 628, 629, 635, 636, 645, 650, 651, 659, 670, 671, 678, 685, 686, 687, 689, 690, 691, 696, 697
                                                  , 700, 702, 703, 708, 718, 720, 724, 726, 731, 738, 741, 742, 743, 746, 750, 754, 755, 759, 772, 778, 781]
                                , 'ThirtySecondsChairStand':[331, 363]
                                , 'TwoMinuteStep': [39, 125, 273, 297, 300, 319, 321, 325, 331, 332, 343, 348, 353, 356, 360, 372, 393, 395, 399
                                                    , 441, 484, 490, 547, 559, 573, 586, 596
                                                    , 601, 603, 608, 610, 614, 628, 629, 643, 644, 649, 650, 651, 652, 669, 674, 679, 685, 686, 687, 690, 697
                                                    , 705, 708, 709, 712, 723, 741, 763]}
        elif self.ActiveDataSet == 'Andersson':
            self.InvalidIndices = {'4':[5, 9, 57, 61, 71, 76, 92, 100, 120, 125, 127, 139, 141, 143, 150, 151, 153]}
        
    
    def LimbParameters(self):
        self.MainJoints = [['ShoulderLeft', 'ElbowLeft'], 
                           ['ElbowLeft', 'WristLeft'],
                           ['ShoulderRight', 'ElbowRight'], 
                           ['ElbowRight', 'WristRight'],
                           ['WristRight','HandRight'],
                           ['SpineBase', 'WristRight'],
                           ['SpineBase', 'ElbowRight']
                           ]
        
#        ['HipLeft', 'KneeLeft'], 
#                           ['KneeLeft', 'AnkleLeft'], 
#                           ['HipRight', 'KneeRight'], 
#                           ['KneeRight', 'AnkleRight']
                      
                     
        
        self.NoMainJoints = len(self.MainJoints)
        
        # A list including 'Phi' and/or 'Theta'
        self.ActiveAngles = ['Phi', 'Theta']
        self.NoAngles = len(self.ActiveAngles)
    
    def HumanFeatures(self):
        if self.ActiveDataSet == 'Cheng':
            self.OverallHumanFeatures= ['SEX',
                                        'AGE',
                                        'BH', 
                                        'BW',
                                        'BMI',
                                        'CC',
                                        'HG',
                                        'OLS/P-1',
                                        'OLS/P-2',
                                        'OLS/P-3',
                                        'FR/P-1',
                                        'FR/P-2',
                                        'FR/P-3',
                                        'STS/P',
                                        'TUG/P-1',
                                        'TUG/P-2',
                                        'TUG/P-3',
                                        'AC/P',
                                        'GS/P-1',
                                        'GS/P-2',
                                        'GS/P-3',
                                        'TMST/P',
                                        'TPA',
                                        'WL',
                                        'EH',
                                        'LWS',
                                        'LST',
                                        'LPA',
                                        'Flevel']
        elif self.ActiveDataSet == 'Andersson':
            self.OverallHumanFeatures= ['Height',
                                        'Weight',
                                        'Age', 
                                        'BMI',
                                        'SEX']
        
    def SmoothingParams(self):
        self.SmoothWindowSize = 100
        self.WinType = 'gaussian'
        self.GaussianStDev = 3
        self.SmoothingStatus = 'True'
    
    def HistogramParameters(self):
        self.NBins = 10
        self.HistRangeForPhi = (0, 360)
        self.HistRangeForTheta = (0, 180)
        
    def FeatureParameters(self):
        self.NFeatures = self.NoMainJoints*self.NoAngles*self.NBins
    
    def ControlParameters(self):
        if self.ActiveDataSet == 'Cheng':
            FilesInitials = {'1':['ArmCurlTestMultiple',1], 
                             '2':['EightFootUpAndGo',3], 
                             '3':['FunctionalReaching',3],
                             '4':['GaitAnalysis',3],
                             '5':['OneLegStance',3],
                             '6':['ThirtySecondsChairStand',1],
                             '7':['TwoMinuteStep',1]}
        elif self.ActiveDataSet == 'Andersson':
            FilesInitials = {'1':['1',1], 
                             '2':['2',1], 
                             '3':['3',1],
                             '4':['4',1],
                             '5':['5',1],
                             '6':['6',1],
                             '7':['7',1]}
        
        self.ActiveExerciseIndices = [4]
        self.ActiveExerciseIndices = [str(element) for element in self.ActiveExerciseIndices]
        
        self.ActiveFiles = []
        NoActiveFiles = 0
        for i in self.ActiveExerciseIndices:
            self.ActiveFiles.append(FilesInitials[i][0])
            NoActiveFiles+=FilesInitials[i][1]
        self.NSamples = self.NoHumans*NoActiveFiles
        
        if self.ActiveDataSet == 'Andersson':
            #This should be a fixed parameter
            self.ActiveFiles = ['4']
        
        FilesIndices = {'1':[1], 
                        '2':[2,3,4], 
                        '3':[5,6,7], 
                        '4':[8,9,10], 
                        '5':[11,12,13], 
                        '6':[14], 
                        '7':[15]}
        self.ActiveFilesIndices = []
        for exercise in self.ActiveExerciseIndices:
            self.ActiveFilesIndices.extend(FilesIndices[str(exercise)])
        self.NoActiveFiles = len(self.ActiveFilesIndices)
        
        if self.ActiveDataSet == 'Cheng':
            self.SlicePercentage = {FilesInitials['1'][0]:(0.4,0.9),
                                    FilesInitials['2'][0]:(0.55,0.95),
                                    FilesInitials['3'][0]:(0.4,0.9),
                                    FilesInitials['4'][0]:(0.85,0.98),
                                    FilesInitials['5'][0]:(0.4,0.9),
                                    FilesInitials['6'][0]:(0.5,0.9),
                                    FilesInitials['7'][0]:(0.4,0.9)}
        elif self.ActiveDataSet == 'Andersson':
            self.SlicePercentage = {FilesInitials['1'][0]:(0.25,0.9),
                                    FilesInitials['2'][0]:(0.25,0.9),
                                    FilesInitials['3'][0]:(0.25,0.9),
                                    FilesInitials['4'][0]:(0.25,0.9),
                                    FilesInitials['5'][0]:(0.25,0.9)}
    
    def ListOfInvalidData(self):
        self.ListOfInvalidData = {'1':[],
                                  '3':[],
                                  '4':[(7,1),
                                       (8,0), (8,1), (8,2),
                                       (9,0), (9,1), (9,2),
                                       (10,1), 
                                       (11,0),
                                       (17,1)]}
    
    def DTWParameters(self):
        if len(self.ActiveExerciseIndices) == 1:
            self.ActiveFileIndex = str(self.ActiveExerciseIndices[0])
    
    def ExtractKeyWords(self, Data):
        self.KeyWords = []
        InitialKeys = Data.Data_DF.columns
        KeyWords = InitialKeys[1:-1:3]
        for i in range (0,KeyWords.size):
            self.KeyWords.extend([KeyWords[i][0:-1]])
    
    def JointsCombinations(self, PP):
        self.KW = {'0':'SpineBase',
              '1':'SpineMid',
              '2':'Neck',
              '3':'Head',
              '4':'ShoulderLeft',
              '5':'ElbowLeft',
              '6':'WristLeft',
              '7':'HandLeft',
              '8':'ShoulderRight',
              '9':'ElbowRight',
              '10':'WristRight',
              '11':'HandRight',
              '12':'HipLeft',
              '13':'KneeLeft',
              '14':'AnkleLeft',
              '15':'FootLeft',
              '16':'HipRight',
              '17':'KneeRight',
              '18':'AnkleRight',
              '19':'FootRight',
              '20':'SpineShoulder',
              '21':'HandTipLeft',
              '22':'ThumbLeft',
              '23':'HandTipRight',
              '24':'ThumbRight'}
        
        if PP.ActiveDataSet == 'Cheng':
            self.JointsExcluded = [21, 22, 23, 24]
        elif PP.ActiveDataSet == 'Andersson':
            self.JointsExcluded = [20, 21, 22, 23, 24]

        # Allowed Combinations are: 'AllJointsCombinations'
        #                           'CustomizedJoints(AllCombinations)'
        #                           'CustomizedJoints(SelectedCombinations)'
        self.JointsCombinationStatus = 'AllJointsCombinations'
        
        if self.JointsCombinationStatus == 'AllJointsCombinations':
            self.JointsKeys = self.KeyWords
            
            self.JointsCombs = {}
            for exercise in range (1,8):
                self.JointsCombs[str(exercise)] = []
                #AllJointsCombinations = []
                for i in range(0,24):
                    for j in range (i+1,25):
                        if (i not in self.JointsExcluded) and (j not in self.JointsExcluded):
                            self.JointsCombs[str(exercise)].append((self.KW[str(i)],self.KW[str(j)]))
                #print(self.JointsCombs[str(exercise)])
        elif self.JointsCombinationStatus == 'CustomizedJoints(AllCombinations)':
            self.JointsKeys = [self.KW['0'],
                               self.KW['8'],
                               self.KW['9'],
                               self.KW['10'],
                               self.KW['4'],
                               self.KW['5'],
                               self.KW['6'],
                               self.KW['13'],
                               self.KW['14'],
                               self.KW['17'],
                               self.KW['18']]
        elif self.JointsCombinationStatus == 'CustomizedJoints(SelectedCombinations)':
#            AllJointsCombinations = []
#            for i in range(0,24):
#                for j in range (i+1,25):
#                    AllJointsCombinations.append((self.KW[str(i)],self.KW[str(j)]))
#            print(AllJointsCombinations)
            
            self.JointsCombs = {'1':[(self.KW['0'],self.KW['10']),
                                    (self.KW['1'],self.KW['10']),
                                    (self.KW['2'],self.KW['10']),
                                    (self.KW['4'],self.KW['10']),
                                    (self.KW['5'],self.KW['10']),
                                    (self.KW['7'],self.KW['10']),
                                    (self.KW['8'],self.KW['10']),
                                    (self.KW['12'],self.KW['10']),
                                    (self.KW['13'],self.KW['10']),
                                    (self.KW['14'],self.KW['10']),
                                    (self.KW['16'],self.KW['10']),
                                    (self.KW['17'],self.KW['10']),
                                    (self.KW['18'],self.KW['10']),
                                    (self.KW['20'],self.KW['10']),
                                    (self.KW['0'],self.KW['9']),
                                    (self.KW['1'],self.KW['9']),
                                    (self.KW['2'],self.KW['9']),
                                    (self.KW['4'],self.KW['9']),
                                    (self.KW['5'],self.KW['9']),
                                    (self.KW['7'],self.KW['9']),
                                    (self.KW['12'],self.KW['9']),
                                    (self.KW['13'],self.KW['9']),
                                    (self.KW['14'],self.KW['9']),
                                    (self.KW['16'],self.KW['9']),
                                    (self.KW['17'],self.KW['9']),
                                    (self.KW['18'],self.KW['9']),
                                    (self.KW['20'],self.KW['9'])],
                                '2':[(self.KW['14'],self.KW['18']),
                                     (self.KW['6'],self.KW['10']),
                                     (self.KW['13'],self.KW['17']),
                                     (self.KW['5'],self.KW['9'])],
                                '3':[(self.KW['8'],self.KW['10']),
                                     (self.KW['1'],self.KW['8']),
                                     (self.KW['9'],self.KW['10'])],
                                '4':[(self.KW['14'],self.KW['18']),
                                     (self.KW['6'],self.KW['10']),
                                     (self.KW['13'],self.KW['17']),
                                     (self.KW['5'],self.KW['9']),
                                     (self.KW['4'],self.KW['11']),
                                     (self.KW['8'],self.KW['7']),
                                     (self.KW['11'],self.KW['19']),
                                     (self.KW['7'],self.KW['15'])],
                                '5':[(self.KW['14'],self.KW['18']),
                                     (self.KW['6'],self.KW['10']),
                                     (self.KW['13'],self.KW['17']),
                                     (self.KW['5'],self.KW['9'])],
                                '6':[(self.KW['5'],self.KW['13']),
                                     (self.KW['9'],self.KW['17']),
                                     (self.KW['13'],self.KW['2']),
                                     (self.KW['17'],self.KW['2'])],
                                '7':[(self.KW['14'],self.KW['18']),
                                     (self.KW['13'],self.KW['19']),
                                     (self.KW['15'],self.KW['19']),
                                     (self.KW['15'],self.KW['17']),
                                     (self.KW['15'],self.KW['17']),
                                     (self.KW['15'],self.KW['19'])]}
                                
                                
                                
                                


#(self.KW['8'],self.KW['10']),
#                                     (self.KW['10'],self.KW['18']),
#                                     (self.KW['9'],self.KW['18']),
#                                     (self.KW['11'],self.KW['17']),
#                                     (self.KW['11'],self.KW['14']),
#                                     (self.KW['0'],self.KW['10']),
#                                     (self.KW['20'],self.KW['10'])
                                
        self.TargetJoints = {'1':[0,4,5,6,8,9,10,12,13,14,15,16,17,18,19],
                             '2':[0,4,5,6,8,9,10,12,13,14,15,16,17,18,19],
                             '4':[0,4,5,6,8,9,10,12,13,14,15,16,17,18,19],
                             '6':[0,4,5,6,8,9,10,12,13,14,15,16,17,18,19],
                             '7':[0,4,5,6,8,9,10,12,13,14,15,16,17,18,19]}
    
    def AnimatingParameters(self):
        FileName = 'EightFootUpAndGo_0620-105511'
        FolderName = 'KN611' 
        self.DataTargetDirectory = self.DataRootDirectory + '\\' + FolderName + '\\'
        self.FileName = FileName + '.csv'
        
        self.AnimationMargin = 0.1
        self.TimeFrameSkip = 20