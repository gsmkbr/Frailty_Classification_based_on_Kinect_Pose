import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from import_of_data import ImportData
from program_parameters import ProgramParameters

PP = ProgramParameters()
PP.GeneralControlParams()
PP.LimbParameters()
PP.SmoothingParams()
PP.HistogramParameters()
PP.FeatureParameters()
PP.ControlParameters()
PP.ListOfInvalidData()
PP.AnimatingParameters()
from human_information import HumanInfo, HumanSelection, HumansWithSpecificExercises
PP.HumanFeatures()
HInfo = HumanInfo(PP)
SelectedHumansIndices = HumanSelection(HInfo, PP)
SelectedHumansIndices = HumansWithSpecificExercises(PP, PP.ActiveFiles, SelectedHumansIndices)
#SelectedHumansIndices = list(np.random.permutation(np.array(SelectedHumansIndices)))

if PP.AnimationStatus == 'ON':
    from pose_animation import PosturalAnimation
    PosturalAnimation(PP)
print('****************')
plt.figure()

if PP.AdditionalFigures == True:
    from additional_figs import AdditionalFigures
    AdditionalFigs = AdditionalFigures(PP.DataRootDirectory)
    AdditionalFigs.FilteringEffect()
    AdditionalFigs.PostureTrace()
    AdditionalFigs.RestJointsPositionAndLabels()

TEMP_FILE = open('TempFile.txt','a')

if PP.ReadingFromFiles == True:
    AllData = []
    counter = 0
    SampleIndex = []
    #for HumanIndex in range (0, PP.NoHumans):
    #for HumanIndex in range (PP.HumanRange[0], PP.HumanRange[1]):
    PP.DistanceHistRange = {}
    for HumanIndex in SelectedHumansIndices:
        ConditionA = not ('ArmCurlTestMultiple' in PP.ActiveFiles and HumanIndex in PP.InvalidIndices['ArmCurlTestMultiple'])
        ConditionD = not (('GaitAnalysis' in PP.ActiveFiles) and (HumanIndex in PP.InvalidIndices['GaitAnalysis']))
        ConditionF = not (('ThirtySecondsChairStand' in PP.ActiveFiles) and (HumanIndex in PP.InvalidIndices['ThirtySecondsChairStand']))
        ConditionG = not (('TwoMinuteStep' in PP.ActiveFiles) and (HumanIndex in PP.InvalidIndices['TwoMinuteStep']))
        ConditionAndersson = not (('4' in PP.ActiveFiles) and (HumanIndex in PP.InvalidIndices['4']))
        #ConditionD = True
        if ConditionA and ConditionD and ConditionF and ConditionG and ConditionAndersson:
            AHumanData = []
            if PP.ActiveDataSet == 'Cheng':
                FolderName = 'KN'+str(HumanIndex+PP.StartHumanCode).zfill(3)
            elif PP.ActiveDataSet == 'Andersson':
                FolderName = 'Person'+str(HumanIndex+PP.StartHumanCode).zfill(3)
            #DataRootDirectory = '..\\..\\..\\Kinect data\\20 Input data'
            DataTargetDirectory = PP.DataRootDirectory+'\\'+FolderName+'\\'
            List_of_Files = listdir(DataTargetDirectory)
            #ListOfExercises = {}
            PP.FeaturesVector = np.array([])
            CNT = {}
            for FileInitial in PP.ActiveFiles:
                CNT[FileInitial] = 0
            for FileName in List_of_Files:
                Condition1 = PP.ActiveDataSet == 'Cheng' and FileName.endswith('.csv')
                Condition2 = PP.ActiveDataSet == 'Andersson' and FileName.endswith('.txt')
                
                if (Condition1 or Condition2):
                    for FileInitial in PP.ActiveFiles:
                        #CNT = 0
                        if FileName.startswith(FileInitial):
                            CNT[FileInitial] += 1
                            if CNT[FileInitial] ==1:
                                print('Human: {}, Exercise: {}'.format(HumanIndex+PP.StartHumanCode, FileInitial))
                                SampleIndex.extend([HumanIndex])
                                Data = ImportData(DataTargetDirectory)
                                Data.DataReading(PP, FileName)
                                Data.MissingDataRemoval()
                                if PP.SmoothingStatus == 'True':
                                    Data.DataSmoothing(PP)
                                
                                if PP.DataPortionUsage == 'SliceOfData':
                                    Data.DataSlicing(PP, FileInitial)
                                elif PP.DataPortionUsage == 'IsolatedCycles':
                                    Data.IsolateCycles(PP, FileInitial, HumanIndex)
                                    if Data.Weight<0.76:
                                        TEMP_FILE.write(str(HumanIndex)+', ')
                                    Data.DataResampling()
                                #if PP.ActivatingDataIsolation == True:
                                #    Data.IsolateCycles(PP, FileInitial)
                                AHumanData.append(Data.Data_DF)
                                Data.OrientationCalc(PP)
                                Data.AngularVelocity(PP)
                                PP.ExtractKeyWords(Data)
                                PP.JointsCombinations(PP)
                                Data.JointCosDissimilarity(PP, FileInitial)
                                Data.JointTriangleArea(PP, FileInitial)
                                Data.LinearVelocity(PP, FileInitial)
                                Data.DistanceCalculations(PP,FileInitial)
                                Data.HistBasedFeatures(PP,counter,FileInitial)
                                Data.FFTBaseFeatures(PP)
                                Data.GeneralFeatures(HInfo, HumanIndex, PP)
                                
                                Data.NoFeatures()
                                Data.ActivatingFeatures(PP)
            Data.NoTotalFeatures = PP.FeaturesVector.size
            if counter == 0:
                Features = np.zeros((PP.NSamples, Data.NoTotalFeatures))
            Features[counter,:] = PP.FeaturesVector
            
            if counter == 0:
                Targets = np.zeros((PP.NSamples,1))
            #Targets[counter,0] = HInfo.iloc[HumanIndex]['Flevel']
            Targets[counter,0] = HInfo.iloc[HumanIndex][PP.ClassificationTarget]
            counter+=1
            print ('Human: {} & File: {}'.format(HumanIndex+1, counter))
                            
    
            AllData.append(AHumanData)
    
    #NoSamples = len(SampleIndex)
    NoSamples = len(SelectedHumansIndices)
    Features = np.delete(Features, range(NoSamples, PP.NSamples), axis=0)
    Targets = np.delete(Targets, range(NoSamples, PP.NSamples), axis=0)
    
    FeaturesDFOrig = pd.DataFrame(Features)
    TargetsDFOrig = pd.DataFrame(Targets)
    Targets.astype('int')
    
    if PP.ExcelReadWrite == 'Active':
        ExcelWriter = pd.ExcelWriter(PP.DataRootDirectory + '\\ExtractedFeatures.xlsx')
        SampleIndexDF = pd.DataFrame(SampleIndex)
        SampleIndexDF.to_excel(ExcelWriter, sheet_name = 'SampleIndex', header=False, index = False)
        FeaturesDFOrig.to_excel(ExcelWriter, sheet_name = 'Features', header=False, index = False)
        TargetsDFOrig.to_excel(ExcelWriter, sheet_name = 'Targets', header=False, index = False)
        ExcelWriter.save()
    #FeaturesDF.to_excel(excel_writer = PP.DataRootDirectory + '\\ExtractedFeatures.xlsx', header=False, index=False, sheet_name=0)
    #TargetsDF.to_excel(excel_writer = PP.DataRootDirectory + '\\ExtractedFeatures.xlsx', header=False, index=False, sheet_name=1)
else:
    #SampleIndexDF = pd.read_excel(PP.DataRootDirectory + '\\ExtractedFeatures.xlsx', header=None, sheet_name='SampleIndex')
    if PP.ExcelReadWrite == 'Active':
        FeaturesDFOrig = pd.read_excel(PP.DataRootDirectory + '\\ExtractedFeatures.xlsx', header=None, sheet_name='Features')
        TargetsDFOrig = pd.read_excel(PP.DataRootDirectory + '\\ExtractedFeatures.xlsx', header=None, sheet_name='Targets')
        NoSamples = FeaturesDFOrig.shape[0]

TEMP_FILE.close()

for TotalIterator in range (1, PP.NoExecutionsWithoutFileReading+1):

    # Slicing specific number of subjects from the overall dataframe 
    from math import floor
    #NoSamples = floor(FeaturesDFOrig.shape[0]*PP.ActiveSubjectsPercentage/100)
    NoSamples = min(FeaturesDFOrig.shape[0], PP.NoSamplesForAnalysis)
    FeaturesDF = FeaturesDFOrig.copy()
    TargetsDF = TargetsDFOrig.copy()
    FeaturesDF = FeaturesDF.iloc[0:NoSamples]
    TargetsDF = TargetsDF.iloc[0:NoSamples]
    
    #PP.ExtractKeyWords(Data)
    #PP.JointsCombinations()
    
    from sklearn.preprocessing import robust_scale, StandardScaler
    Columns = FeaturesDF.columns
    #FeaturesDF = robust_scale(FeaturesDF)
    if PP.ScalingStatus == True:
        Scaler = StandardScaler()
        FeaturesDF = Scaler.fit_transform(FeaturesDF)
        #FeaturesDF.iloc[:,640:1440] = minmax_scale(FeaturesDF.iloc[:,640:1440], feature_range=(0,1))
        FeaturesDF = pd.DataFrame(FeaturesDF, columns=Columns)
    
    PP.DTWParameters()
    if PP.DTWActivity == True:
        from dtw_based_learning import DTWBasedLearning
        DTW = DTWBasedLearning(AllData)
        from fastdtw import fastdtw
        JDFeatureMatrices = []
        for HumanIndex in range (0, len(AllData)):
            if AllData[HumanIndex] != []:
                NoExercises = len(AllData[HumanIndex])
                for ExerciseIter in range (0, NoExercises):
                    
                    if (HumanIndex, ExerciseIter) not in PP.ListOfInvalidData[PP.ActiveFileIndex]:
                        DTW.JDFeatureVectors(HumanIndex, ExerciseIter, PP, PP.ActiveFileIndex)
                        JDFeatureMatrices.append(DTW.FeatureVectors)
        
        NValidSamples = len(JDFeatureMatrices)
        DistMatrix = np.zeros((NValidSamples, NValidSamples+1))    
        for row in range(0, NValidSamples):
            for col in range(row+1, NValidSamples):
                print(row, col)
                Distance = 0
                for JointPair in JDFeatureMatrices[row]:
                    Dist, path = fastdtw(JDFeatureMatrices[row][JointPair], JDFeatureMatrices[col][JointPair])
                    Distance += Dist
                DistMatrix[row,col] = Distance
                DistMatrix[col,row] = DistMatrix[row,col]
            DistMatrix[row,NValidSamples] = TargetsDF.iloc[row,0]
        
        TargetSlice = DistMatrix
        NVotes = 5
        NActiveSamples = TargetSlice.shape[0]
        ACCURACY = np.zeros((NActiveSamples,1))
        for k in range(0, NActiveSamples):
            TruePrediction = 0
            IndexForTest = k
            for i in range(0, NActiveSamples):
                #TargetLabel = TargetSlice[i,1]
                TargetLabel = DistMatrix[i,NActiveSamples]
                Slice = np.delete(TargetSlice, i, axis=0)
                SliceSorted = Slice[np.argsort(Slice[:,IndexForTest])]
                Votes = {'True':0, 'False':0}
                for j in range(0, NVotes):
                    if SliceSorted[j,-1] == TargetLabel:
                        Votes['True'] += 1
                    else:
                        Votes['False'] += 1
                if Votes['True'] > Votes['False']:
                    TruePrediction += 1
            
            PredAccuracy = TruePrediction/NActiveSamples
            ACCURACY[k,0] = PredAccuracy
            print('Iteration {}: Accuracy: {}'.format(k+1, PredAccuracy))
    
    
    
    #from sklearn.neighbors import KNeighborsClassifier
    #KNN = KNeighborsClassifier()
    
    
    
    from feature_preprocessing import Preprocess
    Feature = Preprocess(FeaturesDF)
    Feature.NullFeatureDrop()
    FeaturesDF = Feature.FeaturesNew.copy()
    
    from sklearn import decomposition
    
    if PP.PCAActivation == True:
        NFeatures = FeaturesDF.shape[1]
        NComponents = min(NoSamples, NFeatures//3)
        PCA = decomposition.PCA(n_components=NComponents)
        PCA.fit(FeaturesDF)
        FeaturesDF = PCA.transform(FeaturesDF)
        FeaturesDF = pd.DataFrame(FeaturesDF)
    
    #FeaturesDF = pd.DataFrame(np.random.rand(60,76))
    
    #RandPerm = np.random.permutation(PP.NoHumans)
    #NoHumansForTest = 1
    #NoIterations = PP.NoHumans // NoHumansForTest
    #KNNTestScores = []
    #SVCTestScores = []
    #MLPTestScores = []
    #for iteration in range(0, NoIterations):
    #    TestSampleIndices = []
    #    for i in range (0, NoHumansForTest):
    #        StartIndex = RandPerm[iteration*NoHumansForTest+i]*PP.NoActiveFiles
    #        EndIndex = StartIndex + PP.NoActiveFiles
    #        FileRange = list(np.arange(StartIndex, EndIndex))
    #        TestSampleIndices.extend(FileRange)
    #    
    #    TestInputs = FeaturesDF.iloc[TestSampleIndices]
    #    TestTargets = TargetsDF.iloc[TestSampleIndices]
    #    TrainInputs = FeaturesDF.drop(TestSampleIndices, axis=0, inplace=False)
    #    TrainTargets = TargetsDF.drop(TestSampleIndices, axis=0, inplace=False)
    
    if PP.DataSplit == 'Manual':
        TestPercentage = 5
        NTestSamples = floor(NoSamples*TestPercentage/100)
        NTrainSamples = NoSamples - NTestSamples
        HumanIndices = list(FeaturesDF.index)
        HumanIndicesPermutation = HumanIndices.copy()
        from random import shuffle
        shuffle(HumanIndicesPermutation)
        PermutedIndicesForTraining = HumanIndicesPermutation[0:NTrainSamples]
        PermutedIndicesForTesting = HumanIndicesPermutation[NTrainSamples:]
        
        # Splitting dataset into Train/Test sets
        #TrainingFeaturesDF = FeaturesDF.copy()
        #TrainingTargetsDF = TargetsDF.copy()
        TrainingFeaturesDF = pd.DataFrame(columns=FeaturesDF.columns)
        TrainingTargetsDF = pd.DataFrame(columns=range(0,1))
        TestFeaturesDF = pd.DataFrame(columns=FeaturesDF.columns)
        TestTargetsDF = pd.DataFrame(columns=range(0,1))
        
        for TrainHumanIndex in PermutedIndicesForTraining:
            TrainingFeaturesDF = TrainingFeaturesDF.append(FeaturesDF.iloc[TrainHumanIndex])
            TrainingTargetsDF = TrainingTargetsDF.append(TargetsDF.iloc[TrainHumanIndex])
        
        for TestHumanIndex in PermutedIndicesForTesting:
            #Location = np.where(SampleIndexArray==HumanIndex)
            #TrainingFeaturesDF.drop(TestHumanIndex, axis=0, inplace=True)
            #TrainingTargetsDF.drop(TestHumanIndex, axis=0, inplace=True)
        #    TrainingFeaturesDF = TrainingFeaturesDF.append(FeaturesDF.iloc[TrainHumanIndex])
        #    TrainingTargetsDF = TestTargetsDF.append(TargetsDF.iloc[TrainHumanIndex])
            TestFeaturesDF = TestFeaturesDF.append(FeaturesDF.iloc[TestHumanIndex])
            TestTargetsDF = TestTargetsDF.append(TargetsDF.iloc[TestHumanIndex])
        
        #UniqueAvailableIndices = list(set(SampleIndex))
        #NUniqueSamples = len(UniqueAvailableIndices)
        #TestPercentage = 50
        #NTestSamples = floor(NUniqueSamples*TestPercentage/100)
        #NTrainSamples = NUniqueSamples - NTestSamples
        #from random import shuffle
        #UniqueIndicesPermutation = UniqueAvailableIndices.copy()
        #shuffle(UniqueIndicesPermutation)
        #PermutedIndicesForTraining = UniqueIndicesPermutation[0:NTrainSamples]
        #PermutedIndicesForTesting = UniqueIndicesPermutation[NTrainSamples:]
        #SampleIndexArray = np.array(SampleIndex)
        
        # Splitting dataset into Train/Test sets
        #TrainingFeaturesDF = FeaturesDF.copy()
        #TrainingTargetsDF = TargetsDF.copy()
        #TestFeaturesDF = pd.DataFrame(columns=FeaturesDF.columns)
        #TestTargetsDF = pd.DataFrame(columns=range(0,1))
        #for HumanIndex in PermutedIndicesForTesting:
        #    #Location = np.where(SampleIndexArray==HumanIndex)
        #    TrainingFeaturesDF.drop(HumanIndex, axis=0, inplace=True)
        #    TrainingTargetsDF.drop(HumanIndex, axis=0, inplace=True)
        #    TestFeaturesDF = TestFeaturesDF.append(FeaturesDF.iloc[Location[0]])
        #    TestTargetsDF = TestTargetsDF.append(TargetsDF.iloc[Location[0]])
        
        #from sklearn.model_selection import train_test_split
        #TrainInputs, TestInputs, TrainTargets, TestTargets = train_test_split(FeaturesDF, TargetsDF, test_size = 0.1, random_state = 20, stratify = TargetsDF)
        
        #RandPerm = np.random.permutation(NoSamples)
        ValidationPercentage = 10
        NoHumansForValidation = floor(NTrainSamples*ValidationPercentage/100)
        NoIterations = NTrainSamples // NoHumansForValidation
        KNNTestScores = []
        KNNConfusionMatrix = []
        SVCTestScores = []
        MLPTestScores = []
        bag_clfTrainingScores = []
        bag_clfValidationScores = []
        #CLFValidationScores = {'KNNCV':[],'SVCCV':[],'MLP':[],'EnsembleCLF':[]}
        CLFTrainingScores = {'KNN':[],'SVC':[],'MLP':[],'EnsembleCLF':[]}
        CLFValidationScores = {'KNN':[],'SVC':[],'MLP':[],'EnsembleCLF':[]}
        
        NFeatures = FeaturesDF.shape[1]
        
        from sklearn.metrics import confusion_matrix
        for iteration in range(0, NoIterations):
            ValidationInputs = pd.DataFrame(columns=FeaturesDF.columns)
            TrainInputs = pd.DataFrame(columns=FeaturesDF.columns)
            #TrainInputs = TrainingFeaturesDF.copy()
            #TrainInputs = pd.DataFrame(columns=FeaturesDF.columns)
            ValidationTargets = pd.DataFrame(columns=range(0,1))
            TrainTargets = pd.DataFrame(columns=range(0,1))
            #TrainTargets = TrainingTargetsDF.copy()
            #TrainTargets = pd.DataFrame(columns=range(0,1))
            #Range = range(iteration*NoHumansForTest, (iteration+1)*NoHumansForTest)
            StartIndex = iteration*NoHumansForValidation
            EndIndex = (iteration+1)*NoHumansForValidation
            print(StartIndex, EndIndex)
            HumansValidationIndex = PermutedIndicesForTraining[StartIndex:EndIndex]
            
            ValidationInputs = TrainingFeaturesDF.iloc[StartIndex:EndIndex]
            ValidationTargets = TrainingTargetsDF.iloc[StartIndex:EndIndex]
            TrainInputs = TrainingFeaturesDF.drop(HumansValidationIndex, axis=0, inplace=False)
            TrainTargets = TrainingTargetsDF.drop(HumansValidationIndex, axis=0, inplace=False)
            #print(HumansTestIndex)
        #    for HumanIndex in HumansValidationIndex:
        #        Location = np.where(SampleIndexArray==HumanIndex)
        #        
        #        ValidationInputs = ValidationInputs.append(FeaturesDF.iloc[Location[0]])
        #        ValidationTargets = ValidationTargets.append(TargetsDF.iloc[Location[0]])
        #        print(ValidationInputs.shape)
        #        #FeaturesDFDuplicate = FeaturesDF.copy()
        #        #TargetsDFDuplicate = TargetsDF.copy()
        #        TrainInputs.drop(Location[0], axis=0, inplace=True)
        #        TrainTargets.drop(Location[0], axis=0, inplace=True)
                #TrainInputs = FeaturesDF.drop(Location[0], axis=0, inplace=False)
                #TrainTargets = TargetsDF.drop(Location[0], axis=0, inplace=False)
                
        #        TrainDF = TrainInputs.copy()
        #        TrainDF['Target'] = TrainTargets.iloc[:,0]
        #        TrainDF = TrainDF.iloc[np.random.permutation(len(TrainDF))]
        #        TrainInputs = TrainDF.drop('Target', axis=1)
        #        TrainTargets = TrainDF['Target']
            if iteration == 0:
                from sklearn.ensemble import RandomForestClassifier
                forest_clf = RandomForestClassifier(n_estimators=5000, max_leaf_nodes=16, n_jobs=-1 )
                forest_clf.fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
                for feature in zip(range(0, TrainInputs.shape[1]), forest_clf.feature_importances_):
                    print(feature)
                from sklearn.feature_selection import SelectFromModel
                MaxNoFeatures = 250
                sfm = SelectFromModel(forest_clf, max_features = MaxNoFeatures, threshold=0.00005)
                sfm.fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
            TrainInputs = sfm.transform(TrainInputs)
            ValidationInputs = sfm.transform(ValidationInputs)
            
            if iteration == 0:
                from keras.models import Sequential
                from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout, AlphaDropout
                from keras.regularizers import l1, l2, l1_l2
                from keras.callbacks import ModelCheckpoint, EarlyStopping
                from keras.constraints import MaxNorm
                DLClassifier = Sequential()
                
                #DLClassifier.add(Flatten(input_shape=[1,TrainingFeaturesDF.shape[1]]))
        #        DLClassifier.add(BatchNormalization())
                DLClassifier.add(AlphaDropout(rate=0.5))
                DLClassifier.add(Dense(32,
                                       kernel_initializer = 'he_normal',
                                       kernel_regularizer=l1_l2(0.01,0.01),
                                       kernel_constraint=MaxNorm(0.1),
                                       use_bias = False,
                                       input_shape = TrainInputs[0].shape))
                DLClassifier.add(BatchNormalization())
                DLClassifier.add(Activation('selu'))
                DLClassifier.add(AlphaDropout(rate=0.5))
                DLClassifier.add(Dense(64,
                                       kernel_initializer = 'he_normal',
                                       kernel_regularizer=l1_l2(0.01,0.01),
                                       kernel_constraint=MaxNorm(0.1),
                                       use_bias = False))
                DLClassifier.add(BatchNormalization())
                DLClassifier.add(Activation('selu'))
                DLClassifier.add(AlphaDropout(rate=0.5))
                DLClassifier.add(Dense(64,
                                       kernel_initializer = 'he_normal',
                                       kernel_regularizer=l1_l2(0.01,0.01),
                                       kernel_constraint=MaxNorm(0.1),
                                       use_bias = False))
                DLClassifier.add(BatchNormalization())
                DLClassifier.add(Activation('selu'))
                DLClassifier.add(AlphaDropout(rate=0.5))
                DLClassifier.add(Dense(32,
                                       kernel_initializer = 'he_normal',
                                       kernel_regularizer=l1_l2(0.01,0.01),
                                       kernel_constraint=MaxNorm(0.1),
                                       use_bias = False))
                DLClassifier.add(BatchNormalization())
                DLClassifier.add(Activation('selu'))
                DLClassifier.add(Dense(1, activation = 'softmax'))
                DLClassifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
                N_Train = NTrainSamples
                N_Val = NoHumansForValidation
                CheckPoint = ModelCheckpoint("my_keras_model.h5",save_best_only=True)
                EarlyStopping = EarlyStopping(patience=10,restore_best_weights=True)
                DLClassifier.fit(np.array(TrainInputs),
                                 np.array(TrainTargets.astype('int')),
                                 epochs=1000,
                                 validation_data=(np.array(ValidationInputs), np.array(ValidationTargets.astype('int'))),
                                 batch_size=64,
                                 callbacks=[CheckPoint, EarlyStopping])
                
                
        #        TrainPrediction = bag_clf.predict(TrainInputs)
        #        bagTrainingScore = accuracy_score(TrainTargets.astype('int'), TrainPrediction.astype('int'))
        #        ValidationPrediction = bag_clf.predict(ValidationInputs)
        #        bagValidationScore = accuracy_score(ValidationTargets.astype('int'), ValidationPrediction.astype('int'))
        #        bag_clfTrainingScores.append(bagTrainingScore)
        #        bag_clfValidationScores.append(bagValidationScore)
        #    
            
            
            from sklearn.neighbors import KNeighborsClassifier
            KNN = KNeighborsClassifier()
            from sklearn.model_selection import RandomizedSearchCV
            params = {'n_neighbors':range (3, min(NoSamples//2, 20))}
            KNN_CV = RandomizedSearchCV(KNN, params, cv = 5, n_iter = 30)
            
            from sklearn.svm import SVC
            SVC = SVC(kernel='poly', C=1, gamma = 'scale', probability=True)
            params = {'C':np.logspace(-3, 2, num=1000), 'gamma':np.logspace(-3, 2, num=1000)}
            SVC_CV = RandomizedSearchCV(SVC, params, cv = 5, n_iter = 200)
            
            from sklearn.neural_network import MLPClassifier
            MLP = MLPClassifier(hidden_layer_sizes = (30,30,30,30,30), activation = 'relu')
            
        #    Estimators = [('KNNCV', KNN_CV),
        #                  ('SVCCV', SVC_CV), 
        #                  ('MLP', MLP)]
            Estimators = [('KNN', KNN),
                          ('SVC', SVC), 
                          ('MLP', MLP)]
            from sklearn.ensemble import VotingClassifier
            VotingCLF = VotingClassifier(estimators=Estimators, voting='soft')
            VotingCLF.fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
            
            from sklearn.metrics import accuracy_score
            #for CLF in (('KNNCV',KNN_CV), ('SVCCV',SVC_CV), ('MLP',MLP), ('EnsembleCLF',VotingCLF)):
            for CLF in (('KNN',KNN), ('SVC',SVC), ('MLP',MLP), ('EnsembleCLF',VotingCLF)):
                CLF[1].fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
                TrainPrediction = CLF[1].predict(TrainInputs)
                CLFTrainingScore = accuracy_score(TrainTargets.astype('int'), TrainPrediction.astype('int'))
                ValidationPrediction = CLF[1].predict(ValidationInputs)
                CLFValidationScore = accuracy_score(ValidationTargets.astype('int'), ValidationPrediction.astype('int'))
                print(CLF[1].__class__.__name__, CLFTrainingScore)
                print(CLF[1].__class__.__name__, CLFValidationScore)
                #CLFTestScores.append(CLFTestScore)
                CLFTrainingScores[CLF[0]].append(CLFTrainingScore)
                CLFValidationScores[CLF[0]].append(CLFValidationScore)
            
            from sklearn.ensemble import BaggingClassifier
            from sklearn.tree import DecisionTreeClassifier
            bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                                        n_estimators=5000,
                                        max_samples=0.25,
                                        bootstrap=True,
                                        bootstrap_features=True,
                                        max_features=0.25,
                                        n_jobs=-1)
        #    bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
        #                      n_estimators=500,
        #                      max_samples=1.0,
        #                      bootstrap=True,
        #                      n_jobs=-1)
            from sklearn.ensemble import RandomForestClassifier
            #bag_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
            bag_clf.fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
            TrainPrediction = bag_clf.predict(TrainInputs)
            bagTrainingScore = accuracy_score(TrainTargets.astype('int'), TrainPrediction.astype('int'))
            ValidationPrediction = bag_clf.predict(ValidationInputs)
            bagValidationScore = accuracy_score(ValidationTargets.astype('int'), ValidationPrediction.astype('int'))
            bag_clfTrainingScores.append(bagTrainingScore)
            bag_clfValidationScores.append(bagValidationScore)
            
            
            
        #    from sklearn.neighbors import KNeighborsClassifier
        #    KNN = KNeighborsClassifier()
        #    from sklearn.model_selection import RandomizedSearchCV
        #    params = {'n_neighbors':range (3, min(NoSamples//2, 20))}
        #    KNN_CV = RandomizedSearchCV(KNN, params, cv = 5)
        #    KNN_CV.fit(TrainInputs, np.ravel(TrainTargets))
        #    #print('Best KNN parameters: {}'.format(KNN_CV.best_params_))
        #    print('Best KNN CV score: {}'.format(KNN_CV.best_score_))
        #    TestPrediction = KNN_CV.predict(TestInputs)
        #    TestPrediction.astype('int')
        #    from sklearn.metrics import accuracy_score
        #    print(TestTargets)
        #    print(TestPrediction)
        #    KNNTestScore = accuracy_score(TestTargets.astype('int'), TestPrediction.astype('int'))
        #    print(KNNTestScore)
        #    ConfusionMatrix = confusion_matrix(TestTargets.astype('int'), TestPrediction.astype('int'))
        #    #print('The KNN test score is {}'.format(KNNTestScore))
        #    KNNTestScores.append(KNNTestScore)
        #    KNNConfusionMatrix.append(ConfusionMatrix)
        #    
        #    from sklearn.svm import SVC
        #    SVC = SVC()
        #    params = {'C':np.logspace(-3, 2, num=1000), 'gamma':np.logspace(-3, 2, num=1000)}
        #    SVC_CV = RandomizedSearchCV(SVC, params, cv = 5)
        #    SVC_CV.fit(TrainInputs, np.ravel(TrainTargets))
        #    #print('Best SVC parameters: {}'.format(SVC_CV.best_params_))
        #    print('Best SVC CV score: {}'.format(SVC_CV.best_score_))
        #    TestPrediction = SVC_CV.predict(TestInputs)
        #    TestPrediction.astype('int')
        #    SVCTestScore = accuracy_score(TestTargets.astype('int'), TestPrediction.astype('int'))
        #    #print('The SVC test score is {}'.format(SVCTestScore))
        #    SVCTestScores.append(SVCTestScore)
        #    
        #    
        #    from sklearn.neural_network import MLPClassifier
        #    MLP = MLPClassifier(hidden_layer_sizes = (30,30,30,30,30), activation = 'logistic')
        #    MLP.fit(TrainInputs, TrainTargets)
        #    #params = {'C':np.logspace(-3, 2, num=100), 'gamma':np.logspace(-3, 2, num=100)}
        #    #SVC_CV = RandomizedSearchCV(SVC, params, cv = 5)
        #    #SVC_CV.fit(TrainInputs, np.ravel(TrainTargets))
        #    #print('Best SVC parameters: {}'.format(SVC_CV.best_params_))
        #    #print('Best SVC CV score: {}'.format(SVC_CV.best_score_))
        #    TestPrediction = MLP.predict(TestInputs)
        #    TestPrediction.astype('int')
        #    MLPTestScore = accuracy_score(TestTargets.astype('int'), TestPrediction.astype('int'))
        #    #print('The SVC test score is {}'.format(SVCTestScore))
        #    MLPTestScores.append(MLPTestScore)
            print('Learning: Stage {} out of {}'.format(iteration, NoIterations))
        
        
        TatalCLFTrainingScore = {}
        TatalCLFValidationScore = {}
        #for CLF in ('KNNCV','SVCCV','MLP','EnsembleCLF'):
        for CLF in ('KNN','SVC','MLP','EnsembleCLF'):
            ScoreVector = np.array(CLFTrainingScores[CLF])
            TatalCLFTrainingScore[CLF] = ScoreVector.mean()
            print('The '+CLF+' train score is {}'.format(TatalCLFTrainingScore[CLF]))
            ScoreVector = np.array(CLFValidationScores[CLF])
            TatalCLFValidationScore[CLF] = ScoreVector.mean()
            print('The '+CLF+' validation score is {}'.format(TatalCLFValidationScore[CLF]))
        ScoreVector = np.array(bag_clfTrainingScores)
        TotalBagTrainingScore = ScoreVector.mean()
        print('The baging Train score is {}'.format(TotalBagTrainingScore))
        ScoreVector = np.array(bag_clfValidationScores)
        TotalBagValidationScore = ScoreVector.mean()
        print('The baging validation score is {}'.format(TotalBagValidationScore))
        
        #TotalKNNTestScore = np.array(KNNTestScores).mean()
        #TotalSVCTestScore = np.array(SVCTestScores).mean()
        #TotalMLPTestScore = np.array(MLPTestScores).mean()
        #print('The KNN test score is {}'.format(TotalKNNTestScore))
        #print('The SVC test score is {}'.format(TotalSVCTestScore))
        #print('The MLP test score is {}'.format(TotalMLPTestScore))
        
        #TestInputs = pd.DataFrame(columns=FeaturesDF.columns)
        #TestTargets = pd.DataFrame(columns=range(0,1))
        #
        #for HumanIndex in PermutedIndicesForTesting:
        #    Location = np.where(SampleIndexArray==HumanIndex)
        #    TestInputs = TestInputs.append(FeaturesDF.iloc[Location[0]])
        #    TestTargets = TestTargets.append(TargetsDF.iloc[Location[0]])
        
        TestInputs = TestFeaturesDF.copy()
        TestInputs = sfm.transform(TestInputs)
        TestTargets = TestTargetsDF.copy()
        
        
        #CLFTestScores = {'KNNCV':[],'SVCCV':[],'MLP':[],'EnsembleCLF':[]}
        CLFTestScores = {'KNN':[],'SVC':[],'MLP':[],'EnsembleCLF':[]}
        #for CLF in (('KNNCV',KNN_CV), ('SVCCV',SVC_CV), ('MLP',MLP), ('EnsembleCLF',VotingCLF)):
        for CLF in (('KNN',KNN), ('SVC',SVC), ('MLP',MLP), ('EnsembleCLF',VotingCLF)):
            TestPrediction = CLF[1].predict(TestInputs)
            #CLFTestScore = accuracy_score(TestTargets.astype('int'), TestPrediction.astype('int'))
            #print(CLF[1].__class__.__name__, CLFValidationScore)
            #CLFTestScores.append(CLFTestScore)
            CLFTestScores[CLF[0]] = accuracy_score(TestTargets.astype('int'), TestPrediction.astype('int'))
            print('The '+CLF[0]+' test score is {}'.format(CLFTestScores[CLF[0]]))
            print(' Test Targets: {}'.format(np.array(TestTargets).flatten()))
            print(' Test Prediction by '+CLF[0]+': {}\n'.format(TestPrediction))
        
        bagTestPrediction = bag_clf.predict(TestInputs)
        bagTestScore = accuracy_score(TestTargets.astype('int'), bagTestPrediction.astype('int'))
        print('The bagging test score is {}'.format(bagTestScore))
        print('TestTargets: {}'.format(np.array(TestTargets).flatten()))
        print('TestPrediction by bagging: {}'.format(bagTestPrediction))
        
        
        #SVC = SVC(C = 10, gamma=0.05)
        #SVC.fit(XTrain, YTrain)
        
        #KNNTestAccuracy = KNN.score(XTest, YTest)
        #SVCTestAccuracy = SVC.score(XTest, YTest)
        #print('The accuracy of KNN is {}'.format(KNNTestAccuracy))
        #print('The accuracy of SVC is {}'.format(SVCTestAccuracy))
        
        
    
    
    elif PP.DataSplit == 'Automatic':
        FeaturesForCV = FeaturesDF.copy()
        TargetsForCV = TargetsDF.copy()
        if PP.FeatureSelection == True:
            from sklearn.ensemble import RandomForestClassifier
            forest_clf = RandomForestClassifier(n_estimators=5000, max_leaf_nodes=16, n_jobs=-1 )
            forest_clf.fit(FeaturesForCV, np.ravel(TargetsForCV.astype('int')))
            for feature in zip(range(0, FeaturesForCV.shape[1]), forest_clf.feature_importances_):
                print(feature)
            from sklearn.feature_selection import SelectFromModel
            MaxNoFeatures = PP.NSelectedFeatures
            sfm = SelectFromModel(forest_clf, max_features = MaxNoFeatures, threshold=0.00005)
            sfm.fit(FeaturesForCV, np.ravel(TargetsForCV.astype('int')))
            FeaturesForCV = sfm.transform(FeaturesForCV)
        
        from sklearn.model_selection import train_test_split
    #    TrainInputs, TestInputs, TrainTargets, TestTargets \
    #            = train_test_split(FeaturesForCV, TargetsForCV, test_size = PP.TestPercentage/100, random_state = 20, stratify = TargetsDF)
    #    NoActiveSamples = floor(NoSamples*PP.ActiveSubjectsPercentage/100)
    #    FeaturesSliced = FeaturesForCV[0:NoActiveSamples, :]
    #    TargetsSliced = TargetsForCV[0:NoActiveSamples]
    #    
    #    TrainInputs, TestInputs, TrainTargets, TestTargets \
    #            = train_test_split(FeaturesSliced, TargetsSliced, test_size = PP.TestPercentage/100, random_state = 20)
        
        TrainInputs, TestInputs, TrainTargets, TestTargets \
                = train_test_split(FeaturesForCV, TargetsForCV, test_size = PP.TestPercentage/100, random_state = 20)
        
        if PP.CrossValidationSearch == True:
            from sklearn.neighbors import KNeighborsClassifier
            KNN = KNeighborsClassifier(weights='distance')
            from sklearn.model_selection import RandomizedSearchCV
            params = {'n_neighbors':range (3, NoSamples//2)}
            KNN_CV = RandomizedSearchCV(KNN, params, cv = PP.NCVFolds, n_iter = NoSamples//2)
            
            from sklearn.svm import SVC
            SVCLF = SVC(probability=True)
            params = {'kernel':['poly', 'rbf', 'sigmoid', 'linear'], 'C':np.logspace(-3, 0, num=1000), 'gamma':np.logspace(-3, 0, num=1000)}
            SVC_CV = RandomizedSearchCV(SVCLF, params, cv = PP.NCVFolds, n_iter = 500)
            
            from sklearn.neural_network import MLPClassifier
            MLP = MLPClassifier()
            from hyperparams_config import MLPLayersConfig
            TargetConfig = [(1,20), (2,50), (3,150)]
            nNeuronsRange = [5,50]
            HiddenLayersSizes = MLPLayersConfig(TargetConfig, nNeuronsRange)
            params = {'hidden_layer_sizes':HiddenLayersSizes,
                      'activation':['relu', 'tanh', 'logistic']}
            MLP_CV = RandomizedSearchCV(MLP, params, cv = PP.NCVFolds, n_iter = 100)
            
            from sklearn.ensemble import BaggingClassifier
            from sklearn.tree import DecisionTreeClassifier
            bag_clf = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True, bootstrap_features=True, n_jobs = 4)
            params = {'n_estimators':np.arange(100, 5101, 500),
                      'max_samples':np.arange(0.1, 0.96, 0.1),
                      'max_features':np.arange(0.1, 0.96, 0.1)}
            bag_clf_CV = RandomizedSearchCV(bag_clf, params, cv = PP.NCVFolds, n_iter = 10)
            
            TrainScores = {}
            ValidationScores = {}
            TestScores = {}
            TestPredictions = {}
            from sklearn.metrics import accuracy_score
            for CLF in (('KNNCV',KNN_CV), ('SVCCV',SVC_CV), ('MLPCV',MLP_CV), ('BAGCV',bag_clf_CV)):    #, ('EnsembleCLFCV',VotingCLF_CV)
            #for CLF in (('KNN',KNN), ('SVC',SVC), ('MLP',MLP), ('EnsembleCLF',VotingCLF)):
                CLF[1].fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
                TrainPrediction = CLF[1].predict(TrainInputs)
                CLFTrainingScore = accuracy_score(TrainTargets.astype('int'), TrainPrediction.astype('int'))
                #ValidationPrediction = CLF[1].predict(ValidationInputs)
                #CLFValidationScore = accuracy_score(ValidationTargets.astype('int'), ValidationPrediction.astype('int'))
                print(CLF[1].__class__.__name__, CLFTrainingScore)
                TrainScores[CLF[0]]=CLFTrainingScore
                #print(CLF[1].__class__.__name__, CLFValidationScore)
                #CLFTestScores.append(CLFTestScore)
                #if CLF[0] != 'EnsembleCLF':
                print('Best {} parameters: {}'.format(CLF[0],CLF[1].best_params_))
                print('Best {} score: {}'.format(CLF[0], CLF[1].best_score_))
                ValidationScores[CLF[0]]=CLF[1].best_score_
                TestPredictions[CLF[0]] = CLF[1].predict(TestInputs)
                TestScore = accuracy_score(TestTargets.astype('int'), TestPredictions[CLF[0]].astype('int'))
                print('The {} test score is {}'.format(CLF[0], TestScore))
                TestScores[CLF[0]]=TestScore
            
            
            BestKNN = KNeighborsClassifier(n_neighbors=KNN_CV.best_params_['n_neighbors'])
            BestSVC = SVC(kernel = SVC_CV.best_params_['kernel'],
                          C = SVC_CV.best_params_['C'], 
                          gamma = SVC_CV.best_params_['gamma'], 
                          probability=True)
            BestMLP = MLPClassifier(hidden_layer_sizes=MLP_CV.best_params_['hidden_layer_sizes'], 
                                    activation=MLP_CV.best_params_['activation'])
            BestBagging = BaggingClassifier(DecisionTreeClassifier(), 
                                            n_estimators=bag_clf_CV.best_params_['n_estimators'], 
                                            max_samples=bag_clf_CV.best_params_['max_samples'], 
                                            max_features=bag_clf_CV.best_params_['max_features'], 
                                            bootstrap=True,
                                            bootstrap_features=True)
            Estimators = [('KNNCV', BestKNN),
                          ('SVCCV', BestSVC), 
                          ('MLP', BestMLP), 
                          ('Bag', BestBagging)]
            
            from sklearn.ensemble import VotingClassifier
            VotingCLF = VotingClassifier(estimators=Estimators, voting = 'soft', n_jobs = 4)
            VotingCLF.fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
            TrainPrediction = VotingCLF.predict(TrainInputs)
            VotingCLFTrainingScore = accuracy_score(TrainTargets.astype('int'), TrainPrediction.astype('int'))
            print(VotingCLF.__class__.__name__, VotingCLFTrainingScore)
            TrainScores['VotCLF'] = VotingCLFTrainingScore
            from sklearn.model_selection import cross_val_score
            VotingCLF_ValScore = cross_val_score(VotingCLF, TrainInputs, np.ravel(TrainTargets.astype('int')), cv=PP.NCVFolds)
            VotingCLF_ValScore = np.array(VotingCLF_ValScore).mean()
            print('The VotingCLF validation score is {}'.format(VotingCLF_ValScore))
            ValidationScores['VotCLF'] = VotingCLF_ValScore
            TestPredictions['VotCLF'] = VotingCLF.predict(TestInputs)
            TestScore = accuracy_score(TestTargets.astype('int'), TestPredictions['VotCLF'].astype('int'))
            print('The VotingCLF test score is {}'.format(TestScore))
            TestScores['VotCLF'] = TestScore
            
        elif PP.CrossValidationSearch == False:
            from sklearn.neighbors import KNeighborsClassifier
            KNN_CV = KNeighborsClassifier(n_neighbors=10)
            
            from sklearn.svm import SVC
            SVC_CV = SVC(kernel='rbf', C=0.7636, gamma=0.00959, probability=True)
            
            from sklearn.neural_network import MLPClassifier
            MLP_CV = MLPClassifier(activation='relu', hidden_layer_sizes=(39,30,6))
            
            from sklearn.ensemble import BaggingClassifier
            from sklearn.tree import DecisionTreeClassifier
            bag_clf_CV = BaggingClassifier(DecisionTreeClassifier(),
                                        bootstrap=True, bootstrap_features=True, n_jobs = 4, 
                                        n_estimators = 2850, 
                                        max_samples = 0.7, 
                                        max_features = 0.7)
            
            Estimators = [('KNNCV', KNN_CV),
                          ('SVCCV', SVC_CV), 
                          ('MLPCV', MLP_CV), 
                          ('BAGCV', bag_clf_CV)]
            from sklearn.ensemble import VotingClassifier
            VotingCLF = VotingClassifier(estimators=Estimators, voting='soft')
            
            TrainScores = {}
            ValidationScores = {}
            TestScores = {}
            TestPredictions = {}
            from sklearn.metrics import accuracy_score
            for CLF in (('KNNCV',KNN_CV), ('SVCCV',SVC_CV), ('MLPCV',MLP_CV), ('BAGCV',bag_clf_CV), ('VotCLF',VotingCLF)):    #
            #for CLF in (('KNN',KNN), ('SVC',SVC), ('MLP',MLP), ('EnsembleCLF',VotingCLF)):
                CLF[1].fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
                TrainPrediction = CLF[1].predict(TrainInputs)
                CLFTrainingScore = accuracy_score(TrainTargets.astype('int'), TrainPrediction.astype('int'))
                #ValidationPrediction = CLF[1].predict(ValidationInputs)
                #CLFValidationScore = accuracy_score(ValidationTargets.astype('int'), ValidationPrediction.astype('int'))
                print(CLF[1].__class__.__name__, CLFTrainingScore)
                TrainScores[CLF[0]]=CLFTrainingScore
                #print(CLF[1].__class__.__name__, CLFValidationScore)
                #CLFTestScores.append(CLFTestScore)
                #if CLF[0] != 'EnsembleCLF':
                #print('Best {} parameters: {}'.format(CLF[0],CLF[1].best_params_))
                #print('Best {} score: {}'.format(CLF[0], CLF[1].best_score_))
                from sklearn.model_selection import cross_val_score
                CLF_ValScore = cross_val_score(CLF[1], TrainInputs, np.ravel(TrainTargets.astype('int')), cv=PP.NCVFolds)
                CLF_ValScore = np.array(CLF_ValScore).mean()
                ValidationScores[CLF[0]]=CLF_ValScore
                print('The {} validation score is {}'.format(CLF[0], CLF_ValScore))
                TestPredictions[CLF[0]] = CLF[1].predict(TestInputs)
                TestScore = accuracy_score(TestTargets.astype('int'), TestPredictions[CLF[0]].astype('int'))
                print('The {} test score is {}'.format(CLF[0], TestScore))
                TestScores[CLF[0]]=TestScore
    
        #OutputFileName = ''.join(Data.ActiveFeatures)
        OutputFileName = 'Results.txt'
        file = open(OutputFileName, 'w')
        #file = open('Results\\Validation_new\\'+OutputFileName+'-'+str(TotalIterator)+'.txt', 'w')
    
        file.write('{}\nFINAL RESULTS\n{}'.format('~'*30, '~'*30))
        file.write('\nParameters\n{}\n'.format('-'*20))
        file.write('Classification Target: {}\n'.format(PP.ClassificationTarget))
        file.write('Active Exercise: {}\n'.format(FileInitial))
        file.write('NoSamples: {}\n'.format(NoSamples))
        file.write('TestPercentage: {}\n'.format(PP.TestPercentage))
        file.write('NoCVFolds: {}\n'.format(PP.NCVFolds))
        file.write('NoSelectedFeatures: {}\n'.format(PP.NSelectedFeatures))
        file.write('Active Features: {}\n'.format(Data.ActiveFeatures))
        file.write('\nClassifier\tTrainScore\tCVScore\t\tTestScore\n')
        file.write('___________________________________________________________\n')
        for CLF in (('KNNCV',KNN_CV), ('SVCCV',SVC_CV), ('MLPCV',MLP_CV), ('BAGCV',bag_clf_CV), ('VotCLF',VotingCLF)):
            file.write('{0}\t\t{1:5.3f}\t\t{2:5.3f}\t\t{3:5.3f}\n'.format(CLF[0], TrainScores[CLF[0]], ValidationScores[CLF[0]], TestScores[CLF[0]]))
        
        from sklearn.metrics import classification_report, confusion_matrix
        for CLF in (('KNNCV',KNN_CV), ('SVCCV',SVC_CV), ('MLPCV',MLP_CV), ('BAGCV',bag_clf_CV), ('VotCLF',VotingCLF)):
            ConfusionMatrix = confusion_matrix(TestTargets.astype('int'), TestPredictions[CLF[0]].astype('int'))
            file.write ('~~~~~~~~~~~~~~~~~~\nConfusion Matrix for {}\n{}\n\n'.format(CLF[0], ConfusionMatrix))
            ClassificationReport = classification_report(TestTargets.astype('int'), TestPredictions[CLF[0]].astype('int'), digits = 3)
            file.write ('~~~~~~~~~~~~~~~~~~\nClassification Report for {}\n{}\n\n'.format(CLF[0], ClassificationReport))
            
        file.close()
        
        #Classifiers = {'KNN':KNN_CV, 'SVC':SVC_CV, 'MLP':MLP_CV, 'BAG':bag_clf_CV, 'VC':VotingCLF}
        ComplementryPlots = False
        if ComplementryPlots == True:
            plt.figure()
            from sklearn.model_selection import RandomizedSearchCV
            from hyperparams_config import MLPLayersConfig
            if PP.ActiveDataSet == 'Andersson':
                NSamples = np.arange(19,141, 15)
            
                AnderssonKNNResults = {'KNN':[(20,96.0132),
                                            (35,93.9048),
                                            (50,92.591),
                                            (65,91.0786),
                                            (80,90.6192),
                                            (95,89.0869),
                                            (110,88.9652),
                                            (125,88.327),
                                            (140,87.4304)], 
                                        'SVC':[(20,96.5496), 
                                            (35,92.6333),
                                            (50,92.0943),
                                            (65,89.827),
                                            (80,89.904),
                                            (95,88.4511),
                                            (110,88.0514),
                                            (125,87.0753),
                                            (140,86.2384)],
                                        'MLP':[(20,96.0132),
                                            (35,92.2757),
                                            (50,91.7368),
                                            (65,89.4098),
                                            (80,89.3278),
                                            (95,87.9147),
                                            (110,87.3758),
                                            (125,86.3402),
                                            (140,85.4834)]}
            
                X = {'KNN':[], 'SVC':[], 'MLP':[]}
                Y = {'KNN':[], 'SVC':[], 'MLP':[]}
                for nSamples in NSamples:
                    Scores = {'KNN':[], 'SVC':[], 'MLP':[]}
                    NIterations = 20
                    for iteration in range(0,NIterations):
                        FeaturesSliced = FeaturesForCV[0:nSamples, :]
                        TargetsSliced = TargetsForCV[0:nSamples]
                        TrainInputs, TestInputs, TrainTargets, TestTargets \
                            = train_test_split(FeaturesSliced, TargetsSliced, test_size = PP.TestPercentage/100, stratify = TargetsSliced)
                        KNN = KNeighborsClassifier(weights='distance')
                        params = {'n_neighbors':range (3, nSamples//3)}
                        KNN_CV = RandomizedSearchCV(KNN, params, cv = PP.NCVFolds, n_iter = 20)
                        
                        SVCLF = SVC(probability=True)
                        params = {'kernel':['poly', 'rbf', 'sigmoid', 'linear'], 'C':np.logspace(-3, 0, num=1000), 'gamma':np.logspace(-3, 0, num=1000)}
                        SVC_CV = RandomizedSearchCV(SVCLF, params, cv = PP.NCVFolds, n_iter = 500)
                        
                        MLP = MLPClassifier(activation='relu')
                        TargetConfig = [(1,80)]
                        nNeuronsRange = [5,50]
                        HiddenLayersSizes = MLPLayersConfig(TargetConfig, nNeuronsRange)
                        params = {'hidden_layer_sizes':HiddenLayersSizes}
                        MLP_CV = RandomizedSearchCV(MLP, params, cv = PP.NCVFolds, n_iter = 10)
            #            MLP_CV = MLPClassifier(activation='relu', hidden_layer_sizes=(30,30,30))
                        
                        CLF = [('KNN',KNN_CV), ('SVC',SVC_CV), ('MLP',MLP_CV)]
                        #CLF = [('KNN',KNN), ('SVC',SVCLF), ('MLP',MLP)]
                        for clf in CLF:
                            clf[1].fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
                            #CLF_ValScore = cross_val_score(clf[1], TrainInputs, np.ravel(TrainTargets.astype('int')), cv=PP.NCVFolds)
                            #CLF_ValScore = np.array(CLF_ValScore).mean()
                            #Scores[clf[0]].append(CLF_ValScore)
                            Scores[clf[0]].append(clf[1].best_score_)
                            print(nSamples, iteration, clf[0])
                    CLF_ValScore = {}
                    styles = [('KNN','rs', '-'), ('SVC','bs', '-'), ('MLP','gs', '-')]
                    for STYLE in styles:
                        CLF_ValScore[STYLE[0]] = np.array(Scores[STYLE[0]]).mean()
                    #ValidationScores[CLF[0]]=CLF_ValScore
                    #print('nSamples: {}, Validation score: {}'.format(nSamples, CLF_ValScore))
                        X[STYLE[0]].extend([nSamples+1])
                        Y[STYLE[0]].extend([CLF_ValScore[STYLE[0]]*100])
                for STYLE in styles:
                    plt.plot(X[STYLE[0]], Y[STYLE[0]], STYLE[1], linestyle=STYLE[2], label=STYLE[0]+' (Present Methodology)')
                    #plt.plot(X, Y, label=STYLE[0]+' (Andersson & Araujo)')
                    plt.legend()
                        #plt.plot(nSamples+1, CLF_ValScore[STYLE[0]]*100, STYLE[1])
                    #plt.plot(nSamples, CLF.best_score_, 'rs')
                
                styles = [('KNN','ro', '--'), ('SVC','bo', '--'), ('MLP','go', '--')]
                for STYLE in styles:
                    X = []
                    Y = []
                    for index in AnderssonKNNResults[STYLE[0]]:
                        X.extend([index[0]])
                        Y.extend([index[1]])
                        #plt.plot(index[0], index[1], STYLE[1])
                    plt.plot(X, Y, STYLE[1], linestyle=STYLE[2], label=STYLE[0]+' (Andersson & Araujo)')
                    #plt.plot(X, Y, label=STYLE[0]+' (Andersson & Araujo)')
                    plt.legend()
                plt.xlabel('Number of subjects')
                plt.ylabel('Classifier Accuracy (%)')
                #plt.legend(['KNN (Andersson & Araujo)', 'SVC (Andersson & Araujo)', 'MLP (Andersson & Araujo)'])
            
            elif PP.ActiveDataSet == 'Cheng':
                NTotalSamples = len(SelectedHumansIndices)
                StartIndex = 24
                NoOutputs = 7
                Delta = (NTotalSamples - StartIndex)//NoOutputs
                NSamples = np.arange(24, NTotalSamples, Delta)    
                X = {'KNN':[], 'SVC':[], 'MLP':[], 'BC':[], 'VC':[]}
                Y = {'KNN':[], 'SVC':[], 'MLP':[], 'BC':[], 'VC':[]}
                for nSamples in NSamples:
                    Scores = {'KNN':[], 'SVC':[], 'MLP':[], 'BC':[], 'VC':[]}
                    NIterations = 1
                    for iteration in range(0,NIterations):
                        FeaturesSliced = FeaturesForCV[0:nSamples, :]
                        TargetsSliced = TargetsForCV[0:nSamples]
                        TrainInputs, TestInputs, TrainTargets, TestTargets \
                            = train_test_split(FeaturesSliced, TargetsSliced, test_size = PP.TestPercentage/100, stratify = TargetsSliced)
                        KNN = KNeighborsClassifier(weights='distance')
                        params = {'n_neighbors':range (3, nSamples//3)}
                        KNN_CV = RandomizedSearchCV(KNN, params, cv = PP.NCVFolds, n_iter = 20)
                        
                        SVCLF = SVC(probability=True)
                        params = {'kernel':['poly', 'rbf', 'sigmoid', 'linear'], 'C':np.logspace(-3, 0, num=1000), 'gamma':np.logspace(-3, 0, num=1000)}
                        SVC_CV = RandomizedSearchCV(SVCLF, params, cv = PP.NCVFolds, n_iter = 500)
                        
                        MLP = MLPClassifier(activation='relu')
                        TargetConfig = [(1,10), (2,20), (3,50)]
                        nNeuronsRange = [5,50]
                        HiddenLayersSizes = MLPLayersConfig(TargetConfig, nNeuronsRange)
                        #TargetConfig = [(1,80)]
                        #nNeuronsRange = [5,50]
                        #HiddenLayersSizes = MLPLayersConfig(TargetConfig, nNeuronsRange)
                        params = {'hidden_layer_sizes':HiddenLayersSizes}
                        MLP_CV = RandomizedSearchCV(MLP, params, cv = PP.NCVFolds, n_iter = 10)
            #            MLP_CV = MLPClassifier(activation='relu', hidden_layer_sizes=(30,30,30))
                        
                        from sklearn.ensemble import BaggingClassifier
                        from sklearn.tree import DecisionTreeClassifier
                        BC = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True, bootstrap_features=True, n_jobs = 4)
                        params = {'n_estimators':np.arange(100, 5101, 250),
                                  'max_samples':np.arange(0.1, 0.96, 0.05),
                                  'max_features':np.arange(0.1, 0.96, 0.05)}
                        BC_CV = RandomizedSearchCV(BC, params, cv = PP.NCVFolds, n_iter = 5)
            
                        CLF = [('KNN',KNN_CV), ('SVC',SVC_CV), ('MLP',MLP_CV), ('BC',BC_CV)]
                        
                        for clf in CLF:
                            clf[1].fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
                            #CLF_ValScore = cross_val_score(clf[1], TrainInputs, np.ravel(TrainTargets.astype('int')), cv=PP.NCVFolds)
                            #CLF_ValScore = np.array(CLF_ValScore).mean()
                            #Scores[clf[0]].append(CLF_ValScore)
                            Scores[clf[0]].append(clf[1].best_score_)
                            print(nSamples, iteration, clf[0])
                            print('ValScore: {}'.format(clf[1].best_score_))
                    
                        BestKNN = KNeighborsClassifier(n_neighbors=KNN_CV.best_params_['n_neighbors'])
                        BestSVC = SVC(kernel = SVC_CV.best_params_['kernel'],
                                      C = SVC_CV.best_params_['C'], 
                                      gamma = SVC_CV.best_params_['gamma'], 
                                      probability=True)
                        BestMLP = MLPClassifier(hidden_layer_sizes=MLP_CV.best_params_['hidden_layer_sizes'])
                        BestBagging = BaggingClassifier(DecisionTreeClassifier(), 
                                                        n_estimators=BC_CV.best_params_['n_estimators'], 
                                                        max_samples=BC_CV.best_params_['max_samples'], 
                                                        max_features=BC_CV.best_params_['max_features'], 
                                                        bootstrap=True,
                                                        bootstrap_features=True)
                        Estimators = [('KNNCV', BestKNN),
                                      ('SVCCV', BestSVC), 
                                      ('MLP', BestMLP), 
                                      ('Bag', BestBagging)]
                        
        #                from sklearn.ensemble import VotingClassifier
                        VotingCLF = VotingClassifier(estimators=Estimators, voting = 'soft', n_jobs = 4)
                        VotingCLF.fit(TrainInputs, np.ravel(TrainTargets.astype('int')))
        #                from sklearn.model_selection import cross_val_score
                        VotingCLF_ValScore = cross_val_score(VotingCLF, TrainInputs, np.ravel(TrainTargets.astype('int')), cv=PP.NCVFolds)
                        VotingCLF_ValScore = np.array(VotingCLF_ValScore).mean()
                        Scores['VC'].append(VotingCLF_ValScore)
                        print(nSamples, iteration, 'VC')
                        print('ValScore: {}'.format(VotingCLF_ValScore))
                    
                    CLF_ValScore = {}
                    styles = [('KNN','rs', '-'), ('SVC','bs', '-'), ('MLP','gs', '-'), ('BC','cs', '-'), ('VC','ms', '-')]
                    for STYLE in styles:
                        CLF_ValScore[STYLE[0]] = np.array(Scores[STYLE[0]]).mean()
                    #ValidationScores[CLF[0]]=CLF_ValScore
                    #print('nSamples: {}, Validation score: {}'.format(nSamples, CLF_ValScore))
                        X[STYLE[0]].extend([nSamples+1])
                        Y[STYLE[0]].extend([CLF_ValScore[STYLE[0]]*100])
                for STYLE in styles:
                    plt.plot(X[STYLE[0]], Y[STYLE[0]], STYLE[1], linestyle=STYLE[2], label=STYLE[0])
                    #plt.plot(X, Y, label=STYLE[0]+' (Andersson & Araujo)')
                    plt.legend()
                        #plt.plot(nSamples+1, CLF_ValScore[STYLE[0]]*100, STYLE[1])
                    #plt.plot(nSamples, CLF.best_score_, 'rs')
                plt.xlabel('Number of subjects')
                plt.ylabel('Classifier Accuracy (%)')
                