def HumanInfo(PP):
    import pandas as pd
    import numpy as np
    Info = pd.read_excel(PP.DataRootDirectory + '\\' + 'General_Info.xlsx')
    OverallHumanFeatures = Info[PP.OverallHumanFeatures]
    #Encoding Categorical data into digits
    OverallHumanFeatures['SEX'] = OverallHumanFeatures['SEX'].astype('category')
    OverallHumanFeatures['SEX'] = OverallHumanFeatures['SEX'].cat.codes
    
    return OverallHumanFeatures

def HumanSelection(HInfo, PP):
    import numpy as np
    TargetKey = PP.ClassificationTarget
    NoTotalHumans = HInfo.shape[0]
    if PP.SamplingStatus == 'UninormlyDistributedTargets':
        NoFrailty = {'0':0, '1':0, '2':0}
        #for FrailtyLevel in HInfo[TargetKey]:
        for FrailtyLevel in HInfo[TargetKey][PP.HumanRange[0]:PP.HumanRange[-1]+1]:
            if FrailtyLevel == 0:
                NoFrailty['0'] += 1
            elif FrailtyLevel == 1:
                NoFrailty['1'] += 1
            #elif FrailtyLevel == 2:
            #    NoFrailty['2'] += 1
        
        #MinNoFrailtyLevel = min(NoFrailty['0'], NoFrailty['1'], NoFrailty['2']) - 5
        MinNoFrailtyLevel = min(NoFrailty['0'], NoFrailty['1']) - 1
        #RandomFrailyLists = {'0':[], '1':[], '2':[]}
        RandomFrailyLists = {'0':[], '1':[]}
        RandPerm = np.random.permutation(PP.HumanRange[-1]-PP.HumanRange[0]) + (PP.HumanRange[0])
        counter = 0
        while len(RandomFrailyLists['0']) < MinNoFrailtyLevel:
            RandIndex = RandPerm[counter]
            if HInfo[TargetKey][RandIndex] == 0:
                RandomFrailyLists['0'].extend([RandIndex])
            counter += 1
        
        counter = 0
        while len(RandomFrailyLists['1']) < MinNoFrailtyLevel:
            RandIndex = RandPerm[counter]
            if HInfo[TargetKey][RandIndex] == 1:
                RandomFrailyLists['1'].extend([RandIndex])
            counter += 1
        print(RandomFrailyLists)
            
        #while len(RandomFrailyLists['2']) < MinNoFrailtyLevel:
        #    RandIndex = RandPerm[counter]
        #    if HInfo[TargetKey][RandIndex] == 2:
        #        RandomFrailyLists['2'].extend([RandIndex])
        #    counter += 1
                
        SelectedIndexList = []
        SelectedIndexList.extend(RandomFrailyLists['0'])
        SelectedIndexList.extend(RandomFrailyLists['1'])
        #SelectedIndexList.extend(RandomFrailyLists['2'])
    elif PP.SamplingStatus == 'RandomlyDistributedTargets':
        RandPerm = np.random.permutation(NoTotalHumans)
        SelectedIndexList = list(RandPerm[0:PP.NoRequiredSamples])
    elif PP.SamplingStatus == 'CustomizedRange':
        SelectedIndexList = list(range(PP.HumanRange[0],PP.HumanRange[-1]))
    return SelectedIndexList

def HumansWithSpecificExercises(PP,TargetFilesInitial, OriginalSelectedIndices):
    from os import listdir
    SelectedHumansIndices = []
    for HumanIndex in range(PP.HumanRange[0], PP.HumanRange[1]):
        
        if HumanIndex in OriginalSelectedIndices:
            
            if PP.ActiveDataSet == 'Cheng':
                FolderName = 'KN'+str(HumanIndex+PP.StartHumanCode).zfill(3)
                #DataRootDirectory = '..\\..\\..\\Kinect data\\20 Input data'
                DataTargetDirectory = PP.DataRootDirectory+'\\'+FolderName+'\\'
            elif PP.ActiveDataSet == 'Andersson':
                FolderName = 'Person'+str(HumanIndex+PP.StartHumanCode).zfill(3)
                DataTargetDirectory = PP.DataRootDirectory+'\\'+FolderName+'\\'
            List_of_Files = listdir(DataTargetDirectory)
            ListOfAvailableExercises = {}
            for FileName in List_of_Files:
                if PP.ActiveDataSet == 'Cheng' and FileName.endswith('.csv'):
                    for FileInit in TargetFilesInitial:
                        counter = 0
                        if FileName.startswith(FileInit):
                            counter += 1
                            if counter == 1:
                                ListOfAvailableExercises[FileInit] = 1
                elif PP.ActiveDataSet == 'Andersson' and FileName.endswith('.txt'):
                    for FileInit in TargetFilesInitial:
                        counter = 0
                        if FileName.startswith(FileInit):
                            counter += 1
                            if counter == 1:
                                ListOfAvailableExercises[FileInit] = 1
            if len(ListOfAvailableExercises) == len(TargetFilesInitial):
                SelectedHumansIndices.extend([HumanIndex])
        
    return SelectedHumansIndices