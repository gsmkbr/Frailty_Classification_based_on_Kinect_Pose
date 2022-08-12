class DTWBasedLearning:
    def __init__(self, AllData):
        self.AllData = AllData
        
#    def FeatureVectors(self, HumanIndex, ExerciseIter, RefKey1, RefKey2):
#        import numpy as np
#        RefJoint1 = {'X':np.array(self.AllData[HumanIndex][ExerciseIter][RefKey1+'X']), 
#                      'Y':np.array(self.AllData[HumanIndex][ExerciseIter][RefKey1+'Y']), 
#                      'Z':np.array(self.AllData[HumanIndex][ExerciseIter][RefKey1+'Z'])}
#        RefJoint2 = {'X':np.array(self.AllData[HumanIndex][ExerciseIter][RefKey2+'X']), 
#                      'Y':np.array(self.AllData[HumanIndex][ExerciseIter][RefKey2+'Y']), 
#                      'Z':np.array(self.AllData[HumanIndex][ExerciseIter][RefKey2+'Z'])}
#        self.JointsDistance = np.sqrt(np.power((RefJoint1['X'] - RefJoint2['X']),2)+np.power((RefJoint1['Y'] - RefJoint2['Y']),2)+np.power((RefJoint1['Z'] - RefJoint2['Z']),2))
    
    def JDFeatureVectors(self, HumanIndex, ExerciseIter, PP, ActiveFileIndex):
        import pandas as pd
        from dtw_functions import FeatureVector, JointPairs
        #KW = PP.KeyWords
#        if HumanIndex==0 and ExerciseIter==0:
#            self.FeatureVectors = []
        self.FeatureVectors = {}
        if PP.JointsCombinationStatus == 'AllJointsCombinations'\
                or PP.JointsCombinationStatus == 'CustomizedJoints(AllCombinations)':
            KW = PP.JointsKeys
            for FirstKeyIndex in range (0, len(KW)):
                for SecondKeyIndex in range (FirstKeyIndex+1, len(KW)):
                    FirstKey = KW[FirstKeyIndex]
                    SecondKey = KW[SecondKeyIndex]
                    JointsDistanceVector = FeatureVector(self.AllData, HumanIndex, ExerciseIter, RefKey1 = FirstKey, RefKey2 = SecondKey)
                    JointPair = FirstKey + ' & ' + SecondKey
                    self.FeatureVectors[JointPair] = JointsDistanceVector
        elif PP.JointsCombinationStatus == 'CustomizedJoints(SelectedCombinations)':
            NCombinations = len(PP.JointsCombs[PP.ActiveFileIndex])
            for CombIndex in range (0, NCombinations):
                FirstKey = PP.JointsCombs[PP.ActiveFileIndex][CombIndex][0]
                SecondKey = PP.JointsCombs[PP.ActiveFileIndex][CombIndex][1]
                JointsDistanceVector = FeatureVector(self.AllData, HumanIndex, ExerciseIter, RefKey1 = FirstKey, RefKey2 = SecondKey)
                JointPair = FirstKey + ' & ' + SecondKey
                self.FeatureVectors[JointPair] = JointsDistanceVector