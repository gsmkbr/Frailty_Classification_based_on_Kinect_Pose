def FeatureVector(AllData, HumanIndex, ExerciseIter, RefKey1, RefKey2):
    import numpy as np
    RefJoint1 = {'X':np.array(AllData[HumanIndex][ExerciseIter][RefKey1+'X']), 
                  'Y':np.array(AllData[HumanIndex][ExerciseIter][RefKey1+'Y']), 
                  'Z':np.array(AllData[HumanIndex][ExerciseIter][RefKey1+'Z'])}
    RefJoint2 = {'X':np.array(AllData[HumanIndex][ExerciseIter][RefKey2+'X']), 
                  'Y':np.array(AllData[HumanIndex][ExerciseIter][RefKey2+'Y']), 
                  'Z':np.array(AllData[HumanIndex][ExerciseIter][RefKey2+'Z'])}
    JointsDistance = np.sqrt(np.power((RefJoint1['X'] - RefJoint2['X']),2)+np.power((RefJoint1['Y'] - RefJoint2['Y']),2)+np.power((RefJoint1['Z'] - RefJoint2['Z']),2))
    return JointsDistance

def JointPairs(PP):
    KW = PP.KeyWords
    JointPairs = []
    for FirstKeyIndex in range (0, len(KW)):
        for SecondKeyIndex in range (FirstKeyIndex+1, len(KW)):
            FirstKey = KW[FirstKeyIndex]
            SecondKey = KW[SecondKeyIndex]
            JointPairs.extend([FirstKey + ' & ' + SecondKey])
    return JointPairs