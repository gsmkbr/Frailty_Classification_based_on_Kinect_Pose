class Preprocess:
    def __init__(self, Features):
        self.FeaturesInit = Features
        
    def NullFeatureDrop(self):
        import numpy as np
        NewFeatures = self.FeaturesInit.copy()
        NFeatures = len(self.FeaturesInit.columns)
        NSamples = self.FeaturesInit.index.size
        for sample in range (NFeatures-1,-1,-1):
            FeatureVector = np.array(self.FeaturesInit[sample])
            NonZeroElements = np.where(FeatureVector != 0)
            BinraryVector = np.copy(FeatureVector)
            BinraryVector[NonZeroElements]=1
            NoNonZero = np.sum(BinraryVector)
            if NoNonZero < 0.5*NSamples:
                NewFeatures.drop(sample, axis=1, inplace=True)
        self.FeaturesNew = NewFeatures.copy()