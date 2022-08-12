def MLPLayersConfig(TargetConfig, nNeuronsRange):
    from random import randint
    nNeuronsMin = nNeuronsRange[0]
    nNeuronsMax = nNeuronsRange[1]
    LayersConfig = []
    for Config in TargetConfig:
        nLayers = Config[0]
        nConfigs = Config[1]
        for config in range(0, nConfigs):
            TempList = []
            for layer in range(0,nLayers):
                nNeuronsLocal = randint(nNeuronsMin, nNeuronsMax)
                TempList.append(nNeuronsLocal)
            LayersConfig.append(tuple(TempList))
    return LayersConfig

TargetConfig = [(1,3), (2,4), (3,5)]
nNeuronsRange = [5,50]
LayersConfig = MLPLayersConfig(TargetConfig, nNeuronsRange)
print(LayersConfig)