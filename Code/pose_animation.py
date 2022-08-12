def PosturalAnimation(PP):
    from import_of_data import ImportData
    
    Data = ImportData(PP.DataTargetDirectory)
    Data.DataReading(PP, PP.FileName)
    Data.MissingDataRemoval()

    import matplotlib.pyplot as plt
    from PostDraw import PosturalDrawing
    Fig, ax = plt.subplots(1,3)
    
    import numpy as np
    XRange = [np.min(Data.Data_NP[:,range(1,75,3)])-PP.AnimationMargin, np.max(Data.Data_NP[:,range(1,75,3)])+PP.AnimationMargin]
    YRange = [np.min(Data.Data_NP[:,range(2,75,3)])-PP.AnimationMargin, np.max(Data.Data_NP[:,range(2,75,3)])+PP.AnimationMargin]
    ZRange = [np.min(Data.Data_NP[:,range(3,75,3)])-PP.AnimationMargin, np.max(Data.Data_NP[:,range(3,75,3)])+PP.AnimationMargin]
    for t in range (0, Data.Data_DF.shape[0], PP.TimeFrameSkip):
        C1 = 'X'
        C2 = 'Y'
        Labels = ['X(m)', 'Y(m)']
        R1 = XRange
        R2 = YRange
        PosturalDrawing(ax[0], C1, C2, R1, R2, Labels, Data.Data_DF, t)
        
        C1 = 'Z'
        C2 = 'Y'
        Labels = ['Z(m)', 'Y(m)']
        R1 = ZRange
        R2 = YRange
        PosturalDrawing(ax[1], C1, C2, R1, R2, Labels, Data.Data_DF, t)
        
        C1 = 'X'
        C2 = 'Z'
        Labels = ['X(m)', 'Z(m)']
        R1 = XRange
        R2 = ZRange
        PosturalDrawing(ax[2], C1, C2, R1, R2, Labels, Data.Data_DF, t)
        