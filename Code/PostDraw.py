def PosturalDrawing(ax, C1, C2, R1, R2, Labels, Data_DF, t):
    import matplotlib.pyplot as plt
    ax.clear()
    ax.set_xlim(R1[0],R1[1])
    ax.set_ylim(R2[0],R2[1])
    ax.set_aspect('equal')
    ax.set_xlabel(Labels[0])
    ax.set_ylabel(Labels[1])
    x1 = [Data_DF['ShoulderLeft'+C1][t], Data_DF['ElbowLeft'+C1][t], Data_DF['WristLeft'+C1][t], Data_DF['HandLeft'+C1][t]]
    y1 = [Data_DF['ShoulderLeft'+C2][t], Data_DF['ElbowLeft'+C2][t], Data_DF['WristLeft'+C2][t], Data_DF['HandLeft'+C2][t]]
    # plotting the left hand 
    ax.plot(x1, y1)
    x1 = [Data_DF['ShoulderRight'+C1][t], Data_DF['ElbowRight'+C1][t], Data_DF['WristRight'+C1][t], Data_DF['HandRight'+C1][t]]
    y1 = [Data_DF['ShoulderRight'+C2][t], Data_DF['ElbowRight'+C2][t], Data_DF['WristRight'+C2][t], Data_DF['HandRight'+C2][t]]
    # plotting the right hand 
    ax.plot(x1, y1)
    x1 = [Data_DF['SpineBase'+C1][t], Data_DF['SpineMid'+C1][t], Data_DF['Neck'+C1][t], Data_DF['Head'+C1][t]]
    y1 = [Data_DF['SpineBase'+C2][t], Data_DF['SpineMid'+C2][t], Data_DF['Neck'+C2][t], Data_DF['Head'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1)
    x1 = [Data_DF['ShoulderLeft'+C1][t], Data_DF['Neck'+C1][t], Data_DF['ShoulderRight'+C1][t], Data_DF['SpineBase'+C1][t], Data_DF['ShoulderLeft'+C1][t]]
    y1 = [Data_DF['ShoulderLeft'+C2][t], Data_DF['Neck'+C2][t], Data_DF['ShoulderRight'+C2][t], Data_DF['SpineBase'+C2][t], Data_DF['ShoulderLeft'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1)
    x1 = [Data_DF['HipLeft'+C1][t], Data_DF['KneeLeft'+C1][t], Data_DF['AnkleLeft'+C1][t], Data_DF['FootLeft'+C1][t]]
    y1 = [Data_DF['HipLeft'+C2][t], Data_DF['KneeLeft'+C2][t], Data_DF['AnkleLeft'+C2][t], Data_DF['FootLeft'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1)
    x1 = [Data_DF['HipRight'+C1][t], Data_DF['KneeRight'+C1][t], Data_DF['AnkleRight'+C1][t], Data_DF['FootRight'+C1][t]]
    y1 = [Data_DF['HipRight'+C2][t], Data_DF['KneeRight'+C2][t], Data_DF['AnkleRight'+C2][t], Data_DF['FootRight'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1)
    x1 = [Data_DF['HipRight'+C1][t], Data_DF['SpineBase'+C1][t], Data_DF['HipLeft'+C1][t], Data_DF['HipRight'+C1][t]]
    y1 = [Data_DF['HipRight'+C2][t], Data_DF['SpineBase'+C2][t], Data_DF['HipLeft'+C2][t], Data_DF['HipRight'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1)
    plt.pause(1e-17)
    

def PosturalDrawing2(ax, C1, C2, R1, R2, Labels, Data_DF, t, COLOR, Transparency):
    import matplotlib.pyplot as plt
    ax.set_xlim(R1[0],R1[1])
    ax.set_ylim(R2[0],R2[1])
    ax.set_aspect('equal')
    ax.set_xlabel(Labels[0], fontsize=14)
    ax.set_ylabel(Labels[1], fontsize=14)
    x1 = [Data_DF['ShoulderLeft'+C1][t], Data_DF['ElbowLeft'+C1][t], Data_DF['WristLeft'+C1][t], Data_DF['HandLeft'+C1][t]]
    y1 = [Data_DF['ShoulderLeft'+C2][t], Data_DF['ElbowLeft'+C2][t], Data_DF['WristLeft'+C2][t], Data_DF['HandLeft'+C2][t]]
    # plotting the left hand 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    x1 = [Data_DF['ShoulderRight'+C1][t], Data_DF['ElbowRight'+C1][t], Data_DF['WristRight'+C1][t], Data_DF['HandRight'+C1][t]]
    y1 = [Data_DF['ShoulderRight'+C2][t], Data_DF['ElbowRight'+C2][t], Data_DF['WristRight'+C2][t], Data_DF['HandRight'+C2][t]]
    # plotting the right hand 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    x1 = [Data_DF['SpineBase'+C1][t], Data_DF['SpineMid'+C1][t], Data_DF['Neck'+C1][t], Data_DF['Head'+C1][t]]
    y1 = [Data_DF['SpineBase'+C2][t], Data_DF['SpineMid'+C2][t], Data_DF['Neck'+C2][t], Data_DF['Head'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    x1 = [Data_DF['ShoulderLeft'+C1][t], Data_DF['Neck'+C1][t], Data_DF['ShoulderRight'+C1][t], Data_DF['SpineBase'+C1][t], Data_DF['ShoulderLeft'+C1][t]]
    y1 = [Data_DF['ShoulderLeft'+C2][t], Data_DF['Neck'+C2][t], Data_DF['ShoulderRight'+C2][t], Data_DF['SpineBase'+C2][t], Data_DF['ShoulderLeft'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    x1 = [Data_DF['HipLeft'+C1][t], Data_DF['KneeLeft'+C1][t], Data_DF['AnkleLeft'+C1][t], Data_DF['FootLeft'+C1][t]]
    y1 = [Data_DF['HipLeft'+C2][t], Data_DF['KneeLeft'+C2][t], Data_DF['AnkleLeft'+C2][t], Data_DF['FootLeft'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    x1 = [Data_DF['HipRight'+C1][t], Data_DF['KneeRight'+C1][t], Data_DF['AnkleRight'+C1][t], Data_DF['FootRight'+C1][t]]
    y1 = [Data_DF['HipRight'+C2][t], Data_DF['KneeRight'+C2][t], Data_DF['AnkleRight'+C2][t], Data_DF['FootRight'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    x1 = [Data_DF['HipRight'+C1][t], Data_DF['SpineBase'+C1][t], Data_DF['HipLeft'+C1][t], Data_DF['HipRight'+C1][t]]
    y1 = [Data_DF['HipRight'+C2][t], Data_DF['SpineBase'+C2][t], Data_DF['HipLeft'+C2][t], Data_DF['HipRight'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency, label = 'Frame #'+str(t))
    #ax.legend()
    
    plt.pause(1e-17)
    
def PosturalDrawing3(ax, C1, C2, R1, R2, Labels, Data_DF, t, COLOR, Transparency, SymbProp, MS):
    import matplotlib.pyplot as plt
    ax.set_xlim(R1[0],R1[1])
    ax.set_ylim(R2[0],R2[1])
    ax.set_aspect('equal')
    ax.set_xlabel(Labels[0])
    ax.set_ylabel(Labels[1])
    Counter = 1
    x1 = [Data_DF['ShoulderLeft'+C1][t], Data_DF['ElbowLeft'+C1][t], Data_DF['WristLeft'+C1][t], Data_DF['HandLeft'+C1][t]]
    y1 = [Data_DF['ShoulderLeft'+C2][t], Data_DF['ElbowLeft'+C2][t], Data_DF['WristLeft'+C2][t], Data_DF['HandLeft'+C2][t]]
    # plotting the left hand 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    ax.plot(x1, y1, SymbProp, alpha=Transparency, markersize=MS)
    Shift = [(-0.05,0),(-0.05,0),(-0.05,0),(-0.05,-0.03)]
    for index in range(0, len(x1)):
        XY = x1[index]+Shift[index][0], y1[index]+Shift[index][1]
        ax.annotate(str(Counter), xy=XY, xytext=XY)
        Counter+=1
    x1 = [Data_DF['ShoulderRight'+C1][t], Data_DF['ElbowRight'+C1][t], Data_DF['WristRight'+C1][t], Data_DF['HandRight'+C1][t]]
    y1 = [Data_DF['ShoulderRight'+C2][t], Data_DF['ElbowRight'+C2][t], Data_DF['WristRight'+C2][t], Data_DF['HandRight'+C2][t]]
    # plotting the right hand 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    ax.plot(x1, y1, SymbProp, alpha=Transparency, markersize=MS)
    Shift = [(0.03,0),(0.03,0),(0.03,0),(0,-0.07)]
    for index in range(0, len(x1)):
        XY = x1[index]+Shift[index][0], y1[index]+Shift[index][1]
        ax.annotate(str(Counter), xy=XY, xytext=XY)
        Counter+=1
    x1 = [Data_DF['SpineBase'+C1][t], Data_DF['SpineMid'+C1][t], Data_DF['SpineShoulder'+C1][t], Data_DF['Neck'+C1][t], Data_DF['Head'+C1][t]]
    y1 = [Data_DF['SpineBase'+C2][t], Data_DF['SpineMid'+C2][t], Data_DF['SpineShoulder'+C2][t], Data_DF['Neck'+C2][t], Data_DF['Head'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    ax.plot(x1, y1, SymbProp, alpha=Transparency, markersize=MS)
    Shift = [(0,-0.07),(0.03,0),(0.03,-0.04),(0.04,0),(0.04,0)]
    for index in range(0, len(x1)):
        XY = x1[index]+Shift[index][0], y1[index]+Shift[index][1]
        ax.annotate(str(Counter), xy=XY, xytext=XY)
        Counter+=1
    x1 = [Data_DF['ShoulderLeft'+C1][t], Data_DF['Neck'+C1][t], Data_DF['ShoulderRight'+C1][t], Data_DF['SpineBase'+C1][t], Data_DF['ShoulderLeft'+C1][t]]
    y1 = [Data_DF['ShoulderLeft'+C2][t], Data_DF['Neck'+C2][t], Data_DF['ShoulderRight'+C2][t], Data_DF['SpineBase'+C2][t], Data_DF['ShoulderLeft'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    ax.plot(x1, y1, SymbProp, alpha=Transparency, markersize=MS)

    x1 = [Data_DF['HipLeft'+C1][t], Data_DF['KneeLeft'+C1][t], Data_DF['AnkleLeft'+C1][t], Data_DF['FootLeft'+C1][t]]
    y1 = [Data_DF['HipLeft'+C2][t], Data_DF['KneeLeft'+C2][t], Data_DF['AnkleLeft'+C2][t], Data_DF['FootLeft'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    ax.plot(x1, y1, SymbProp, alpha=Transparency, markersize=MS)
    Shift = [(-0.06,0.02),(-0.07,0),(-0.08,0),(0,-0.10)]
    for index in range(0, len(x1)):
        XY = x1[index]+Shift[index][0], y1[index]+Shift[index][1]
        ax.annotate(str(Counter), xy=XY, xytext=XY)
        Counter+=1
    x1 = [Data_DF['HipRight'+C1][t], Data_DF['KneeRight'+C1][t], Data_DF['AnkleRight'+C1][t], Data_DF['FootRight'+C1][t]]
    y1 = [Data_DF['HipRight'+C2][t], Data_DF['KneeRight'+C2][t], Data_DF['AnkleRight'+C2][t], Data_DF['FootRight'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency)
    ax.plot(x1, y1, SymbProp, alpha=Transparency, markersize=MS)
    Shift = [(0.03,0),(0.03,0),(0.04,0),(0,-0.07)]
    for index in range(0, len(x1)):
        XY = x1[index]+Shift[index][0], y1[index]+Shift[index][1]
        ax.annotate(str(Counter), xy=XY, xytext=XY)
        Counter+=1
    x1 = [Data_DF['HipRight'+C1][t], Data_DF['SpineBase'+C1][t], Data_DF['HipLeft'+C1][t], Data_DF['HipRight'+C1][t]]
    y1 = [Data_DF['HipRight'+C2][t], Data_DF['SpineBase'+C2][t], Data_DF['HipLeft'+C2][t], Data_DF['HipRight'+C2][t]]
    # plotting the Spine, neck and head 
    ax.plot(x1, y1, c=COLOR, linewidth=3, alpha=Transparency, label = 'Frame #'+str(t))
    ax.plot(x1, y1, SymbProp, alpha=Transparency, markersize=MS)
           
    plt.pause(1e-17)