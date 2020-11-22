dataframes = fourier
#Dataframe for final values 
#Finding the 5 biggest peak in the specified range of frequency data
#The for loop runs through a range of heights(prominence) until 5 biggest peak are found
harmonics = []
resistance = np.linspace(10000,10000,400)

for i,s in zip(dataframes, np.arange(200)): #Number of Monte Carlo simulations 
    data = i
    print(s)
    for j in np.linspace(5,10000,1000): #frequency values range 
        for i in range(15,120): #range of heights for peaks 
            x = data.iloc[:int(j),1]
            peaks1, properties = scipy.signal.find_peaks(x, prominence=i, width=1)
            properties["prominences"], properties["widths"]
            if len(peaks1) == 1:
                break
        if len(peaks1) == 1:
                break
    
    #corresponding frequency for first 5 harmonics \
    for i in [2,3,4,5]:
        peaks1 = np.append(peaks1,peaks1[0]*i)
    peaks_1 = peaks1
    for k in [1,2,3,4]:
        interval = []
        for r in np.linspace(peaks_1[k]-5,peaks_1[k]+5,11):
            interval = np.append(interval,data.iloc[int(r),1])
        j = np.argmax(interval)   
        
        p = peaks_1[k]-5+j 
        peaks_1[k] = p
        
    x = data.iloc[:600,1]
    plt.plot(x)
    plt.plot(peaks_1, x[peaks_1], "x")
    #plt.semilogx()
    plt.vlines(x=peaks_1, ymin=x[peaks_1] - properties["prominences"],
                ymax = x[peaks_1], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                xmax=properties["right_ips"], color = "C1")
    plt.show()
        
    frequencies = data.iloc[peaks_1,0]
    VoltageGain_dB = data.iloc[peaks_1,1]
    
    
    #list for first 5 harmonics
    #frequency
    b = [12,resistance[s]] # The zero here is the LABEL given to the data 
    for i in range(len(frequencies)):
        b.append(frequencies.iloc[i])
    for i in range(len(frequencies)):
        b.append(VoltageGain_dB.iloc[i])
    harmonics.append(b)
    
df = pd.DataFrame(harmonics, columns =['label','Resistance',
                                       'Harm_1_freq',
                                       'Harm_2_freq',
                                       'Harm_3_freq',
                                       'Harm_4_freq',
                                       'Harm_5_freq',
                                       'Harm_1_Amplitude_dB',
                                       'Harm_2_Amplitude_dB',
                                       'Harm_3_Amplitude_dB',
                                       'Harm_4_Amplitude_dB',
                                       'Harm_5_Amplitude_dB',
                                      ], dtype = float) 
del df['Resistance']
print(df)
