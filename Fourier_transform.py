import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import control

fourier = list(0 for i in range(0,200)) 
for j in range(200):
    data_no = dataframes[j]
    data = data_no.iloc[20000:,:]
    data1 = []
    for i in range(len(data.iloc[:,1])):
        data1.append(data.iloc[i,1])

    t = np.linspace(0,-float(data.iloc[0,0])+float(data.iloc[-1,0]),len(data1))
    s = data1

    fft = np.fft.fft(s)
    T = t[1] - t[0]  # sampling interval 
    N = len(data1)

    # 1/T = frequency
    f = np.linspace(0, 1 / T, N)
    x1 = f[:N // 2]
    y1 = control.mag2db(np.abs(fft)[:N // 2] * 1 / N)
    x = x1[:500]
    y = y1[:500]
    plt.figure(figsize=(15,4))
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.plot(x,y)
    print(j)
    plt.show()
    
    d = {'Frequency':x,'Amplitude':y}
    data12 = pd.DataFrame(d)
    fourier[j] = data12
