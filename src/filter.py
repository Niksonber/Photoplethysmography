import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks 

import matplotlib.pyplot as plt

class cutFreqFilter:
    def __init__(self, fs = 30, fmin=30/60.0, fmax=240/60.0):
        self.fs = fs
        self.fmin = fmin
        self.fmax = fmax
        self.n = None
        self.lastMax = None

    def filter(self, x):
        self.n = x.size
        yFFT = fft(x)/self.n
        idxMin = int(self.fmin*self.n/self.fs)
        idxMax = int(self.fmax*self.n/self.fs)
        # plt.plot(yFFT[1, idxMin:idxMax])
        # plt.show()
        return yFFT[:, idxMin:idxMax]

    def maxFreq(self, x):
        y = x.copy()
        idx = find_peaks(y[1])[0]
        m = idx * self.fs/self.n + 30.0/60
        if self.lastMax is None:
            self.lastMax = np.argmax(y[1]) * self.fs/self.n + 30.0/60
        else:
            i = np.argmin(np.abs(m-self.lastMax))
            if np.abs(m[i]-self.lastMax) < 20/60:
                self.lastMax = m[i]
        return self.lastMax

    