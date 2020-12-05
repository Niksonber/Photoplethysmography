"""
Based in https://github.com/akcarsten/Independent_Component_Analysis

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

""" classe para ICA"""
class ICA:
    def __init__(self, thresold=1e-10, iterations=1000):
        self.thresold  = thresold
        self.iterations = iterations

    """ normalize signal to mean 0 """
    def removeBias(self, s):
        m = np.mean(s, axis=1).reshape(s.shape[0],1)
        return s-m, m

    """ normalize the independent componentes of signal to variance 1 """
    def whiten(self, s):
        cov = np.cov(s)
        P, D, Pm1 = np.linalg.svd(cov) # A = PDP^-1

        # normalize signal by x/sqrt(sigma), where sigma is variance
        # with remove bias, we have (x-m)/sqrt(sigma)
        wM = np.dot(P, np.dot(np.diag(1.0/np.sqrt(D)),P.T))  # P D^(-1/2) P^T (note D is diagonal matrix)
        x = np.dot(wM, s)

        return x, wM

    def __g(self, x):
        return np.tanh(x)

    def __gp(self, x):
        return 1 - np.tanh(x)**2

    def preprocess(self, s):
        x, _ = self.removeBias(s)
        x, _ = self.whiten(x)
        return x

    def process(self, s):
        n_signals = s.shape[0]
        # initialize weights
        W = 2*np.random.random((n_signals,n_signals))-1
        for c in range(n_signals):
            #w = W[c, :].copy()
            w = W[c, :].copy().reshape(n_signals, 1)
            # normalize
            w = w / np.linalg.norm(w)

            for i in range(self.iterations):
                # Update weights
                #wNew = (s*self.__g(np.dot(w,s))).mean(axis=1) - self.__gp(np.dot(w,s)).mean(axis=1)*w
                wNew = (s * self.__g(np.dot(w.T, s))).mean(axis=1) - self.__gp(np.dot(w.T, s)).mean() * w.squeeze()
                # Decorrelate weights
                if c>0:
                    wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
                wNew = wNew / np.linalg.norm(wNew)
                # Calculate limit condition
                distance = np.abs(np.abs((wNew * w).sum()) - 1)
                if distance < self.thresold :
                    break

                w = wNew

            W[c, :] = w.T

        x = np.dot(W,s)

        return x, W

    def run(self, s):
        x = self.preprocess(s)
        x, _ = self.process(x)
        return x
  

if __name__=="__main__":
    t = np.linspace(0,20,100)
    xi = np.array([np.sin(t),
                  signal.sawtooth(2.12*t),])

    M =  np.array([[0.5, 1],
                   [1, 0.4]])

    s = np.dot(M, xi)

    ica = ICA()
    x = ica.run(s)


    fig, axs = plt.subplots(4,2)
    axs[0,0].plot(xi[0])
    axs[0,1].plot(xi[1])
    axs[1,0].plot(s[0,:])
    axs[1,1].plot(s[1,:])
    axs[2,0].plot(x[0,:])
    axs[2,1].plot(x[1,:])


    ica2 = FastICA(n_components=2)
    S_ = ica2.fit_transform(s.T).T

    axs[3,0].plot(S_[0,:])
    axs[3,1].plot(S_[1,:])
    plt.show()