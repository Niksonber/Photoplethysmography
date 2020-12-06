import numpy as np 
import cv2
import time

from filter import cutFreqFilter
from ICA import ICA
from ROIDetection import HaarDetection

class HRMeasure:
    def __init__(self, videoCapture=0):
        self.__capture = cv2.VideoCapture(videoCapture)
        self.__capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.roidetector = HaarDetection()
        self.ica = ICA()
        self.filter = cutFreqFilter(fs=30/4)
        
        self.im = None
        self.s = []
        self.rate = None

    def getNewImage(self):
        r, self.im = self.__capture.read()
        if r: self.im = cv2.GaussianBlur(self.im, (7,7), 3)
        return r

    def loop(self):
        l = time.time()
        while self.__capture.isOpened():
            if(time.time()-l < 4/30):
                continue
            #print("time {}".format(time.time()-l))
            l = time.time()
            if(self.getNewImage()):
                roi = self.roidetector.detect(self.im)
                if roi !=None:
                    roi = roi[0]
                    self.s.append(roi.mean(axis=0).mean(axis=0).tolist())
                    if len(self.s) >= 30/4* 6:
                        s = np.array(self.s[1:]).T
                        i = self.ica.run(s)
                        f = self.filter.filter(s)
                        m = self.filter.maxFreq(f)#[1]
                        #print(self.filter.maxFreq(f))
                        print("Heart rate: {}\r".format(m*60), end='')
                        self.rate = m
                        self.s = self.s[1:]

                cv2.imshow('im1', self.im)
                if cv2.waitKey(1) == ord('q'):
                    break


if __name__== "__main__":
    hrm = HRMeasure()
    hrm.loop()