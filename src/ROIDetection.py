import os
import numpy as np 
import cv2
from abc import ABC, abstractmethod
from scipy.fft import fft

cv2Path = os.path.dirname(os.path.abspath(cv2.__file__))

class ROIDetection(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, im):
        pass

class HaarDetection(ROIDetection):
    def __init__(self):
        super().__init__()
        self.faceCascade = cv2.CascadeClassifier(cv2Path + '/data/haarcascade_frontalface_default.xml')
        self.eyeCascade = cv2.CascadeClassifier(cv2Path + '/data/haarcascade_eye.xml')

    def detect(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            im = cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            faceGray = gray[y:y+h, x:x+w]
            faceColor = im[y:y+h, x:x+w]
            eyes = self.eyeCascade.detectMultiScale(faceGray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(faceColor,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            maxY = max(eyes[0][1], eyes[1][1])
            roiForehead = faceColor[maxY:maxY-eyes[0][3]][eyes[0][0]+eyes[0][2]//2:eyes[1][0]+eyes[1][2]//2]
            cv2.rectangle(faceColor,(eyes[0][0]+eyes[0][2]//2,maxY),(eyes[1][0]+eyes[1][2]//2,maxY-eyes[0][3]),(0,0,255),2)

            roiNose= faceColor[maxY : maxY+eyes[0][3]][eyes[0][0]+eyes[0][2] : eyes[1][0]]
            cv2.rectangle(faceColor,(eyes[0][0],maxY),(eyes[1][0]+eyes[0][2], maxY+eyes[0][3]),(255,0,255),2)
            
            angle = np.arctan2(eyes[0][1]- eyes[1][1], eyes[0][0]- eyes[1][0])
            M = cv2.getRotationMatrix2D((im.shape[0]//2, im.shape[1]//2), angle*180.0/np.pi, 1) 
            rotated = cv2.warpAffine(im, M, (im.shape[0], im.shape[1])) 
            
            cv2.imshow('img',im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return (roiForehead, roiNose)

        

        


if __name__ == "__main__":
    im = cv2.imread('data/ob.jpeg')
    h = HaarDetection()
    h.detect(im)