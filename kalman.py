import cv2 as cv
import numpy as np
import sys

# Instantiate OCV kalman filter
class KalmanFilter:
    def __init__(self):
        self.kf = cv.KalmanFilter(4, 2)
        # self.kf.measurementMatrix = 1. * np.ones((17, 2))
        # self.kf.transitionMatrix = 1. * np.ones((17, 4))
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def update(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted
    
    # def update(self, x, y):
    #     measured = np.array([[np.float32(x)], [np.float32(y)]])
    #     self.kf.correct(measured)