import os
import cv2
import statistics
import torch
import pyrealsense2.pyrealsense2 as rs
import time
import argparse
import struct
from turtle import color
import numpy as np

matplotlib.use('TKAgg')
# Disable tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ColorDetect:
    def __init__(self):
        self.blueLow = np.array([62, 43, 140])
        self.blueUp = np.array([110, 255, 255])
        self.redLow = np.array([0,7,232])
        self.redUp = np.array([55, 255, 255])
        

    def checkColor(self, colorFrame):
        hsv = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2HSV)
        blueMask = cv2.inRange(hsv, self.blueLow, self.blueUp)
        redMask = cv2.inRange(hsv, self.redLow, self.redUp)
        redContours, _ = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blueContours, _ = cv2.findContours(blueMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(redContours) > 0 and len(blueContours) > 0:
            redA = 0
            blueA = 0
            for c in redContours:
               redA += cv2.contourArea(c)
            for c in blueContours:
                blueA += cv2.contourArea(c)
            if redA > blueA:
                return 'r'
            else:
                return 'b'
        elif len(redContours) > 0:
            return 'r'
        return 'b'

class Capture:
    def __init__(self, dc=None, cameraIndex=0, isRealsense=True):
        # Check if realsense class depth camera object is passed or an integer for the index of a regular camera
        self.cap = cv2.VideoCapture("vid.mp4")
        self.loadModel()
        self.robotList = []
        self.colorDetected = ColorDetect()

    def loadModel(self):
        # or yolov5m, yolov5l, yolov5x, custom
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Algorithm/pt_files/best.pt')


    def processFrame(self, colorImage, debug=False, display=False):
        confThres = 0.25  # Confidence threshold
        # Get bounding boxes
        results = self.model(colorImage)

        # Post process bounding boxes
        rows = results.pandas().xyxy[0].to_numpy()

        detectionsRows = results.pandas().xyxy
        # Go through all detections

        for i in range(len(rows)):
            if len(rows) > 0:
                # Get the bounding box of the first object (most confident)
                xMin, yMin, xMax, yMax, conf, cls, label = rows[i]

                if debug:
                    print("({},{}) \n\n\n ({},{})".format(xMin, yMin, xMax, yMax))

                if display and self.colorDetected.checkColor(colorImage) == "b":
                    bbox = [xMin, yMin, xMax, yMax]
                    cv2.rectangle(colorImage, (int(xMin), int(yMin)), (int(xMax), int(yMax)), (0, 255, 0), 2)  # Draw with green color

                    # Display the label with the confidence
                    labelConf = label + " " + str(conf)
                    cv2.putText(colorImage, labelConf, (int(xMin), int(yMin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('RealSense', colorImage)
        cv2.waitKey(1)


    def capturePipeline(self, debug=False, display=False):
        while True:
            # Get frame from camera
            try:
               ret, colorImage = self.cap.read()
            except:
                print("Error getting frame")

            if ret:
                key = cv2.waitKey(1)
                if key == 27:
                    break

                # Frame is valid
                self.processFrame(colorImage=colorImage, debug=debug, display=display)


captureStream = Capture()
captureStream.capturePipeline(debug=True, display=True)