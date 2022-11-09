import cv2
import statistics
import torch
import pyrealsense2.pyrealsense2 as rs
import numpy as np

class DepthCamera:

    # Constructor
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.loadModel()
        
        # Get device product line for setting a supporting resolution
        pipelineWrapper = rs.pipelineWrapper(self.pipeline)
        pipelineProfile = config.resolve(pipelineWrapper)
        device = pipelineProfile.get_device()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        depthSensor = device.query_sensors()[0]
        depthSensor.set_option(rs.option.laser_power, 0)

        # Start streaming
        self.pipeline.start(config)

    def loadModel(self): 
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./Algorithm/pt_files/best.pt')

    # Get Depth and Color Frame
    def getFrame(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except:
            return False, None, None
        depthFrame = frames.get_depth_frame()
        colorFrame = frames.get_color_frame()

        depthImage = np.asanyarray(depthFrame.get_data())
        colorImage = np.asanyarray(colorFrame.get_data())

        if not depthFrame or not colorFrame:
            return False, None, None

        return True, depthImage, colorImage

    def release(self):
        self.pipeline.stop()


    def getCoordinates(self, frame, model, depthFrame):   
        results = self.model(frame)
       
        detectRows = results.pandas().xyxy
        for i in range(len(detectRows)):
            rows = rows[i].to_numpy()

        if len(rows) != 0:
            values = []
            for i in range(len(rows)):
                xMin, yMin, xMax, yMax = rows[i]
            for x in range(xMin - 1, xMax):
                for y in range(yMin - 1, yMax):
                    values.append(depthFrame[y, x])

            med = statistics.median(values)
            return (xMin, yMin, xMax, yMax, med)
        return None

    def show_frame(self,colorFrame, depthFrame, depth, coordinates):
        # Display Text for distance
        if coordinates != None:
            cv2.putText(colorFrame,"Median: {}mm".format(depth), (coordinates[0], coordinates[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))

            # Display Rectangle overlay
            cv2.rectangle(colorFrame, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 0, 255), 10)
            cv2.rectangle(depthFrame, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 0, 255), 10)

        # Show Both
        cv2.imshow("Video", colorFrame)
        cv2.imshow("Video_Depth", depthFrame)

    def offsetCalc(self, xCoord, yCoord, xRes, yRes):
        yCoord = yRes-yCoord
        centerx, centery = xRes/2.0, yRes/2.0

        moveX= xCoord-centerx
        moveY = yCoord-centery
        if(moveX != 0):
            moveX /= centerx
        if(moveY != 0):
            moveY /= centery

        return(moveX, moveX)



cam = DepthCamera()
while True:
    # Start Video Capture
    try:
        ret, depthFrame, colorFrame = cam.getFrame()
    except:
        print("Error getting frame")

    # If frame is not empty
    if ret:
        key = cv2.waitKey(1)
        if key == 27:
            break
        # Get coordinates from color frame
        try:
            coordinates = cam.getCoordinates(colorFrame, cam.model, depthFrame)
        except:
            print("Error getting coordinates\n")

        if coordinates != None:
            #print("In: ", coordinates)
            cam.show_frame(colorFrame, depthFrame, coordinates[4], coordinates)

           # try:
            finalCords = cam.offsetCalc((coordinates[0]+coordinates[2])/2, (coordinates[1]+coordinates[3])/2, 640, 480)
            print("Offset: ", finalCords[0] ,' , ', finalCords[1])
            