import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time


model = YOLO('yolov8s.pt')
capture = cv2.VideoCapture('input.mp4')

classesFile = open("coco.txt", "r")
fileData = classesFile.read()
classList = fileData.split("\n")

tracker = Tracker()
passingVehicles = set()
lineOnePassingTime = {}
lineTwoPassingTime = {}
distanceBetweenLines = 10   # in meters

frameCount = 0
skipFrames = 4

lineOneY = 323
lineTwoY = 367
offset = 8

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # skipping frames
    frameCount += 1
    if frameCount == skipFrames:
        frameCount = 0
    else:
        continue

    frame = cv2.resize(frame, (1040, 500))

    results = model.predict(frame)

    dataTable = pd.DataFrame(results[0].boxes.data).astype("float")

    coordinates = []

    for index, row in dataTable.iterrows():
        # bound box coordinates
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        # object class
        objectClass = classList[int(row[5])]
        # detecting only vehicles
        if (objectClass in ['car', 'bicycle', 'motorcycle', 'truck', 'bus']):
            coordinates.append([x1, y1, x2, y2])

    boundBoxes = tracker.update(coordinates)

    for boundBox in boundBoxes:
        x3, y3, x4, y4, id = boundBox

        # center of the bound box
        vehicleCenterX = int(x3+x4)//2
        vehicleCenterY = int(y3+y4)//2

        # calculations for vehicles moving down
        if lineOneY < vehicleCenterY+offset and lineOneY > vehicleCenterY-offset:
            lineOnePassingTime[id] = time.time()
        if id in lineOnePassingTime:
            if lineTwoY < vehicleCenterY+offset and lineTwoY > vehicleCenterY-offset:
                elapsedTime = time.time()-lineOnePassingTime[id]
                speedKph = (distanceBetweenLines / elapsedTime) * 3.6

                passingVehicles.add(id)
                cv2.circle(frame, (vehicleCenterX, vehicleCenterY),
                           4, (0, 0, 225), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.putText(frame, f'{int(speedKph)} km/h', (x3, y3),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        # calculations for vehicles moving up
        if lineTwoY < vehicleCenterY+offset and lineTwoY > vehicleCenterY-offset:
            lineTwoPassingTime[id] = time.time()
        if id in lineTwoPassingTime:
            if lineOneY < vehicleCenterY+offset and lineOneY > vehicleCenterY-offset:
                elapsedTime = time.time()-lineTwoPassingTime[id]
                speedKph = (distanceBetweenLines / elapsedTime) * 3.6

                passingVehicles.add(id)
                cv2.circle(frame, (vehicleCenterX, vehicleCenterY),
                           4, (0, 0, 225), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.putText(frame, f'{int(speedKph)} km/h', (x3, y3),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.line(frame, (274, lineOneY), (824, lineOneY), (255, 255, 255), 1)
    cv2.line(frame, (177, lineTwoY), (927, lineTwoY), (255, 255, 255), 1)

    vehicleCount = len(passingVehicles)
    cv2.putText(frame, str(vehicleCount), (50, 100),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
