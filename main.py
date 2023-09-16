import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


model = YOLO('yolov8s.pt')
capture = cv2.VideoCapture('input.mp4')

classesFile = open("coco.txt", "r")
fileData = classesFile.read()
classList = fileData.split("\n")

area = [(270, 238), (294, 280), (592, 226), (552, 207)]
tracker = Tracker()
passingVehicles = set()

while True:
    ret, frame = capture.read()
    if not ret:
        break
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
        cx = int(x3+x4)//2
        cy = int(y3+y4)//2

        # checking for the bound box is within the area
        results = cv2.pointPolygonTest(
            np.array(area, np.int32), ((cx, cy)), False)

        if results >= 0:
            passingVehicles.add(id)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 225), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
            cv2.putText(frame, str(id), (x3, y3),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    vehicleCount = len(passingVehicles)
    cv2.putText(frame, str(vehicleCount), (50, 100),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 255, 0), 3)
    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
