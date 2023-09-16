import cv2
import pandas as pd
from ultralytics import YOLO


model = YOLO('yolov8s.pt')
capture = cv2.VideoCapture('input.mp4')

classesFile = open("coco.txt", "r")
fileData = classesFile.read()
classList = fileData.split("\n")

while True:
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1040, 500))

    results = model.predict(frame)

    dataTable = pd.DataFrame(results[0].boxes.data).astype("float")

    for index, row in dataTable.iterrows():
        # bound box coordinates
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        # object class
        objectClass = classList[int(row[5])]
        # displaying bound box and object class
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, str(objectClass), (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
