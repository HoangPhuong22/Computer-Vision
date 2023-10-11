import cv2
import HandTrackingModule as htm
import Utils as ut
import math
import numpy as np

#Video
cap = cv2.VideoCapture(0)
Detecter = htm.HandTracking(detecterCon = 0.8)
cap.set(3, 1280)
cap.set(4, 960)
#Find Function
#x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2) # y = Ax^2 + Bx + C

while True:
    success, image = cap.read()
    image, bbox, lmList = Detecter.findHands(image, draw = True)
    # draw box
    if bbox:
        image = ut.connerReact(image, bbox)
    #lmList
    if lmList:
        x1, y1 = lmList[5]
        x2, y2 = lmList[17]
        xt, yt = bbox[0], bbox[1]
        print(xt, yt)
        distance = int(math.sqrt((y2 - y1)**2 + (x2 - x1) ** 2))
        A, B, C = coff
        distanceCM = A*distance**2 + B*distance + C
        image = ut.putTexRect(image, f'{int(distanceCM)} cm', (xt - 20, yt - 20))
        print(distanceCM, distance)

    cv2.imshow('ZeroCoder', image)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()