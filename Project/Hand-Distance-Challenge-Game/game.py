import random
import cv2
import HandTrackingModule as htm
import Utils as ut
import math
import numpy as np
import time




#Video
cap = cv2.VideoCapture(0)
Detecter = htm.HandTracking(detecterCon = 0.8)
cap.set(3, 1280)
cap.set(4, 960)
#Find Function
#x is the raw distance y is the value in cm
X = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
Y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(X, Y, 2) # y = Ax^2 + Bx + C



#Game variable
cx, cy = 250, 250
color = (255, 0, 255)
counter = 0
#Score
score = 0

#Time
timeStart = time.time()
totalTime = 20

#loop
while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if time.time() - timeStart < totalTime:
        image, bbox, lmList = Detecter.findHands(image, draw = True)
        # draw box
        if bbox:
            image = ut.connerReact(image, bbox)
        #lmList
        if lmList:
            x, y, w, h = bbox
            x1, y1 = lmList[5]
            x2, y2 = lmList[17]
            xt, yt = bbox[0], bbox[1]

            distance = int(math.sqrt((y2 - y1)**2 + (x2 - x1) ** 2))
            A, B, C = coff
            distanceCM = A*distance**2 + B*distance + C
            image = ut.putTexRect(image, f'{int(distanceCM)} cm', (xt - 20, yt - 20))

            if distanceCM < 40:
                if x < cx < x + w and y < cy < y + h:
                    color = (0, 255, 0)
                    counter = 1
            else : color = (255, 0, 255)


        if counter:
            counter += 1
            color = (0, 255, 0)
            if counter == 3:
                cx = random.randint(100, 1100)
                cy = random.randint(100, 600)
                color = (255, 0, 255)
                counter = 0
                score += 1

        #draw button

        cv2.circle(image, (cx, cy),30, color, -1)
        cv2.circle(image, (cx, cy), 10, (255, 255, 255), -1)
        cv2.circle(image, (cx, cy), 20, (255, 255, 255), 2)
        cv2.circle(image, (cx, cy), 30, (50, 50, 50), 1)

        #Game HUB

        ut.putTexRect(image, f'Time: {int(totalTime - (time.time() - timeStart))}',
                      (1000, 75), scale = 3, offset=20)
        ut.putTexRect(image, f'Score: {str(score).zfill(2)}', (50, 75), scale=3, offset=20)

    else:
        ut.putTexRect(image, 'Game Over', (400, 300), scale=5, offset=30, thickness=7)
        ut.putTexRect(image, f'Your Score: {score}', (440, 400), scale=3, offset=20)
        ut.putTexRect(image, 'Press R to restart or P to Out', (270, 475), scale=3, offset=20)

    #Show
    cv2.imshow('ZeroCoder', image)
    key = cv2.waitKey(1)
    if key == ord('r'):
        timeStart = time.time()
        score = 0
    elif key == ord('p'):
        break
cap.release()
cv2.destroyAllWindows()