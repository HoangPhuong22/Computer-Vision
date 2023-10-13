import cv2
import pickle
import numpy as np
import Utils as ut
#Video feed
cap = cv2.VideoCapture('./File/carPark.mp4')
width, height = 107, 48
with open('./File/CarParkPos', 'rb') as f:
    posList = pickle.load(f)

def checkParkingSpace(imagePro):
    spaceCounter = 0
    for pos in posList:
        x,y = pos
        imgCrop = imagePro[y: y+height, x:x+width]
        # cv2.imshow(str(x*y), imgCrop)
        count = cv2.countNonZero(imgCrop)

        if count < 900:
            color = (0, 255, 0)
            thinkness = 5
            spaceCounter+=1
        else:
            color = (0, 0, 255)
            thinkness = 2

        cv2.rectangle(image, pos, (pos[0] + width, pos[1] + height), color, thinkness)
        ut.putTexRect(image, str(count), (x, y + height - 3), scale=1.5, thickness=1, offset=0,
                      colorR=color)
    ut.putTexRect(image, f'Free : {str(spaceCounter)}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20,
                  colorR=(0, 200, 0))
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    succes, image = cap.read()
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(
        imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 16
    )
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    checkParkingSpace(imgDilate)
    cv2.imshow('image', image)
    # cv2.imshow('imageBlur', imgBlur)
    # cv2.imshow('imgageThres', imgMedian)
    key = cv2.waitKey(10)
    if key == ord('p'):
        break