import SelfiSegmentationModule as sg
import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = sg.SelfieSegmentation(model = 0)

listImg = os.listdir("./Images")
imgList = []
print(listImg)
for imgPath in listImg:
    img = cv2.imread(f"./Images/{imgPath}")
    imgList.append(img)
print(len(imgList))
indexImg = 0
while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    imageOut = segmentor.removeBG(image,imgList[indexImg],cutThreshold=0.8)

    cv2.imshow('image', imageOut)

    key = cv2.waitKey(1)
    size = len(imgList)
    if key == ord('p'):
        break
    elif key == ord('d'):
        indexImg += 1
        indexImg = (indexImg%size + size)%size
    elif key == ord('a'):
        indexImg -= 1
        indexImg = (indexImg % size + size) % size

cap.release()
cv2.destroyAllWindows()