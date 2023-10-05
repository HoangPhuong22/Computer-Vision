import Hands_Tracking_Module as htm
import Utils
import cv2
from Rectangle_update import Rectangle
import numpy as np

detecter = htm.HanTracker(detectionCon= 0.8)
cap = cv2.VideoCapture(0)

#Set height(4), width(3)
cap.set(3, 1280)
cap.set(4, 720)

#Biến màu
colorR = (255, 0, 0)

# Tạo 5 hình chữ nhật lên màn hình
recList = []
for i in range(5):
    recList.append(Rectangle(center=[250 * i + 150 , 150]))


while True:
    succes, image = cap.read()

    image = cv2.flip(image, 1) # Lật ngược camera

    image, bbox, lmList = detecter.findHands(image, draw = True)
    if lmList:
        image, length = detecter.finDistance(lmList[8], lmList[12], image)
        if length < 40:
            cursor = lmList[8]
            for i in recList:
                i.update(cursor) # Update vị trí center của hình chữ nhật

        #Box
        x, y, w, h = bbox[1] # vẽ box cho bàn tay
        image = Utils.connerReact(image, (x - 20, y - 20, w + 40, h + 40))
        print(length)

    # Draw
    #Xử lí làm mờ
    imgNew = np.zeros_like(image, np.uint8)
    for rec in recList:
        cx, cy = rec.getCenter()
        w_rec, h_rec = rec.getSize()
        cv2.rectangle(imgNew, (cx - w_rec//2, cy - h_rec//2), (cx + w_rec//2, cy + h_rec // 2), colorR, -1)
        Utils.connerReact(imgNew, (cx - w_rec//2, cy - h_rec//2, w_rec, h_rec))
    out = image.copy()
    alpha = 0.1
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(image, alpha, imgNew, 1 - alpha, 0)[mask]

    #Show
    cv2.imshow('ZeroCoder', out)

    #Out
    if cv2.waitKey(1) & 0xFF == ord('p'):
        break
cap.release()
cv2.destroyAllWindows()