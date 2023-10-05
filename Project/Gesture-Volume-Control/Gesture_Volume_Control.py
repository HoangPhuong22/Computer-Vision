import cv2
import time
import numpy as np
import Hands_Tracking_Module as hml
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detecter = hml.Hantracker(detectionCon= 0.5)
ptime = 0

########################################################################
#Volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)

max_volume, min_volume = volume.GetVolumeRange()[1], volume.GetVolumeRange()[0]
##########################################################################
volper = 400
pt = 0
while True:
    success, image = cap.read()
    if not success:
        break
    image = detecter.findHands(image)
    lmList = detecter.findPosition(image)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        c1, c2 = (x1 + x2)//2, (y1 + y2)//2
        cv2.line(image, (x1, y1), (x2, y2), (0, 238, 218), 3)
        cv2.circle(image, (x1, y1), 10, (51, 255 , 51), -1)
        cv2.circle(image, (x2, y2), 10, (51, 255 , 51), -1)
        cv2.circle(image, (c1, c2), 10, (51, 255, 51), -1)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 15 : 
            cv2.circle(image, (c1, c2), 10, (0, 238, 218), -1)
        #set volume
        vol = np.interp(length, [15, 200], [min_volume, max_volume])
        volper = np.interp(length, [15, 200], [400, 150])
        pt = np.interp(length, [15, 200], [0, 100])
        volume.SetMasterVolumeLevel(vol, None)
    
    cv2.rectangle(image,(10, 150), (110, 400),(255, 128, 0), 2)
    cv2.rectangle(image,(10, int(volper)), (110, 400),(255, 128, 0), -1)
    cv2.putText(image,f'{int(pt)} %' , (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 0), 2)
    #FPS
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(image,f'FPS : {int(fps)}' , (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 128, 0), 2)
    cv2.putText(image,f'ZeroCoder' , (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 204, 153), 2)
    cv2.imshow('ZeroCoder', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()