import cv2
import mediapipe as mp
import math

class HandTracking():
    def __init__(self, mode = False, maxHand = 1, comPlexity = 1, detecterCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.comPlexity = comPlexity
        self.detecterCon = detecterCon
        self.trackingCon = trackingCon
        self.mpHands = mp.solutions.hands
        self.Hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHand,
            model_complexity= self.comPlexity,
            min_detection_confidence= self.detecterCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw =mp.solutions.drawing_utils
        self.results = None
    def findHands(self, image, draw = True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.Hands.process(imageRGB)
        height, width  = image.shape[:2]
        lmList = []
        xList = []
        yList = []
        bbox = []
        if self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.multi_hand_landmarks[0], self.mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                px, py = int(lm.x * width), int(lm.y * height)
                lmList.append([px, py])
                xList.append(px)
                yList.append(py)
                if draw :
                    cv2.circle(image, (px, py), 3, (255, 0, 255), -1)
            x, y = min(xList), min(yList)
            w, h = max(xList) - x ,  max(yList) - y
            bbox= [x, y , w, h]
        return image, bbox, lmList

    def findPosition(self, image, p1, p2, draw = True):
        x1, y1 = p1
        x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)
        if draw:
            cv2.circle(image, (x1, y1), 3, (255, 255, 0), -1)
            cv2.circle(image, (x2, y2), 3, (255, 255, 0), -1)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 3)
            return image, length
        else: return length