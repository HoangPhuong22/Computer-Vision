import cv2
import mediapipe as mp

class Hantracker():
    def __init__(self, mode = False, maxHands = 1, detectionCon = 0.5, modelComplexity = 1, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplexity = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
    def findHands(self, image, draw = True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        
        if self.results.multi_hand_landmarks:
            for handDraw in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handDraw, self.mpHands.HAND_CONNECTIONS)
                    
        return image
    def findPosition(self, image, handNo = 0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(hands.landmark):
                height, width, digital_image = image.shape
                lmx, lmy = int(lm.x*width), int(lm.y*height)
                lmList.append([id, lmx, lmy])
                if draw:
                    cv2.circle(image, (lmx, lmy), 5, (0, 238, 218), -1)
        return lmList