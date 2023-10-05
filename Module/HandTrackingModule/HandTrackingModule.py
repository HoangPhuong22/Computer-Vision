"""
HandTrackingModule
by : ZeroCoder
Name : Hoang Van Phuong
"""
import cv2
import mediapipe as mp
import math

class HandDetectionModule():
    def __init__(self, mode = False, maxhands = 2, model_complexity = 1, min_detectionCon = 0.5, min_trackingCon = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.model_complexity = model_complexity
        self.min_detection = min_detectionCon
        self.min_tracking = min_trackingCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = mode,
            max_num_hands = maxhands,
            model_complexity = model_complexity,
            min_detection_confidence = min_detectionCon,
            min_tracking_confidence = min_trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
    def findHands(self , image, draw = True, flipType = True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        allHands = []
        height, width = image.shape[:2]
        if self.results.multi_hand_landmarks:
            #for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
            for i in range(len(self.results.multi_handedness)):
                handType = self.results.multi_handedness[i]
                handLms = self.results.multi_hand_landmarks[i]
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    px, py = int(lm.x * width), int(lm.y * height)
                    cv2.circle(image, (px, py), 3, (255, 215, 0), -1)
                    mylmList.append([px, py])
                    xList.append(px)
                    yList.append(py)
                #find box
                xMin, xMax = min(xList), max(xList)
                yMin, yMax = min(yList), max(yList)
                boxH, boxW = yMax - yMin, xMax - xMin
                box = xMin, yMin, boxW, boxH
                #center box
                cx, cy = box[0] + (box[2] // 2), box[1] + (box[3] // 2)
                myHand['lmList'] = mylmList
                myHand['box'] = box
                myHand['center'] = (cx, cy)
                
                if flipType:
                    if handType.classification[0].label == 'Right':
                        myHand['type'] = 'Left'
                    else : myHand['type'] = 'Right'
                else:
                    myHand['type'] = handType.classification[0].label
                allHands.append(myHand)
                #draw
                if draw : 
                    # self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(image, (box[0] - 20, box[1] - 20), (box[0] + box[2] + 20, box[1] + box[3] + 20), (91, 189, 43), 2)
                    cv2.putText(image, myHand['type'], (box[0] - 20, box[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (91, 189, 43), 2)
        if draw:
            return allHands, image
        else : return allHands
    def countFingers(self, myHand):
        lmList = myHand['lmList']
        count_ones, count_zeros = 0, 0
        if self.results.multi_hand_landmarks:
            if lmList[self.tipIds[0]][0] > lmList[self.tipIds[4]][0]:
                if lmList[self.tipIds[0]][0] > lmList[self.tipIds[0] - 1][0]:
                    count_ones += 1
                else: count_zeros += 1
            else : 
                if lmList[self.tipIds[0]][0] < lmList[self.tipIds[0] - 1][0]:
                    count_ones += 1
                else: count_zeros += 1
            for index in range(1 , 5):
                if lmList[self.tipIds[index]][1] > lmList[self.tipIds[index] - 2][1]:
                    count_zeros += 1
                else : count_ones += 1
        return (count_zeros, count_ones)
    def findDistances(self, p_ones, p_two, image = None):
        x1, y1 = p_ones
        x2, y2 = p_two
        cx, cy = (x1 + x2)//2 , (y1 + y2)//2
        length = math.hypot(x1 - x2, y1 - y2)
        info = (x1, y1, x2, y2, cx, cy)
        if image is not None:
            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, image
        else:
            return length, info 
    def drawFinger(self, image, hands, img_x, img_y, number):
        finger = 0
        for hand in hands:
            finger += self.countFingers(hand)[number]
        cv2.putText(image, f'Finger : {finger}',(img_x, img_y),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return image
          
def main():
    
    cap = cv2.VideoCapture(0)
    detector = HandDetectionModule(min_detectionCon = 0.8, maxhands = 3)
    while True:
        success, image = cap.read()
        if not success:
            break
        height, width = image.shape[:2]
        hands, image = detector.findHands(image, draw = True, flipType = True)
        image = detector.drawFinger(image, hands, 10, height - 30, 1)
        cv2.imshow('HandtrackingModule by ZeroCoder', image)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()