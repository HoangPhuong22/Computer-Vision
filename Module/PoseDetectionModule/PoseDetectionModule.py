import cv2
import mediapipe as mp
import math

class PoseDetection():
    def __init__(self, mode=False,smoothlm = True, enable = False, smoothsg = True, detectorCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.smoothlm = smoothlm
        self.enable = enable
        self.smoothsg = smoothsg
        self.detectorCon = detectorCon
        self.trackingCon = trackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode = self.mode,
            smooth_landmarks = self.smoothlm,
            enable_segmentation = self.enable,
            smooth_segmentation= self.smoothsg,
            min_detection_confidence = self.detectorCon,
            min_tracking_confidence = self.trackingCon
        )

    def findPose(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return image
    def finPosition(self, image, draw = True, boxWithHands = True):
        lmList = []
        boxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width = image.shape[:2]
                x, y = int(lm.x * width) , int(lm.y * height)
                lmList.append([id, x , y])

            distance = abs(lmList[12][1] - lmList[11][1]) // 2

            if boxWithHands:
                x1 = lmList[16][1] - distance
                x2 = lmList[15][1] + distance
            else:
                x1 = lmList[12][1] - distance
                x2 = lmList[11][1] - distance
            y1 = lmList[1][2] - distance
            y2 = lmList[30][2] + distance

            boxinfo = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            boxInfo = {"box" : boxinfo, "center":(cx, cy)}

            if draw:
                cv2.rectangle(image, (boxinfo[0], boxinfo[1]), (boxinfo[0] + boxinfo[2], boxinfo[1] + boxinfo[3]), (255, 0, 0), 2)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
        return image, lmList, boxInfo

def main():
    height = 640
    width = 480
    # cap = cv2.VideoCapture(0) #WEBCAM
    cap = cv2.VideoCapture('D:\Computer-Vision\Video-Test\M10.mp4')
    poseDetector = PoseDetection()
    while True:
        success, image = cap.read()
        if not success:
            break
        image = poseDetector.findPose(image)
        image, lmList, boxInfo = poseDetector.finPosition(image)
        cv2.imshow('Pose detector by Zero Coder', image)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
