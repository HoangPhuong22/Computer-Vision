import cv2
import mediapipe as mp
import math

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5, model = 0):
        self.minDetectionCon = minDetectionCon
        self.model = model
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            min_detection_confidence = self.minDetectionCon,
            model_selection= self.model
        )
        self.results = None
    def findFaceDetection(self, image, draw = True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imageRGB)
        inFo = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                pointbox = {}
                ih, iw = image.shape[:2]
                boxinfo = detection.location_data.relative_bounding_box
                x, y, width, height = int(iw * boxinfo.xmin), int(ih * boxinfo.ymin), int(iw * boxinfo.width), int(ih * boxinfo.height)
                box = (x, y, width, height)
                cx, cy = box[0] + box[2]//2 , box[1] + box[3]//2
                lmList = []
                for landmark in detection.location_data.relative_keypoints:
                    xx, yy = int(landmark.x * iw), int(landmark.y * ih)
                    lmList.append([xx, yy])
                pointbox = {'id' : id, 'box': box, 'center': (cx, cy), 'lmList' : lmList}
                inFo.append(pointbox)

                if draw:
                    cv2.rectangle(image, box, (255, 0, 255), 2)
                    cv2.putText(image, f'{int(detection.score[0] * 100)}%', (box[0], box[1] - 20), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255, 0, 255), 2)
        return inFo, image

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        success, image = cap.read()
        lmList, image = detector.findFaceDetection(image, draw = True)
        print(lmList)
        cv2.imshow('Zero Coder', image)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()