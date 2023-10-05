import cv2
import mediapipe as mp

class FaceMeshDetector():
    def __init__(self, mode = False, maxFace = 1, refineLm = False, minDetectionCon = 0.5, minTrackingCon = 0.5):
        self.mode = mode
        self.maxFace = maxFace
        self.refineLm = refineLm
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.FaceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode= self.mode,
            max_num_faces= self.maxFace,
            refine_landmarks=self.refineLm,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence= self.minTrackingCon
        )
        self.DrawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius= 2)
        self.results = None
    def findFaceMesh(self, image, draw = True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.FaceMesh.process(imageRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for face in self.results.multi_face_landmarks:
                myList = []
                if draw:
                    self.mpDraw.draw_landmarks(image, face, self.mpFaceMesh.FACEMESH_CONTOURS, self.DrawSpec, self.DrawSpec)
                for id, lm in enumerate(face.landmark):
                    ih, iw = image.shape[:2]
                    x, y = int(iw * lm.x) , int(ih * lm.y)
                    myList.append([x, y])
                faces.append(myList)

        return image, faces

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFace=2)
    while True:
        success, image = cap.read()
        image, faces = detector.findFaceMesh(image)
        if faces:
            print(faces[0])
        cv2.imshow("ZeroCoder", image)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break


if __name__ == "__main__":
    main()