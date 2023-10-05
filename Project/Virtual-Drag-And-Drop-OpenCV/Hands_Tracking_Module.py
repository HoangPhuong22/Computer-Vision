import cv2
import mediapipe as mp
import math

class HanTracker():
    def __init__(self, mode = False, maxhands = 1, modelComplexity = 1, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode, # Xác định chế độ của hình ảnh. True là hình ảnh tĩnh, False là hình ảnh thực
            max_num_hands =  self.maxhands, # Số bàn tay có thể xử lý
            model_complexity = self.modelComplexity, # Độ phức tạp của mô hình
            min_detection_confidence = self.detectionCon, # Độ tin cậy hệ thống phát hiện tay
            min_tracking_confidence = self.trackingCon # Độ tin cậy hệ thống phải đạt được để theo dõi tay
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
    def findHands(self, image, draw = True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển về dạng hình ảnh màu RGB
        self.results = self.hands.process(imageRGB) # Xử lí bắt bàn tay
        height, width = image.shape[:2] # Lấy ra chiều cao và rộng của khung hình
        lmList = [] # Lưu tọa độ các điểm trên bàn tay
        xList = [] # Lưu toa độ x các điểm trên bàn tay
        yList = [] # Lưu tọa độ y các điểm trên bàn tay
        bbox = [] # Lưu tọa độ các điểm quan trọng cx, cy, x, y, w, h
        if self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.results.multi_hand_landmarks[0], self.mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark): # Lấy ra id, tọa độ của bàn tay được phát hiện
                px , py = int(lm.x * width), int(lm.y * height) # Chuyển tọa độ về dạng chuẩn theo khung hình
                lmList.append([px, py])
                xList.append(px)
                yList.append(py)
                cv2.circle(image, (px, py), 3, (204, 0, 102), -1)

            x, y = min(xList), min(yList)
            w, h = max(xList) - x, max(yList) - y
            cx, cy = x + w//2, y + h//2
            bbox.append((cx, cy))
            bbox.append((x, y, w, h))
        return image, bbox, lmList
    def finDistance(self, p_one, p_two, image = None):
        x1, y1 = p_one
        x2, y2 = p_two
        cx, cy = (x1 + x2)//2 , (y1 + y2)//2
        length = math.hypot(x2 - x1, y2 - y1)
        if image is not None:
            cv2.circle(image, (x1, y1), 5, (255, 0, 255), -1)
            cv2.circle(image, (x2, y2), 5, (255, 0, 255), -1)
            cv2.line(image, (x1, y1), (x2, y2), (0, 204, 0), 3)
            cv2.circle(image, (cx, cy), 5, (255, 0, 255), -1)
            return  image, length
        else: return length