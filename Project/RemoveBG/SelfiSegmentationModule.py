import cv2
import mediapipe as mp
import numpy as np


class SelfieSegmentation():
    def __init__(self, model=1):
        self.model = model
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(
            model_selection=self.model
        )
        self.mpDraw = mp.solutions.drawing_utils

    def removeBG(self, image, imageBG=(255, 255, 255), cutThreshold=0.1):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi thành RGB để thực hiện
        results = self.selfieSegmentation.process(imageRGB)
        """
        :results.segmentation_mask: Đây là mặt nạ phân đoạn, là một mảng đa chiều
         chứa thông tin về phân đoạn vùng khuôn mặt và nền. Trong mặt nạ này, 
         các điểm ảnh thuộc vùng khuôn mặt thường được đánh dấu là một giá trị 
         cụ thể (thường là 1 hoặc True), trong khi các điểm ảnh thuộc vùng nền 
         có giá trị khác (thường là 0 hoặc False). Bạn có thể sử dụng mặt nạ này
          để xác định vùng khuôn mặt trong ảnh.

        :results.image: Đây là ảnh đầu ra sau khi quá trình phân đoạn được thực 
          hiện. Nó hiển thị vùng khuôn mặt được phân đoạn với nền đã được loại bỏ.
        """
        condition = np.stack(
            (results.segmentation_mask,) * 3,
            axis=-1) > cutThreshold
        """
        Trong trường hợp này condition sẽ là một mảng cùng kích thước
        với results.segmentation. Chứa True ở những vị trí thõa mãn
        False ở những vị trí không thõa mãn.
        """

        if isinstance(imageBG, tuple):
            """
            :Kiểm tra xem imageBG phải là tuple không, nếu không  phải thì
            tức là thay thế bằng phông nên
            """
            _imageBG = np.zeros(image.shape, dtype=np.uint8)
            _imageBG[:] = imageBG  # Gán màu
            imgOut = np.where(condition, image, _imageBG)
            # Những vị trí = True thì là vùng khuân mặt lấy trong image
            # Những vị trí = False thì lấy màu trong _imageBG
        else:
            imgOut = np.where(condition, image, imageBG)
            # Những vị trí = True thì là vùng khuân mặt lấy trong image
            # Những vị trí = False thì lấy phông nền trong _imageBG
        return imgOut
# def main():
#     cap = cv2.VideoCapture(0)
#     cap.set(3, 640)
#     cap.set(4, 480)
#
#     segmentor = SelfieSegmentation(model= 0)
#     while True:
#         success, image = cap.read()
#         imgOut = segmentor.removeBG(image, imageBG=(255,0, 0), cutThreshold=0.1)
#         cv2.imshow('ZeroCoder', imgOut)
#
#         if cv2.waitKey(1) & 0xFF == ord('p'):
#             break
#
# if __name__ == "__main__":
#     main()import  cv2
# import mediapipe as mp
# import numpy as np
#
# class SelfieSegmentation():
#     def __init__(self, model = 1):
#         self.model = model
#         self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
#         self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(
#             model_selection=self.model
#         )
#         self.mpDraw = mp.solutions.drawing_utils
#
#     def removeBG(self, image, imageBG = (255, 255, 255), cutThreshold = 0.1):
#         imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Chuyển đổi thành RGB để thực hiện
#         results = self.selfieSegmentation.process(imageRGB)
#         """
#         :results.segmentation_mask: Đây là mặt nạ phân đoạn, là một mảng đa chiều
#          chứa thông tin về phân đoạn vùng khuôn mặt và nền. Trong mặt nạ này,
#          các điểm ảnh thuộc vùng khuôn mặt thường được đánh dấu là một giá trị
#          cụ thể (thường là 1 hoặc True), trong khi các điểm ảnh thuộc vùng nền
#          có giá trị khác (thường là 0 hoặc False). Bạn có thể sử dụng mặt nạ này
#           để xác định vùng khuôn mặt trong ảnh.
#
#         :results.image: Đây là ảnh đầu ra sau khi quá trình phân đoạn được thực
#           hiện. Nó hiển thị vùng khuôn mặt được phân đoạn với nền đã được loại bỏ.
#         """
#         condition = np.stack(
#             (results.segmentation_mask,) * 3,
#             axis = -1) > cutThreshold
#         """
#         Trong trường hợp này condition sẽ là một mảng cùng kích thước
#         với results.segmentation. Chứa True ở những vị trí thõa mãn
#         False ở những vị trí không thõa mãn.
#         """
#
#         if isinstance(imageBG, tuple):
#             """
#             :Kiểm tra xem imageBG phải là tuple không, nếu không  phải thì
#             tức là thay thế bằng phông nên
#             """
#             _imageBG = np.zeros(image.shape, dtype= np.uint8)
#             _imageBG[:] = imageBG # Gán màu
#             imgOut = np.where(condition, image, _imageBG)
#             # Những vị trí = True thì là vùng khuân mặt lấy trong image
#             #Những vị trí = False thì lấy màu trong _imageBG
#         else:
#             imgOut = np.where(condition, image, imageBG)
#             # Những vị trí = True thì là vùng khuân mặt lấy trong image
#             # Những vị trí = False thì lấy phông nền trong _imageBG
#         return imgOut
# # def main():
# #     cap = cv2.VideoCapture(0)
# #     cap.set(3, 640)
# #     cap.set(4, 480)
# #
# #     segmentor = SelfieSegmentation(model= 0)
# #     while True:
# #         success, image = cap.read()
# #         imgOut = segmentor.removeBG(image, imageBG=(255,0, 0), cutThreshold=0.1)
# #         cv2.imshow('ZeroCoder', imgOut)
# #
# #         if cv2.waitKey(1) & 0xFF == ord('p'):
# #             break
# #
# # if __name__ == "__main__":
# #     main()