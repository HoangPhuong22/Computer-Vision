import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('Image_Basic/ElonMusk.jpg') # load file
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB) # Đổi thành RGB
imgTest = face_recognition.load_image_file('Image_Basic/BillGates.jpg') # load file
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB) # Đổi thành RGB

faceLoc = face_recognition.face_locations(imgElon)[0] # Nhận diện khuân mặt đầu tiên tìm thấy, trả về các tọa độ top right bootom left
encodeElon = face_recognition.face_encodings(imgElon)[0] # Mã hóa
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2) # Vẽ box

faceLocTest = face_recognition.face_locations(imgTest)[0] # Nhận diện khuân mặt đầu tiên tìm thấy, trả về các tọa độ top right bootom left
encodeTest = face_recognition.face_encodings(imgTest)[0] # Mã hóa
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2) # Vẽ box

results = face_recognition.compare_faces([encodeElon], encodeTest) # So sánh 2 khuôn mặt
faceDis = face_recognition.face_distance([encodeElon], encodeTest) # Tính khoảng cách hai khuân mặt, khoảng cách càng lớn thì càng khác nhau

print(results, faceDis)
#Puttext
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50),cv2.FONT_HERSHEY_DUPLEX, 1, (250, 0, 250), 2)
#Show
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)

cv2.waitKey(0)