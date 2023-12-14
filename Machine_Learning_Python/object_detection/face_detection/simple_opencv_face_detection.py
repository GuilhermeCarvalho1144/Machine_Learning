import cv2

test_image_path = '/mnt/1b0f3c80-0858-4ed5-a19d-c13144d4a615/Database/Face_detection/WIDER_test/images/3--Riot/3_Riot_Riot_3_1.jpg'

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


test_img = cv2.imread(test_image_path)

test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(test_img_gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imshow('test img ', test_img)
cv2.waitKey()
