import numpy as np
import cv2
import lib_contour_LV as lib

#Чтение изображения с контуром
video_with_contour = cv2.VideoCapture('N120131007162257079.avi')    
ret, frame = video_with_contour.read()
video_with_contour.release()
image_contour = lib.get_contour_image(frame)
cv2.imshow('image',image_contour)
#Чтение изображения УЗИ-видеопетли
video = cv2.VideoCapture('N120131007093530930.avi')
ret, frame = video.read()
video.release()
cv2.imshow('segment',lib.get_segment_image(frame))
# Выделим маску
#res = cv2.bitwise_and(frame,frame, mask = mask)
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#frame = cv2.medianBlur(frame,5)
#ret,thr = cv2.threshold(frame, 1, 255,cv2.THRESH_BINARY)
#cv2.imshow('frame',thr)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)

#cv2.imwrite('1.png', frame)
k = cv2.waitKey(0)


