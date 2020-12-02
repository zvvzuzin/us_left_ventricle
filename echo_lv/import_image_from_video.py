## Разбивание видео на отдельные изображения

import numpy as np
import cv2

cap_1 = cv2.VideoCapture('N120131007093530930.avi')
cap_2 = cv2.VideoCapture('N120131007162257079.avi')

while(cap_1.isOpened()): 
    ret, frame_1 = cap_1.read() # читаем кадр из видео 
    ret, frame_2 = cap_2.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # можем преобразовать изображение в серое 

    cv2.imshow('frame_1', frame_1) # показываем текущий кадр
    cv2.imshow('frame_2', frame_2)
    #time.sleep(0.03) # здесь можем управлять частотой кадров 
    if cv2.waitKey(0) & 0xFF == ord('q'): # если нужно, то завершаем программу 
        break 

cap_1.release() 
cap_2.release()
cv2.destroyAllWindows() 
