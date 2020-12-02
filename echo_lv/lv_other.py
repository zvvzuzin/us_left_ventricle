import numpy as np
import cv2
import scipy.interpolate as inter
import matplotlib.pyplot as plt

def get_area_contour_image(image):
    # Выделение контура на новых видеозаписях
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Определим диапазон оранжевого цвета в HSV
    #lower_orange = np.array([14,0,0])
    #upper_orange = np.array([19,255,255])
    lower_orange = np.array([5,0,0])
    upper_orange = np.array([25,255,255])
    # Выделим оранжевый контур 
    image = cv2.inRange(hsv, lower_orange, upper_orange)
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(image, mask, (0,0), 255)
    # Увеличим немного края контура дилатацией, приближенные к исходному контуру
    image = cv2.dilate(cv2.bitwise_not(image),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=1)
    # Вернем контур пересечение изображений, это и будет итоговым контуром 
    return image

def get_segment_image(image):
    # Функция написана под изображения с круговым сегментом в центре
    # На выходе контур сегмента и изображение
    # Бинаризация изображения 
    image = preproccesing_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[np.where(image > 0)] = 255
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    #ret,image = cv2.threshold(image,1,255,cv2.THRESH_BINARY)
    segm = image.copy()
    h, w = image.shape[:2]    
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(segm, mask, (h//2,w//2),0)
    cv2.floodFill(segm, mask, (h//3,w//2),0)
    cv2.floodFill(segm, mask, (3*h//4,w//2),0)
    
    image = cv2.bitwise_xor(image, segm)
    segm = cv2.bitwise_not(image)
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(segm, mask, (0,0), 0)
    image += segm

    return image

def get_segment_contour(im_segm):
    im2,contours,hierarchy = cv2.findContours(im_segm, 1, 2)
    flag = -1
    area = 0
    for it in contours:
        flag += 1
        M = cv2.moments(it)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if np.abs(im_segm.shape[0]/2. - cy) < im_segm.shape[0]*0.2 \
            and np.abs(im_segm.shape[1]/2. - cx) < im_segm.shape[1]*0.2: 
                if cv2.contourArea(it) > area:
                    cont = it
                    area = cv2.contourArea(it)
                    index = flag
    return cont

def get_adjust_area_image(image_cont, image_orig):
    # Изображение с контуром подгоняется под изобрыжение исходного кадра.
    # На выходе маска с изображением контура подогнанный под исходное изображение.
    
    # Выделяем сегменты с предварительной обработкой изображений
    segm_cont = get_segment_image(image_cont)
    segm_orig = get_segment_image(image_orig)
    
    # Выделяем контуры и находим крайнюю правую и левую точки
    cont = get_segment_contour(segm_cont)
    leftmost_cont = cont[cont[:,:,0].argmin()][0]
    rightmost_cont = cont[cont[:,:,0].argmax()][0]
    topmost_cont = cont[cont[:,:,1].argmin()][0]
    botmost_cont = cont[cont[:,:,1].argmax()][0]
    
    cont = get_segment_contour(segm_orig)
    leftmost_orig = cont[cont[:,:,0].argmin()][0]
    rightmost_orig = cont[cont[:,:,0].argmax()][0]
    topmost_orig = cont[cont[:,:,1].argmin()][0]
    botmost_orig = cont[cont[:,:,1].argmax()][0]
    
    # Вычисляем коэффициент отношения длины между точками оригинального изображения с сегментом 
    # к длине между точками сегмента изображения с контуром. На этот коэффициент надо будет увеличить изображение с контуром.
    koef_1 = 1. * np.sqrt(np.sum((leftmost_orig - rightmost_orig) ** 2))/np.sqrt(np.sum((leftmost_cont - rightmost_cont) ** 2))
    koef_2 = 1. * np.sqrt(np.sum((topmost_orig - botmost_orig) ** 2))/np.sqrt(np.sum((topmost_cont - botmost_cont) ** 2))
    koef = (koef_2 + koef_2) / 2.

    # Вычислим вектор смещения изображения контура относительно оригинального
    #dis_vec_1 = [np.uint16(leftmost_orig[0] - koef * leftmost_cont[0]),np.uint16(leftmost_orig[1] - koef * leftmost_cont[1])]
    #dis_vec_2 = [np.uint16(topmost_orig[0] - koef * topmost_cont[0]),np.uint16(topmost_orig[1] - koef * topmost_cont[1])]
    #dis_vec_1 = leftmost_orig - koef * leftmost_cont
    #dis_vec_2 = rightmost_orig - koef * rightmost_cont
    dis_vec_3 = topmost_orig - koef * topmost_cont
    dis_vec_4 = botmost_orig - koef * botmost_cont
    dis_vec = np.uint16((dis_vec_3 + dis_vec_4) / 2.)
    
    # Получим бинарный кадр контура с изображения
    image_area = cv2.bitwise_and(get_area_contour_image(image_cont), segm_cont)
    
    # Увеличим кадр так, чтобы сегменты были одного размера
    image_area = cv2.resize(image_area, None, fx = koef, fy = koef, interpolation = cv2.INTER_AREA)
    image_area[np.where(image_area > 0)] = 255
        
    image = np.zeros(image_orig.shape[:2], dtype = np.uint8)
    image[dis_vec[1]: dis_vec[1] + image_area.shape[0], \
          dis_vec[0]: dis_vec[0] + image_area.shape[1]] = image_area
    
    return image

def cart2pol(x, y, cent_x, cent_y):
    rho = np.sqrt((x - cent_x) ** 2 + (y - cent_y) ** 2)
    phi = np.arctan2( y - cent_y,  x - cent_x)
    return rho, phi

def pol2cart(phi, rho, cent_x, cent_y):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x + cent_x, y + cent_y

def get_contour_points(im_cont, num = None):
    cont_y, cont_x = np.where(im_cont == 255)
    cent_x = np.mean(cont_x) 
    cent_y = np.mean(cont_y)
    
    cont_rho, cont_phi = cart2pol(cont_x, cont_y, cent_x, cent_y)
    ind_srt = np.argsort(cont_phi)
    cont_phi = cont_phi[ind_srt]
    cont_rho = cont_rho[ind_srt]
    
    delta = 10
    last_win = np.mean(cont_rho[:delta])
    ext_max_phi = np.array([], dtype = np.float)
    ext_max_rho = np.array([], dtype = np.float)
    ext_max_ind = np.array([], dtype = np.uint16)
    sign = ''
    for it in np.arange(delta,len(cont_rho),3):
        next_win = np.mean(cont_rho[it - delta:it + delta])
        if sign == '+' and next_win - last_win < 0:
            sign = '-'
            ind_lc_max = np.where(cont_rho[it - delta: it + delta] == np.max(cont_rho[it - delta: it + delta + 1]))
            ext_max_phi = np.append(ext_max_phi,(cont_phi[it - delta + ind_lc_max]))
            ext_max_rho = np.append(ext_max_rho,(cont_rho[it - delta + ind_lc_max]))
            ext_max_ind = np.append(ext_max_ind, it - delta + ind_lc_max)
        elif sign == '-' and next_win - last_win > 0:
            sign = '+'
        elif sign == '' and next_win - last_win > 0:
            sign = '+'
        elif sign == '' and next_win - last_win < 0:
            sign = '-'
        last_win = next_win
        
    cont_phi = np.hstack((cont_phi[ext_max_ind[1]:],cont_phi[:ext_max_ind[0]]+2*np.pi))
    cont_rho = np.hstack((cont_rho[ext_max_ind[1]:],cont_rho[:ext_max_ind[0]]))
    ext_max_phi[0] += 2*np.pi
    if num:
        itr = np.linspace(0, len(cont_phi)-1, num = num, endpoint = True, dtype = np.uint16)
        cont_x, cont_y = pol2cart(cont_phi[itr], cont_rho[itr], cent_x, cent_y)
    else:
        cont_x, cont_y = pol2cart(cont_phi, cont_rho, cent_x, cent_y)

    return np.uint16(cont_x), np.uint16(cont_y)

def interp_contour_points(cont_x, cont_y, frame_orig):
    itr_more = np.linspace(0, len(cont_x)-1, num = 1000, endpoint = True, dtype = np.float)
    cont_more_x = inter.interp1d(np.arange(len(cont_x)), cont_x, kind = 'cubic')(itr_more)
    cont_more_y = inter.interp1d(np.arange(len(cont_y)), cont_y, kind = 'cubic')(itr_more)

    itr_base = np.linspace(0, 1, num = 500, endpoint = True, dtype = np.float)
    cont_base_x = inter.interp1d(np.array([0,1]), np.array([cont_x[0], cont_x[-1]]), kind = 'linear')(itr_base)
    cont_base_y = inter.interp1d(np.array([0,1]), np.array([cont_y[0], cont_y[-1]]), kind = 'linear')(itr_base)

    image = np.zeros(frame_orig.shape[:2], dtype = np.uint8)
    image[np.uint16(cont_more_y),np.uint16(cont_more_x)] = 255
    image[np.uint16(cont_base_y),np.uint16(cont_base_x)] = 255

    return image

def area2cont(im_area):
    return cv2.bitwise_and(cv2.dilate(im_area,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=1), \
                          cv2.bitwise_not(im_area))

def cont2area(im_cont):
    h, w = im_cont.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_cont, mask, (0,0), 255)
    return cv2.dilate(cv2.bitwise_not(im_cont),cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=1)
    
def preproccesing_image(image):
    # Убираем зеленые кардиограммы, так как они могут мешать при обработке кадров
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Определим диапазон зеленого цвета в HSV
    lower_green = np.array([58,0,0])
    upper_green = np.array([62,255,255])

    hsv = cv2.inRange(hsv, lower_green, upper_green)
    image[np.where(hsv > 0)] = 0
    
    # Убираем желтые надписи, которые соприкасаются с круговым сегментов на кадрах старого УЗИ-сканера
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Определим диапазон желтого цвета в HSV
    lower_yellow = np.array([29,0,0])
    upper_yellow = np.array([31,255,255])

    hsv = cv2.inRange(hsv, lower_yellow, upper_yellow)
    image[np.where(hsv > 0)] = 0
    
    # Убираем бирюзовую кардиограмму, которая соприкасается с круговым сегментов на кадрах старого УЗИ-сканера
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Определим диапазон желтого цвета в HSV
    lower_yellow = np.array([86,0,0])
    upper_yellow = np.array([88,255,255])

    hsv = cv2.inRange(hsv, lower_yellow, upper_yellow)
    image[np.where(hsv > 0)] = 0
    
    return image 

def smoothing_area(im_area): 
    x, y = get_contour_points(area2cont(im_area))
    return cont2area(interp_contour_points(x, y, im_area))

