import numpy as np
import cv2
# import scipy.interpolate as inter
from scipy import interpolate
import matplotlib.pyplot as plt


def cart2pol(x, y, cent_x, cent_y, norm=2):
    '''
    return phi, rho
    '''
    rho = (abs(x - cent_x) ** norm + abs(y - cent_y) ** norm) ** (1/norm)
    phi = np.arctan2(y - cent_y,  x - cent_x)
    return phi, rho


def pol2cart(phi, rho, cent_x, cent_y):
    '''
    return x, y
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x + cent_x, y + cent_y

 
def sort2d(cont_phi, cont_rho):
    phi, rho = cont_phi.copy(), cont_rho.copy()
    min_phi, max_phi, min_rho, max_rho = min(phi), max(phi - min(phi)), min(rho), max(rho - min(rho))
    rho = (rho - min_rho) / max_rho
    phi = (phi - min_phi) / max_phi
    phi = list(phi)
    rho = list(rho)
    sort_phi = []
    sort_rho = []
    index = np.argmin(phi)
    sort_phi.append(phi.pop(index))
    sort_rho.append(rho.pop(index))
    while phi and rho:
        dist = 10
        near_index = None
        for index, (el_phi, el_rho) in enumerate(zip(phi, rho)):
            new_dist = (el_phi - sort_phi[-1]) ** 2 + (el_rho - sort_rho[-1]) ** 2
            if new_dist < dist and new_dist < 0.03:
                near_index = index
                dist = new_dist
        if near_index is not None:
            sort_phi.append(phi.pop(near_index))
            sort_rho.append(rho.pop(near_index))
        else:
            break
    return np.array(sort_phi) * max_phi + min_phi, np.array(sort_rho) * max_rho + min_rho
    
    

def get_contour_points(im, base_points=None, top_point=None, space='cart', kind='whole', center=None, num=None):
    '''
    center = [None, 'base', tuple]
    kind = ['whole', 'contour']
    form = ['polar', 'cart']
    '''
    im_cont = im.copy()
    im_cont = area2cont(im_cont)

    h = 1000
    w = 1000
    k_w = w / im_cont.shape[1]
    k_h = h / im_cont.shape[0]
    cont_y, cont_x = np.where(im_cont != 0)
    
    if not (base_points and top_point):
        top_point, base_l_point, base_r_point, orig_cent_x, orig_cent_y = get_main_points(im_cont, space='cart')
    else:
        base_l_point, base_r_point = base_points[0], base_points[1]
        orig_cent_x, orig_cent_y = np.mean(cont_x), np.mean(cont_y)
    
    cont_x, cont_y = cont_x * k_w, cont_y * k_h
    cent_x, cent_y = np.mean(cont_x), np.mean(cont_y)
    
    cont_phi, cont_rho = cart2pol(cont_x, cont_y, cent_x, cent_y)
    base_l_point = cart2pol(base_l_point[0]* k_w, base_l_point[1]* k_h, cent_x, cent_y)
    base_r_point = cart2pol(base_r_point[0]* k_w, base_r_point[1]* k_h, cent_x, cent_y)
    top_point = cart2pol(top_point[0]* k_w, top_point[1]* k_h, cent_x, cent_y)

    if kind == 'contour':
        cont_rho = cont_rho[(cont_phi <= min(base_l_point[0], base_r_point[0])) | (cont_phi >= max(base_l_point[0], base_r_point[0]))]
        cont_phi = cont_phi[(cont_phi <= min(base_l_point[0], base_r_point[0])) | (cont_phi >= max(base_l_point[0], base_r_point[0]))]
    
    if type(center) == tuple:
        cont_x, cont_y = pol2cart(cont_phi, cont_rho, cent_x, cent_y)
        cent_x, cent_y = center[0], center[1]
        cont_phi, cont_rho = cart2pol(cont_x, cont_y, cent_x, cent_y)
    elif type(center) == str:
        if center == 'base':
            cont_x, cont_y = pol2cart(cont_phi, cont_rho, cent_x, cent_y)
            top_point = pol2cart(top_point[0], top_point[1], cent_x, cent_y)
            base_l_point = pol2cart(base_l_point[0], base_l_point[1], cent_x, cent_y)
            base_r_point = pol2cart(base_r_point[0], base_r_point[1], cent_x, cent_y)
            cent_x, cent_y = (base_l_point[0] + base_r_point[0]) / 2, (base_l_point[1] + base_r_point[1]) / 2
            cont_phi, cont_rho = cart2pol(cont_x, cont_y, cent_x, cent_y)
            base_l_point = cart2pol(base_l_point[0], base_l_point[1], cent_x, cent_y)
            base_r_point = cart2pol(base_r_point[0], base_r_point[1], cent_x, cent_y)
            top_point = cart2pol(top_point[0], top_point[1], cent_x, cent_y)
            
    cont_phi[cont_phi >= max(base_l_point[0], base_r_point[0])] -= 2*np.pi
#     cont_phi, cont_rho = sort2d(cont_phi, cont_rho)
    indexes = np.argsort(cont_phi)
    cont_phi = cont_phi[indexes]
    cont_rho = cont_rho[indexes]
    
    if num:
        if num % 2:
            cont_phi_l =  cont_phi[cont_phi <= top_point[0]]
            cont_rho_l =  cont_rho[cont_phi <= top_point[0]]
            cont_phi_r =  cont_phi[cont_phi >= top_point[0]]
            cont_rho_r =  cont_rho[cont_phi >= top_point[0]]
            cont_phi = np.concatenate((
                np.array([base_l_point[0]]),
                cont_phi_l[np.linspace(0, len(cont_phi_l)-1, num // 2 + 1, dtype=int)][1:-1],
                np.array([top_point[0]]),
                cont_phi_r[np.linspace(0, len(cont_phi_r)-1, num // 2 + 1, dtype=int)][1:-1],
                np.array([base_r_point[0]]),
            ))
            cont_rho = np.concatenate((
                np.array([base_l_point[1]]),
                cont_rho_l[np.linspace(0, len(cont_rho_l)-1, num // 2 + 1, dtype=int)][1:-1],
                np.array([top_point[1]]),
                cont_rho_r[np.linspace(0, len(cont_rho_r)-1, num // 2 + 1, dtype=int)][1:-1],
                np.array([base_r_point[1]]),
            ))
        else:
            cont_phi = cont_phi[np.linspace(0, len(cont_phi)-1, num, dtype=int)]
            cont_rho = cont_rho[np.linspace(0, len(cont_rho)-1, num, dtype=int)]
        
    cont_x, cont_y = pol2cart(cont_phi, cont_rho, cent_x, cent_y)
    cont_x, cont_y = cont_x / k_w, cont_y / k_h
    cont_phi, cont_rho = cart2pol(cont_x, cont_y, orig_cent_x, orig_cent_y)
        
    if space == 'cart':
        cont_x, cont_y = pol2cart(cont_phi, cont_rho, orig_cent_x, orig_cent_y)
        return np.uint16(cont_x), np.uint16(cont_y), orig_cent_x, orig_cent_y
    elif space == 'polar':
        return cont_phi, cont_rho, orig_cent_x, orig_cent_y
    else:
        return


def mean_vector(cont_x, cont_y, x_point, y_point):
    vec_x, vec_y = [], []
    for x, y in zip(cont_x - x_point, cont_y - y_point):
        cur_dist = np.linalg.norm((x, y))
        if cur_dist != 0:
            vec_x.append(x / cur_dist)
            vec_y.append(y / cur_dist)
    return np.mean(vec_x), np.mean(vec_y)

def variance_contour(cont_x, cont_y, vec_x, vec_y):
    var = []
    for x, y in zip(cont_x, cont_y):
        var.append(np.linalg.norm((x - vec_x * (x * vec_x + y * vec_y) / (vec_x**2 + vec_y**2), 
                                   y - vec_y * (x * vec_x + y * vec_y) / (vec_x**2 + vec_y**2))))
        
    return np.mean(var)
    
    
def get_main_points(im, space='cart'):
    '''
    return apex_point, base_left_point, base_right_point
    '''
    im_cont = im.copy()
    im_cont = area2cont(im_cont)
    cont_y, cont_x = np.where(im_cont != 0)
    cent_x, cent_y = np.mean(cont_x), np.mean(cont_y)

    h = 1000
    w = 1000
    k_w = w / im_cont.shape[1]
    k_h = h / im_cont.shape[0]

    new_cont_x, new_cont_y = cont_x * k_w, cont_y * k_h
    cont_phi, cont_rho = cart2pol(new_cont_x, 
                                  new_cont_y, 
                                  np.mean(new_cont_x), 
                                  np.mean(new_cont_y))
    indexes = np.argsort(cont_phi)
    cont_phi = cont_phi[indexes]
    cont_rho = cont_rho[indexes]
    pos_indexes = cont_phi >= 0
    neg_indexes = cont_phi < 0
    index = np.argmax(cont_rho[neg_indexes])
    top_x, top_y = pol2cart(cont_phi[neg_indexes][index], 
                            cont_rho[neg_indexes][index], 
                            np.mean(new_cont_x), 
                            np.mean(new_cont_y))
    dict_base = {}
    for i in range(-100, 101):
        base_phi, base_rho = cart2pol(new_cont_x, 
                                      new_cont_y, 
                                      top_x+i, 
                                      top_y)
    #     plt.scatter(base_phi, base_rho)
        index = np.argmax(base_rho)
        if index in dict_base.keys():
            dict_base[index] += 1
        else:
            dict_base[index] = 1
    
    count = 0        
    for k in dict_base.keys():
        if count < dict_base[k]:
            index = k
            count = dict_base[k]    
    base_x, base_y = new_cont_x[index], new_cont_y[index]
    #pol2cart(base_phi[index], 
          #                    base_rho[index], 
          #                    top_x, 
          #                    top_y)
    
    base_phi, base_rho = cart2pol(base_x, 
                                  base_y, 
                                  np.mean(new_cont_x), 
                                  np.mean(new_cont_y))
    
    check_r_phi = cont_phi[pos_indexes][cont_phi[pos_indexes] < base_phi]
    check_r_rho = cont_rho[pos_indexes][cont_phi[pos_indexes] < base_phi]
    check_l_phi = cont_phi[pos_indexes][cont_phi[pos_indexes] > base_phi]
    check_l_rho = cont_rho[pos_indexes][cont_phi[pos_indexes] > base_phi]
    
    r_x, r_y = pol2cart(check_r_phi, 
                        check_r_rho, 
                        np.mean(new_cont_x), 
                        np.mean(new_cont_y))

    l_x, l_y = pol2cart(check_l_phi, 
                        check_l_rho, 
                        np.mean(new_cont_x), 
                        np.mean(new_cont_y))

    var_l = variance_contour(l_x - base_x, l_y - base_y, l_x[0] - l_x[-1], l_y[0] - l_y[-1])
    var_r = variance_contour(r_x - base_x, r_y - base_y, r_x[0] - r_x[-1], r_y[0] - r_y[-1])
    length = 10
    var_near_l = variance_contour(l_x[:length] - base_x, l_y[:length] - base_y, l_x[0] - l_x[length-1], l_y[0] - l_y[length-1])
    var_near_r = variance_contour(r_x[-length:] - base_x, r_y[-length:] - base_y, r_x[-1] - r_x[-length], r_y[-1] - r_y[-length])
    base_l_x, base_l_y, base_r_x, base_r_y = None, None, None, None

    if var_r > var_l:
        check_phi = check_r_phi
        check_rho = check_r_rho
        check_x, check_y = r_x, r_y
        indexes = np.array([i for i, (x, y) in enumerate(zip(check_x, check_y)) if np.linalg.norm((x - base_x, y - base_y)) < 200], dtype=int)
        norm_vec_x, norm_vec_y = mean_vector(np.array(check_x)[indexes], np.array(check_y)[indexes], base_x, base_y)
        norm_vec_x, norm_vec_y = norm_vec_y, -norm_vec_x
        base_l_x, base_l_y = base_x / k_w, base_y / k_h
        edge_x, edge_y = check_x[0], check_y[0]
    else:
        check_phi = check_l_phi
        check_rho = check_l_rho
        check_x, check_y = l_x, l_y
        indexes = np.array([i for i, (x, y) in enumerate(zip(check_x, check_y)) if np.linalg.norm((x - base_x, y - base_y)) < 200], dtype=int)
        norm_vec_x, norm_vec_y = mean_vector(np.array(check_x)[indexes], np.array(check_y)[indexes], base_x, base_y)
        norm_vec_x, norm_vec_y = -norm_vec_y, norm_vec_x
        base_r_x, base_r_y = base_x / k_w, base_y / k_h
        edge_x, edge_y = check_x[-1], check_y[-1]
    
    k = 10
    c_x, c_y = base_x + k * norm_vec_x, base_y + k * norm_vec_y
    while np.linalg.norm((c_x - base_x, c_y - base_y)) <= np.linalg.norm((edge_x - c_x, edge_y - c_y)) or c_y > top_y:
        if k > 1000:
            break
        k += 1
        c_x, c_y = base_x + k * norm_vec_x, base_y + k * norm_vec_y
    
    check_phi, check_rho = cart2pol(check_x, 
                                    check_y,
                                    c_x,
                                    c_y,
                                   )
   
    check_phi[check_phi <= -np.pi/2] += 2*np.pi
    
    index = np.argmax(check_rho)
    base_x, base_y = check_x[index], check_y[index]

    if base_r_x is None and base_r_y is None:
        base_r_x, base_r_y = base_x / k_w, base_y / k_h

    elif base_l_x is None and base_l_y is None:
        base_l_x, base_l_y = base_x / k_w, base_y / k_h
    
    cont_phi, cont_rho = cart2pol(cont_x, cont_y, (base_l_x + base_r_x) / 2, (base_l_y + base_r_y) / 2)
    index = np.argmax(cont_rho)
    apex_phi, apex_rho = cont_phi[index], cont_rho[index]
    apex_x, apex_y = pol2cart(apex_phi, 
                              apex_rho, 
                              (base_l_x + base_r_x) / 2, 
                              (base_l_y + base_r_y) / 2)
        
    
    if space == 'polar':
        apex_phi, apex_rho = cart2pol(apex_x, 
                                      apex_y, 
                                      cent_x, 
                                      cent_y)
        base_r_phi, base_r_rho = cart2pol(base_r_x, 
                                          base_r_y, 
                                          cent_x, 
                                          cent_y)
        base_l_phi, base_l_rho = cart2pol(base_l_x, 
                                          base_l_y, 
                                          cent_x, 
                                          cent_y)
        return (apex_phi, apex_rho), (base_l_phi, base_l_rho), (base_r_phi, base_r_rho), cent_x, cent_y
    elif space == 'cart':
        return (apex_x, apex_y), (base_l_x, base_l_y), (base_r_x, base_r_y), cent_x, cent_y
    else:
        return
    


def interpolate_contour(x, y, with_base=True, k=2):
    
    x_tck = interpolate.splrep(np.linspace(0,1,len(x)), x, k=k)
    y_tck = interpolate.splrep(np.linspace(0,1,len(x)), y, k=k)

    full_x = interpolate.splev(np.linspace(0,1,10000), x_tck)
    full_y = interpolate.splev(np.linspace(0,1,10000), y_tck)
    
    if with_base:
        x_base = interpolate.splrep(np.linspace(0,1,2), x[np.array([-1,0])], k=1)
        y_base = interpolate.splrep(np.linspace(0,1,2), y[np.array([-1,0])], k=1)
        full_x_base = interpolate.splev(np.linspace(0,1,2000), x_base)
        full_y_base = interpolate.splev(np.linspace(0,1,2000), y_base)
    else:
        full_x_base = []
        full_y_base = []
    return np.concatenate((full_x, full_x_base)).astype(int), np.concatenate((full_y, full_y_base)).astype(int)


def choose_nearest_to_mean(x_points, y_points):
    mean_x, mean_y = np.mean(x_points),np.mean(y_points)
    dist = np.Inf
    for x, y in zip(x_points, y_points):
        new_dist = ((mean_x - x) ** 2 + (mean_y - y) ** 2) ** (1/2)
        if dist > new_dist:
            dist = new_dist.copy()
            choose_x = x
            choose_y = y
    return choose_x, choose_y


def choose_nearest_to_median(x_points, y_points):
    mean_x, mean_y = np.median(x_points),np.median(y_points)
    dist = np.Inf
    for x, y in zip(x_points, y_points):
        new_dist = ((mean_x - x) ** 2 + (mean_y - y) ** 2) ** (1/2)
        if dist > new_dist:
            dist = new_dist.copy()
            choose_x = x
            choose_y = y
    return choose_x, choose_y


def area2cont(im_area):
    im_cont = im_area.copy()
    y, x = np.where(im_cont != 0)
    for i, j in zip(x, y):
        if im_area[j+1, i] != 0 and im_area[j-1, i] != 0 and im_area[j, i-1] != 0 and im_area[j, i+1] != 0:
            im_cont[j, i] = 0
    return im_cont


def cont2area(im_cont):
    im_area = im_cont.copy()
    fill = np.max(im_area)
    y, x = np.where(im_area != 0)
    pxl = (int(np.mean(y)), int(np.mean(x)))
    Q = [pxl]
    inside_contour = True
    while Q:
        Q_ns = []
        pt = Q.pop()
        if im_cont[pt] == 0:
            Q_ns.append(pt)
        e = 1
        w = -1
        pt_e = (pt[0], pt[1]+e)
        pt_w = (pt[0], pt[1]+w)
        if pt_e[1] == im_area.shape[1] or pt_w[1] < 0:
            inside_contour = False
            continue
        while im_area[pt_e] == 0:
            Q_ns.append(pt_e)
            e += 1
            pt_e = (pt[0], pt[1]+e)
            if pt_e[1] == im_area.shape[1]:
                inside_contour = False
                break
        while im_area[pt_w] == 0:
            Q_ns.append(pt_w)
            w -= 1
            pt_w = (pt[0], pt[1]+w)
            if pt_w[1] < 0:
                inside_contour = False
                break
        while Q_ns:
            pt_ns = Q_ns.pop()
            im_area[pt_ns] = fill
            pt_n = (pt_ns[0]-1, pt_ns[1])
            pt_s = (pt_ns[0]+1, pt_ns[1])
            if pt_s[0] == im_area.shape[0] or pt_n[0] < 0:
                inside_contour = False
                break
            if im_area[pt_n] == 0:
                Q.append(pt_n)
            if im_area[pt_s] == 0:
                Q.append(pt_s)
    if inside_contour:
        return im_area
    return abs(im_area - fill)
###



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
    contours,hierarchy = cv2.findContours(im_segm, 1, 2)
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

