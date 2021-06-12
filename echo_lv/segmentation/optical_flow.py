import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian
from echo_lv.lv import get_contour_points, area2cont, cont2area, interpolate_contour

def window_image(img, cent_point, window):
        y0 = int(np.round(cent_point[0]) - window // 2)
        y1 = int(np.round(cent_point[0]) + window // 2 + 1)
        x0 = int(np.round(cent_point[1]) - window // 2)
        x1 = int(np.round(cent_point[1]) + window // 2 + 1)
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if y1 > img.shape[0]:
            y1 = img.shape[0]
        if x1 > img.shape[1]:
            x1 = img.shape[1]    
        img = img[y0:y1, x0:x1]
        if img.shape[0] != window:
            if y0 == 0:
                img = np.concatenate((np.zeros((window - img.shape[0], img.shape[1])), img), axis=0)
            elif y1 == img.shape[0]:
                img = np.concatenate((img, np.zeros((window - img.shape[0], img.shape[1]))), axis=0)
        if img.shape[1] != window:
            if x0 == 0:
                img = np.concatenate((np.zeros((img.shape[0], window - img.shape[1])), img), axis=1)
            elif x1 == img.shape[1]:
                img = np.concatenate((img, np.zeros((img.shape[0], window - img.shape[1]))), axis=1)
        return img

class LucasKanade:
    def __init__(self, gauss_layers = 0, window = 7):
        self.gauss_layers = gauss_layers
        self.window = window
        
        
    def get_points(self, imgs, points):
        points_results = [points]
        for ind in range(0, len(imgs)-1):
            r_1_imgs = list(pyramid_gaussian(imgs[ind], max_layer=self.gauss_layers))
            r_2_imgs = list(pyramid_gaussian(imgs[ind+1], max_layer=self.gauss_layers))
            new_points = []
            for point in points:
                flow = np.array([[0], [0]])
                for l, (img_1, img_2) in enumerate(zip(r_1_imgs[::-1], r_2_imgs[::-1])):
                    img1 = window_image(img_1, 
                                        (point[0] / 2 ** (self.gauss_layers - l),
                                         point[1] / 2 ** (self.gauss_layers - l)),
                                        self.window)
                    
                    img2 = window_image(img_2, 
                                        ((point[0] + flow[1]) / 2 ** (self.gauss_layers - l),
                                         (point[1] + flow[0]) / 2 ** (self.gauss_layers - l)), 
                                        self.window)
                        
                    f_y, f_x = np.gradient(img1)
                    f_t = img1 - img2
                    A = np.array([[np.sum(f_x ** 2), np.sum(f_x * f_y)],
                                 [np.sum(f_x * f_y), np.sum(f_y ** 2)]])
                    B = np.array([[np.sum(f_x * f_t)],
                                [np.sum(f_y * f_t)]
                                 ])
                    solv_flow = np.linalg.lstsq(A, B, rcond=None)[0]#np.matmul(np.linalg.inv(A), B)
                    flow = 2 * (flow + solv_flow)
                    
                new_points.append((point[0] + int(flow[1]), point[1] + int(flow[0])))
            points = new_points
            points_results.append(points)
        return points_results
    
    
    def predict(self, imgs, true_msk):
        cont_x, cont_y, *_ = get_contour_points(area2cont(true_msk), kind='contour', num = 9)
        points = [(y, x) for x, y in zip(cont_x, cont_y)]
        results = self.get_points(imgs, points)
        msks = []
        for res in results:
            x = [p[1] for p in res]
            y = [p[0] for p in res]
            mask = np.zeros((512,512))
            p_x, p_y = interpolate_contour(np.array(x), np.array(y),)
            mask[p_y, p_x] = 1
            msks.append(cont2area(mask))
        return msks
            
    