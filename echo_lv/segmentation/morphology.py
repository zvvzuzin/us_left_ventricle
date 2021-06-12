import numpy as np
from ..lv import  get_main_points, get_contour_points, interpolate_contour, cont2area, area2cont, pol2cart
import cv2
from scipy.stats import multivariate_normal

class Morph_segmentation:
    def __init__(self, img, msk):
        self.fit(img, msk)
     
    
    def fit(self, img, msk):
        self.gauss = cv2.blur(msk, (50,50))
        temp_img = self.gauss * (1 - img)
        lv_img = temp_img * msk
        counts, med_bins, *_ = np.histogram(lv_img[lv_img != 0].ravel(), bins=100)

        cdf = np.cumsum(counts) / np.sum(counts)
        self.thresh = [(med_bins[it] + med_bins[it+1]) / 2 for it in range(len(med_bins) - 1)][np.argmax(cdf > 0.10)]
     
    
    def predict(self, img):
        temp_img = self.gauss * (1 - img)
        bw = temp_img > self.thresh

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats((255*bw).astype(np.uint8), connectivity=4)
        stats[0,-1] = 0
        label = np.argmax(stats[:,-1])
        bw = labels == label

        inv_bw = np.bitwise_not(bw)
        mask = np.zeros((inv_bw.shape[0]+2, inv_bw.shape[1]+2), np.uint8)
        _, holes_bw, *_ = cv2.floodFill((255*inv_bw).astype(np.uint8), mask, (0,0), 0)
        bw = bw.astype(float) + holes_bw.astype(float) / 255 
        
        kernel = np.ones((15,15))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations = 5)
        
        apex, base_l, base_r, *_ = get_main_points(area2cont(bw), space='polar')
        phi, rho, cent_x, cent_y = get_contour_points(area2cont(bw), kind='contour', space='polar')
        phi[phi > (base_r[0] + base_l[0]) / 2] -= 2*np.pi

        delta = np.pi / 8
        new_phi, new_rho = [], []
        for i in np.linspace(min(phi), max(phi), 1000):
            new_phi.append(i)

            indexes = (phi > i - delta / 2) & (phi < i + delta / 2)
            new_rho.append(np.mean(rho[indexes]))

        indexes = np.argsort(new_phi)
        new_phi = np.array(new_phi)[indexes]
        new_rho = np.array(new_rho)[indexes]
        x, y = pol2cart(new_phi, new_rho, cent_x, cent_y)
        x, y = interpolate_contour(x, y)
        bw = np.zeros(bw.shape)
        bw[y.astype(int),x.astype(int)] = 1
        return cont2area(bw)