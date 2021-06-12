import numpy as np
from scipy.ndimage import filters
from skimage.measure import find_contours

class GAC():
    def __init__(self, msk, w = (0.7, 0.2, 0.1), sigma=2, iters = 500, C = 10, v= 1., dt = 1.):
        self.cur_iter = 0
        self.final_iter = iters
        self.msk = msk
        self.w = w
        self.v = v
        self.dt = dt
        self.C = C
        self.sigma = sigma
        self.shape = msk.shape

        
    def init_phi(self, msk):
        init_radius = 10
        mask_area = - np.ones((2 * init_radius, 2 * init_radius))
        for y, x in zip(*np.where(mask_area == -1)):
            if ((x - init_radius) ** 2 + (y - init_radius) ** 2) ** (1/2) <= 10:
                mask_area[y, x] = 1
        phi = - np.ones(msk.shape[:2])
        y, x = np.where(msk != 0)
        centr_x, centr_y = np.mean(x), np.mean(y)
        phi[int(centr_y - init_radius):int(centr_y + init_radius), int(centr_x - init_radius):int(centr_x + init_radius)] = mask_area
        return phi
        
    
    def gradient(self, img):
        return np.gradient(img)

    
    def norm(self, img, axis=0):
        return np.sqrt(np.sum(np.square(img), axis=axis))
    

    def velocity(self, img):
        return 1. / (1. + self.C * self.norm(self.gradient(img)))
    
    
    def curvature(self, f):
        fy, fx = self.gradient(f)
        norm = np.sqrt(fx**2 + fy**2)
        Nx = fx / (norm + 1e-8)
        Ny = fy / (norm + 1e-8)
        return self.div(Nx, Ny)


    def div(self, fx, fy):
        fyy, fyx = self.gradient(fy)
        fxy, fxx = self.gradient(fx)
        return fxx + fyy


    def dot(self, x, y, axis=0):
        return np.sum(x * y, axis=axis)
    
    
    def predict(self, img, contin=False, it=None):
        if not contin or not self.cur_iter:
            self.cur_iter = 0
            self.phi = self.init_phi(self.msk)
        
        if it is not None:
            self.final_iter = it
        
        img = filters.gaussian_filter(img, sigma=self.sigma)
        g = self.velocity(img)
        dg = self.gradient(g)

        
        for i in range(self.cur_iter, self.final_iter):
            dphi = self.gradient(self.phi)
            dphi_norm = self.norm(dphi)
            kappa = self.curvature(self.phi)

            smoothing = g * kappa * dphi_norm
            balloon = g * self.v * dphi_norm 
            attachment = self.dot(np.concatenate([np.expand_dims(d, axis=0) for d in dphi], axis=0), 
                                  np.concatenate([np.expand_dims(d, axis=0) for d in dg], axis=0))

            dphi_t = self.w[0]*smoothing + self.w[1]*balloon + self.w[2]*attachment

            self.phi = self.phi + self.dt * dphi_t
            
        self.cur_iter = self.final_iter    
        return self.phi >= 0
    

class Level_Set():
    def __init__(self, msk, sigma=2, iters = 500, C = 10, v= 1., dt = 1.):
        self.cur_iter = 0
        self.final_iter = iters
        self.msk = msk
        self.phi = self.init_phi(self.msk)
        self.v = v
        self.dt = dt
        self.C = C
        self.sigma = sigma
        self.shape = msk.shape

        
    def init_phi(self, msk):
        init_radius = 10
        mask_area = - np.ones((2 * init_radius, 2 * init_radius))
        for y, x in zip(*np.where(mask_area == -1)):
            if ((x - init_radius) ** 2 + (y - init_radius) ** 2) ** (1/2) <= 10:
                mask_area[y, x] = 1
        phi = - np.ones(msk.shape[:2])
        y, x = np.where(msk != 0)
        centr_x, centr_y = np.mean(x), np.mean(y)
        phi[int(centr_y - init_radius):int(centr_y + init_radius), int(centr_x - init_radius):int(centr_x + init_radius)] = mask_area
        return phi
        
    
    def gradient(self, img):
        return np.gradient(img)

    
    def norm(self, img, axis=0):
        return np.sqrt(np.sum(np.square(img), axis=axis))
    

    def velocity(self, img):
        return 1. / (1. + self.C * self.norm(self.gradient(img)))
    
    
    def predict(self, img, contin=False, it=None):
        if not contin:
            self.cur_iter = 0
            self.phi = self.init_phi(self.msk)
        
        if it is not None:
            self.final_iter = it
        
        img = filters.gaussian_filter(img, sigma=self.sigma)
        g = self.velocity(img)
        
        
        for i in range(self.cur_iter, self.final_iter):
            dphi = self.gradient(self.phi)
            dphi_norm = self.norm(dphi)

            dphi_t = g * dphi_norm

            self.phi = self.phi + self.dt * dphi_t
        
        self.cur_iter = self.final_iter    
        return self.phi >= 0