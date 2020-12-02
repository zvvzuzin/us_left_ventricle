import numpy as np

def jaccard(im_true, im_pred):
    return len(np.where(im_true * im_pred != 0)[0]) / (len(np.where(im_true != 0)[0]) + len(np.where(im_pred != 0)[0]) - len(np.where(im_true * im_pred != 0)[0]))
    
def dice(im_true, im_pred):
    return 2 * len(np.where(im_true * im_pred != 0)[0]) / (len(np.where(im_true != 0)[0]) + len(np.where(im_pred != 0)[0]))