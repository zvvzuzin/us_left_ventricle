import os
import numpy as np
from tqdm import tqdm

import cv2

def y1x1y2x2_2_cycxhw(box):
    cy = (box[0] + box[2]) / 2
    cx = (box[1] + box[3]) / 2
    h = box[2] - box[0]
    w = box[3] - box[1]
    if all([True if it <= 1 else False for it in box]):
        return [cy, cx, h, w]
    else:
        return [int(cy), int(cx), int(h), int(w)]
    
def cycxhw_2_y1x1y2x2(box):
    y1 = box[0] - box[2] / 2
    x1 = box[1] - box[3] / 2
    y2 = box[0] + box[2] / 2
    x2 = box[1] + box[3] / 2
    if all([True if it <= 1 else False for it in box]):
        return [y1, x1, y2, x2]
    else:
        return [int(y1), int(x1), int(y2), int(x2)]


class LV_dataset:
    
    def __init__(self,
                 dataset_path='/home/vasily/datasets/US/all', 
                 task='segmentation', 
                 random_state=17, 
                 test_size=0.2, 
                 folds=None, 
                 lv_crop_ratio=1.2, 
                 lv_crop_aspect_ratio=.5,
                ):
        # Initalized all variables.       
        
        np.random.seed(random_state)
        self._dataset_path = dataset_path
#         self._img_size = ()
#         self._channels = channels
#         self._random_state = random_state
        self._task = task
        
        if test_size:
            self._test_size = max(0, test_size)
        else:
            self._test_size = None
                
        if folds:
            self._folds = max(0, folds)
        else:
            self._folds = None
                
        path_to_images = os.path.join(self._dataset_path, 'images')
        path_to_masks = os.path.join(self._dataset_path, 'labels')
        info = self.get_dataset_info(path_to_images, path_to_masks)
        
        if lv_crop_ratio or lv_crop_aspect_ratio:
            self.add_cropping_info(info, lv_crop_ratio=lv_crop_ratio, lv_crop_aspect_ratio=lv_crop_aspect_ratio)
        
        self.info = self.split_data(info)
#         self._fit_data_info()
        # Spliting data on train, valid, test subsets
#         self._indexes_split_data()
        
#         num_train, num_test = train_test_split(np.arange(len(name_patients)), test_size=conf.SPLIT, random_state=17)
    
    
    
    def _chech_catalogues(self, path_to_images, path_to_masks):
        # Check catalogues that are correct.
        
        for it, ((path_img, dirs_img, files_img), (path_msk, dirs_msk, files_msk)) in enumerate(zip(os.walk(path_to_images), os.walk(path_to_masks))):
            if dirs_img != dirs_msk and files_img != files_msk:
                print('Image and masks catalogues are different!')
                return 0
        print('Images and masks catalogues are the same.')
        return 1

    
    def get_dataset_info(self, path_to_images, path_to_masks):
        if self._chech_catalogues(path_to_images, path_to_masks):
            info = []
            for cat in np.sort(os.listdir(path_to_images)):
                for pat in np.sort(os.listdir(os.path.join(path_to_images, cat))):
                    obj1, obj2 = [], []
                    for obj in np.sort(os.listdir(os.path.join(path_to_images, cat, pat))):
                        obj1.append(os.path.join(path_to_images, cat, pat, obj))
                        obj2.append(os.path.join(path_to_masks, cat, pat, obj))
                    info.append({'name': pat, 'category' : cat, 'img_size': cv2.imread(obj1[0]).shape[:2], 'num_frames': len(obj1), 'img_path' : obj1, 'msk_path' : obj2})
            return np.array(info)
    
    
    def add_cropping_info(self, info, lv_crop_ratio=1.2, lv_crop_aspect_ratio=.45):
        # crop_aspect_ratio - the propotional relationship between width and height
        
            
        if lv_crop_ratio:
            lv_crop_ratio = max(1, lv_crop_ratio)
            for patient in tqdm(info, total=len(info)):
                boxes = np.ndarray((len(patient['msk_path']), 4))
                for i, msk_path in enumerate(patient['msk_path']):
                    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                    y, x = np.where(msk != 0)
#                     h, w = msk.shape
                    boxes[i] = min(y), min(x), max(y), max(x)
                box_y1x1y2x2 = [min(boxes[:,0]), 
                                min(boxes[:,1]), 
                                max(boxes[:,2]), 
                                max(boxes[:,3])]
                box_cycxhw = y1x1y2x2_2_cycxhw(box_y1x1y2x2)
                cur_ratio = box_cycxhw[3] / box_cycxhw[2]
                if cur_ratio > lv_crop_aspect_ratio:
                    box_cycxhw[2] = box_cycxhw[3] / lv_crop_aspect_ratio
                elif cur_ratio < lv_crop_aspect_ratio:
                    box_cycxhw[3] = box_cycxhw[2] * lv_crop_aspect_ratio
                box_cycxhw[2] *= lv_crop_ratio
                box_cycxhw[3] *= lv_crop_ratio
                patient['bbox'] = cycxhw_2_y1x1y2x2(box_cycxhw)
                
    
    def split_data(self, info):
        # Splitting data on train, valid and test subsets.
        
        result_info = {}
        if self._test_size:
            shuffle_info = info[np.random.permutation(len(info))]
            info_train = shuffle_info[:int(len(info) * (1 - self._test_size))]
            info_test = shuffle_info[int(len(info) * (1 - self._test_size)):]
            result_info['train'] = info_train
            result_info['test'] = info_test
        else:
            result_info['train'] = info
            
#         if self._folds:
#             if self._folds > 1:
#                 self.kf = KFold(n_splits=self._folds, shuffle=True, random_state=self._random_state)
#                 self._indx_train, self._indx_valid = [], []
#                 for indx_train, indx_valid in self.kf.split(np.arange(len(self._all_indx_train))):
#                     self._indx_train.append(self._all_indx_train[indx_train])
#                     self._indx_valid.append(self._all_indx_train[indx_valid])
#             else:
#                 self._indx_train = self._all_indx_train
        return result_info


    def fit_info(self, subset='train', categories=None, shuffle=False, random_state=1):
        info_data = []
        try:
            for patient in self.info[subset]:
                if categories:
                    if not patient['category'] in categories:
                        continue
                for img_path, msk_path in zip(patient['img_path'], patient['msk_path']):
                    info_data.append({'img_path' : img_path, 'msk_path' : msk_path, 'bbox' : patient['bbox'], 'img_size' : patient['img_size']})

            if shuffle:
                np.random.seed(random_state)
                np.random.shuffle(info_data)

            return info_data
        
        except:
            return None
            
    
    def print_information(self):
        
#         print('All data')
#         for it in self.info:
            
#         print(len([]))
        
        if self._folds:
            for it, (indx_train, indx_valid) in enumerate(zip(self._indx_train, self._indx_valid)):
                cat, num_cat = np.unique(self._cat_pat[indx_valid], return_counts=True)
                print(10*'-', it + 1, ' fold', 10*'-')
                print(cat)
                print(num_cat)
                print(len(sum(self._paths_img[indx_valid], [])))
                
        pathes = []
        categories = []
        for patient in self.info['train']:
            pathes.append(patient['img_path'])
            categories.append(patient['category'])
        cat, num_cat = np.unique(categories, return_counts=True)
        print(10*'-', 'training subset', 10*'-')
        print(cat)
        print(num_cat)
        print(len(sum(pathes, [])))
        
        pathes = []
        categories = []
        for patient in self.info['test']:
            pathes.append(patient['img_path'])
            categories.append(patient['category'])
        cat, num_cat = np.unique(categories, return_counts=True)
        print(10*'-', 'test subset', 10*'-')
        print(cat)
        print(num_cat)
        print(len(sum(pathes, [])))
        
    
    def get_data(self, subset='train', img_shape=(256, 128)):
        
        try:
            prepare_info = self.fit_info(subset=subset)
            images = np.ndarray((len(prepare_info), *img_shape, 1))
            masks = np.ndarray((len(prepare_info), *img_shape, 1))
            for i, item in enumerate(tqdm(prepare_info, total=len(prepare_info))):
                img = cv2.imread(item['img_path'], cv2.IMREAD_GRAYSCALE)[max(0, item['bbox'][0]):min(item['img_size'][0], item['bbox'][2]), 
                                                                         max(0, item['bbox'][1]):min(item['img_size'][1], item['bbox'][3])]
                msk = cv2.imread(item['msk_path'], cv2.IMREAD_GRAYSCALE)[max(0, item['bbox'][0]):min(item['img_size'][0], item['bbox'][2]), 
                                                                         max(0, item['bbox'][1]):min(item['img_size'][1], item['bbox'][3])]
                images[i,...,0] = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
                masks[i,...,0] = cv2.resize(msk, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)

            return images, masks
        except:
            print('problem')
            print(item)
