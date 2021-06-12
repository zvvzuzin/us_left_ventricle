import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import cv2
from skimage.transform import resize

class LV_EKB_Dataset:
    
    def __init__(self,
                 dataset_path='/home/vasily/datasets/us_ekb', 
                 task='segmentation',
                 img_size=None,
                 patient_cat = {'Norma_Old', 'Norma', 'Pathology_Old', 'DKMP'},
                 only_first_frames = False,
                 random_state=17, 
                 train_ratio=None,
                 valid_ratio=None, 
                 normalize = False,
                 folds=None,
                 shuffle=True,
                 lv_crop_ratio=None, #1.2
                 lv_crop_aspect_ratio=None, #0.5
                ):
        # Initalized all variables.       
        
        np.random.seed(random_state)
        self._dataset_path = dataset_path
        self._task = task
        self.categories = patient_cat
        self.lv_crop_ratio = lv_crop_ratio
        self.lv_crop_aspect_ratio = lv_crop_aspect_ratio
        self.folds = None
        self.shuffle = shuffle
        self.img_size = img_size
        self.random_state = random_state
        self.normalize = normalize
        self.only_first_frames = only_first_frames
        
        if self.folds:
            self.num_folds = max(0, folds)
            self.train_ratio = None
            self.valid_ratio = None
            self.test_ratio = None
        else:
            self.num_folds = None
            if train_ratio:
                self.train_ratio = train_ratio
                if valid_ratio:
                    self.valid_ratio = valid_ratio
                    self.test_ratio = 1 - train_ratio - valid_ratio
                else:
                    self.valid_ratio = 0
                    self.test_ratio = 1 - train_ratio
            else:
                self.train_ratio = 1
                self.valid_ratio = 0
                self.test_ratio = 0
                
        info = self.get_dataset_info(self._dataset_path)
        
        if lv_crop_ratio or lv_crop_aspect_ratio:
            self.add_cropping_info(info, lv_crop_ratio=lv_crop_ratio, lv_crop_aspect_ratio=lv_crop_aspect_ratio)
    
    
    def _chech_catalogues(self, path_to_dataset):
        # Check catalogues that are correct.
        path_to_images = os.path.join(path_to_dataset, 'images')
        path_to_masks = os.path.join(path_to_dataset, 'labels')
        for it, ((path_img, dirs_img, files_img), (path_msk, dirs_msk, files_msk)) in enumerate(zip(os.walk(path_to_images), os.walk(path_to_masks))):
            if dirs_img != dirs_msk or files_img != files_msk:
                print('Image and masks catalogues are different!')
                return False
        print('Dataset is correct.')
        return True

    
    def get_dataset_info(self, path_to_dataset):
        if self._chech_catalogues(path_to_dataset):
            self.df_images = pd.DataFrame(columns=['patient', 'category', 'img_shape', 'obj_name', 'bbox'])
            self.df_patients = pd.DataFrame(columns=['patient', 'category', 'img_shape', 'num_frames'])
            
            for category in np.sort(os.listdir(os.path.join(path_to_dataset, 'images'))):

                if not category in self.categories:
                    continue
                
                for patient in np.sort(os.listdir(os.path.join(path_to_dataset, 'images', category))):
                    num_frame = 0
                    img_size = None
                    if self.lv_crop_ratio or self.lv_crop_aspect_ratio:
                        box = self.get_cropping(os.path.join(path_to_dataset, 'labels', category, patient, obj))
                    else:
                        box = None
                        
                    for it, obj in enumerate(np.sort(os.listdir(os.path.join(path_to_dataset, 'images', category, patient)))):
                        num_frame += 1
                        if it == 0:
#                             try:
                            img_size = tuple(cv2.imread(os.path.join(path_to_dataset, 'images', category, patient, obj)).shape[:2])
#                             except:
#                                 print(os.path.join(path_to_dataset, 'images', category, patient, obj))
#                         msk = cv2.imread(os.path.join(path_to_dataset, 'labels', category, patient, obj), cv2.IMREAD_GRAYSCALE)
                        self.df_images = self.df_images.append({'patient' : patient, 
                                                                'category' : category,
                                                                'img_shape' : img_size,
                                                                'obj_name' : obj,
                                                                'bbox' : box,
                                                                }, ignore_index=True)
                        if self.only_first_frames:
                            break
        
                    
                    self.df_patients = self.df_patients.append({'patient' : patient,
                                                                'category' : category,
                                                                'img_shape' : img_size,
                                                                'num_frames' : num_frame,
                                                               }, ignore_index=True)
            
            if self.shuffle:
                self.df_images = self.df_images.sample(frac=1, random_state=self.random_state)
                
            return
    
    
    def get_cropping(self, msk_pat_path, lv_crop_ratio=1.2, lv_crop_aspect_ratio=.45):
        # crop_aspect_ratio - the propotional relationship between width and height
        
        if lv_crop_ratio:
            lv_crop_ratio = max(1, lv_crop_ratio)
            boxes = np.ndarray((len(patient['msk_path']), 4))
            for i, msk_path in enumerate(os.listdir(msk_pat_path)):
                msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                y, x = np.where(msk != 0)
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
            return cycxhw_2_y1x1y2x2(box_cycxhw)
                
    
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
        
    
    def get_data(self, subset='train', img_shape=(256, 256)):
        
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
    
    def get_sequence(self, patient, category):
        objects = self.df_images[(self.df_images['patient'] == patient) & (self.df_images['category'] == category)].sort_values('obj_name')
        imgs, msks = [], []
        for i, obj in objects.iterrows():
            img = cv2.imread(os.path.join(self._dataset_path, 'images', obj['category'], obj['patient'], obj['obj_name']), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(os.path.join(self._dataset_path, 'labels', obj['category'], obj['patient'], obj['obj_name']), cv2.IMREAD_GRAYSCALE)
            if obj['bbox']:
                img = img[max(0, obj['bbox'][0]):min(obj['img_size'][0], obj['bbox'][2]), 
                          max(0, obj['bbox'][1]):min(obj['img_size'][1], obj['bbox'][3])]
                msk = msk[max(0, obj['bbox'][0]):min(obj['img_size'][0], obj['bbox'][2]), 
                          max(0, obj['bbox'][1]):min(obj['img_size'][1], obj['bbox'][3])]

            if self.img_size:
                img = resize(img, self.img_size, preserve_range=True, anti_aliasing=True, order=1)
                msk = resize(msk, self.img_size, preserve_range=True, anti_aliasing=False, order=0)
            
            if self.normalize:
                img = img / 255
                msk = msk / 255
                
            imgs.append(img)
            msks.append(msk)
            
        return imgs, msks
    
    
    def __len__(self):
        return self.df_images.shape[0]
            
    
    def __getitem__(self, index):
        obj = self.df_images.iloc[index]
        img = cv2.imread(os.path.join(self._dataset_path, 'images', obj['category'], obj['patient'], obj['obj_name']), 
                         cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(self._dataset_path, 'labels', obj['category'], obj['patient'], obj['obj_name']), 
                         cv2.IMREAD_GRAYSCALE)
        if obj['bbox']:
            img = img[max(0, obj['bbox'][0]):min(obj['img_size'][0], obj['bbox'][2]), 
                      max(0, obj['bbox'][1]):min(obj['img_size'][1], obj['bbox'][3])]
            msk = msk[max(0, obj['bbox'][0]):min(obj['img_size'][0], obj['bbox'][2]), 
                      max(0, obj['bbox'][1]):min(obj['img_size'][1], obj['bbox'][3])]
        
        if self.img_size:
            img = resize(img, self.img_size, preserve_range=True, order=1)
            msk = resize(msk, self.img_size, preserve_range=True, anti_aliasing=False, order=0)
        
        if self.normalize:
            img = img / 255
            msk = msk / 255
            
        return np.expand_dims(img, axis=0), np.expand_dims(msk, axis=0)