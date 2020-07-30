import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import SimpleITK as sitk

class LV_Dataset:
    
    def __init__(self,
                 dataset_path='/home/vasily/datasets/us_camus/', 
#                  subset='train',
                 random_state=17, 
#                  test_size=0.2, 
                 folds=9,
#                  lv_crop_ratio=1.2, 
#                  lv_crop_aspect_ratio=.5,
                ):
        # Initalized all variables.       
        
        np.random.seed(random_state)
        self.dataset_path = os.path.join(dataset_path, 'training')
                
        if folds:
            self.folds = max(0, folds)
        else:
            self.folds = None
        
        self.df = self.get_patient_info()
        self.calc_info()
    
    def get_patient_info(self):
        df = pd.DataFrame(columns=['patient', 'image_quality', 'ef', 'fold', 'img_shapes', 'img_names', 'msk_names'])
        for patient in np.sort(os.listdir(self.dataset_path)):
            info = []
            img_shapes = []
            img_names = []
            msk_names = []
            for file in os.listdir(os.path.join(self.dataset_path, patient)):
                if file.endswith('cfg'):
                    f = open(os.path.join(self.dataset_path, patient, file))
                    info.append([it.split()[-1] for it in f.readlines() if it.find('ImageQuality') >= 0 or it.find('LVef') >= 0])
                elif file.endswith('gt.mhd'):
                    f = open(os.path.join(self.dataset_path, patient, file), 'r')
                    for line in [it.split() for it in f.readlines()]:
                        if line[0].startswith('DimSize'):
                            img_shape = []
                            for element in line:
                                try:
                                    img_shape.append(int(element))
                                except:
                                    pass
                        elif line[0].startswith('ElementDataFile'):
                            msk_file_name = line[-1]
                            img_file_name = msk_file_name[:-7] + msk_file_name[-4:]
                    img_shapes.append(img_shape)
                    img_names.append(img_file_name)
                    msk_names.append(msk_file_name)
        
            if info:
                if info[0][0] == 'Poor' or info[1][0] == 'Poor':
                    img_quality = 'Poor'
                elif info[0][0] == 'Medium' or info[1][0] == 'Medium':
                    img_quality = 'Medium'
                elif info[0][0] == 'Good' or info[1][0] == 'Good':
                    img_quality = 'Good'

                if info[0][1] != info[1][1]:
                    print('Problems with EF')
                else:
                    ef = float(info[0][1])

                df = df.append({'patient' : patient, 'image_quality' : img_quality, 'ef': ef, 'img_shapes': img_shapes, 'img_names': img_names, 'msk_names': msk_names}, ignore_index=True) 
        
        if self.folds:
            self.num_patient_in_fold = df['image_quality'].value_counts() / self.folds
            for fold in range(self.folds):
                count = {'Good':0, 'Medium':0, 'Poor':0}
                for index, row in df.iterrows():
                    if not pd.isna(row['fold']):
                        continue
                    quality = row['image_quality']
                    if count[quality] < self.num_patient_in_fold[quality]:
                        count[quality] += 1
                        df['fold'].at[index] = fold
                    if sum(list(count.values())) == 50:
                        break
    
        return df
    
    def calc_info(self):
        
        self.ef_dist = {'<=45': 0, '>=55': 0, 'else': 0}
        self.ef_dist['<=45'] = len(self.df[(self.df['ef'] <= 45)]) / len(self.df)
        self.ef_dist['>=55'] = len(self.df[(self.df['ef'] >= 55)]) / len(self.df)
        self.ef_dist['else'] = len(self.df[(self.df['ef'] > 45) & (self.df['ef'] < 55)]) / len(self.df)
        
        self.quality_dist = dict(self.df['image_quality'].value_counts() / len(self.df))
        
        self.ef_fold_dist = {}
        for fold in range(self.folds):
            self.ef_fold_dist[fold] = {'<=45': 0, '>=55': 0, 'else': 0}
            self.ef_fold_dist[fold]['<=45'] = len(self.df[(self.df['fold'] == fold) & (self.df['ef'] <= 45)]) / len(self.df) * self.folds
            self.ef_fold_dist[fold]['>=55'] = len(self.df[(self.df['fold'] == fold) & (self.df['ef'] >= 55)]) / len(self.df) * self.folds
            self.ef_fold_dist[fold]['else'] = len(self.df[(self.df['fold'] == fold) & (self.df['ef'] > 45) & (self.df['ef'] < 55)]) / len(self.df) * self.folds
           
        for fold in range(self.folds):
            self.quality_fold_dist = dict(self.df[self.df['fold'] == fold]['image_quality'].value_counts() / len(self.df) * self.folds )
            
    def get_pathes_train_val(self, fold):
        img_train_path = [os.path.join(self.dataset_path, it.split('_')[0], it) for it in sum(list(self.df[self.df['fold'] != fold]['img_names']),[])]
        msk_train_path = [os.path.join(self.dataset_path, it.split('_')[0], it) for it in sum(list(self.df[self.df['fold'] != fold]['msk_names']),[])]
        img_valid_path = [os.path.join(self.dataset_path, it.split('_')[0], it) for it in sum(list(self.df[self.df['fold'] == fold]['img_names']),[])]
        msk_valid_path = [os.path.join(self.dataset_path, it.split('_')[0], it) for it in sum(list(self.df[self.df['fold'] == fold]['msk_names']),[])]
        train_shape = sum(list(self.df[self.df['fold'] != fold]['img_shapes']), [])
        valid_shape = sum(list(self.df[self.df['fold'] == fold]['img_shapes']), [])
        train = zip(img_train_path, msk_train_path, train_shape)
        valid = zip(img_valid_path, msk_valid_path, valid_shape)
        
        return list(train), list(valid)

class Data(Dataset):
    def __init__(self, pathes_shapes, normalize=True, img_size=(256, 512)):
        self.pathes = pathes_shapes
        self.img_size = img_size
        self.normalize = normalize
            
    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, index):
        f = open(self.pathes[index][0], 'rb')
        img = np.fromfile(f, dtype=np.uint8)
        f = open(self.pathes[index][1], 'rb')
        msk = np.fromfile(f, dtype=np.uint8)
        shape = self.pathes[index][2]
        img = cv2.resize(img.reshape((shape[1], shape[0], shape[2])), self.img_size, cv2.INTER_LINEAR) / 255 #- 0.5
        msk = cv2.resize(msk.reshape((shape[1], shape[0], shape[2])), self.img_size, interpolation=0)
        if self.normalize:
            img -= 0.5
            img *= 2
        return np.expand_dims(img, axis=0).astype(np.float32), msk.astype(np.int)