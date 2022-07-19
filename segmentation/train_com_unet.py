import numpy as np
from tqdm import tqdm
from echo_lv.data import LV_CAMUS_Dataset, LV_EKB_Dataset
from echo_lv.metrics import dice as dice_np
from echo_lv.utils import AverageMeter
from echo_lv.segmentation.cnn import UNet

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import sigmoid
from torchvision import datasets, transforms, models
import segmentation_models_pytorch as smp
import pandas as pd


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

random_state = 17
torch.manual_seed(random_state)
torch.cuda.manual_seed(random_state)
torch.backends.cudnn.deterministic = True

batch = 4
epochs = 100
folds = 9

lv_camus = LV_CAMUS_Dataset(img_size = (388,388), classes = {0, 1}, folds=folds)

weight = 10 * torch.ones((1,1,388,388), device=device).to(device)
criterion = smp.utils.losses.BCEWithLogitsLoss(pos_weight=weight).to(device)
# criterion = smp.utils.losses.DiceLoss(activation='sigmoid')# + smp.utils.losses.BCEWithLogitsLoss(pos_weight=weight) 
dice = smp.utils.metrics.Fscore(activation='sigmoid', threshold=None).to(device)#Dice()
iou = smp.utils.metrics.IoU(activation='sigmoid', threshold=None).to(device)

for fold in range(0, folds):
    header = True
    
    model = UNet(n_channels = 1, n_classes = 1, bilinear=False).to(device)
    
    optimizer = torch.optim.SGD([
            {'params': model.parameters(), 'lr': 1e-4, 'momentum' : 0.99},   
        ])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    t = tqdm(total=epochs, 
            bar_format='{desc} | {postfix[0]}/'+ str(epochs) +' | ' +
            '{postfix[1]} : {postfix[2]:>2.4f} | {postfix[3]} : {postfix[4]:>2.4f} | {postfix[5]} : {postfix[6]:>2.4f} |' +
            '{postfix[7]} : {postfix[8]:>2.4f} | {postfix[9]} : {postfix[10]:>2.4f} | {postfix[11]} : {postfix[12]:>2.4f} |',
            postfix=[0, 'loss', 0, 'dice_lv', 0,  'jaccard_lv', 0,
                     'val_loss', 0, 'val_dice_lv', 0, 'val_jaccard_lv', 0], 
            desc = 'Train common unet on fold ' + str(fold),
            position=0, leave=True
         )            
        
       
    for epoch in range(0, epochs):
        average_total_loss = AverageMeter()
        average_dice = AverageMeter()
        average_jaccard = AverageMeter()
        
#         torch.cuda.empty_cache()
        model.train()

        t.postfix[0] = epoch + 1
        
        lv_camus.set_state('train', fold)
        train_loader = DataLoader(lv_camus, batch_size=batch, shuffle=True, num_workers=2)
        for data in train_loader:

            inputs, masks, *_ = data
            shape = inputs.shape
            inputs = torch.cat([torch.zeros((shape[0], shape[1], shape[2], 92), dtype=float), inputs, torch.zeros((shape[0], shape[1], shape[2], 92), dtype=float)], axis=3)
            shape = inputs.shape
            inputs = torch.cat([torch.zeros((shape[0], shape[1], 92, shape[3]), dtype=float), inputs, torch.zeros((shape[0], shape[1], 92, shape[3]), dtype=float)], axis=2)
            inputs=inputs.to(device).float()
            masks=masks.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, masks)
            d = dice(outputs, masks)
            j = iou(outputs, masks)

            average_total_loss.update(loss.data.item())
            average_dice.update(d.item())
            average_jaccard.update(j.item())

            loss.backward()
            optimizer.step()

            t.postfix[2] = average_total_loss.average()
            t.postfix[4] = average_dice.average()
            t.postfix[6] = average_jaccard.average()
            t.update(n=1)

        # validation
        average_val_total_loss = AverageMeter()
        average_val_dice = AverageMeter()
        average_val_jaccard = AverageMeter()
        
        model.eval()
        
        lv_camus.set_state('valid', fold)
        valid_loader = DataLoader(lv_camus, batch_size=batch, shuffle=False, num_workers=1)
        for data in valid_loader:
            inputs, masks, *_ = data
            shape = inputs.shape
            inputs = torch.cat([torch.zeros((shape[0], shape[1], shape[2], 92), dtype=float), inputs, torch.zeros((shape[0], shape[1], shape[2], 92), dtype=float)], axis=3)
            shape = inputs.shape
            inputs = torch.cat([torch.zeros((shape[0], shape[1], 92, shape[3]), dtype=float), inputs, torch.zeros((shape[0], shape[1], 92, shape[3]), dtype=float)], axis=2)
            inputs=inputs.to(device).float()
            masks=masks.to(device).float()

            outputs = model(inputs)

            loss = criterion(outputs, masks)
            d = dice(outputs, masks)
            j = iou(outputs, masks)

            average_val_total_loss.update(loss.data.item())
            average_val_dice.update(d.item())
            average_val_jaccard.update(j.item())

            t.postfix[8] = average_val_total_loss.average()
            t.postfix[10] = average_val_dice.average()
            t.postfix[12] = average_val_jaccard.average()
            t.update(n=0)
        
        result = [optimizer.param_groups[0]['lr'], average_total_loss.average(), average_dice.average(), average_jaccard.average(),
                  average_val_total_loss.average(), average_val_dice.average(), average_val_jaccard.average()
                 ]
        df = pd.DataFrame(np.array([result]), columns=['lr', 'loss', 'dice', 'jaccard', 
                                                     'val_loss', 'val_dice', 'val_jaccard'])
        df.to_csv('cnn/com_unet/result_cunet_'+ str(fold) +'.csv', mode='a', header=header, index=False,)
        header=None
        scheduler.step()
    t.close()
    