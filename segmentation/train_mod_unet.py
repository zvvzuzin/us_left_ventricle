import numpy as np
from tqdm import tqdm
from echo_lv.data import LV_CAMUS_Dataset, LV_EKB_Dataset
from echo_lv.metrics import dice as dice_np
from echo_lv.utils import AverageMeter
from echo_lv.segmentation.cnn import mUNet

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

batch = 8
epochs = 50
folds = 9

lv_camus = LV_CAMUS_Dataset(img_size = (512,512), classes = {0, 1}, folds=folds)

weight = 10 * torch.ones((1,1,512,512), device=device)
criterion = smp.utils.losses.DiceLoss(activation='sigmoid') + smp.utils.losses.BCEWithLogitsLoss(pos_weight=weight) 
dice = smp.utils.metrics.Fscore(activation='sigmoid', threshold=None).to(device)#Dice()
iou = smp.utils.metrics.IoU(activation='sigmoid', threshold=None).to(device)
result = []


for fold in range(0, folds):
    header = True
    
    model = mUNet(in_channels = 1, out_channels = 1, dropout=0.0).to(device)
    
    optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'lr': 1e-4, 'betas' : (0.95, 0.99)},   
        ])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    t = tqdm(total=epochs, 
            bar_format='{desc} | {postfix[0]}/'+ str(epochs) +' | ' +
            '{postfix[1]} : {postfix[2]:>2.4f} | {postfix[3]} : {postfix[4]:>2.4f} | {postfix[5]} : {postfix[6]:>2.4f} |' +
            '{postfix[7]} : {postfix[8]:>2.4f} | {postfix[9]} : {postfix[10]:>2.4f} | {postfix[11]} : {postfix[12]:>2.4f} |',
            postfix=[0, 'loss', 0, 'dice_lv', 0,  'jaccard_lv', 0,
                     'val_loss', 0, 'val_dice_lv', 0, 'val_jaccard_lv', 0], 
            desc = 'Train modified unet on fold ' + str(fold),
            position=0, leave=True
         )            
        
       
    for epoch in range(0, epochs):
        average_total_loss = AverageMeter()
        average_dice = AverageMeter()
        average_jaccard = AverageMeter()
        
        model.train()
#         torch.cuda.empty_cache()

        t.postfix[0] = epoch + 1
        
        lv_camus.set_state('train', fold)
        train_loader = DataLoader(lv_camus, batch_size=batch, shuffle=True, num_workers=2)
        for data in train_loader:
            inputs, masks, *_ = data

            inputs=inputs.to(device).float()
            masks=masks.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, masks)

            average_total_loss.update(loss.data.item())
            average_dice.update(dice(outputs, masks).item())
            average_jaccard.update(iou(outputs, masks).item())

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
#         torch.cuda.empty_cache()
        
        lv_camus.set_state('valid', fold)
        valid_loader = DataLoader(lv_camus, batch_size=batch // 2, shuffle=False, num_workers=1)
        for data in valid_loader:
            inputs, masks, *_ = data

            inputs=inputs.to(device).float()
            masks=masks.to(device).float()

            outputs = model(inputs)

            loss = criterion(outputs, masks)

            average_val_total_loss.update(loss.data.item())
            average_val_dice.update(dice(outputs, masks).item())
            average_val_jaccard.update(iou(outputs, masks).item())

            t.postfix[8] = average_val_total_loss.average()
            t.postfix[10] = average_val_dice.average()
            t.postfix[12] = average_val_jaccard.average()
            t.update(n=0)
            
        
        result = [optimizer.param_groups[0]['lr'], 
                  average_total_loss.average(), 
                  average_dice.average(), 
                  average_jaccard.average(),
                  average_val_total_loss.average(), 
                  average_val_dice.average(), 
                  average_val_jaccard.average()
                 ]
        df = pd.DataFrame(np.array([result]), columns=['lr', 'loss', 'dice', 'jaccard', 
                                                     'val_loss', 'val_dice', 'val_jaccard'])
        df.to_csv('cnn/mod_unet/result_munet_'+ str(fold) +'.csv', mode='a', header=header, index=False,)
        header=None
        scheduler.step()
    t.close()
    