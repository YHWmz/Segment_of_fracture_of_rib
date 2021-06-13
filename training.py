# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:03:44 2021

@author: 10513
"""

import numpy as np
from U_net import U_net
from metric import Diceloss, IOU, G_Diceloss, Focalloss,IOUloss,dicelossg 
import torch
from tqdm import tqdm
import nibabel as nib
import os
import matplotlib.pyplot as plt
from torch import sigmoid
import preprocess as ppc
from metric import cal_rec_pre
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
os.environ["CUDA_VISIBLE_DEVICES"]="1,0"

def main():
    train_data_path = '/GPFS/data/yuhaowang/Ribfrac_tmp/ribfrac-train/data/'
    train_datalist = os.listdir(train_data_path)
    train_label_path = '/GPFS/data/yuhaowang/Ribfrac_tmp/ribfrac-train/label/'
    train_labellist = os.listdir(train_label_path)
    model_path = './model_weights_jieduan1.pth'
    
    config = []
    config.append(4); config.append(3)
    config.append([1,32,64,128])
    config.append([96,192,384])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U_net(config)
    #model_weights = torch.load(model_path)
    #model.load_state_dict(model_weights)
    model = torch.nn.DataParallel(model, device_ids=[0,1]).to(device)
    #model = U_net(config).to(device)
    
    opt = torch.optim.Adam(model.parameters(),lr = 0.0005,weight_decay=0.001)
    #scheduler = ReduceLROnPlateau(opt, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=6000,gamma=0.2)
    opt.load_state_dict(opt.state_dict()) 
    scheduler.load_state_dict(scheduler.state_dict())
    epoches = 30
    batch_size = 4

    loss_func1 = dicelossg().to(device)
    loss_func2 = torch.nn.BCEWithLogitsLoss().to(device)
    torch.cuda.empty_cache()
    Processor = ppc.preprocesser(24,(64,64,64))
    
    print('============= start training =============')
    
    print('epoch','%25s'%'Loss','%25s'%'dice','%25s'%'recall','%25s'%'precision')
    for i in range(epoches):
        turns = 0
        loss_mean = 0
        rec_mean = 0
        pre_mean = 0
        dic_mean = 0
        for data_name in tqdm(train_datalist):
            label_name = data_name.split('-')[0]+'-label.nii.gz'
            img = nib.load(train_data_path+data_name)
            label = nib.load(train_label_path+label_name)
            label = np.array(label.dataobj).transpose((2,0,1))
            img = np.array(img.dataobj).transpose((2,0,1))
            index = np.random.choice(np.arange(len(train_datalist)), size=2,\
                                    replace=False)
            tmp0 = np.array(nib.load(train_data_path+train_datalist[index[0]]).dataobj).transpose((2,0,1))
            tmp1 = np.array(nib.load(train_data_path+train_datalist[index[1]]).dataobj).transpose((2,0,1))
            img = np.concatenate([img,tmp0,tmp1],axis=0)
            tmp0 = np.array(nib.load(train_label_path+train_datalist[index[0]].split('-')[0]+'-label.nii.gz').dataobj).transpose((2,0,1))
            tmp1 = np.array(nib.load(train_label_path+train_datalist[index[1]].split('-')[0]+'-label.nii.gz').dataobj).transpose((2,0,1))
            label = np.concatenate([label,tmp0,tmp1],axis=0)
            train_loader = Processor.process(img,label,batch_size) 

            for img_tmp, lab_tmp in train_loader:
                opt.zero_grad()
                img_tmp = torch.FloatTensor(img_tmp).to(device)
                lab_tmp = torch.FloatTensor(lab_tmp).to(device)
                output = model.forward(img_tmp)
                loss1 = loss_func1(output,lab_tmp)
                loss2 = loss_func2(output,lab_tmp)
                loss = loss1*0.5 + loss2 
                
                loss.backward()
                opt.step()
                scheduler.step()
                #scheduler.step(loss)
                dice = 1-loss1.item()
                rec, pre = cal_rec_pre(output, lab_tmp)
                loss_mean = (loss_mean + loss.item())
                rec_mean = (rec + rec_mean)
                pre_mean = (pre + pre_mean)
                dic_mean = (dice + dic_mean)
                turns += 1
                if(turns % 100 == 0):
                    print(i,'%25.5f'%(loss_mean/turns),'%25.5f'%(dic_mean/turns),'%25.5f'%(rec_mean/turns),'%25.5f'%(pre_mean/turns))
                    turns = 0
                    loss_mean = 0
                    rec_mean = 0
                    pre_mean = 0
                    dic_mean = 0
        torch.save(model.module.state_dict(), "./model_weights.pth") 
                   

if __name__ == '__main__':
    main()


