# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:49:50 2021

@author: 10513
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from torch.nn import MaxPool3d,Upsample,AvgPool3d
import torch
import os
from skimage.measure import regionprops
from numpy.lib.type_check import _imag_dispatcher
from tqdm import tqdm
from torch import sigmoid
import torch.utils.data as Data
from skimage import measure
import cv2
import torchvision.transforms as transforms
from numpy import seterr
from scipy import ndimage
from skimage.morphology import disk, remove_small_objects
import os
seterr(all='raise')

#将三维的原图与原label输入，以dataloader的形式输出
class preprocesser():
    def __init__(self, num = 4, crop_size = (64,64,64)):
        super(preprocesser,self).__init__()
        self.num_sample = num
        self.crop_size = crop_size
        self.bone_thre = 0
        
    def process(self,img, label, minibatch=3):
        #img = BoneExtract(img)
        img = imgnomalize(img)
        
        img_sample, label_sample, bone_thre = \
            get_sample(img,label,crop_size=self.crop_size, num_sample=self.num_sample)
            
        if(self.bone_thre != 0):
            self.bone_thre = (self.bone_thre+bone_thre)/2
        else:
            self.bone_thre = bone_thre
        label_sample[label_sample>=1]=1
        dataset = Data.TensorDataset(torch.FloatTensor(img_sample),torch.FloatTensor(label_sample))
        train_loader = Data.DataLoader(dataset=dataset, batch_size=minibatch, shuffle=False, num_workers=0)
        return train_loader

        
class val_preprocesser():
    def __init__(self):
        super(val_preprocesser,self).__init__()
        
    def process(self, img):
        img = imgnomalize(img)
        threshold = 0.454810158599513/2
        
        return img,threshold

#预处理1：骨头部分（ROI）的提取
#让图片只剩下肋骨与脊柱以及附近组织
def BoneExtract(imgs):
    imgs[:,:,400:-1]=-1024
    img_mask = bone_mask(imgs, mode = 5)
    imgs = imgs*img_mask
    imgs[img_mask==0] = -1024
    imgs[imgs<-1024]=-1024
    imgs[imgs>1024]=1024
    return imgs


def bone_mask(img,mode = 5):
    # mode 用来设置filter的大小：3,5,7....
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.Tensor(img).unsqueeze(0).unsqueeze(0).to(device)
    maxpool = MaxPool3d(mode,stride = 1,padding = mode // 2).to(device)
    img = maxpool(img)
    if mode >= 5 :
        threshold = 300
    else:
        threshold = 200
    img = img.cpu().detach_().numpy().squeeze((0,1))
    img_mask = np.where(img>threshold,1,0)
    return img_mask.astype(np.uint8)

#预处理2：归一化
#将三维np数据放入即可进行minmax归一化，返回numpy
def imgnomalize(imgs):
    imgs[imgs<-200] = -200
    imgs[imgs>1024] = 1024
    minv = np.min(imgs)
    maxv = np.max(imgs)
    imgs = imgs - minv
    imgs = imgs / (maxv-minv)
    imgs = imgs*2 - 1
    return imgs

def tmptmp(cen, crop_size, shape):
    a = [cen[0]-crop_size[0]//2,cen[0]+crop_size[0]//2]
    if(cen[0]+crop_size[0]//2>shape[0]):
        a[1] = shape[0]-1
        a[0] = a[1] - crop_size[0]
    elif(cen[0]-crop_size[0]//2<0):
        a[0] = 0
        a[1] = a[0] + crop_size[0]
    
    b = [cen[1]-crop_size[1]//2,cen[1]+crop_size[1]//2]
    if(cen[1]+crop_size[1]//2>shape[1]):
        b[1] = shape[1]-1
        b[0] = b[1] - crop_size[1]
    elif(cen[1]-crop_size[1]//2<0):
        b[0] = 0
        b[1] = b[0] + crop_size[1]
    
    c = [cen[2]-crop_size[2]//2,cen[2]+crop_size[2]//2]
    if(cen[2]+crop_size[2]//2>shape[2]):
        c[1] = shape[2]-1
        c[0] = c[1] - crop_size[2]
    elif(cen[2]-crop_size[2]//2<0):
        c[0] = 0
        c[1] = c[0] + crop_size[2]
    
    return a,b,c

#返回5D的正例训练数据
def get_pos_sample(img, label, pos_cen,num_pos_sample, crop_size=(48,64,64)):
    img_pos = np.zeros((num_pos_sample,1,crop_size[0],crop_size[1],crop_size[2]))
    label_pos = np.zeros((num_pos_sample,1,crop_size[0],crop_size[1],crop_size[2]))
    index = np.random.choice(np.arange(len(pos_cen)), size=num_pos_sample,\
                                     replace=False)
    for i in range(len(index)):
        cen = pos_cen[index[i]]
        a,b,c = tmptmp(cen,crop_size,img.shape)
        img_pos[i,0] = img[a[0]:a[1],\
                           b[0]:b[1],\
                           c[0]:c[1]]
        label_pos[i,0] = label[a[0]:a[1],\
                           b[0]:b[1],\
                           c[0]:c[1]]
    bone_thre = np.mean(img_pos)
    return img_pos, label_pos, bone_thre

def get_neg_sample(img, label, pos_cen, num_neg_sample, crop_size = (48,64,64),num_sample = 4):
    neg_cen = get_neg_centriod(pos_cen, img.shape, crop_size, num_sample)
    img_neg = np.zeros((num_neg_sample,1,crop_size[0],crop_size[1],crop_size[2]))
    label_neg = np.zeros((num_neg_sample,1,crop_size[0],crop_size[1],crop_size[2]))
    index = np.random.choice(np.arange(len(neg_cen)), size=num_neg_sample,\
                                     replace=False)
    for i in range(len(index)):
        cen = neg_cen[index[i]]
        a,b,c = tmptmp(cen,crop_size,img.shape)
        img_neg[i,0] = img[a[0]:a[1],\
                           b[0]:b[1],\
                           c[0]:c[1]]
        label_neg[i,0] = label[a[0]:a[1],\
                           b[0]:b[1],\
                           c[0]:c[1]]
    return img_neg, label_neg

def get_sample(img, label,crop_size=(48,64,64), num_sample=4):
    pos_cen = get_pos_centriod(label)
    
    if(len(pos_cen)>=num_sample):
        img_pos, label_pos, bone_thre = get_pos_sample(img, label, pos_cen, len(pos_cen), crop_size)
        img_neg, label_neg = get_neg_sample(img, label, pos_cen, num_neg_sample=len(pos_cen)//4+1, crop_size=crop_size, num_sample=num_sample)
    else:
        num_pos_sample = min(len(pos_cen),num_sample-1)
        img_pos, label_pos, bone_thre = get_pos_sample(img, label, pos_cen, num_pos_sample, crop_size)
        img_neg, label_neg = get_neg_sample(img, label, pos_cen, num_sample-num_pos_sample, crop_size, num_sample)
    
    img_sample = np.concatenate((img_pos,img_neg),axis=0)
    label_sample = np.concatenate((label_pos, label_neg), axis=0)
    return img_sample, label_sample, bone_thre

#输入3D标注数据，返回list，存储着每个标注凸包的质心
def get_pos_centriod(label):
    pos_cen = []
    for prop in regionprops(label):
        pos_cen.append(tuple([round(x) for x in prop.centroid]))
    return pos_cen

#主要是为了避免正样本与对称负样本不够
#shape为图片的大小，crop_size为patch三维的大小，cen_num为需要的中心个数
def get_spine_centriod(shape, crop_size, cen_num):
    spin_cen = []
    x_min = 260
    x_max = 400
    y_min = 200
    y_max = 300
    z_min = crop_size[0]
    z_max = shape[0] - crop_size[0]
    for i in range(cen_num):
        tmp = (np.random.randint(z_min, z_max),\
               np.random.randint(y_min, y_max),\
                   np.random.randint(x_min, x_max))
        spin_cen.append(tmp)
    return spin_cen

#得到负样本的中心
#pos_cen为正样本的中心，shape为图片大小，crop_size
#为patch三维的大小，num_sample是每张图需要采样的图片数 
def get_neg_centriod(pos_cen, shape, crop_size, num_sample = 4):
    sym_neg_cen = []
    #因为骨头在沿y轴方向是对称的，因此取正样本关于y轴对称的区域就能得到包含肋骨的负样本
    for pos in pos_cen:
        sym_neg_cen.append((pos[0],512 - pos[1], pos[2]))
    if(len(pos_cen) < num_sample//2):
        tmp = num_sample - len(pos_cen) * 2
    else:
        tmp = num_sample//4 + 1
    spin_cen = get_spine_centriod(shape,crop_size, tmp)
    neg_cen = sym_neg_cen + spin_cen
    
    return neg_cen


