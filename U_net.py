# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:09:30 2021

@author: DELL
"""

import torch
from torch.nn.init import xavier_normal_
from torch import nn

class mirror_conv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size):
        super(mirror_conv,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size)
        #xavier_normal_(self.conv.weight)
        
    def forward(self,img):
        pad = (self.kernel_size - 1)//2
        up,down = img[:,:,:pad,:,:],img[:,:,-pad:,:,:]
        up = torch.flip(up,dims = [2]); down = torch.flip(down,dims = [2])
        img = torch.cat([up,img,down],dim = 2)
        
        up,down = img[:,:,:,:pad,:],img[:,:,:,-pad:,:]
        up = torch.flip(up,dims = [3]); down = torch.flip(down,dims = [3])
        img = torch.cat([up,img,down],dim = 3)
        
        up,down = img[:,:,:,:,:pad],img[:,:,:,:,-pad:]
        up = torch.flip(up,dims = [4]); down = torch.flip(down,dims = [4])
        img = torch.cat([up,img,down],dim = 4)
        
        return self.conv(img)
    
class circular_conv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size):
        super(circular_conv,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channel, out_channel, kernel_size)
        xavier_normal_(self.conv.weight)
        
    def forward(self,img):
        pad = self.kernel_size - 1
        up,down = img[:,:,:pad,:,:],img[:,:,-pad:,:,:]
        img = torch.cat([down,img,up],dim = 2)
        
        up,down = img[:,:,:,:pad,:],img[:,:,:,-pad:,:]
        img = torch.cat([down,img,up],dim = 3)
        
        up,down = img[:,:,:,:,:pad],img[:,:,:,:,-pad:]
        img = torch.cat([down,img,up],dim = 4)
        
        return self.conv(img)
        
class conv_block(nn.Module):
    def __init__(self,block,depth,in_channel):
        super(conv_block,self).__init__()
        '''
        depth指的是第几级conv
        layer_num指的是在一个级内从输入到输出所需要的卷积层数
        in_channel指的是第一个输入的图像的channel
        out_channel指的是第一个输出图像的channel
        '''
        if block == 'encoder':
            if depth == 0:
                channel1 = [in_channel,16]
                channel2 = [16,32]
            else:
                channel1 = [in_channel,in_channel]
                channel2 = [in_channel,in_channel*2]
        else:
            channel1 = [in_channel,in_channel // 3]
            channel2 = [in_channel//3, in_channel // 3]
            
        self.sequential = nn.Sequential()
        for i in range(2):
            #self.sequential.add_module('conv_{}_{}'.format(depth,i),
            #                            mirror_conv(channel1[i], channel2[i], 3)) #U-Net论文里面是3
            self.sequential.add_module('conv_{}_{}'.format(depth,i),
                                       nn.Conv3d(channel1[i], channel2[i], 3,padding=1))
            #xavier_normal_(self.sequential['conv_{}_{}'.format(depth,i)].weight)
            xavier_normal_(self.sequential[i*3].weight)
            self.sequential.add_module('BN_{}_{}'.format(depth,i),module = nn.BatchNorm3d(channel2[i]))
            self.sequential.add_module('Relu_{}_{}'.format(depth,i),module = nn.LeakyReLU())# nn.Relu()
            '''
            输入输出的channel数需要再考虑
            由于conv在decoder和encoder都会用
            所以在decoder和encoder的channel数可能不一样
            '''
    
    def forward(self,data):
        return self.sequential(data)
    
class encoder_block(nn.Module):
    def __init__(self,depth_num,in_channel):
        super(encoder_block,self).__init__()
        '''
        depth_num指的是encoder的级数
        inchannel指的是第i级的第一层卷积输入的channel,list
        out_channel指的是第i级的第一层卷积的输出的channel,list
        '''
        self.module_dict = nn.ModuleDict()
        self.layer_num = [2] * depth_num #待定参数
        for i in range(depth_num):
            self.module_dict['encoder_{}'.format(i)] = conv_block('encoder',i, 
                                                                  in_channel[i])
            if i != 0:
                self.module_dict['pooling_{}'.format(i)] = nn.MaxPool3d(2,stride = 2)
         
    def forward(self,data,depth):
        if depth != 0:
            data = self.module_dict['pooling_{}'.format(depth)](data)
        return self.module_dict['encoder_{}'.format(depth)](data)
    
class decoder_block(nn.Module):
    def __init__(self,depth_num,in_channel):
        super(decoder_block,self).__init__()
        '''
        depth_num指的是encoder的级数
        inchannel指的是第i级的第一层卷积输入的channel,list
        out_channel指的是第i级的第一层卷积的输出的channel,list
        '''
        self.module_dict = nn.ModuleDict()
        self.layer_num = [2] * depth_num
        for i in range(depth_num):
            self.module_dict['decoder_{}'.format(i)] = conv_block('decoder',i,
                                                                  in_channel[i])
            #self.module_dict['upsample_{}'.format(i)] = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=True)
            self.module_dict['upsample_{}'.format(i)] = nn.ConvTranspose3d(in_channel[i]*2//3,in_channel[i]*2//3,
                                                                    kernel_size=2,
                                                                    stride = 2)
            xavier_normal_(self.module_dict['upsample_{}'.format(i)].weight)                                                  
            self.module_dict['batch_normal_{}'.format(i)] = nn.BatchNorm3d(in_channel[i]*2//3)
    def forward(self,data,encoder,depth):
        data = self.module_dict['upsample_{}'.format(depth)](data)
        data = self.module_dict['batch_normal_{}'.format(depth)](data)
        data = torch.cat([encoder,data],dim = 1)
        return self.module_dict['decoder_{}'.format(depth)](data)
    


    
class U_net(nn.Module):
    def __init__(self,config):
        super(U_net,self).__init__()
        #torch.manual_seed(55)
        #torch.cuda.manual_seed_all(55)
        self.encoder_depth = config[0]
        self.decoder_depth = config[1]
        self.encoder_channel = config[2]
        self.decoder_channel = config[3]
        
        self.encoder = encoder_block(self.encoder_depth, self.encoder_channel)
        self.decoder = decoder_block(self.decoder_depth, self.decoder_channel)
        self.last_conv = nn.Conv3d(self.decoder_channel[0]//3 , 1, 1)
        xavier_normal_(self.last_conv.weight)
        
    def forward(self,data):
        encoder_out = {}
        depth = [i for i in range(self.encoder_depth)]
        for i in depth:
            data = self.encoder.forward(data, i)
            encoder_out['depth_{}'.format(i)] = data
        depth = [i for i in range(self.decoder_depth)]
        depth = depth[::-1]
        data = encoder_out['depth_{}'.format(self.encoder_depth-1)]
        for i in depth:
            data = self.decoder.forward(data,encoder_out['depth_{}'.format(i)],i)
        data = self.last_conv(data)
        return data
        
