import torch
from torch import nn
import numpy as np

def cal_rec_pre(score,label):
    num = label.size(0)
    score = score.view(num,-1)
    label= label.view(num,-1).cpu().detach_().numpy()
    prediction = np.where(score.cpu().detach_().numpy()>0.1,1,0)
    p = label + prediction
    s = label - prediction
    TP = np.where(p==2,1,0).sum(1)
    FP = np.where(s==-1,1,0).sum(1)
    FN = np.where(s==1,1,0).sum(1)
    
    rec = (TP/(TP + FN + 1e-8)).mean()
    pre = (TP/(TP+FP+1e-8)).mean()
    return rec, pre

class Diceloss(nn.Module):
    def __init__(self):
        super(Diceloss,self).__init__()
        
    def forward(self, result, label):
        num = label.size(0)
        smooth = 1
        
        result = torch.sigmoid(result)
        rflat = result.view(num, -1)
        lflat = label.view(num, -1)
        
        intersection = (rflat * lflat)
 
        score = (2. * intersection.sum(1) + smooth) / (rflat.sum(1) + lflat.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class IOU(nn.Module):
    def __init__(self):
        super(IOU,self).__init__()
        
    def forward(self, result, label):
        num = label.size(0)
        
        result = torch.sigmoid(result)
        rflat = result.view(num, -1)
        lflat = label.view(num, -1)
        
        intersection = (rflat * lflat)
 
        score = intersection.sum(1) / (rflat.sum(1) + lflat.sum(1) - intersection.sum(1))
        return score

class IOUloss(nn.Module):
    def __init__(self):
        super(IOUloss,self).__init__()
        
    def forward(self, result, label):
        num = label.size(0)
        
        result = torch.sigmoid(result)
        rflat = result.view(num, -1)
        lflat = label.view(num, -1)
        
        intersection = (rflat * lflat)
 
        score = (intersection.sum(1) + 1) / (rflat.sum(1) + lflat.sum(1) - intersection.sum(1) + 1)
        score = 1 - score.sum() / num
        return score
        
class G_Diceloss(nn.Module):
    def __init__(self):
        super(G_Diceloss,self).__init__()
    def forward(self, result, label):
        result = torch.sigmoid(result)
        m = result.shape[2]
        intersection = (result * label).sum((1,3,4))
        union = (result + label).sum((1,3,4))
        weight = label.sum((1,3,4))
        weight = 1/(weight+1)
        tmp = weight.cpu().detach_().numpy()
        score = 1-((intersection*weight).sum(1)/(m*(union*weight).sum(1))).sum()
        return score

class Focalloss(nn.Module):
    def __init__(self):
        super(Focalloss,self).__init__()
    def forward(self,result,label,gamma = 1,alpha = 0.99):
        result = torch.sigmoid(result)
        num = label.size(0)
        rflat = result.view(num, -1)
        lflat = label.view(num, -1)
        log_res = alpha*torch.log(rflat+1e-8)*lflat
        log_o_res = (1-alpha)*torch.log(1-rflat+1e-8)*(1-lflat)
        #score = -(torch.pow(1-rflat,gamma)*log_res+torch.pow(rflat,gamma)*log_o_res).mean()
        score = -(log_res+log_o_res).mean()
        return score

class Recall_loss(nn.Module):
    def __init__(self):
        super(Recall_loss,self).__init__()
    def forward(self,result,label):
        result = torch.sigmoid(result)
        intersection = (result * label)
        score = 1-(intersection.sum())/(label.sum()+1)
        #m = result.shape[0]
        #intersection = (result * label).sum((1,2,3,4))
        #score = 1-(intersection.sum(0)/(m*label.sum((1,2,3,4))+1)).sum()/64/64/64
        return score
    
class Precision_loss(nn.Module):
    def __init__(self):
        super(Precision_loss,self).__init__()
    def forward(self,result,label):
        result = torch.sigmoid(result)
        intersection = (result * label)
        score = 1-(intersection.sum())/(result.sum()+1)
        #m = result.shape[0]
        #intersection = (result * label).sum((1,2,3,4))
        #score = 1-(intersection.sum(0)/(m*result.sum((1,2,3,4))+1)).sum()/64/64/64
        return score


class dicelossg(nn.Module):
    def __init__(self, image=False):
        super().__init__()
        self.image = image

    def forward(self, x, y):
        x = x.sigmoid()
        i, u = [t.flatten(1).sum(1) if self.image else t.sum() for t in [x * y, x + y]]

        dc = (2 * i + 1) / (u + 1)
        dc = 1 - dc.mean()
        return dc
