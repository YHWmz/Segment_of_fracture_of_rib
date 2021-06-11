import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm
from U_net import U_net
import os
import nibabel as nib
import preprocess as ppc
from itertools import product
os.environ["CUDA_VISIBLE_DEVICES"]="3,2"

def postprocess(pred):
    
    pro_threshold = 0.1
    pred[pred<pro_threshold] = 0
    
    size_threshold = 100
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_threshold)
    pred = np.where(pred_bin, pred, 0)

    return pred



def img_prediction(img, threshold, crop_size, model, device):
    dim_coords = [list(range(0, dim, 64 // 2))[1:-1]+ [dim - 64 // 2] for dim in img.shape]
    centers = list(product(*dim_coords))
    cropz,cropx,cropy = crop_size
    pred = np.zeros(img.shape)
    for cenz,cenx,ceny in tqdm(centers):
        img_tmp = img[cenz-cropz//2:cenz+cropz//2, cenx-cropx//2:cenx+cropx//2, ceny-cropy//2:ceny+cropy//2].reshape(1,1,cropz,cropx,cropy)
        if(cenx-cropx//2<64 or cenx+cropx//2>448 or ceny+cropy//2>420 or ceny-cropy//2<100):
             pred[cenz-cropz//2:cenz+cropz//2, cenx-cropx//2:cenx+cropx//2, ceny-cropy//2:ceny+cropy//2] = 0
             continue
        #img_tmp = img_tmp.transpose((0,1,4,2,3))
        img_tmp = torch.FloatTensor(img_tmp).to(device)
        output = torch.sigmoid(model.forward(img_tmp)).cpu().detach().numpy().reshape(cropz,cropx,cropy)
        #output = output.transpose((1,2,0))
        pred[cenz-cropz//2:cenz+cropz//2, cenx-cropx//2:cenx+cropx//2, ceny-cropy//2:ceny+cropy//2]\
             = np.where(pred[cenz-cropz//2:cenz+cropz//2, cenx-cropx//2:cenx+cropx//2, ceny-cropy//2:ceny+cropy//2] > 0, \
                 np.mean((output,pred[cenz-cropz//2:cenz+cropz//2, cenx-cropx//2:cenx+cropx//2, ceny-cropy//2:ceny+cropy//2]),axis=0),\
                     output)
    pred = postprocess(pred)
    pred = pred.transpose((1,2,0))
    
    return pred

def make_submission_files(pred, image_id, affine):
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    # placeholder for label class since classifaction isn't included
    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info

def predict(model_path, data_path = '/GPFS/data/yuhaowang/Ribfrac_tmp/ribfrac-val/data/', crop_size = (64,64,64),store_path='/GPFS/data/yuhaowang/3dunet/pred_ours11'):
    config = []
    config.append(4); config.append(3)
    config.append([1,32,64,128])
    config.append([96,192,384])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = U_net(config)
    model.eval()
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)
    model = torch.nn.DataParallel(model, device_ids=[0,1]).to(device)

    val_data_path = data_path
    valid_datalist = os.listdir(val_data_path)


    pred_result_list = []
    with torch.no_grad():
        for data_name in tqdm(valid_datalist):
            label_name = data_name.split('-')[0]+'-label.nii.gz'
            img = nib.load(val_data_path+data_name)
            img_affine = img.affine
            img = np.array(img.dataobj).transpose((2,0,1))
            bone_threshold = 200
            P = ppc.val_preprocesser()
            img,threshold = P.process(img)
            pred = img_prediction(img, threshold, crop_size, model, device)
            pred_image, pred_result = make_submission_files(pred, data_name.split('-')[0],img_affine)
            pred_result_list.append(pred_result)
            pred_path = os.path.join(store_path, f"{data_name.split('-')[0]}.nii.gz")
            nib.save(pred_image, pred_path)
            
    pred_info = pd.concat(pred_result_list, ignore_index=True)
    pred_info.to_csv(os.path.join(store_path, "ribfrac-val-pred.csv"),index=False)

def main():
    #模型参数路径
    model_path = './model_weights_ours1.pth'    
    #预测结果的存储路径
    store_path = '/GPFS/data/yuhaowang/3dunet/pred_test'
    #需要预测的图像的路径
    data_path = '/GPFS/data/yuhaowang/Ribfrac_tmp/ribfrac-test-images/'
    crop_size = (64,64,64)
    predict(model_path, data_path , crop_size , store_path)
    #predict(model_path, data_path = '/GPFS/data/yuhaowang/3dunet/smalldata/', crop_size = (64,64,64),store_path)

if __name__ == '__main__':
    main()
