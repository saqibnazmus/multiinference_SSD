import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.DnCNN import DnCNN
from utils import *
from dataloader import DataLoader_test
from torchmetrics import PeakSignalNoiseRatio
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='bsd68', help='test dataset')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--root_dir", type=str, default='./test', help='root directory of test images')
parser.add_argument("--batch_size", type=float, default=4, help='noise level used on test set')
opt = parser.parse_args()




if __name__ == "__main__":
    dataset_test = DataLoader_test(opt.root_dir)
    testloader = DataLoader(dataset = dataset_test, 
                              num_workers=4,
                              batch_size =2 ,
                              shuffle = False,
                              pin_memory=False, 
                              drop_last=True)
    
    x = next(iter(testloader))


    
    
    print('Loading model .......\n')
    model =  DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model.pth')))
    model.eval()
    
    for images in tqdm(testloader):
        clean, fixed_AWGN, fixed_poisson, var_AWGN, var_poisson = images[0], images[1], images[2], images[3], images[4]
        with torch.no_grad():
            fixed_AWGN_pred, fixed_poisson_pred, var_AWGN_pred, var_poisson_pred = model(fixed_AWGN), model(fixed_poisson), model(var_AWGN), model(var_poisson) 
            psnr = PeakSignalNoiseRatio()
            psnr_fixed_AWGN = psnr(fixed_AWGN_pred, clean)
            psnr_fixed_poisson = psnr(fixed_poisson_pred, clean)
            psnr_var_AWGN = psnr(var_AWGN_pred, clean)
            psnr_var_poisson = psnr(var_poisson_pred, clean)
            print("psnr for fixed Gaussian:", psnr_fixed_AWGN)
            print("psnr for fixed Poisson:", psnr_fixed_poisson)
            print("psnr for variable Gaussian:", psnr_var_AWGN)
            print("psnr for variable Poisson:", psnr_var_poisson)
    
