import os
import argparse
import numpy as np
import torch
import glob
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.DnCNN import DnCNN
import torch.fft
from datetime import datetime
from dataloader import DataLoader_ImageNet_train, DataLoader_ImageNet_val
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio
from dataset import prepare_training_dataset







parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument('--w_decay', type=float, default=1e-8)
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--dataset_train_dir", type=str,  default='./ImageNet/train')
parser.add_argument("--dataset_val_dir", type=str,  default='./ImageNet/validation')
parser.add_argument("--root_dir", type=str,  default='./data')
parser.add_argument("--outf", type=str, default="./weights", help='path of log files')



opt = parser.parse_args()



if __name__ == "__main__":
    print("preparing training dataset..........\n")
    
    prepare_training_dataset(opt.dataset_train_dir, opt.dataset_val_dir, opt.root_dir)
    
    
    
    print("Loading dataset......\n")
    MODEL_PATH = opt.outf
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    dataset_train = DataLoader_ImageNet_train(opt.root_dir)
    TrainingLoader = DataLoader(dataset=dataset_train,
                            num_workers=8,
                            batch_size=opt.batchSize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)
    
    dataset_val = DataLoader_ImageNet_val(opt.root_dir)
    valLoader = DataLoader(dataset=dataset_val,
                            num_workers=8,
                            batch_size=opt.batchSize,
                            shuffle=True,
                            pin_memory=False,
                            drop_last=True)
    
    
    psnr = PeakSignalNoiseRatio().cuda()
    model =  DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model = model.cuda()
    criterion = nn.MSELoss(size_average=False)
    criterion.cuda()
    num_epoch = opt.epochs 
    ratio = num_epoch/ 100
    optimizer = optim.Adam(model.parameters(), lr = opt.lr, weight_decay = opt.w_decay)
    print("Batchsize={}, number of epoch={}".format(opt.batchSize, opt.epochs))
    now = datetime.now()
    print('Start training.....',now.strftime("%H:%M:%S"))
    for epoch in tqdm(range(num_epoch), total=num_epoch):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        epoch_loss = 0.0
        psnr_train = 0.0    
        for inp,tar in TrainingLoader:
            noisy_input = torch.FloatTensor(inp)
            noisy_input = noisy_input.cuda()
            noisy_target = torch.FloatTensor(tar)
            noisy_target = noisy_target.cuda()
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            Pred1 = model(noisy_input)
            loss_1 = criterion(Pred1, noisy_target)
            Pred2 = model(Pred1)
            loss_2 = criterion(Pred2, noisy_target)
            Pred3 = model(Pred2)
            Pred4 = model(Pred3)
            loss_3 = criterion(Pred3, Pred4)
            loss_4 = criterion(Pred4, noisy_target)
            loss = loss_1 + loss_2 + loss_3 + loss_4
            epoch_loss += loss/4
            loss.backward()
            optimizer.step()
            psnr_train += psnr(Pred4, noisy_target)
        psnr_avg_train = psnr_train / len(TrainingLoader)
        print(f"epoch: {epoch+1}, psnr_avg_train: {psnr_avg_train}, loss: {epoch_loss/len(TrainingLoader)}")
        model.eval()
        psnr_val = 0
        for i, (image, clean) in enumerate(valLoader):
            val_input = torch.FloatTensor(image).cuda()
            val_clean = torch.FloatTensor(clean).cuda()
            with torch.no_grad():
                out_val = model(val_input)
            psnr_val += psnr(out_val, val_clean) 
        psnr_avg = psnr_val/len(valLoader)  
        print("\n[epoch %d] PSNR_avg: %.4f" % (epoch+1, psnr_avg))
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'model.pth'))
    now = datetime.now()
    print('Total training time.....',now.strftime("%H:%M:%S"))
            

            
            
            
            
    