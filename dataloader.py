import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
from PIL import Image



class DataLoader_ImageNet_train(Dataset):
    def __init__(self, data_dir):
        super(DataLoader_ImageNet_train, self).__init__()
        self.data_dir = data_dir
        self.input = glob.glob(os.path.join(self.data_dir, "train", "input", "*"))
        self.input.sort()
        self.target = glob.glob(os.path.join(self.data_dir, "train", "target", "*"))
        self.target.sort()
        self.set = list(zip(self.input, self.target))
        
    def __getitem__(self,index):
        img_input, img_target = self.set[index]
        img_input, img_target = Image.open(img_input), Image.open(img_target)
        transformer = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((128,128))])
        return transformer(img_input), transformer(img_target)
    def __len__(self):
        return len(self.set)
    

class DataLoader_ImageNet_val(Dataset):
    def __init__(self, root_dir):
        super(DataLoader_ImageNet_val, self).__init__()
        self.root_dir = root_dir 
        self.val_clean = glob.glob(os.path.join(self.root_dir, "validation", "clean", "*"))
        self.val_clean.sort()
        self.val_noisy = glob.glob(os.path.join(self.root_dir, "validation", "noisy", "*"))
        self.val_noisy.sort()
        self.set = list(zip(self.val_noisy, self.val_clean)) 
    def __getitem__(self,index):
        img_input, img_target = self.set[index] 
        img_input, img_target = Image.open(img_input), Image.open(img_target) 
        transformer = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize((128, 128))])
        return transformer(img_input), transformer(img_target)
    def __len__(self):
        return len(self.set)
    
class DataLoader_test(Dataset):
    def __init__(self, root_dir, dataset_names=["bsd68", "kodak24", "set14"]):
        super(DataLoader_test, self).__init__()
        self.root_dir = root_dir
        self.clean_dirs = [os.path.join(root_dir,path) for path in dataset_names]
        self.noisy_dirs = [os.path.join(root_dir,path + "_noisy") for path in dataset_names]
        self.gen_img_array = self.gen_img_array()
        self.transforms =  transforms.Compose([transforms.ToTensor()
                                          ])

    def gen_img_array(self):
        noisyfiles = []
        for dirs in self.noisy_dirs:
            for roots, subs, files in os.walk(dirs):
                for file in files:
                    noisyfiles.append(os.path.join(roots,file))
        noisyfiles.sort()
        
        cleanfiles = []
        for dirs in self.clean_dirs:
            for roots, subs, files in os.walk(dirs):
                for file in files:
                    cleanfiles.append(os.path.join(roots,file))

        cleanfiles.sort()
        
        imgarray = []
        for i in cleanfiles:
            imgs = [] 
            datasetname_clean = i.split('\\')[-2]
            imgname_clean = i.split('\\')[-1]
            imgs.append(i)
            for j in noisyfiles:
                noisyfolder = j.split('\\')[-3]
                datasetname_noisy = noisyfolder.split('_')[0]
                imgname_noisy = j.split('\\')[-1]
                if datasetname_clean == datasetname_noisy and imgname_clean == imgname_noisy:
                    imgs.append(j)
            imgarray.append(imgs)
        return imgarray
    def __getitem__(self, index):
        imgtensors = []
        for path in self.gen_img_array[index]:
            x = Image.open(path)
            x = self.transforms(x)
            imgtensors.append(x)
        return imgtensors
    def __len__(self):
        return len(self.gen_img_array)
