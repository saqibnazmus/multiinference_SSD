import cv2
import numpy as np
import platform
import glob
import os
from utils import create_folder, writefile, applynoise, copyfile
delimeter = "\\" if platform.system() == "Windows" else "/"


def add_noise(input_img,sigma):
    noise = np.random.normal(scale=sigma/255, size=input_img.shape)
    add_noise = np.clip(input_img + noise, 0, 1)
    return np.float32(add_noise)



class noise_addition:
    def fixed_AWGN(input_img, sigma):
        noise = np.random.normal(scale=sigma/255, size=input_img.shape)
        add_noise = np.clip(input_img + noise, 0, 1)
        return np.float32(add_noise)
    def var_AWGN(input_img, sigma_1, sigma_2):
        sigma = np.random.randint(sigma_1, sigma_2)
        noise = np.random.normal(scale=sigma/255, size=input_img.shape)
        add_noise = np.clip(input_img + noise, 0, 1)
        return np.float32(add_noise)
    def fixed_poisson(input_img, lamb):
        lam = lamb*torch.ones((shape[0], 1, 1, 1), device=input_img.device)
        noised = torch.poisson(lam * input_img) / lam
        return noised
    def var_poisson(input_img, min_lam, max_lamb):
        lam = torch.rand(size=(shape[0], 1, 1, 1), device=input_img.device)*(max_lam-min_lam) + min_lam
        noised = torch.poisson(lam * input_img) / lam
        return noised


    
def prepare_training_dataset(dataset_train_dir, dataset_val_dir, save_dir):
    clean_train = glob.glob(os.path.join(dataset_train_dir,'*'))
    clean_train.sort()
    
    noisy_input_dir = create_folder(os.path.join(save_dir,"train"),"input")
    noisy_target_dir = create_folder(os.path.join(save_dir,"train"), "target")
    for i in clean_train:
        im_path = i
        im_name = im_path.split(delimeter)[-1]
        img = cv2.imread(i)
        y1 = np.uint8(255*noise_addition.var_AWGN(img/255, sigma_1=5, sigma_2=25))
        z1 = np.uint8(255*noise_addition.fixed_AWGN(y1/255, sigma=20))
        cv2.imwrite(os.path.join(noisy_input_dir,im_name), z1)
        y2 = np.uint8(255*noise_addition.var_AWGN(img/255, sigma_1=5, sigma_2=25))
        z2 = np.uint8(255*noise_addition.fixed_AWGN(y2/255, sigma=25))
        cv2.imwrite(os.path.join(noisy_target_dir,im_name), z2)
    print("training noisy data prepared")
    
    clean_val = glob.glob(os.path.join(dataset_val_dir,'*'))
    clean_val.sort()
    noisy_val_dir = create_folder(os.path.join(save_dir,"validation"), "noisy")
    clean_val_dir = create_folder(os.path.join(save_dir,"validation"), "clean")
    copyfile(dataset_val_dir, clean_val_dir)
    for k in clean_val:
        val_path = k
        val_name = val_path.split(delimeter)[-1]
        img_val = cv2.imread(k)
        y_val = np.uint8(255*noise_addition.var_AWGN(img_val/255, sigma_1=5, sigma_2=25))
        z_val = np.uint8(255*noise_addition.fixed_AWGN(y_val/255, sigma=25))
        target_dir_val = os.path.join(noisy_val_dir, val_name)
        cv2.imwrite(target_dir_val, z_val)
    print("validation noisy data prepared")
    
                                    
def prepare_testdataset(data_dir, dataset_name):
    files = glob.glob(os.path.join(data_dir, dataset_name, '*.png'))
    files.sort()
    noisy_dir = create_folder(data_dir, dataset_name + "_noisy")
    for i in range(len(files)):
        im_path = i
        im_name = im_path.split('\\\\')[-1]
        img = cv2.imread(files[i])
        applynoise(noise_addition.fixed_AWGN(img/255,sigma=25), "fixed_AWGN")
        applynoise(noise_addition.var_AWGN(img/255,sigma_1=5, sigma_2=25), "var_AWGN")
        applynoise(noise_addition.fixed_poisson(img/255,30), "fixed_poisson")
        applynoise(noise_addition.var_poisson(img/255,30,50), "var_poisson") 
    