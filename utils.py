import torch.fft
import torch
import numpy as np
from shutil import copy2
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import glob
import cv2


def create_folder(data_dir, target_dir):
    target = os.path.join(data_dir, target_dir)
    if not os.path.exists(target):
        os.mkdir(target)
    return target


def writefile(target_dir, filename, data):
    filepath = os.path.join(target_dir, filename)
    cv2.imwrite(filepath, data)


def applynoise(noise_fn, noisy_dir, im_name, dest):
    """
    Save a noisy image array to disk under noisy_dir/dest/im_name.

    Args:
        noise_fn  : float array in [0, 1] — the noisy image
        noisy_dir : root directory that holds noise-level sub-folders
        im_name   : filename to write (e.g. '0001.png')
        dest      : sub-folder name inside noisy_dir (e.g. 'sigma25')
    """
    noisy = np.uint8(255 * noise_fn)
    target_folder = create_folder(noisy_dir, dest)
    writefile(target_folder, im_name, noisy)


def copyfile(source_dir, dest_dir):
    sources = glob.glob(os.path.join(source_dir, '*'))
    for i in sources:
        filename = os.path.basename(i)          # cross-platform (replaces split('\\'))
        filepath = os.path.join(dest_dir, filename)
        copy2(i, filepath)
