# reference:
# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/datasets.py
import json
import glob
import random
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F


from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.utils import *


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names
'''
def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)
'''

class ListDataset(Dataset):
    def __init__(self, model_type, data_dir, json_file='anno_data.json',
                 img_size=416,
                 augmentation=None,min_size=1):
        """
        Vehicle detection dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            model_type (str): model name specified in config file
            list_path (str): dataset list textfile path
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
        """
        self.model_type = model_type
        self.img_size = img_size
        self.max_labels = 100
        self.min_size = min_size
        self.data_dir = data_dir
        self.json_file = json_file
        self.lrflip = augmentation['LRFLIP']
        self.jitter = augmentation['JITTER']
        self.random_placing = augmentation['RANDOM_PLACING']
        self.hue = augmentation['HUE']
        self.saturation = augmentation['SATURATION']
        self.exposure = augmentation['EXPOSURE']
        self.random_distort = augmentation['RANDOM_DISTORT']

        self.img_list = glob.glob(os.path.join(self.data_dir, '*.jpg'))
        self.img_list.extend(glob.glob(os.path.join(self.data_dir,'*.png')))

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """

        # load image and preprocess
        img_path = self.img_list[index % len(self.img_list)]
        img = cv2.imread(img_path)
        assert img is not None
        img, info_img = preprocess(img, self.img_size, jitter=self.jitter,
                                   random_placing=self.random_placing)
        if self.random_distort:
            img = random_distort(img, self.hue, self.saturation, self.exposure)

        img = np.transpose(img / 255., (2, 0, 1))

        lrflip = False
        if np.random.rand() > 0.5 and self.lrflip == True:
            lrflip = True
        
        if lrflip:
            img = np.flip(img, axis=2).copy()

        # load labels
        json_open = open(os.path.join(self.data_dir, self.json_file), 'r')
        json_load = json.load(json_open)
        annotations = json_load[os.path.basename(img_path)]['regions']

        labels = []
        for anno in annotations:
            if anno['bb'][2] > self.min_size and anno['bb'][3] > self.min_size:
                labels.append([])
                labels[-1].append(anno['class_id'])
                labels[-1].extend(anno['bb'])

        padded_labels = np.zeros((self.max_labels, 5))
        if len(labels) > 0:
            labels = np.stack(labels).astype(np.float64)
            if 'YOLO' in self.model_type:
                labels = label2yolobox(labels, info_img, self.img_size, lrflip)
            padded_labels[range(len(labels))[:self.max_labels]
                          ] = labels[:self.max_labels]
        padded_labels = torch.from_numpy(padded_labels)

        return img, padded_labels, info_img, index
    

    def __len__(self):
        return len(self.img_list)