from __future__ import print_function

import os
import os.path
import os.path as osp
import numpy as np
import random
import pickle
import math
import sys
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

# Set the appropriate paths of the datasets here.
# _TIERED_IMAGENET_DATASET_DIR = './data/tieredimagenet/' # compressed file


# def load_data(file):
#     try:
#         with open(file, 'rb') as fo:
#             data = pickle.load(fo)
#         return data
#     except:
#         with open(file, 'rb') as f:
#             u = pickle._Unpickler(f)
#             u.encoding = 'latin1'
#             data = u.load()
#         return data

# file_path = {'train':[os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'train_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'train_labels.pkl')],
#              'val':[os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'val_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR,'val_labels.pkl')],
#              'test':[os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'test_images.npz'), os.path.join(_TIERED_IMAGENET_DATASET_DIR, 'test_labels.pkl')]}

# class tieredImageNet(data.Dataset):
#     def __init__(self, setname, args):
#         assert(setname=='train' or setname=='val' or setname=='test')
#         image_path = file_path[setname][0]
#         label_path = file_path[setname][1]

#         data_train = load_data(label_path)
#         labels = data_train['labels']
#         self.data = np.load(image_path)['images'] # [num_images, 84, 84, 3]

#         label = []
#         lb = -1
#         self.wnids = []
#         for wnid in labels:
#             if wnid not in self.wnids:
#                 self.wnids.append(wnid)
#                 lb += 1
#             label.append(lb)

#         self.label = label
#         self.num_class = len(set(label))

#         mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
#         std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
#         normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        
#         # Transformation
#         if args.model_type == 'ConvNet':
#             image_size = 84
#             self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             normalize])
#         elif args.model_type == 'ResNet' and setname == 'train':
#             image_size = 80            
#             self.transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             normalize])            
#             # self.transform = transforms.Compose([
#             # transforms.CenterCrop(image_size),
#             # transforms.ToTensor()])
#         elif args.model_type == 'ResNet' and setname != 'train':
#             image_size = 80            
#             self.transform = transforms.Compose([            
#             transforms.CenterCrop(image_size),
#             transforms.ToTensor(),
#             normalize])            
#             # self.transform = transforms.Compose([
#             # transforms.CenterCrop(image_size),
#             # transforms.ToTensor()])   

#         else:
#             raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
        

#     def __getitem__(self, index):
#         img, label = self.data[index], self.label[index]
#         img = self.transform(Image.fromarray(img))
#         # print(img.shape) # [3, 80, 80]
#         return img, label

#     def __len__(self):
#         return len(self.data)


IMAGE_PATH = './data/tieredImageNet/'
SPLIT_PATH = './data/tieredImageNet/split'

class tieredImageNet(data.Dataset):
    """ Usage: 
    """
    def __init__(self, setname, args):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, setname, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if args.model_type == 'ConvNet':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.model_type == 'ResNet':
            image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label