import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/cub/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/cub/split')

# This is for the CUB dataset, which does not support the ResNet encoder now
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)
class CUB(Dataset):

    def __init__(self, setname, args):
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []

        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = np.unique(np.array(label)).shape[0]


        if args.model_type == 'ConvNet':
            image_size = 84
            self.transform = transforms.Compose([
            transforms.Resize((image_size ,image_size ), interpolation = PIL.Image.BICUBIC),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
            
        elif args.model_type == 'ResNet' :
            image_size = 80
            if setname == 'train':
                self.transform = transforms.Compose([
                # transforms.Resize(92, interpolation = PIL.Image.BICUBIC),
                # transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            else:
                self.transform = transforms.Compose([            
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])    

        elif args.model_type == 'ResNet18':
            image_size = 224
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]            
            if setname == 'train' or setname == 'all':
                self.transform = transforms.Compose([                    
                    transforms.RandomResizedCrop(image_size),                    
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),                   
                    transforms.Normalize(mean, std)])
            else:
                self.transform = transforms.Compose([
                    transforms.Scale((int(image_size * 1.15), int(image_size*1.15))),            
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])

        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label            

