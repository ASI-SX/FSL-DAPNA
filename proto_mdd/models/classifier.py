import torch
import torch.nn as nn
import numpy as np
from proto_mdd.utils import euclidean_metric
import torch.nn.functional as F
from torchvision import models
    
class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == 'ConvNet':
            hdim = 64
            from proto_mdd.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.model_type == 'ResNet':
            hdim = 640
            from proto_mdd.networks.resnet import ResNet as ResNet
            self.encoder = ResNet()      
        elif args.model_type == 'ResNet18':
            hdim = 512
            from proto_mdd.networks.resnet_pytorch import resnet18
            self.encoder = resnet18(pretrained = False)
        else:
            raise ValueError('')

        self.fc = nn.Linear(hdim, args.num_class)

    def forward(self, data, is_emb = False):
        out = self.encoder(data)
        if not is_emb:
            out = self.fc(out)
        return out
    
    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        
        query = self.encoder(data_query)
        logits = euclidean_metric(query, proto)
        return logits