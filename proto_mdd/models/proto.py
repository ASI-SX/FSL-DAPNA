import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from proto_mdd.utils import euclidean_metric

class proto_net(nn.Module):

    def __init__(self, args, dropout=0.2):
        super().__init__()
        if args.model_type == 'ConvNet':
            from proto_mdd.networks.convnet import ConvNet
            self.encoder = ConvNet()
            z_dim = 64
        elif args.model_type == 'ResNet':
            from proto_mdd.networks.resnet import ResNet
            self.encoder = ResNet()
            z_dim = 640            
        elif args.model_type == 'ResNet18':
            hdim = 512
            from proto_mdd.networks.resnet_pytorch import resnet18
            self.encoder = resnet18(pretrained = False)            
            z_dim = 512
        else:
            raise ValueError('')
        self.z_dim = z_dim
        self.args = args

    def forward(self, support, query, input_type = "data"):
        if input_type == "data":
            # feature extraction
            support = self.encoder(support)
            query = self.encoder(query) 
            # get mean of the support
            proto = support.reshape(self.args.shot, -1, support.shape[-1]).mean(dim=0) # N x d
            logitis = euclidean_metric(query, proto)
                  
            return support, query, logitis

        elif input_type == "feature":                
            # get mean of the support
            proto = support.reshape(self.args.shot, -1, support.shape[-1]).mean(dim=0) # N x d            
            logitis = euclidean_metric(query, proto)                    
            return logitis

