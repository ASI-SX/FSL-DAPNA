import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from proto_mdd.dataloader.samplers import CategoriesSampler, CategoriesSamplerOurs
from proto_mdd.models.proto import proto_net
from proto_mdd.models.proto_attention import proto_attention_net
from proto_mdd.models.mdd import MDDNet, mdd_loss
from proto_mdd.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval, log, setup_seed
from tensorboardX import SummaryWriter

setup_seed(666)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=5)    
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ConvNet', 'ResNet'])
    parser.add_argument('--dataset', type=str, default='TieredImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])    
    parser.add_argument('--base_net_path', type=str, default="./TieredImageNet-ResNet-proto-5-10/0.0001_128_1.0_0.0_0.0/max_acc.pth")    
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--proto_attention', default = 0, type = int) 
    parser.add_argument('--num_test_episodes', default = 2000, type = int)
    args = parser.parse_args()
    args.temperature = 1
    args.head = 1

    set_gpu(args.gpu)
    if args.dataset == 'MiniImageNet':
        from proto_mdd.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from proto_mdd.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from proto_mdd.dataloader.tiered_imagenet import tieredImageNet as Dataset       
    else:
        raise ValueError('Non-supported Dataset.')
    
    if args.proto_attention:
        base_net = proto_attention_net(args, dropout = 0.5)        
    else:
        base_net = proto_net(args, dropout = 0.5)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        base_net = base_net.cuda()
        
    test_set = Dataset('test', args)
    sampler = CategoriesSampler(test_set.label, args.num_test_episodes, args.way, args.shot + args.query)
    loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    test_acc_record = np.zeros((args.num_test_episodes,))
    model_dict = base_net.state_dict()
    pretrain_dict = torch.load(args.base_net_path)['params']
    # print(pretrain_dict.keys())    
    pretrain_dict = {'encoder.' + k[15:]:v for k,v in pretrain_dict.items()} 
    pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in  model_dict}
    print(pretrain_dict.keys())  
    base_net.load_state_dict(pretrain_dict, False)
    base_net.eval()

    ave_acc = Averager()
    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
        
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            k = args.way * args.shot
            data_shot, data_query = data[:k], data[k:]
            _, _, logits = base_net(data_shot, data_query)
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            if i % 100 == 0:                
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))