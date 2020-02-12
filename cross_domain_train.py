import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from proto_mdd.dataloader.samplers import CategoriesSampler, CategoriesSamplerOurs
from proto_mdd.models.proto import proto_net
from proto_mdd.models.proto_attention import proto_attention_net
from proto_mdd.models.mdd import MDDNet, mdd_loss
from proto_mdd.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval, log, setup_seed

from baseline_data.datamgr import SimpleDataManager, SetDataManager


setup_seed(666)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=600)
    parser.add_argument('--train_way', type=int, default=5)    
    parser.add_argument('--val_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_mul', type=float, default=10) # lr is the basic learning rate, while lr * lr_mul is the lr for other parts   
    parser.add_argument('--temperature', type=float, default=128)        
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--class_num', default=64, type=int)
    parser.add_argument('--srcweight', default=-1, type=int)
    parser.add_argument('--lambda_pre_fsl_loss', default= 1, type=float)
    parser.add_argument('--lambda_da', default=1, type=float)    
    parser.add_argument('--lambda_new_fsl_loss', default = 1, type=float)    
    parser.add_argument('--proto_attention', default = 0, type = int)  
    parser.add_argument('--model_type', type=str, default='ResNet18', choices=['ConvNet', 'ResNet', 'ResNet18'])
    parser.add_argument('--dataset', type=str, default='cross', choices=['MiniImageNet', 'CUB', 'TieredImageNet','cross'])
    parser.add_argument('--init_weights', type = str, default= None)   
    parser.add_argument('--head', type=int, default=1)
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--print_i_mdd', default=0, type= int)
    parser.add_argument('--num_train_episodes', default = 100, type = int)
    args = parser.parse_args()    

    if args.dataset == 'MiniImageNet':
        args.class_num = 64
        args.width = 1024
        args.srcweight = 4
        is_cen = False
    elif args.dataset == 'TieredImageNet':
        args.class_num = 351
        args.width = 1024
        args.srcweight = 4
        args.num_train_episodes = 1000
        is_cen = False
    elif args.dataset == 'CUB':
        args.class_num = 100
        args.width = 1024
        args.srcweight = 4
        is_cen = False
    elif args.dataset == 'cross':
        args.class_num = 100
        args.width = 1024
        args.srcweight = 4
        args.lr = 0.01
        args.lr_mul =  0.1
        is_cen = False

    else:
        print('Dataset not supported!')
        exit()
    

    set_gpu(args.gpu)
    if args.proto_attention:
        save_path1 = '-'.join([args.dataset, args.model_type, 'proto_atten', str(args.shot), str(args.train_way)])
    else:
        save_path1 = '-'.join([args.dataset, args.model_type, 'proto', str(args.shot), str(args.train_way)])
    save_path2 = '_'.join([str(args.lr),str(args.temperature), str(args.lambda_pre_fsl_loss), str(args.lambda_da), str(args.lambda_new_fsl_loss)])

    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path)
    train_log_file_path = os.path.join(args.save_path, 'train_log.txt')    
    val_log_file_path = os.path.join(args.save_path, 'val_log.txt')
    log(train_log_file_path, str(vars(args)))
    log(val_log_file_path, str(vars(args)))


    if args.dataset == 'MiniImageNet':       
        from proto_mdd.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from proto_mdd.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImageNet':
        from proto_mdd.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'cross':
        from proto_mdd.dataloader.mini_imagenet_pre import MiniImageNet as Dataset_mini
        from proto_mdd.dataloader.cub import CUB as Dataset_cub
    else:
        raise ValueError('Non-supported Dataset.')
    
    if args.dataset == 'cross':            
        train_file = '~/filelists/miniImagenet/all.json'
        val_file = '~/filelists/CUB/val.json'
        image_size = 224
        base_datamgr = SetDataManager(image_size, n_query = args.query,  n_way = args.train_way * 2, n_support = args.shot)
        train_loader = base_datamgr.get_data_loader( train_file , aug = True)                
        val_datamgr  = SetDataManager(image_size, n_query = args.query, n_way = args.val_way, n_support = args.shot)
        val_loader   = val_datamgr.get_data_loader( val_file, aug = False) 
    else:
        trainset = Dataset('train', args)
        train_sampler = CategoriesSamplerOurs(trainset.label, args.num_train_episodes, args.train_way, args.shot + args.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)
        valset = Dataset('val', args)
        val_sampler = CategoriesSampler(valset.label, 600, args.val_way, args.shot + args.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    
    if args.proto_attention:
        base_net = proto_attention_net(args, dropout = 0.5)        
    else:
        base_net = proto_net(args, dropout = 0.5)

    mdd_net = MDDNet(base_net=args.model_type, use_bottleneck=True,
                bottleneck_dim=args.width, width=args.width,
                class_num=args.class_num).cuda()


    # parameter list to optimize
    base_net_param_list = [{'params': base_net.encoder.parameters()}]
    if args.proto_attention:
        base_net_param_list = base_net_param_list + [{'params': base_net.slf_attn.parameters(), 'lr': args.lr * args.lr_mul}]
                                     
    mdd_net_param_list = mdd_net.get_parameter_list()
    mdd_net_param_list = [{'params':x['params'], 'lr': x['lr'] * args.lr * args.lr_mul} for x in mdd_net_param_list]   

    parameter_list = base_net_param_list + mdd_net_param_list

    if args.model_type == 'ConvNet':
        optimizer = torch.optim.Adam(parameter_list, lr=args.lr)
    elif args.model_type == 'ResNet' or args.model_type == "ResNet18":
        optimizer = torch.optim.SGD(parameter_list, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)        
    
    # load pre-trained model (no FC weights)
    base_net_dict = base_net.state_dict()
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights)['state_dict']
        # remove weights for FC
        # pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()} # for cross-domain setting
        pretrained_dict = {'encoder.' + k[7:]: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in base_net_dict and base_net_dict[k].shape == pretrained_dict[k].shape}
        print(pretrained_dict.keys())
        base_net_dict.update(pretrained_dict) 
    base_net.load_state_dict(base_net_dict, False)           
    
    if args.train_way >= 10:
    	base_net.encoder = torch.nn.DataParallel(base_net.encoder, device_ids = [0,1]).cuda()
    	base_net = base_net.cuda()
    else:
    	base_net = base_net.cuda()
    
    def save_model(name):
        torch.save(dict(params=base_net.state_dict()), osp.join(args.save_path, name + '.pth'))
        torch.save(dict(params=mdd_net.state_dict()), osp.join(args.save_path, name + '_mdd.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0
    
    timer = Timer()
    global_count = 0    
        
    label = torch.arange(args.train_way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor)    
    if torch.cuda.is_available():
        label = label.cuda()       
            
    for epoch in range(1, args.max_epoch + 1):
        if args.dataset != 'cross':
            lr_scheduler.step()
        base_net.train()
        mdd_net.train()
        tl = Averager()
        ta = Averager()
            
        for i, batch in enumerate(train_loader, 1):
            args.print_i_mdd = i
            global_count = global_count + 1

            n_imgs = args.train_way * (args.shot + args.query) # 5*(5+15) = 100            
            data, index_label = batch[0].cuda(), batch[1].cuda()
            data_1 = data[:args.train_way]
            data_2 = data[args.train_way:]
            index_label_1 = index_label[:args.train_way]
            index_label_2 = index_label[args.train_way:]
            data_1 = data_1.permute(1,0,2,3,4)
            data_2 = data_2.permute(1,0,2,3,4)
            data_1 = data_1.reshape([-1] + list(data_1.shape[-3:]))
            data_2 = data_2.reshape([-1] + list(data_2.shape[-3:]))
            index_label_1 = index_label_1.permute(1,0)
            index_label_2 = index_label_2.permute(1,0)
            index_label_1 = index_label_1.reshape([-1])
            index_label_2 = index_label_2.reshape([-1])
            data = torch.cat([data_1, data_2], dim = 0)
            index_label = torch.cat([index_label_1, index_label_2])                                                
            # print(index_label)
            # prototypical network part
            # the first FSL loss i.e. (2 * N)-train_way K-shot learning
            
            pre_data_src = data[:n_imgs]
            pre_data_tgt = data[n_imgs:]

            p = args.shot * args.train_way
            pre_data_src_shot = pre_data_src[:p]
            pre_data_src_query = pre_data_src[p:]
            pre_data_tgt_shot = pre_data_tgt[:p]
            pre_data_tgt_query = pre_data_tgt[p:]
            pre_data_shot = torch.cat([pre_data_src_shot, pre_data_tgt_shot], dim = 0)
            pre_data_query = torch.cat([pre_data_src_query, pre_data_tgt_query], dim = 0)
            pre_fea_shot, pre_fea_query, pre_logits = base_net(pre_data_shot, pre_data_query)

            pre_label_fsl_s = torch.arange(args.train_way).repeat(args.query)
            pre_label_fsl_t = torch.arange(args.train_way, 2 * args.train_way).repeat(args.query)
            pre_label_fsl = torch.cat([pre_label_fsl_s, pre_label_fsl_t], dim = 0)
            pre_label_fsl = pre_label_fsl.type(torch.cuda.LongTensor)

            pre_fsl_loss = F.cross_entropy(pre_logits, pre_label_fsl)            
            pre_fsl_acc = count_acc(pre_logits, pre_label_fsl)

            # rearrange the feature index
            pre_fea_src_shot = pre_fea_shot[:p]
            pre_fea_tgt_shot = pre_fea_shot[p:]
            pre_fea_src_query = pre_fea_query[:(n_imgs-p)]
            pre_fea_tgt_query = pre_fea_query[(n_imgs-p):]
            pre_src_features = torch.cat([pre_fea_src_shot, pre_fea_src_query], dim = 0)
            pre_tgt_features = torch.cat([pre_fea_tgt_shot, pre_fea_tgt_query], dim = 0)

            # domain adaptation part
            # the second FSL loss i.e. N-train_way K-shot learning
            pre_features = torch.cat([pre_src_features, pre_tgt_features], dim = 0)
            new_features, outputs, outputs_adv = mdd_net(pre_features)

            new_fea_s = new_features[:n_imgs]
            new_fea_t = new_features[n_imgs:]
            
            new_fea_shot_s, new_fea_query_s = new_fea_s[:p], new_fea_s[p:]
            new_fea_shot_t, new_fea_query_t = new_fea_t[:p], new_fea_t[p:]
            new_logits_s = base_net(new_fea_shot_s, new_fea_query_s, input_type = "feature")
            new_logits_t = base_net(new_fea_shot_t, new_fea_query_t, input_type = "feature")
            new_label_fsl = torch.arange(args.train_way).repeat(args.query)
            new_label_fsl = new_label_fsl.type(torch.cuda.LongTensor)            
            new_fsl_loss_s = F.cross_entropy(new_logits_s, new_label_fsl)
            new_fsl_loss_t = F.cross_entropy(new_logits_t, new_label_fsl)
            new_fsl_loss = new_fsl_loss_s + new_fsl_loss_t

            new_fsl_acc_s = count_acc(new_logits_s, new_label_fsl)
            new_fsl_acc_t = count_acc(new_logits_t, new_label_fsl)

            # domain adaptation loss            
            src_idx = list(range(n_imgs))
            tgt_idx = list(range(n_imgs, (2 * n_imgs)))
            # src_support_num = args.train_way * args.shot
            # tgt_support_num = args.train_way * args.query
            # src_idx = list(range(src_support_num))
            # tgt_idx = list(range((2 * n_imgs - tgt_support_num), (2 * n_imgs)))

            transfer_loss = mdd_loss(args, new_features, outputs, outputs_adv, index_label, src_idx, tgt_idx)
            if torch.isnan(transfer_loss):
                print(index_label[src_idx])
                print(index_label[tgt_idx])
                transfer_loss = pre_fsl_loss

            
            # total loss
            total_loss = args.lambda_pre_fsl_loss * pre_fsl_loss + args.lambda_new_fsl_loss * new_fsl_loss + args.lambda_da * transfer_loss

            if i % 25 == 0:
            	log(train_log_file_path, "epoch: {} iter: {} transfer_loss: {:.4f} pre_fsl_loss: {:.4f} new_fsl_loss: {:.4f} total_fsl_loss: {:.4f}".format\
            		(epoch, i, transfer_loss.item(), pre_fsl_loss.item(), new_fsl_loss.item(), total_loss.item()))
            	log(train_log_file_path, "epoch: {} iter: {} fsl_acc_s: {:.4f} fsl_acc_t: {:.4f} pre_fsl_acc: {:.4f}".format\
            		(epoch, i, new_fsl_acc_s, new_fsl_acc_t, pre_fsl_acc))
            	if i% 100 == 0:
            		log(train_log_file_path, "\n")              
            
            tl.add(total_loss.item())
            ta.add(pre_fsl_acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        base_net.eval()
        mdd_net.eval()

        vl = Averager()
        va = Averager()

        label_val = torch.arange(args.val_way).repeat(args.query)
        label_val = label_val.type(torch.cuda.LongTensor)
                    
        with torch.no_grad():
            for i, batch in enumerate(val_loader, 1):                
                data, _ = [_.cuda() for _ in batch]
                data = data.permute(1, 0, 2, 3, 4)
                data = data.reshape([-1] + list(data.shape[-3:]))             
                p = args.shot * args.val_way
                data_shot, data_query = data[:p], data[p:]
                _, _, logits = base_net(data_shot, data_query)
                loss = F.cross_entropy(logits, label_val)
                acc = count_acc(logits, label_val)       
                vl.add(loss.item())
                va.add(acc)

        vl = vl.item()
        va = va.item()             
        log(val_log_file_path,'epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va) \
        	+ ' *** best epoch and acc: {} {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))

        if va >= trlog['max_acc']:
            trlog['max_acc'] = va
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')          
                
        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))    

    # Test Phase
    trlog = torch.load(osp.join(args.save_path, 'trlog'))
    if args.dataset == "cross":
        # test_set = Dataset_cub('test', args)
        # sampler = CategoriesSampler(test_set.label, 10000, args.val_way, args.shot + args.query)
        # loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)       
        test_file = '~/filelists/CUB/novel.json'
        image_size = 224       
        test_datamgr             = SetDataManager(image_size, n_query = args.query, n_way = args.val_way, n_support = args.shot)
        test_loader              = test_datamgr.get_data_loader( test_file, aug = False) 
    else:
        test_set = Dataset('test', args)
        sampler = CategoriesSampler(test_set.label, 2000, args.val_way, args.shot + args.query)
        test_loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)

    if args.dataset == 'cross':
    	test_acc_record = np.zeros((len(test_loader),))
    else:
    	test_acc_record = np.zeros((2000,))

    base_net.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
    base_net.eval()

    ave_acc = Averager()
    label_val = torch.arange(args.val_way).repeat(args.query)
    label_val = label_val.type(torch.cuda.LongTensor)    

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):           
            data, _ = [_.cuda() for _ in batch]
            data = data.permute(1, 0, 2, 3, 4)        
            data = data.reshape([-1] + list(data.shape[-3:]))             
            k = args.val_way * args.shot
            data_shot, data_query = data[:k], data[k:]
            _, _, logits = base_net(data_shot, data_query)
            acc = count_acc(logits, label_val)
            ave_acc.add(acc)
            test_acc_record[i-1] = acc
            if i % 100 == 0:
            	print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
    m, pm = compute_confidence_interval(test_acc_record)
    log(val_log_file_path, 'Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
    log(val_log_file_path, 'Test Acc {:.6f} + {:.6f}'.format(m, pm))      