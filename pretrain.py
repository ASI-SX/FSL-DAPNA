import argparse
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from proto_mdd.models.classifier import Classifier
from proto_mdd.dataloader.samplers import CategoriesSampler
from proto_mdd.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, log
from tensorboardX import SummaryWriter
from tqdm import tqdm

# pre-train backbone
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ngpu', type=int, default=2, help='0 = CPU.')
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImagenet','CUB','cross'])    
    parser.add_argument('--model_type', type=str, default='ResNet', choices=['ConvNet', 'ResNet', 'ResNet18'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 50, 80], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--log_val_file', type=str, default=None)
    parser.add_argument('--way', type=int, default=5)
    args = parser.parse_args()
    pprint(vars(args))
    
    save_path1 = '-'.join([args.dataset, args.model_type, 'Pre'])
    save_path2 = '_'.join([str(args.lr), str(args.gamma)])
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path)
    args.log_val_file = osp.join(args.save_path, 'log_val_file.txt')
    args.train_log_file = osp.join(args.save_path, 'log_train_file.txt')

    if args.dataset == 'MiniImageNet':        
        from proto_mdd.dataloader.mini_imagenet_pre import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from proto_mdd.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImagenet':
        from proto_mdd.dataloader.tiered_imagenet import tieredImageNet as Dataset    
    elif args.dataset == 'cross':
        from proto_mdd.dataloader.mini_imagenet_pre import MiniImageNet as Dataset_mini
        from proto_mdd.dataloader.cub import CUB as Dataset_cub
    else:
        raise ValueError('Non-supported Dataset.')

    if args.dataset == 'cross':
        trainset = Dataset_mini('all', args)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        args.num_class = trainset.num_class
        valset = Dataset_cub('val', args)
        val_sampler = CategoriesSampler(valset.label, 200, args.way, 16) # test on args.way 1-shot 15-query
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)    
        args.shot = 1    
    else:
        trainset = Dataset('train', args)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        args.num_class = trainset.num_class
        valset = Dataset('val', args)
        val_sampler = CategoriesSampler(valset.label, 200, args.way, 16) # test on 16-way 1-shot
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)    
        args.shot = 1
    
    # construct model
    model = Classifier(args)   

    if args.model_type == 'ConvNet':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif args.model_type == 'ResNet' or args.model_type == 'ResNet18':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder')    
    criterion = torch.nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.ngpu  > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))                

    model = model.cuda()
    criterion = criterion.cuda()
    
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
        state = {'epoch': epoch + 1,
                 'args': args,
                 'state_dict': model.state_dict(),
                 'trlog': trlog,
                 'val_acc': trlog['max_acc'],
                 'optimizer' : optimizer.state_dict(),
                 'global_count': global_count}
        
        torch.save(state, osp.join(args.save_path, filename))
        if is_best:
            shutil.copyfile(osp.join(args.save_path, filename), osp.join(args.save_path, 'model_best.pth.tar'))
    
    if args.resume == True:
        # load checkpoint
        # state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))        
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        # resumed_state = {'module.'+k:v for k,v in resumed_state.items()}
        model.load_state_dict(resumed_state)
        trlog = state['trlog']
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_count = state['global_count']
    else:
        init_epoch = 1
        trlog = {}
        trlog['args'] = vars(args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        initial_lr = args.lr
        global_count = 0

    timer = Timer()
    writer = SummaryWriter(logdir=args.save_path) # should change to log_dir for previous version tensorboardX
    for epoch in range(init_epoch, args.max_epoch + 1):
        # refine the step-size
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        model.train()
        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, label = [_.cuda() for _ in batch]
                label = label.type(torch.cuda.LongTensor)
            else:
                data, label = batch
                label = label.type(torch.LongTensor)
            logits = model(data)
            loss = criterion(logits, label)
            acc = count_acc(logits, label)
            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))
            if i % 50 == 0:
                log(args.train_log_file, 'epoch {}, train {}/{}, loss={:.4f} acc={:.4f}\n'.format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()

        # do not do validation in first 500 epoches
        if epoch % 1 == 0:
            model.eval()
            vl = Averager()
            va = Averager()
            if epoch % 10 == 0:
                log(args.log_val_file, 'best epoch {}, current best val acc={:.4f}\n'.format(trlog['max_acc_epoch'], trlog['max_acc']))            
            # test performance with Few-Shot
            label = torch.arange(args.way).repeat(15)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)        
            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader, 1)):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data, _ = batch
                    data_shot, data_query = data[:args.way], data[args.way:] # 16-way test
                    if args.ngpu > 1:
                        logits = model.module.forward_proto(data_shot, data_query, args.way)
                    else:
                        logits = model.forward_proto(data_shot, data_query, args.way)
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    vl.add(loss.item())
                    va.add(acc)

            vl = vl.item()
            va = va.item()
            writer.add_scalar('data/val_loss', float(vl), epoch)
            writer.add_scalar('data/val_acc', float(va), epoch)        
            log(args.log_val_file, 'epoch {}, val, loss={:.4f} acc={:.4f}\n'.format(epoch, vl, va))
    
            if va > trlog['max_acc']:
                trlog['max_acc'] = va
                trlog['max_acc_epoch'] = epoch
                save_model('max_acc')
                save_checkpoint(True)
    
            trlog['train_loss'].append(tl)
            trlog['train_acc'].append(ta)
            trlog['val_loss'].append(vl)
            trlog['val_acc'].append(va)
            save_model('epoch-{}'.format(epoch))
    
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    writer.close()          