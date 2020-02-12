import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def mdd_loss(args, features, outputs, outputs_adv, labels, src_idx, tgt_idx):
        # batch_size = train_way * (shot + query)
        # inputs.size(): [batch_size, 3, h, w]
        # labels.size(): [batch_size]
        # src_idx: image indecies from source domain
        # tgt_idx: image indecies from target domain
        # len(src_idx) + len(tgt_idx) = batch_size
        labels_src = labels[src_idx]
        labels_tgt = labels[tgt_idx]
        src_classifier_criterion = nn.CrossEntropyLoss()

        classifier_loss = src_classifier_criterion(outputs[src_idx, :], labels_src)    

        pred_labels_adv = outputs.max(1)[1] # class labels predicted by the labeling function h_f        
        pred_labels_adv_src = pred_labels_adv[src_idx]
        pred_labels_adv_tgt = pred_labels_adv[tgt_idx]

        classifier_loss_adv_src = src_classifier_criterion(outputs_adv[src_idx, :], pred_labels_adv_src)

        logloss_tgt = torch.log(1 - F.softmax(outputs_adv[tgt_idx, :], dim = 1))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, pred_labels_adv_tgt)

        if classifier_loss_adv_tgt > 100000:
            print("inf error of domain adptation!")
            classifier_loss_adv_tgt = classifier_loss_adv_src

        transfer_loss = args.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt    

        total_loss = classifier_loss + transfer_loss

        if args.print_i_mdd % 50 == 0:
            print('iter {} mdd_classifier_loss = {:.4f}, mdd_transfer_loss = {:.4f}'.format(\
                args.print_i_mdd, classifier_loss.item(), transfer_loss.item()))
       
        return total_loss        

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


def calc_coeff(iter_num, high=0.1, low=0.0, alpha=1.0, max_iter=1000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(- alpha * iter_num / max_iter)) - (high - low) + low)


class classifier(nn.Module):
    def __init__(self, in_dim, width, class_num, grl=False):
        super(classifier, self).__init__()
        self.grl = grl

        self.fc1 = nn.Linear(in_dim, width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(width, class_num)

        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)

        if self.grl:
            self.iter_num = 0
            self.alpha = 1.0
            self.low = 0.0
            self.high = 0.1
            self.max_iter = 1000.0

    def forward(self, x):
        if self.grl:
            if self.training:
                self.iter_num += 1
                coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
                x = x * 1.0
                x.register_hook(grl_hook(coeff))
            else :
                coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
                x = x * 1.0
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        y = self.fc2(x)

        return y


class MDDNet(nn.Module):
    def __init__(self, base_net='ConvNet', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=64, pretrain=False):
        super(MDDNet, self).__init__()
        if base_net == 'ConvNet':
            fea_dim = 64
            self.encoder_layer = nn.Linear(fea_dim, int(fea_dim/2))
            self.decoder_layer = nn.Linear(int(fea_dim/2), fea_dim)
        elif base_net == 'ResNet':
            fea_dim = 640
            self.encoder_layer = nn.Linear(fea_dim, 256)
            self.decoder_layer = nn.Linear(256, fea_dim)
        elif base_net == 'ResNet18':
            fea_dim = 512
            self.encoder_layer = nn.Linear(fea_dim, 256)
            self.decoder_layer = nn.Linear(256, fea_dim)
        else:
            self.encoder_layer = nn.Linear(fea_dim, 256)
            self.decoder_layer = nn.Linear(256, fea_dim)
            fea_dim = 512
            
        self.use_bottleneck = use_bottleneck        
        self.bottleneck_layer_list = [nn.Linear(fea_dim, bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        
        self.classifier1 = classifier(bottleneck_dim, width, class_num, False)
        self.classifier2 = classifier(bottleneck_dim, width, class_num, True)        

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

        ## collect parameters
        # self.parameter_list = [{"params":self.bottleneck_layer.parameters(), "lr":1}, # original:1
        #                     {"params":self.classifier1.parameters(), "lr":1}, #original: 1
        #                     {"params":self.classifier2.parameters(), "lr":1}] #original: 1        

        self.parameter_list = [{"params":self.encoder_layer.parameters(), "lr":1}, # origninal:0.1
        					{"params":self.decoder_layer.parameters(), "lr":1}, # original : 0.1
                            {"params":self.bottleneck_layer.parameters(), "lr":1}, # original:1
                            {"params":self.classifier1.parameters(), "lr":1}, #original: 1
                            {"params":self.classifier2.parameters(), "lr":1}] #original: 1        
   

    def forward(self, features):

        encoder_fea = self.encoder_layer(features)
        decoder_fea = self.decoder_layer(encoder_fea)
        if self.use_bottleneck:
            features_b = self.bottleneck_layer(decoder_fea)
        
        outputs = self.classifier1(features_b)
        outputs_adv = self.classifier2(features_b)
        
        return decoder_fea, outputs, outputs_adv

    def get_parameter_list(self):
        return self.parameter_list