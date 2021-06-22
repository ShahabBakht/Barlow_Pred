import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
from select_backbone import select_resnet, select_mousenet, select_simmousenet, select_monkeynet
from convrnn import ConvGRU


class BP_RNN(nn.Module):
    '''DPC with RNN'''
    def __init__(self, sample_size, num_seq=8, seq_len=5, pred_step=3, network='resnet50', proj_size = 1024, lambd = 5e-3, hp='./SimMouseNet_hyperparams.yaml'):
        super(BP_RNN, self).__init__()
        torch.cuda.manual_seed(233) #233
        print('Using BP-RNN model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.pred_step = pred_step
        self.proj_size = proj_size
        self.lambd = lambd
        
        if network == 'vgg' or network == 'mousenet' or network == 'simmousenet' or network == 'monkeynet':
            self.last_duration = seq_len
        else:
            self.last_duration = int(math.ceil(seq_len / 4))
        
        if network == 'resnet0':
            self.last_size = int(math.ceil(sample_size / 8)) #8
            self.pool_size = 1
        elif network == 'mousenet':
            self.last_size = 16
            self.pool_size = 2 # (2 for all readout, 4 for VISp5 readout)
        elif network == 'simmousenet':
            self.last_size = 16
            self.pool_size = 1
        elif network == 'monkeynet':
            self.last_size = 16
            self.pool_size = 1
        else:
            self.last_size = int(math.ceil(sample_size / 32))
            self.pool_size = 1
            
            
        print('final feature map has size %dx%d' % (self.last_size, self.last_size))
        if network == 'mousenet':
            self.backbone, self.param = select_mousenet()
        elif network == 'simmousenet':
            self.backbone, self.param = select_simmousenet(hp)
        elif network == 'monkeynet':
            self.backbone, self.param = select_monkeynet()
        else:
            self.backbone, self.param = select_resnet(network, track_running_stats=False)
            
        self.param['num_layers'] = 1 # param for GRU
        self.param['hidden_size'] = self.param['feature_size'] # param for GRU

        self.agg = ConvGRU(input_size=self.param['feature_size'],
                               hidden_size=self.param['hidden_size'],
                               kernel_size=1,
                               num_layers=self.param['num_layers'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                                )
        self.projector = nn.Linear(self.param['feature_size']*self.pred_step*self.last_size**2,self.proj_size,bias=False)
        self.bn = nn.BatchNorm1d(self.proj_size, affine=False)
        
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        # block: [B, N, C, SL, W, H]
        ### extract feature ###
        (B, N, C, SL, H, W) = block.shape
        block = block.view(B*N, C, SL, H, W)
        
        feature = self.backbone(block)
         
        del block
        feature = F.avg_pool3d(feature, (self.last_duration, self.pool_size, self.pool_size), stride=(1, self.pool_size, self.pool_size))
        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # before ReLU, (-inf, +inf)
        feature = self.relu(feature) # [0, +inf)
        feature = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size) # [B,N,D,6,6], [0, +inf)
        feature_inf = feature_inf_all[:, N-self.pred_step::, :].contiguous()
        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(feature[:, 0:N-self.pred_step, :].contiguous())
        hidden = hidden[:,-1,:] # after tanh, (-1,1). get the hidden state of last layer, last time step
        
        pred = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:,-1,:]
        pred = torch.stack(pred, 1) # B, pred_step, xxx
        del hidden


        ### Get similarity score ###
        # pred: [B, pred_step, D, last_size, last_size]
        # GT: [B, N, D, last_size, last_size]
        N = self.pred_step
        pred = pred.contiguous().view(B, self.param['feature_size']*self.pred_step*self.last_size**2)
        feature_inf = feature_inf.contiguous().view(B, self.param['feature_size']*self.pred_step*self.last_size**2)
        pred = (self.bn(self.projector(pred)))#.transpose(0,1)
        feature_inf = (self.bn(self.projector(feature_inf)))
#         print(f'std pred: {pred.std(0) }, std gt: {feature_inf.std(0)}')
#         pred = pred.permute(0,1,3,4,2).contiguous().view(B*self.pred_step*self.last_size**2, self.param['feature_size']).transpose(0,1)
#         feature_inf = feature_inf.permute(0,1,3,4,2).contiguous().view(B*N*self.last_size**2, self.param['feature_size'])

#         c = torch.matmul(pred, feature_inf) / B
        
#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag = off_diagonal(c).pow_(2).sum()
#         loss = on_diag + self.lambd * off_diag
#         del feature_inf, pred

#         return loss, on_diag, off_diag
        return pred, feature_inf

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)
        # other resnet weights have been initialized in resnet itself

    def reset_mask(self):
        self.mask = None
    

if __name__ == "__main__":
    
    mydata = torch.FloatTensor(10, 8, 3, 5, 64, 64).to('cuda')
    nn.init.normal_(mydata)
    
    model = BP_RNN(sample_size = 64, network = 'monkeynet').to('cuda')
    
    loss = model(mydata)
    print(loss)
    
#     print(score.shape)