import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('..')
from models.layers import MaskedConv2d, MaskedLinear

__all__ = ['vgg']

defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

#comp_rate_list = [0.5, 0.5, 1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25]# vgg16 final comp rate in paper: efficient ConvNets

default_comp_rate_list = [1, 0.5, 0.5, 1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 1]# plus a input 3-channels not pruned

class vgg(nn.Module):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, use_mask=False, comp_rate_list=None, freeze_weight=False):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        
        if comp_rate_list is None:
            comp_rate_list = [1] * (depth-1)
            print('No Pruning!')
        
        self.cfg = cfg
        self.comp_rate_list = comp_rate_list
        self.freeze_weight = freeze_weight
         
        #self.comp_rate_list = default_comp_rate_list # IF Prune or train pruned baseline
        
        #print('comp_rate_list ', self.comp_rate_list)
        
        self.use_mask = use_mask
        
        self.conv, self.linear = nn.Conv2d, nn.Linear # choose mask type
        
        if self.use_mask:
            self.conv = MaskedConv2d
            self.linear = MaskedLinear
            #self.comp_rate_list = default_comp_rate_list#  Self Fine-tuning
        
        print('comp_rate_list ', self.comp_rate_list)

        self.feature = self.make_layers(cfg, True) # 先不用BN试试看

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        
        
        
        if self.use_mask:
            # comp_rate_list[-2] is the output channels of the final conv layer, i.e, that input channels to the first fc
            self.classifier = self.linear(cfg[-1], num_classes, in_comp_rate=self.comp_rate_list[-2], comp_rate=1) # final fc not pruned
            
            # fc 层 不剪枝
            #self.classifier = nn.Sequential(
            #    self.linear(cfg[-1], 512, in_comp_rate=self.comp_rate_list[-2], comp_rate=1, freeze_weight=self.freeze_weight),
                #nn.BatchNorm1d(512),
            #    nn.ReLU(inplace=True),
            #    self.linear(512, num_classes, in_comp_rate=1, comp_rate=1, freeze_weight=self.freeze_weight)
            #)
            
        else:
            self.classifier = self.linear(cfg[-1], num_classes)
            
            #ins_reduced = int(cfg[-1] * self.comp_rate_list[-2])

            #self.classifier = nn.Sequential(
            #    nn.Linear(ins_reduced, 512),
                #nn.BatchNorm1d(512),
            #    nn.ReLU(inplace=True),
            #    nn.Linear(512, num_classes)
            #)
          
        
        if init_weights:
            self._initialize_weights()
    def initial_from_pretrain(self, pretrained_model):

        for module, module_pretrained in zip(self.feature.children(), pretrained_model.feature.children()):

            if 'Conv' in str(type(module)):
                #print(str(type(module)))
                #print(module.weight.data[0][0][0][:5], '   ', module_pretrained.weight.data[0][0][0][:5])
                module.weight.data.copy_(module_pretrained.weight.data)
                if module.bias:
                    module.bias.data.copy_(module_pretrained.bias.data)
            elif 'BatchNorm' in str(type(module)):
                module.weight.data.copy_(module_pretrained.weight.data)
                module.bias.data.copy_(module_pretrained.bias.data)
                module.running_mean.copy_(module_pretrained.running_mean)
                module.running_var.copy_(module_pretrained.running_var)
                #print(str(type(module)))

        for module, module_pretrained in zip(self.classifier.children(), pretrained_model.classifier.children()):
            if 'Masked' in str(type(module)) or 'Linear' in str(type(module)):
                #print( str(type(module)))
                module.weight.data.copy_(module_pretrained.weight.data)
                module.bias.data.copy_(module_pretrained.bias.data)
            elif 'BatchNorm' in str(type(module)):
                module.weight.data.copy_(module_pretrained.weight.data)
                module.bias.data.copy_(module_pretrained.bias.data)
                module.running_mean.copy_(module_pretrained.running_mean)
                module.running_var.copy_(module_pretrained.running_var)
            #else:
                #print(str(type(module)))


        print('initialize from pretrain model!')


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        
        idx = 0
        print(cfg)
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                #print(v)
                if self.use_mask:
                    conv2d = self.conv(in_channels, v, kernel_size=3, padding=1, bias=False, in_comp_rate=self.comp_rate_list[idx], comp_rate=self.comp_rate_list[idx+1], freeze_weight=self.freeze_weight)    
                    idx += 1
                    #print('idx ', idx)
                else:    
                    #conv2d = self.conv(in_channels, v, kernel_size=3, padding=1, bias=False)
                    ins_reduced = int(in_channels * self.comp_rate_list[idx]) # NOTE: BN must not be used!!
                    outs_reduced = int(v * self.comp_rate_list[idx+1])
                    idx += 1
                    conv2d = self.conv(ins_reduced, outs_reduced, kernel_size=3, padding=1, bias=False)

                if batch_norm:
                    outs_reduced = int(v * self.comp_rate_list[idx])

                    layers += [conv2d, nn.BatchNorm2d(outs_reduced), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
                #print(str(type(m)))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #if self.use_mask:
                #    n = int(n * m.comp_rate) # 乘以通道剪枝比例 
                #print(n)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
                if isinstance(m, MaskedConv2d): # 不同的初始化
                    m.help_w.data.normal_(0, math.sqrt(2. / n))
                    m.help_w_in.data.normal_(0, math.sqrt(2. / n))

                    #m.help_w.data.normal_(0, math.sqrt(1. / n)) # 如果 help_w扩宽输出通道数
                    #m.help_w_1.data.normal_(0, math.sqrt(2. / n))
                    #m.help_w_2.data.normal_(0, math.sqrt(2. / n))
                    
                    #if hasattr(m, 'help_w_3'):
                    #    m.help_w_3.data.normal_(0, math.sqrt(2. / n))
                    pass

                if m.bias is not None:
                    m.bias.data.zero_()
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, MaskedLinear):
                print(str(type(m)))
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':
    #net = vgg(depth=16)
    x = Variable(torch.FloatTensor(16, 3, 32, 32))
    #y = net(x)
    #print(y.data.shape)
    #print(net) 
    net1 = vgg(depth=16, use_mask=True)
    y1 = net1(x)
    print(y1.data.shape)
    print(net1)
    for index, item in enumerate(net1.named_parameters()):
        if len(item[1].size()) == 4:
            print(index, ' ', item[0], ' ', item[1].data.size())
    
    print('\n')
    for idx, m in enumerate(net1.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
            print(idx, str(type(m)))

