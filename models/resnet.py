import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import sys
sys.path.append('..')
from .res_utils import DownsampleA, DownsampleC, DownsampleD
import math
from .layers import MaskedConv2d, MaskedLinear

class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_mask=False, comp_rate=None, freeze_weight=False):
        super(ResNetBasicblock, self).__init__()
        
        self.use_mask = use_mask
        self.comp_rate= 1 if comp_rate is None else comp_rate

         
        self.freeze_weight = freeze_weight
        self.conv = nn.Conv2d

        if self.use_mask:
            self.conv = MaskedConv2d
            self.comp_rate = 1.
        
        self.in_comp_rate = self.comp_rate


        if self.use_mask:
            self.conv_a = self.conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, 
                            bias=False, in_comp_rate=self.in_comp_rate, comp_rate=self.comp_rate, freeze_weight=self.freeze_weight)
            
            self.conv_b = self.conv(planes, planes, kernel_size=3, stride=1, padding=1,
                            bias=False, in_comp_rate=self.in_comp_rate, comp_rate=self.comp_rate, freeze_weight=self.freeze_weight)

        else:
            self.conv_a = self.conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv_b = self.conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn_a = nn.BatchNorm2d(planes)
        self.bn_b = nn.BatchNorm2d(planes)
    
        self.downsample = downsample

    def forward(self, x):
        residual = x
    
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
    
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
    
        if self.downsample is not None:
          residual = self.downsample(x)
        
        #print('x.size ', x.size())
        #print('after downsample, residual.size ', residual.size())
        #print('basicblock.size ', basicblock.size(), '\n')
        
        return F.relu(residual + basicblock, inplace=True)

default_comp_rate_list = [1] + [0.6] * (19) +[1]

class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """
    def __init__(self, block, depth, num_classes, use_mask=False, comp_rate=None, freeze_weight=False):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()
    
        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    
        self.num_classes = num_classes
        
        if comp_rate is None:
            #comp_rate_list = [1] * (depth + 1) # 卷积层 19层, fc层1层, 19 + 1= 20
            comp_rate = 1
            print('No Pruning!')
        
        self.comp_rate= comp_rate
        self.freeze_weight = freeze_weight
         
    
    
        self.use_mask = use_mask
        
        self.conv, self.linear = nn.Conv2d, nn.Linear # choose mask type
        
        if self.use_mask:
            self.conv = MaskedConv2d
            self.linear = MaskedLinear
            self.comp_rate = 1 #  0.6
        
        print('comp_rate ', self.comp_rate)
        
        
        
        if self.use_mask:
            self.conv_1_3x3 = self.conv(3, 16, kernel_size=3,  stride=1, padding=1, bias=False, in_comp_rate=1, comp_rate=self.comp_rate, freeze_weight=self.freeze_weight)    
         
        else:
            self.conv_1_3x3 = self.conv(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn_1 = nn.BatchNorm2d(int(16 * self.comp_rate))
    
        self.inplanes = int(16 * self.comp_rate)
        
        self.stage_1 = self._make_layer(block, int(16 * self.comp_rate), layer_blocks, 1) # layer_blocks 是每一个stage 的 block数量。
        self.stage_2 = self._make_layer(block, int(32 * self.comp_rate), layer_blocks, 2)
        self.stage_3 = self._make_layer(block, int(64 * self.comp_rate), layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        
        if self.use_mask:
           
            self.classifier = self.linear(64*block.expansion, num_classes, in_comp_rate=self.comp_rate, comp_rate=1)
            
        else:
            
            self.classifier = self.linear(64*block.expansion, num_classes)
    
        for m in self.modules():
          if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            #m.bias.data.zero_()
            if isinstance(m, MaskedConv2d): # 不同的初始化
                    
                    m.help_w.data.normal_(0, math.sqrt(2. / n))
                    m.help_w_in.data.normal_(0, math.sqrt(2. / n))

                    #m.help_w.data.normal_(0, math.sqrt(1. / n)) # 如果 help_w扩宽输出通道数
                    #m.help_w_1.data.normal_(0, math.sqrt(2. / n))
                    #m.help_w_2.data.normal_(0, math.sqrt(2. / n))
                    
                    #if hasattr(m, 'help_w_3'):
                    #    m.help_w_3.data.normal_(0, math.sqrt(2. / n))
                    pass
            
          elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
          elif isinstance(m, nn.Linear) or isinstance(m, MaskedLinear):
            init.kaiming_normal_(m.weight)
            m.bias.data.zero_()
            

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            #print('downsample ',self.inplanes, ' ', planes * block.expansion )
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
    
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.use_mask, self.comp_rate, self.freeze_weight))
        
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, self.use_mask, self.comp_rate, self.freeze_weight))
    
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def resnet20(num_classes=10, use_mask=False, comp_rate=None, freeze_weight=False):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 20, num_classes, use_mask, comp_rate, freeze_weight)
    return model

def resnet32(num_classes=10, use_mask=False, comp_rate=None, freeze_weight=False):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 32, num_classes, use_mask, comp_rate, freeze_weight)
    return model

def resnet44(num_classes=10, use_mask=False, comp_rate=None, freeze_weight=False):
    """Constructs a ResNet-44 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 44, num_classes, use_mask, comp_rate, freeze_weight)
    return model

def resnet56(num_classes=10, use_mask=False, comp_rate=None, freeze_weight=False):
    """Constructs a ResNet-56 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 56, num_classes, use_mask, comp_rate, freeze_weight)
    return model

def resnet110(num_classes=10, use_mask=False, comp_rate=None, freeze_weight=False):
    """Constructs a ResNet-110 model for CIFAR-10 (by default)
    Args:
        num_classes (uint): number of classes
    """
    model = CifarResNet(ResNetBasicblock, 110, num_classes, use_mask, comp_rate, freeze_weight)
    return model

if __name__ == '__main__':
    net = resnet56(num_classes=10, use_mask=True)
    x = torch.randn(size=(4,3,32,32))
    output = net(x)
    print(output.size())
    print(output[0])
    
    net.eval()
    output1 = net(x)
    print(output[0])
    l = [m for m in net.named_parameters()]
    print(net)
    for m in net.named_parameters():
        if not isinstance(m ,tuple):
            if len(m.size()) == 2:
                print(m.size(), m)
    print('\n')

    for index, item in enumerate(net.named_parameters()):
        if len(item[1].size()) == 4:
            print(index, ' ', item[0], ' ', item[1].data.size())
    
    print('\n')
    
    for index, m in enumerate(net.modules()):
                        
        if isinstance(m, nn.Conv2d) or isinstance(m, MaskedConv2d):
            print(index, ' ', str(type(m)))
