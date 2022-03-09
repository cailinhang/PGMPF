"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import numpy as np
#from .noisy_label import NoisyNet
import torch.nn.init as init
import torch.nn as nn
import math

class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(context, probs):
        binarized = (probs == torch.max(probs, dim=1, keepdim=True)[0]).float()
        context.save_for_backward(binarized)
        return binarized

    @staticmethod
    def backward(context, gradient_output):
        binarized, = context.saved_tensors
        gradient_output[binarized == 0] = 0
        return gradient_output

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 mask_init='normal', mask_scale=1e-2, in_comp_rate=0.5, comp_rate=0.5, freeze_weight=False):
        
        super(MaskedConv2d, self).__init__(int(in_channels*in_comp_rate), int(out_channels*comp_rate), # pruned number of filters 
            kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        
        #alpha = torch.Tensor([0,1,0,1,1,1,0,1,0]).reshape(1,1, 3, 3)
        
        #alpha = torch.Tensor([1,1,1,1,1,1,1,1,1]).reshape(1,1, 3, 3)
        alpha_vert = torch.Tensor([0,1,0,0,1,0,0,1,0]).reshape(1,1, 3, 3)
        alpha_hori = torch.Tensor([0,0,0,1,1,1,0,0,0]).reshape(1,1, 3, 3)
        
        self.in_comp_rate = in_comp_rate
        self.comp_rate = comp_rate

        self.in_channels = int(in_channels*self.in_comp_rate)
        self.out_channels = int(out_channels*self.comp_rate)
        
        #self.alpha = Parameter(alpha, requires_grad=True)
        self.alpha_vert = Parameter(alpha_vert, requires_grad=False)
        self.alpha_hori = Parameter(alpha_hori, requires_grad=False)

        
        #self.out_channels_in_group_assignment_map = nn.Parameter(torch.Tensor(out_channels * 3, 2))
        #nn.init.normal_(self.out_channels_in_group_assignment_map)
        
        #self.binarize = Binarize.apply
        outs, ins, k, k = self.weight.shape
        
        self.help_w = self.weight.data.new( size=(2*outs, ins, k, k )  ) # 扩宽输出通道数
        self.help_w_in = self.weight.data.new( size=(outs, 2*ins, k, k )  ) # 扩宽输入通道数
        
        #self.help_w = self.weight.data.new( size=(self.weight.size() )  )
        
        #self.help_w_1 = self.weight.data.new( size=(self.weight.size() )  )
        #self.help_w_2 = self.weight.data.new( size=(self.weight.size() )  )

        mask = torch.ones_like(self.weight)
        self.mask = Parameter(mask, requires_grad=False)

        mask_bn = torch.ones(size=(1, outs, 1, 1))
        self.mask_bn = Parameter(mask_bn, requires_grad=False)

        #self.bias_1 = self.weight.data.new( size=(outs, 1 )  )
        #self.bias_2 = self.weight.data.new( size=(outs, 1)  )

        #self.bias_1.data.zero_()
        #self.bias_2.data.zero_()

        #self.help_w.data.copy_(self.weight.data)
        #self.help_w_1.data.copy_(self.weight.data)
        #self.help_w_2.data.copy_(self.weight.data)

        self.help_w = Parameter(self.help_w)
        self.help_w_in = Parameter(self.help_w_in)
        #self.help_w_1 = Parameter(self.help_w_1)
        #self.help_w_2 = Parameter(self.help_w_2)
        #self.bias_1 = Parameter(self.bias_1)
        #self.bias_2 = Parameter(self.bias_2)
        
        self.bn = nn.BatchNorm2d(num_features=self.out_channels, affine=True)
        self.bn_1 = nn.BatchNorm2d(num_features=self.out_channels, affine=True)
        self.bn_2 = nn.BatchNorm2d(num_features=self.out_channels, affine=True)
        
        gamma_init = 0.333
        self.init_gamma(gamma_init)
        
        

        #self.in_comp_rate = 1
        #self.comp_rate = 1
        
        reduced_in_channels = int(self.in_comp_rate * in_channels) 
        reduced_channels = int(self.comp_rate * out_channels) 
       
        self.dimension_reduce_in = self.weight.data.new( size=(reduced_in_channels, self.in_channels*2 ) )
        
        self.dimension_reduce = self.weight.data.new( size=(reduced_channels, self.out_channels*2 )  )
        
        #self.dimension_reduce_2 = self.weight.data.new( size=(reduced_channels, self.out_channels*2 )  )

        nn.init.normal_(self.dimension_reduce_in)
        nn.init.normal_(self.dimension_reduce)
        #nn.init.normal_(self.dimension_reduce_2)
        
        #if mask_init == 'normal':
           
         #   self.dimension_reduce.normal_(0, 0.01)
         #   self.dimension_reduce_in.normal_(0, 0.01)
          
        # mask_real is now a trainable parameter.
        
        self.dimension_reduce_in = Parameter(self.dimension_reduce_in)
        self.dimension_reduce = Parameter(self.dimension_reduce)
        #self.dimension_reduce_2 = Parameter(self.dimension_reduce_2)
        
    
    def init_gamma(self, gamma_value):
        
        #assert (gamma_value - 0.333) < 0.1
        init.constant_(self.bn.weight, gamma_value ) # TODO: gamma would be 1/2 1/4 1/4
        init.constant_(self.bn_1.weight, gamma_value  )
        init.constant_(self.bn_2.weight, gamma_value )
        
        #print('init gamma of square, ver and hor as ', gamma_value)
    
    
        
    
    def get_trunk(self):
        ''' 
            return trunked weight shaped (new_outs, new_ins, k, k) 
            from top-left of self.weight
        '''
        outs, ins, k, k = self.weight.shape
        
        #new_ins = self.dimension_reduce_in.shape[0]
        new_ins = ins
        new_outs = self.dimension_reduce.shape[0]
        
        weight_trunk = self.weight[0:new_outs, 0: new_ins]
          
        return weight_trunk
    
    
    def reset_parameters(self):
        n = self.in_channels
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        
        # (3, 3) or (3*outs, 3)
        #mp = self.binarize(torch.softmax(self.out_channels_in_group_assignment_map, dim=1))
        #mp = torch.softmax(self.out_channels_in_group_assignment_map, dim=1).reshape((1, 3*self.out_channels,2, 1, 1))
        
        #mp = self.binarize(torch.softmax(self.out_channels_in_group_assignment_map, dim=1))
        #mp = mp.reshape((1, 3*self.out_channels,3, 1, 1)) # 二值化

        #mp_0 = mp[:, :self.out_channels] # (1, C, 3, 1, 1)
        #mp_1 = mp[:, self.out_channels: 2 * self.out_channels]
        #mp_2 = mp[:, 2*self.out_channels:]
        #"""
        #print('self.weight ', self.weight.size())
        #out = F.conv2d(x, (self.weight*self.mask), self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
        
        #return out
        #"""
        #if 0 and self.in_channels == self.out_channels:
        #    out_bn = self.bn(out + x)        
        #else:
        #out_bn = self.bn(out)
#
#
        #out_1 = F.conv2d(x, (self.alpha_vert *self.mask* self.help_w_1), self.bias, self.stride,
        #        self.padding, self.dilation, self.groups)
#
        #if 0 and self.in_channels == self.out_channels:
        #    out_1_bn = self.bn_1(out_1 + x)
        #else:
        #out_1_bn = self.bn_1(out_1)
#
        #out_2 = F.conv2d(x, (self.alpha_hori * self.mask * self.help_w_2), self.bias, self.stride,
        #                self.padding, self.dilation, self.groups)
#        
        #if 0 and self.in_channels == self.out_channels:
        #    out_2_bn = self.bn_2(out_2 + x)
        #else:
        #out_2_bn = self.bn_2(out_2)

        # (1, C, 1, 1)
        #out_bn = self.bn(mp_0[:, :, 0] *out + mp_0[:, :,1] * out_1 + mp_0[:, :,2] * out_2)
        #out_1_bn = self.bn_1(mp_1[:, :,0] *out + mp_1[:, :,1] * out_1 + mp_1[:, :,2] * out_2)
        #out_2_bn = self.bn_2(mp_2[:, :,0] *out + mp_2[:, :,1] * out_1 + mp_2[:, :,2] * out_2)

        #out_bn = self.bn(mp_0[:, :, 0] *out + mp_0[:, :,1] * out_1)
        #out_1_bn = self.bn_1(mp_1[:, :,0] *out_1 + mp_1[:, :,1] * out_2)
        #out_2_bn = self.bn_2(mp_2[:, :,0] *out_2 + mp_2[:, :,1] * out)
        
        #return out_bn + out_1_bn + out_2_bn
       
        
        # Mask weights with above mask.
        outs, ins, k, k = self.weight.shape
        
        new_ins = self.dimension_reduce_in.shape[0]
        new_outs = self.dimension_reduce.shape[0]
           
        
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        #start_idx = outs - new_outs 
        
        #mp1 = torch.softmax(self.dimension_reduce, dim=1)
        #mp1 = torch.sigmoid(self.dimension_reduce)
        
        beta = 1 # 控制残差映射的强度

        help_w_1 = torch.matmul(self.dimension_reduce, self.help_w[:2*outs].contiguous().view(2 * outs, -1))
        #help_w_1 += self.bias_1# # w' = Pw + b
        #help_w_1 = beta * help_w_1.view(new_outs, ins, k, k)
        help_w_1 = beta * help_w_1.view(new_outs, ins, k, k)  + self.help_w[:outs]
        
        
        #start_idx_ins = ins - new_ins
        #mp2 = torch.softmax(self.dimension_reduce_2, dim=1)
        #mp2 = torch.sigmoid(self.dimension_reduce_2)
        
        help_w_2 = torch.matmul(self.dimension_reduce_in, self.help_w_in[:, :2*ins].contiguous().view( 2 * ins, -1))
        
        #help_w_2 = beta * help_w_2.view(outs, new_ins, k, k) 
        help_w_2 = beta * help_w_2.view(outs, new_ins, k, k) + self.help_w_in[:, :new_ins]
        
        #help_w_2 = torch.matmul(self.dimension_reduce_2, self.help_w[:2*outs].contiguous().view( 2 * outs, -1))
        #help_w_2 += self.bias_2

        #help_w_2 = beta * help_w_2.view(new_outs, ins, k, k) + self.help_w[outs:]
        
        
        out = F.conv2d(x, (self.weight), self.bias, self.stride,
                self.padding, self.dilation, self.groups)

        out = self.bn(out)
        
        out_1 = F.conv2d(x, ((self.alpha_vert) * help_w_1), self.bias, self.stride,
                self.padding, self.dilation, self.groups)

        out_1 = self.bn_1(out_1)
        
        out_2 = F.conv2d(x, ((self.alpha_hori)  *help_w_2), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
        out_2 = self.bn_2(out_2)
        
        return (out + out_1 + out_2) * self.mask_bn
        
        
#        return F.conv2d(x, (self.weight  + self.alpha_vert * self.help_w_1 + self.alpha_hori * self.help_w_2), self.bias, self.stride,
#                        self.padding, self.dilation, self.groups)
#
#        return F.conv2d(x, (self.weight  + self.alpha_vert * self.help_w_1 + self.alpha_hori * self.help_w_2)/3, self.bias, self.stride,
#                        self.padding, self.dilation, self.groups)




class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mask_init='normal', mask_scale=1e-2, in_comp_rate=0.5, comp_rate=0.5, freeze_weight=False):
        super(MaskedLinear, self).__init__(int(in_features*in_comp_rate), int(out_features*comp_rate), bias) 
        
        #self.in_features = in_features
        #self.out_features = out_features
        #self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        #self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        
        #if bias:
        #    self.bias = Parameter(torch.Tensor(out_features))
        #else:
        #    self.register_parameter('bias', None)
        
        #self.reset_parameters()
        
        self.in_comp_rate = in_comp_rate
        self.comp_rate = comp_rate

        self.in_features = int(self.in_comp_rate * in_features)
        self.out_features = int(self.comp_rate * out_features)


        reduced_in_channels = int(self.in_comp_rate * in_features) 
        reduced_channels = int(self.comp_rate * out_features) 
       
        #self.dimension_reduce_in = self.weight.data.new( size=(reduced_in_channels, self.in_channels ) )
        
        self.dimension_reduce = self.weight.data.new( size=(reduced_channels, self.out_features )  )

        
        if mask_init == 'normal':
           
            self.dimension_reduce.normal_(0, 0.01)
            #self.dimension_reduce_in.normal_(0, 0.01)
          
        # mask_real is now a trainable parameter.
        #self.dimension_reduce_in = Parameter(self.dimension_reduce_in)
        self.dimension_reduce = Parameter(self.dimension_reduce)
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        
        # Mask weights with above mask.
        #outs, ins = self.weight.shape
        
        #new_ins = self.dimension_reduce_in.shape[0]
           
        
        # 维度变换之后必须要使用.contiguous()使得张量在内存连续之后才能调用view函数
        #masked_weight = torch.matmul(self.dimension_reduce_in, self.weight.contiguous().view(ins, -1))
        
        #masked_weight = masked_weight.view(outs, new_ins, k, k) 
        
        #new_outs = self.dimension_reduce.shape[0]
        
        #masked_weight_v2 = torch.matmul(self.dimension_reduce, self.weight.contiguous().view(outs, -1))
        
        #masked_weight_v2 = masked_weight_v2.view(new_outs, ins) 
        
        return F.linear(input, self.weight , self.bias)

