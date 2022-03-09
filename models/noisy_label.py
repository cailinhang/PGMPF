import math
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class NoisyNet(torch.nn.Module):
    def __init__(self, c_in, is_fc=False):
        super(NoisyNet, self).__init__()
        #self.c_mid = c_in*2
        self.c_mid = max(c_in // 4, 4)
        self.is_fc =is_fc

        if self.is_fc:
            self.fc1 = nn.Linear(c_in, self.c_mid)
            self.fc2 = nn.Linear(self.c_mid, c_in)
        else:

            self.conv1 = nn.Conv2d(c_in, self.c_mid, 1, padding=0)
            self.bn1 = nn.BatchNorm2d(self.c_mid)
        
            self.conv2 = nn.Conv2d(self.c_mid, c_in, 1, padding=0)
        
        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, input):
        x = input
        #x = self.relu(self.bn1(self.conv1(x)))
        if self.is_fc:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.conv2(x)
        
        #x = torch.sigmoid(x)
        x = torch.tanh(x)

        return x

def noisy_l2_loss(out, label, p_minus=0.01, p_plus=0.01):
    out = out.reshape(-1) 

    
    #label = label.view(-1).float() # 1 or -1 ==> 1 or 0
    
    #loss = (out - label) * (out - label)

    loss = out * out  

    return torch.sqrt(loss.mean() + 1e-8)

    #neg_label = -label
    neg_label = 1-label 
    
    p_y = p_minus*(label>0).float() + p_plus*(neg_label>0).float()
    
    py = p_plus*(label>0).float() + p_minus*(neg_label>0).float()
    
    # convert 1, -1 label to 0, 1 label
    
    #label01 = 0.5*(label+1) # label == 1 ==>  1 ; label == 0 ==> 0
    label01 = label
    #neg_label01 = 0.5*(neg_label+1)
    neg_label01 = neg_label
    
    loss1 = (1-p_y)*(out-label01)*(out-label01)
    
    loss2 = (-py)*(out-neg_label01)*(out-neg_label01)
    
    loss = (loss1 + loss2)/(1-p_minus-p_plus)
    return loss.mean()
