'''
This program implements mixup and cutout regularizations.

References:
    https://github.com/facebookresearch/mixup-cifar10
    https://github.com/uoguelph-mlrg/Cutout
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def label_smoothing(target, epsilon=0.1):
    
    num_classes = target.shape[-1]
    
    assert num_classes >0
    
    return (1 - epsilon) * target + epsilon/num_classes
    

def mixup_data(x, y, alpha = 1.0, use_cuda=True):
    '''
    Adds convex combinations of training examples.
        
    Args:
        x (Tensor): Inputs.
        y (Tensor): Target labels.
        alpha (int): Mixup interpolation coefficient.
    
    Returns:
        tuple: Mixed inputs, pairs of targets, and lambda.
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    #index = torch.randperm(batch_size).to(self.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''
    Loss function for mixup.
        
    Args:
        criterion (Loss): A loss function.
        pred (Tensor): Predicted outputs.
        y_a, y_b (Tensor): Mixup labels.
        lam (float): lambda coefficient.
    
    Returns:
        Tensor: The value of mixup loss function.
    '''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class Cutout(object):
    ''' Cutout regularization, which randomly masks out one or more patches from an image. '''
    
    def __init__(self, n_holes, length):
        '''
        Initializes a Cutout object.
        
        Args:
            n_holes (int): The number of patches to cut out of each image.
            length (int): The length (in pixels) of each square patch.
        '''
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        '''
        Randomly masks out one or more patches from an image.
        
        Args:
            img (Tensor): A Tensor image of size (C, H, W).
            
        Returns:
            Tensor: An image with n_holes of dimension length x length cut out of it.
        '''
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
