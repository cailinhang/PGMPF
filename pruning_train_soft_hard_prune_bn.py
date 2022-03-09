# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os, sys, math
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
from utils import convert_secs2time, time_string, time_file_str
# from models import print_log
import models
import random
import numpy as np
from collections import OrderedDict
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import warnings
warnings.filterwarnings('ignore')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# Train data
parser.add_argument('--train_data', metavar='DIR', default='/dev/shm/ImageNet/', help='path to train dataset')

# Val data
parser.add_argument('--val_data', metavar='DIR', default='/dev/shm/ImageNet/', help='path to val dataset')
#parser.add_argument('--val_data', metavar='DIR', default='../../../../dev/shm/', help='path to val dataset')

parser.add_argument('--save_dir', type=str, default='./logs', help='Folder to save checkpoints and log.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
# Acceleration
parser.add_argument('--ngpu', type=int, default=2, help='0 = CPU.')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# batch size && lr changed for multi-gpu
parser.add_argument('-b', '--batch-size', default=128*3, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--cos', '--cosine_annealing', default=0, type=int, help='cosine annealing')

parser.add_argument('--mask_0_decay', type=float, default=1, help='The Decay Rate of the Mask0(Pruned Weight).')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=2600*2, type=int, metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')

parser.add_argument('--lambda_hard', type=float, default=0.5, help='hard pruning ratio, 1 - lambda_hard == soft pruning ratio.')

parser.add_argument('--exp_or_poly', type=str, default='poly', help='Exp. or polynomial(x**3) increase of pruing rate')

# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')

# compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
parser.add_argument('--layer_begin', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=3, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--skip_downsample', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_sparse', dest='use_sparse', action='store_true', help='use sparse model as initial or not')
parser.add_argument('--sparse',
                    default='/data/yahe/imagenet/resnet50-rate-0.7/checkpoint.resnet50.2018-01-07-9744.pth.tar',
                    type=str, metavar='PATH', help='path of sparse model')
parser.add_argument('--lr_adjust', type=int, default=30, help='number of epochs that change learning rate')


args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

args.prefix = time_file_str()
#cudnn.benchmark = True

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main(mask_0_decay, manualSeed, manual_lambda_hard, use_resume=False):
    best_prec1 = 0
    
    use_resume = False
    #use_resume = True # start from checkpoint

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    args.manualSeed = manualSeed  # set manualSeed
    args.mask_0_decay = mask_0_decay  # set mask 0 decay
    
    args.lambda_hard = manual_lambda_hard
    
    is_asfp = True
    sfp_type= "sfp"
    
    if is_asfp:
        sfp_type="asfp"

    get_hard_codebook = True
    # get_hard_codebook = False

    gd_or_softhard = 'gd'

    if get_hard_codebook:
        gd_or_softhard = 'soft_hard'
    
    # 在 使用 Gradient Decay 的情况下，soft_to_hard 模式 不再 表示 一部分软剪枝，一部分硬剪枝，
    # 而是同一时刻 所有结点 都是 部分 软剪枝
    
    soft_to_hard_type = 'soft_to_hard' 
    #soft_to_hard_type = 'soft_or_hard'
    
    #grad_decay_type = 'cos_anneal'
    grad_decay_type = 'poly' # gradient decay 用的 衰减类型
    


    args.exp_or_poly = 'exp'

    
    is_poly = True
    if args.exp_or_poly == "exp":
        is_poly = False
    
    # soft hard
    log = open(os.path.join(args.save_dir, 'log_seed_{}_{}_sigma_1e-5_soft_hard_mask_bn_{}_final_lambda_hard_{}_{}_{}_pretrain_lambda_hard_1.txt'.format(args.manualSeed, mask_0_decay, gd_or_softhard, args.lambda_hard, args.exp_or_poly , soft_to_hard_type)), 'w')


    print_log('save dir : {}'.format(args.save_dir), log)

    # version information
    print_log("PyThon  version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch version : {}".format(torch.__version__), log)
    print_log("cuDNN   version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Vision  version : {}".format(torchvision.__version__), log)
    
    # create model
    print_log("=> creating model '{}'".format(args.arch), log)
    print('Use Pretrain ', args.use_pretrain)
    
    model = models.__dict__[args.arch](pretrained=args.use_pretrain)# Not use pretrained model
    
    if args.use_sparse:
        print_log("use sparse == True", log)
        print('use sparse == True')
        model = import_sparse(model)
    
    #print_log("=> Model : {}".format(model), log)
    print_log("=> parameter : {}".format(args), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Mask_0_decay: {}".format(args.mask_0_decay), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("Skip downsample : {}".format(args.skip_downsample), log)
    print_log("Workers         : {}".format(args.workers), log)
    print_log("Learning-Rate   : {}".format(args.lr), log)
    print_log("Use Pre-Trained : {}".format(args.use_pretrain), log)
    print_log("lr adjust : {}".format(args.lr_adjust), log)
    
    print_log("lambda_hard : {}".format(args.lambda_hard), log)
    print_log("exp_or_poly : {}".format(args.exp_or_poly), log)
    #print_log("use regularize: {}".format(args.regularize), log) 
    print_log("cos : {}".format(args.cos), log)
    print_log("grad_decay_type : {}".format(grad_decay_type), log)
    

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # optionally resume from a checkpoint
    # args.resume = r'logs/resnet50-rate-0.6/checkpoint.resnet50.2021-06-19-5902.pth.tar'
    args.resume = r'logs/resnet50-rate-0.6/checkpoint.resnet50.2021-09-25-4028.pth.tar' # cosine_anneal_lr_180_epoch
    print('use_resume ', use_resume)

    if args.resume and use_resume==True:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] # modify the args, effects all experiments starting not from 1
            #args.start_epoch = 0
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        args.start_epoch = 0

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.train_data, 'train')
    valdir = os.path.join(args.val_data, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, log)
        return

    filename = os.path.join(args.save_dir, 'checkpoint.{:}.{:}.pth.tar'.format(args.arch, args.prefix))
    bestname = os.path.join(args.save_dir, 'best.{:}.{:}.pth.tar'.format(args.arch, args.prefix))

    lambda_hard = args.lambda_hard
    
    warm_up = 0
    epoch_iter = 1

    # 最大 epoch 设置 为 0.9 * args.epochs
    cos_annel_func = lambda epoch_t: (epoch_t / warm_up) if epoch_t < warm_up else 0.5 * (math.cos((epoch_t - warm_up)/(0.9*args.epochs*epoch_iter - warm_up) * math.pi) + 1) * (epoch_t < 0.9 * args.epochs)

    
    if soft_to_hard_type == 'soft_to_hard':
        lambda_hard = (1. - 0.) * ( 1. -  (1. - (args.start_epoch + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
        if grad_decay_type.find('cos') >=0:
            lambda_hard = 1 - cos_annel_func(args.start_epoch)
    else:
        assert grad_decay_type == '' # 如果是 soft_or_hard, 则 不使用 gard_decay, 等价于使用 poly
    
    if args.start_epoch >= (args.epochs) * 9/10 or  args.start_epoch >= (args.epochs) * 8/10 and args.use_pretrain: # 对于 预训练模型，初始学习率比较低，需要 提前终止 梯度衰减
        lambda_hard = args.lambda_hard

    m = Mask(model, lambda_hard)

    m.init_length()

    mask_0_decay = args.mask_0_decay
    
    sigma = 1e-3  # smoother mask_0_decay supported by PGMPF
    #sigma = 1e-7  # control the accuracy of mask 0 decay for models  trained from scratch
    
    if args.use_pretrain:
        sigma *= 0.01
        #sigma = 1e-9 # for pretrained model

    print_log("sigma : {}".format(sigma), log)

    # use 0.01 * sigma: to prevent divide 0
    alpha_decay = -np.log(sigma / (args.mask_0_decay + 0.01 * sigma)) / args.epochs

    #coeffi = np.pi / (2 * args.epochs)  # cos() decrease
    # comp_rate = args.rate + mask_0_decay * (1-args.rate)
    D = 1 / 8
    asymptotic_k = np.log(4) / (D * args.epochs)
    

    print('start_epoch ',args.start_epoch)
    
    comp_rate = args.rate + (1 - args.rate) * np.exp(-asymptotic_k*args.start_epoch)
    
    if is_poly:
        comp_rate =1. - (1-args.rate) * ( 1. -  (1. - (args.start_epoch + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
    
    if is_asfp == False:
        comp_rate = args.rate
        
    
    
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("the compression rate now is {:}".format(comp_rate)) # modified

    #val_acc_1 = validate(val_loader, model, criterion, log)

    #print(">>>>> accu before is: {:}".format(val_acc_1))

    m.model = model

    #m.init_mask(comp_rate) # modified
    m.init_mask(comp_rate, get_hard_codebook) # 暂时，beta 是 一个常量，可以设计 成 随着 epoch增加 而 逐渐 增大 的 数字
    
    
    # m.if_zero()

    #exit(0)
    
    mask_0_decay = args.mask_0_decay * np.exp(-alpha_decay * args.start_epoch)
    
    if args.start_epoch > (args.epochs) * 4 / 5:
        mask_0_decay = 0
    
    
    m.do_mask(mask_0_decay)
    model = m.model
    #m.if_zero()
    
    #exit(0)

    if args.use_cuda:
        model = model.cuda()
    val_acc_2 = validate(val_loader, model, criterion, log)
    print(">>>>> accu after is: {:}".format(val_acc_2))

    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        
        #if epoch < 90:
        current_lr = adjust_learning_rate(optimizer, epoch)
        #current_lr = adjust_learning_rate_cosine(optimizer, epoch, args)
    
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.val * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        
        
        print_log(
            ' [{:s}] :: {:3d}/{:3d} ----- cur_lr={:.5f}, [{:s}] {:s}'.format(args.arch, epoch, args.epochs, current_lr, time_string(), need_time),
            log)
        
        #val_acc_0 = validate(val_loader, model, criterion, log)
        # train for one epoch
        
        train(train_loader, model, m, criterion, optimizer, epoch, log, get_hard_codebook)
        
        
        # evaluate on validation set
        val_acc_1 = validate(val_loader, model, criterion, log)
        
        
        if (epoch % args.epoch_prune == 0 or epoch == args.epochs - 1):
            #        if (random.randint(1,args.epoch_prune)==1 or epoch == args.epochs-1):
            m.model = model
            
            if soft_to_hard_type == 'soft_to_hard':
                lambda_hard = (1. - 0.) * ( 1. -  (1. - (epoch + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
                if grad_decay_type.find('cos') >=0:
                    lambda_hard = 1 - cos_annel_func(epoch)

            if epoch >= (args.epochs) * 9/10 or  epoch >= (args.epochs) * 8/10 and args.use_pretrain: # 对于 预训练模型，初始学习率比较低，需要 提前终止 梯度衰减
                lambda_hard = args.lambda_hard

            m.lambda_hard = lambda_hard
            

            mask_0_decay = args.mask_0_decay * np.exp(-alpha_decay * epoch)
           
            
            if epoch >= (args.epochs) * 4 / 5:
                mask_0_decay = 0
            
            if is_asfp == True:
                comp_rate = args.rate + (1 - args.rate) * np.exp(-asymptotic_k * epoch)
                if is_poly:
                    comp_rate =1. -(1- args.rate) * ( 1. -  (1. - (epoch + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
            
            if epoch >= (args.epochs) * 9/10:
                comp_rate = args.rate
                

            print_log('lambda_hard : {}, mask_0_decay : {}, comp rate : {}'.format(m.lambda_hard, mask_0_decay, comp_rate), log)

            #print("comp rate: %.3f" % comp_rate)
            m.init_mask(comp_rate, get_hard_codebook)
            
            m.do_mask(mask_0_decay)
            #m.if_zero()
            model = m.model
            if args.use_cuda:
                model = model.cuda()

        val_acc_2 = validate(val_loader, model, criterion, log)

        # remember best prec@1 and save checkpoint
        is_best = val_acc_2 > best_prec1
        best_prec1 = max(val_acc_2, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename, bestname)
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        # break

    log.close()


def import_sparse(model):
    checkpoint = torch.load(args.sparse)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print("sparse_model_loaded")
    return model


def train(train_loader, model, m, criterion, optimizer, epoch, log, get_hard_codebook):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        #if i>= 3:
            #break
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        # 如果有硬剪枝的话，此处会 屏蔽 被 硬剪枝结点的梯度信息。
        # 软剪枝的结点也可能考虑 对 梯度信息做 自适应的抑制
        
        # 在 网络depth较大时，引入的 梯度噪声 过多，可能 不利于模型优化，类似dropout 丢弃 某些层的 衰减

        for index, p in enumerate(m.model.parameters()):

            if (index in m.mask_index) and abs(m.compress_rate[index] - 1) > 1e-7:
                
                if get_hard_codebook:
                           
                    tensor_mask_hard = m.mat_hard[index].view_as( p.data ).cpu().numpy()

                    #print("tensor_mask.size == ", tensor_mask.shape)
                    
                    grad_tensor = p.grad.data.cpu().numpy()
                    
                    #grad_tensor = np.where(tensor_mask == 0, 0, grad_tensor)
                    
                    grad_tensor = np.where(tensor_mask_hard == 0, 0, grad_tensor)
                    
                    # 对梯度进行衰减， 衰减 系数为 1 - lambda_hard
                    # 需要保证 lambda_hard 的 最终值为 1，否则 就是 软剪枝了。
                    
                    p.grad.data = torch.from_numpy(grad_tensor)
                    if args.use_cuda:
                        p.grad.data = p.grad.data.cuda() 

                else:

                    rand_x = 0 # 默认 rand_x < 0.5 成立
                    rand_x = random.random() # Layer-Wise 
                    
                    rand_x = np.random.random(size=(p.data.shape[0],1,1,1)) # (outs, 1, 1, 1) # channel-wise

                    #if rand_x < 0.5:
                    #    continue # 梯度完全流通
                    
                    #a = p.data.view(m.model_length[index])
                    #if 'mask' in name:
                    #    continue
                    
                    #tensor = p.data.cpu().numpy()
                    #print("tensor == ", tensor.shape)
                    
                    #tensor_mask = m.mat_hard[index].view_as( p.data ).cpu().numpy()
                    # 使用 梯度衰减时，不使用 mat_hard
                    tensor_mask = m.mat[index].view_as( p.data ).cpu().numpy()
                    
                   
                    #print("tensor_mask.size == ", tensor_mask.shape)
                    
                    grad_tensor = p.grad.data.cpu().numpy()
                    
                    #grad_tensor = np.where(tensor_mask == 0, 0, grad_tensor)
                  
                    # 对梯度进行衰减， 衰减 系数为 1 - lambda_hard
                    # 需要保证 lambda_hard 的 最终值为 1，否则 就是 软剪枝了。

                    # rand_x < 0.5时，部分流通 被衰减的梯度，
                    # rand_x >= 0.5时，完全阻止 被衰减的梯度。 hard pruning biased gradient drop
                    
                    grad_tensor = grad_tensor * ( tensor_mask + (1-tensor_mask) * (1-m.lambda_hard) *(rand_x < 0.5))

                    # rand_x < 0.5, 完全流通 梯度
                    # else, 部分 流通 梯度
                    #grad_tensor = grad_tensor * ( tensor_mask + (1-tensor_mask) * (1-m.lambda_hard) )
                    
                    p.grad.data = torch.from_numpy(grad_tensor)
                    if args.use_cuda:
                        p.grad.data = p.grad.data.cuda() 
        
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5), log)
    print_log(' **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
		

def validate(val_loader, model, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #if i >= 5:
        #    break
        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        #target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        '''
        if i % args.print_freq == 0:
            print_log('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5), log)
       '''
    print_log(' **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                           error1=100 - top1.avg), log)

    return top1.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def print_log(print_string, log):
    print("{:}".format(print_string))
    log.write('{:}\n'.format(print_string))
    log.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_adjust))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def adjust_learning_rate_cosine(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


class Mask:
    def __init__(self, model, lambda_hard):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
            
        self.filter_codebook = {}
        self.mat_hard = {} # identify hard pruned nodes
        
       
        self.lambda_hard=lambda_hard
        

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        
        #weight_np[abs(weight_np) >= threshold ] = 1
       
        weight_np[weight_np != 1] = 0

        #print("codebook done")
        return weight_np

    # 梯度 衰减的情况下，不再区分 软剪枝 和 硬 剪枝的 mask， hard_codebook 为 全 1
    def get_filter_codebook_gradient_decay(self, weight_torch,compress_rate,length, index, get_hard_codebook=False):
        # momentum pruning 需要传入 index
        
            
        codebook = np.ones(length)
        hard_codebook = np.ones(length) # hard_codebook 为 全 1

        #get_hard_codebook = False
        #get_hard_codebook = True 

        filter_codebook = np.ones(weight_torch.size(0))  # filter-level codebook
        
        if len( weight_torch.size())==4:
            
            filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))
            filter_hard_pruned_num = int(weight_torch.size()[0]*(1-compress_rate)*self.lambda_hard)
            weight_vec = weight_torch.view(weight_torch.size()[0],-1)
            
            # weight_vec (out_channels, -1)
            # 2-norm
            # dim=1 对 第二维进行reduced
            
            norm2 = torch.norm(weight_vec,2,1) # 当前 通道 L2-norm 重要性
            
            norm2_np = norm2.cpu().numpy() # 只考虑 当前 的 通道 L2-Norm重要性
            #norm2_np = None
              
            # 筛选出 不重要的 结点的 下标 
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            
            filter_index_hard = norm2_np.argsort()[:filter_hard_pruned_num]
            
#            norm1_sort = np.sort(norm1_np)
#            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            
            kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]
            
            # codebook 和 hard_codebook 表示 生成 剪枝策略
            
            for x in range(0,len(filter_index)):
                filter_codebook[filter_index[x]] = 0
                codebook [filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0
                #hard_codebook [filter_index_hard[x] *kernel_length : (filter_index_hard[x]+1) *kernel_length] = 0
            
            if get_hard_codebook:

                for x in range(0,len(filter_index_hard)):
                    hard_codebook [filter_index_hard[x] *kernel_length : (filter_index_hard[x]+1) *kernel_length] = 0

            #print("filter codebook done")
        else:
            pass
        
        
        
        return codebook, hard_codebook, filter_codebook
    
    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            
            # 当前 权值向量 W(out_channels, in_channels, kernelsize, kernelsize) 
            # 需要 被剪掉的 filter 的数量
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            
            # weight_vec (out_channels, -1)
            # 2-norm 
            # dim=1 对 第二维进行reduced
            norm2 = torch.norm(weight_vec, 2, 1)
            
            norm2_np = norm2.cpu().numpy()
            
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            
            # 权值向量 W (out_channels, in_channels, kernel_size, kernel_size)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

            #print("filter codebook done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                # 第一个维度 out_channels
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                    
                else: # 剩下的维度
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        
        if 'vgg' in args.arch:
            cfg_5x = [24, 22, 41, 51, 108, 89, 111, 184, 276, 228, 512, 512, 512]
            cfg_official = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            # cfg = [32, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256]
            cfg_index = 0
            pre_cfg = True
            
            for index, item in enumerate(self.model.named_parameters()):
                self.compress_rate[index] = 1
                
                if len(item[1].size()) == 4:
                    print(item[1].size())
                    if not pre_cfg:
                        self.compress_rate[index] = layer_rate
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index)
                    else:
                        self.compress_rate[index] =  1 - cfg_5x[cfg_index] / item[1].size()[0]
                        self.mask_index.append(index)
                        print(item[0], "self.mask_index", self.mask_index, cfg_index, cfg_5x[cfg_index], item[1].size()[0],
                               )
                        cfg_index += 1
                        
        elif "resnet" in args.arch:
            
            for index, item in enumerate(self.model.parameters()):
                self.compress_rate[index] = 1
                
            for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
                self.compress_rate[key] = layer_rate
                
            if args.arch == 'resnet18':
                # last index include last fc layer
                last_index = 60
                skip_list = [21, 36, 51]
                
            elif args.arch == 'resnet34':
                last_index = 108
                skip_list = [27, 54, 93]
                
            elif args.arch == 'resnet50':
                last_index = 159
                skip_list = [12, 42, 81, 138]
                
            elif args.arch == 'resnet101':
                last_index = 312
                skip_list = [12, 42, 81, 291]
                
            elif args.arch == 'resnet152':
                last_index = 465
                skip_list = [12, 42, 117, 444]
                
            self.mask_index = [x for x in range(0, last_index, 3)]
            
            # skip downsample layer
            if args.skip_downsample == 1:
                
                for x in skip_list:
                    self.compress_rate[x] = 1
                    self.mask_index.remove(x)
                    #print(self.mask_index)
            else:
                pass

    def init_mask(self, layer_rate, get_hard_codebook):

        self.init_rate(layer_rate)
        #for index, item in enumerate(self.model.parameters()):
        for index, (name, item) in enumerate(self.model.named_parameters()):
            
            # 需要被 mask： 剪枝的 层
            if (index in self.mask_index):

                #print('name = ', name, item.data.size())
                #self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],
                #                                           self.model_length[index])
                
                # 生成了 当前的剪枝决策策略, 梯度衰减时，mat_hard 为 全 1
                self.mat[index], self.mat_hard[index], self.filter_codebook[name] =  \
                       self.get_filter_codebook_gradient_decay(item.data, 
                                                self.compress_rate[index],
                                                self.model_length[index], index, get_hard_codebook) # 动量剪枝 需要 传入 index
                       
                self.mat[index] = self.convert2tensor(self.mat[index])
                self.mat_hard[index] = self.convert2tensor(self.mat_hard[index])
                self.filter_codebook[name] = self.convert2tensor(self.filter_codebook[name])
                
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
                    self.mat_hard[index] = self.mat_hard[index].cuda()
                    self.filter_codebook[name] = self.filter_codebook[name].cuda()
            
            else:
                #print('name = ', name, item.data.size())
                pass
                    
        #print("mask Ready")

    def do_mask(self, mask_0_decay):
        pre_filter_codebook = None
        # for index, item in enumerate(self.model.parameters()):
        for index, (name, item) in enumerate(self.model.named_parameters()):

            if (index in self.mask_index):
                #print('name = ', name)
                a = item.data.view(self.model_length[index])
                
                #b = a * self.mat[index]*(1 -mask_0_decay) + a*mask_0_decay
                if abs(mask_0_decay) < 1e-4:  
                    b = a * self.mat[index]
                else:
                    b = a * ( self.mat[index] +  mask_0_decay*( 1 - self.mat[index] )* self.mat_hard[index]  )
                # 只有 mask_0_decay 等于 0 的 时候，才算是 真正的 剪枝。
                
                item.data = b.view(self.model_size[index])
            
            else:
                if 'bn' in name and 'running' not in name:  # only decay the weight, bias of BN layer
                    conv_name = name.replace('bn', 'conv')
                    
                    # the name of the first conv 
                    # NOTE: change according to the network, 
                    # only for resnet-56 for cifar-10/100
                    #if 'conv_1' in conv_name and 'stage' not in conv_name and 'layer' not in conv_name:
                    #    conv_name = conv_name.replace('conv_1', 'conv_1_3x3')
                    
                    if conv_name not in self.filter_codebook:
                        #print(name, ' conv_name ', conv_name, ' not in self.filter_codebook')
                    
                        
                        if 'bias'  in conv_name:
                            cur_filter_codebook = pre_filter_codebook  # use pre_filter_codebook of bn_x.weight
                        else:
                            continue
                    else:
                        cur_filter_codebook = self.filter_codebook[conv_name]

                    a = item.data 
                    # decay the bn parameters
                    b = a * ( cur_filter_codebook  + mask_0_decay * (1 - cur_filter_codebook))

                    #item.data = a
                    item.data = b

                    pre_filter_codebook = cur_filter_codebook
                    
                    if 'bias' in name:
                        pre_filter_codebook = None # reset to None
                
           
        #print("mask Done")

    def if_zero(self):
        
    
        for index, item in enumerate(self.model.parameters()):
            #if(index in self.mask_index):
            if index in [x for x in range(args.layer_begin, args.layer_end + 1, args.layer_inter)]:
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("layer: %d, number of nonzero weight is %d, zero is %d" % (
                    index, np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    
    mask_0_decay_list = [0.0]
    mask_0_decay_list = [1.0]
    
    #lambda_hard_list = [0.0] 
    
    lambda_hard_list = [1.0] # 使用 momentum importance 剪枝时，目前只关注 软剪枝，不关注 硬剪枝。
    
    used_seed = {}
    
    used_seed = {}
    for t in range(1):# 实验次数
  
        
        for lambda_hard in lambda_hard_list:
       
            manualSeed = random.randint(1, 10000)

            while used_seed.__contains__(manualSeed):
                manualSeed = random.randint(1, 10000)
            used_seed[manualSeed] = 1 # 标记为使用过的种
            
            
            random.seed(manualSeed)
            torch.manual_seed(manualSeed)
            if args.use_cuda:
                torch.cuda.manual_seed_all(manualSeed)
            
            for i in range(len(mask_0_decay_list)):
                print("i == ", i, " mask_0_decay[i] == ", mask_0_decay_list[i])
    
   
                main(mask_0_decay_list[i], manualSeed, i==0)# only resume one time
                #main(mask_0_decay_list[i], manualSeed, False)
                # break

