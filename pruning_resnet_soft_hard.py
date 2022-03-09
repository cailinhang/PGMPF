from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
import numpy as np
import torch.nn as nn
#import losses_torch
from regularization import mixup_data, mixup_criterion, Cutout, label_smoothing, LabelSmoothingCrossEntropy
import models.layers as layers

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

layer_end_dict = {'resnet20': 54, 'resnet32':90, 'resnet56':162, 'resnet110':324}

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# data_path: default cifar-10
#parser.add_argument('--data_path', type=str, default='../../zh/data', help='Path to dataset')
parser.add_argument('--data_path', type=str, default='../../dataset/', help='Path to dataset')
#parser.add_argument('data_path', type=str, default='../../ssh/cnn-gen-0.2/data/', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet20', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--mask_0_decay', type=float, default=0.5, help='The Decay Rate of the Mask0(Pruned Weight).')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
# Regularization
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')

parser.add_argument('--lambda_hard', type=float, default=0.5, help='hard pruning ratio, 1 - lambda_hard == soft pruning ratio.')

parser.add_argument('--regularize', type=str, default='', choices=['', 'mixup', 'cutout', 'smoothcutout', 'cutout_alpha'], help='use mixup or cutout or smoothcutout.')

parser.add_argument('--n_holes_cutout', type = int, default = 1, help = 'Number of holes to cut out from image.')
parser.add_argument('--length_cutout', type = int, default = 16, help = 'Length of the holes in cutout.')# cifar10 cutout 12, cifar100 cutout 6

parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')

parser.add_argument('--exp_or_poly', type=str, default='poly', help='Exp. or polynomial(x**3) increase of pruing rate')
# Checkpoints
parser.add_argument('--print_freq', default=400, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--use_pretrain', default=False, type=int, help='use pre-trained model or not')


# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=3, help='number of data loading workers (default: 2)')

# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
#compress rate
parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')

parser.add_argument('--layer_begin', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1,  help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1,  help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1,  help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')


args = parser.parse_args()
print(" args.ngpu ", args.ngpu, "  torch.cuda.is_available() ",  torch.cuda.is_available())
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
print(" use_cuda ",args.use_cuda)

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main(mask_0_decay, manualSeed, manual_lambda_hard):
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    
    args.manualSeed = manualSeed # set manualSeed
   
    args.lambda_hard = manual_lambda_hard
    #is_asfp = False
    args.layer_end = layer_end_dict[ args.arch ]
    #args.use_pretrain = True
    
    args.regularize = 'cutout'
    
    if args.dataset == 'cifar10':
        args.length_cutout = 12
        
    elif args.dataset == 'cifar100':
        args.length_cutout = 6

    is_asfp = True
    sfp_type= "sfp"
    if is_asfp:
        sfp_type="asfp"
    
    soft_to_hard_type = 'soft_to_hard'
    #soft_to_hard_type = 'soft_or_hard'
    #args.exp_or_poly = 'exp'

    is_poly = True
    if args.exp_or_poly == "exp":
        is_poly = False
        
    log = open(os.path.join(args.save_path, 'log_seed_{}_{}_lambda_hard_{}_{}_{}_{}.txt'.format(args.manualSeed, mask_0_decay, args.lambda_hard, args.exp_or_poly , soft_to_hard_type, args.regularize)), 'w')
    print_log('save path : {}'.format(args.save_path), log)

    args.mask_0_decay = mask_0_decay # set mask 0 decay
    #args.manualSeed = manualSeed # set manualSeed

    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Mask 0 Decay: {}".format(args.mask_0_decay), log) # mask 0 decay
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("Use Pre-Trained : {}".format(args.use_pretrain), log)
    print_log("lambda_hard : {}".format(args.lambda_hard), log)
    print_log("exp_or_poly : {}".format(args.exp_or_poly), log)
    print_log("use regularize: {}".format(args.regularize), log) 
    
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        #mean = [0.4914, 0.4822, 0.4465]
        #std = [ 0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        #mean = [0.4914, 0.4822, 0.4465]
        #std = [ 0.2023, 0.1994, 0.2010]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)


    
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    
    if args.regularize == 'cutout':
        train_transform.transforms.append(Cutout(n_holes = args.n_holes_cutout,
                                                     length = args.length_cutout))

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif args.dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    
    num_classes = 10 if args.dataset == 'cifar10' else 100

    #net = models.__dict__[args.arch](num_classes=num_classes, use_mask=False)
    
    net = models.__dict__[args.arch](num_classes=num_classes, use_mask=True, freeze_weight=False)
    
    #net = models.__dict__[args.arch](num_classes)
    #print_log("=> network :\n {}".format(net), log)
    #"""
    if args.use_pretrain:# use pretrain model
        print_log("=> loading checkpoint '{}'".format(r"logs/cifar10_resnet56/cifar10_resnet56_rate1.0/model_best.pth.tar" ), log)
        checkpoint = torch.load(r"logs/cifar10_resnet56_rate1.0/model_best.pth.tar" )# resnet20
        #checkpoint = torch.load(r"pretrain/cifar10-resnet56-f5939a66.pth" )# resnet56
        #net.load_state_dict(checkpoint['state_dict'] ) # load from pretrain/
        net = checkpoint['state_dict'] # load from log/
        print_log("=> loaded checkpoint ", log)
    #"""
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))



    # define loss function (criterion) and optimizer
    #if use_label_smooth:
        #criterion = LabelSmoothingCrossEntropy()
    #else:
    criterion = torch.nn.CrossEntropyLoss()
        
    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)

    """
    if args.use_pretrain:# use pretrain model
        print_log("=> loading checkpoint '{}'".format(r"logs/cifar10_resnet56_rate1.0/asfp/model_best_baseline_not_pruned.pth.tar" ), log)
        checkpoint = torch.load(r"logs/cifar10_resnet56_rate1.0/asfp/checkpoint.pth.tar" )# resnet56 acc 93%
        checkpoint = torch.load(r"logs/cifar10_resnet56_rate1.0/asfp/model_best_baseline_not_pruned.pth.tar" )# resnet56 acc 81%
        #net.load_state_dict(checkpoint['state_dict'] )
        net = checkpoint['state_dict']
        print_log("=> loaded checkpoint ", log)
   """ 
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print ('function took %0.3f ms' % ((time2-time1)*1000.0))
        return

    lambda_hard = args.lambda_hard
    if soft_to_hard_type == 'soft_to_hard':
        lambda_hard = (1. - 0.) * ( 1. -  (1. - (0 + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
    m=Mask(net, lambda_hard)
        
    
    mask_0_decay = args.mask_0_decay

    sigma = 1e-5  # control the accuracy of mask 0 decay
    # use 0.01 * sigma: to prevent divide 0
    alpha_decay = -np.log(sigma / (args.mask_0_decay + 0.01 * sigma)) / args.epochs

    #coeffi = np.pi/(2*args.epochs) # cos() decrease
    #comp_rate = args.rate + mask_0_decay * (1-args.rate)
    #D = args.D
    D = 1/8
    asymptotic_k = np.log(4)/( D * args.epochs)
    #comp_rate = 1 - (1 - args.rate) * ( 1 - np.exp(-asymptotic_k*0))
    comp_rate = args.rate + (1 - args.rate) * np.exp(-asymptotic_k*0)
    if is_poly:
        comp_rate =1. - (1-args.rate) * ( 1. -  (1. - (0 + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
    if is_asfp == False:
        comp_rate = args.rate
    
    print("-"*10+"one epoch begin"+"-"*10)
    print("the compression rate now is %f" % comp_rate)

    val_acc_1,   val_los_1   = validate(test_loader, net, criterion, log)

    print(" accu before is: %.3f %%" % val_acc_1)
    
    #return 

    m.model = net
    
    m.init_mask(comp_rate)
    
    mask_0_decay = args.mask_0_decay * np.exp(-alpha_decay * 0)
    #mask_0_decay += args.mask_0_decay * np.cos(coeffi*0)
    #mask_0_decay /=2
    #mask_0_decay = args.mask_0_decay
    #slope = -args.mask_0_decay / (args.epochs - 1) #  斜率
    # alpha0 + slope*t
    m.do_mask(mask_0_decay)
    net = m.model
    
    if args.use_cuda:
        net = net.cuda()    
    val_acc_2,   val_los_2   = validate(test_loader, net, criterion, log)
    print(" accu after is: %s %%" % val_acc_2)
    

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(train_loader, net, m, criterion, optimizer, epoch, log)

        # evaluate on validation set
        val_acc_1,   val_los_1   = validate(test_loader, net, criterion, log)
        #print('epoch ', epoch, 'epoch_prune  ',args.epoch_prune)
        if (epoch % args.epoch_prune ==0 or epoch == args.epochs-1):
            m.model = net
            #m.init_mask(1.0)
            if soft_to_hard_type == 'soft_to_hard':
                lambda_hard = (1. - 0.) * ( 1. -  (1. - (epoch + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
            m.lambda_hard = lambda_hard
            print("lambda_hard : %.3f" % m.lambda_hard)
            mask_0_decay = args.mask_0_decay * np.exp(-alpha_decay * epoch)
            #mask_0_decay = args.mask_0_decay + slope*epoch # linear decay
            
            if epoch >= (args.epochs) * 9/10 :
                mask_0_decay = 0
            # when mask_0_decay is set to zero, comp_rate is args.rate
            #comp_rate =  args.rate + mask_0_decay * (1-args.rate) # compress_rate from 1  to args.rate

            if is_asfp == True:
                comp_rate = args.rate + (1 - args.rate) * np.exp(-asymptotic_k * epoch)
                if is_poly:
                    comp_rate =1. -(1- args.rate) * ( 1. -  (1. - (epoch + 1.)/(args.epochs+1e-7) ) ** 3  ) # x^3 递增
            if epoch >= (args.epochs) * 19/20:
                comp_rate = args.rate
            #if epoch < args.epochs -1: # only weight decay
                #comp_rate = 1.0

            print("compress rate: %.3f" % comp_rate)
            m.init_mask(comp_rate)
           
            #m.do_mask(0.0)


            #m.init_mask(comp_rate)
            #mask_0_decay += args.mask_0_decay* np.cos(coeffi * epoch)
            #mask_0_decay /=2
            #mask_0_decay = args.mask_0_decay + slope*epoch # linear decay

                
            m.do_mask(mask_0_decay)
           
            net = m.model
            if args.use_cuda:
                net = net.cuda()  
            
        val_acc_2,   val_los_2   = validate(test_loader, net, criterion, log)
    
        
        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

    log.close()

# train function (forward, backward, update)
def train(train_loader, model, m, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            #target = target.cuda( async=True)
            #input = input.cuda()
            input, target = input.cuda(), target.cuda()
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
        
        """
        for index, p in enumerate(m.model.parameters()):
            if(index in m.mask_index):
                #a = p.data.view(m.model_length[index])
                #if 'mask' in name:
                #    continue
                
                #tensor = p.data.cpu().numpy()
                #print("tensor == ", tensor.shape)
                tensor_mask = m.mat_hard[index].view_as( p.data ).cpu().numpy()
                #print("tensor_mask.size == ", tensor_mask.shape)
                
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor_mask == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor)
                if args.use_cuda:
                   p.grad.data = p.grad.data.cuda() 
              
        """

        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            #target = target.cuda(async=True)
            #input = input.cuda()
            input, target = input.cuda(), target.cuda()
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

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
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
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item() )
    return res


class Mask:
    def __init__(self,model,lambda_hard):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        #self.mask_0_decay = {}
        self.mat = {}
        self.mat_hard = {} # identify hard pruned nodes
        self.mat_hard_bn = {} # hard pruned nodes bn
        self.model = model
        self.mask_index = []
        self.lambda_hard=lambda_hard
        #self.alpha_decay = 0
    
    def get_codebook(self, weight_torch,compress_rate,length):

        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()
    
        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        # Pruning threshold
        threshold = weight_sort[int (length * (1-compress_rate) )]

        weight_np [weight_np <= -threshold  ] = 1
        weight_np [weight_np >= threshold  ] = 1
        weight_np [weight_np !=1  ] = 0
        
        #print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch,compress_rate,length):
        codebook = np.ones(length)
        hard_codebook = np.ones(length)
        if len( weight_torch.size())==4:
            filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))
            filter_hard_pruned_num = int(weight_torch.size()[0]*(1-compress_rate)*self.lambda_hard)
            weight_vec = weight_torch.view(weight_torch.size()[0],-1)
            # weight_vec (out_channels, -1)
            # 2-norm
            # dim=1 对 第二维进行reduced
            norm2 = torch.norm(weight_vec,2,1)
            norm2_np = norm2.cpu().numpy()
            filter_index = norm2_np.argsort()[:filter_pruned_num]
            filter_index_hard = norm2_np.argsort()[:filter_hard_pruned_num]
#            norm1_sort = np.sort(norm1_np)
#            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]
            for x in range(0,len(filter_index)):
                codebook [filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0
                #hard_codebook [filter_index_hard[x] *kernel_length : (filter_index_hard[x]+1) *kernel_length] = 0
            for x in range(0,len(filter_index_hard)):
                hard_codebook [filter_index_hard[x] *kernel_length : (filter_index_hard[x]+1) *kernel_length] = 0

            #print("filter codebook done")
        else:
            pass
        return codebook, hard_codebook
    
    def get_filter_codebook_v1(self, m, compress_rate):
        
        alpha_vert = m.alpha_vert.data
        alpha_hori = m.alpha_hori.data
        
        weight = m.weight.data
        
        outs, ins, k, k = weight.size()
        
        beta = 1
        new_ins = m.dimension_reduce_in.shape[0]
        new_outs = m.dimension_reduce.shape[0]

        help_w_1 = torch.matmul(m.dimension_reduce, m.help_w[:2*outs].contiguous().view(2 * outs, -1))
        help_w_1 = beta * help_w_1.view(new_outs, ins, k, k)  + m.help_w[:outs]
        help_w_1 = help_w_1.data

        help_w_2 = torch.matmul(m.dimension_reduce_in, m.help_w_in[:, :2*ins].contiguous().view( 2 * ins, -1))
        help_w_2 = beta * help_w_2.view(outs, new_ins, k, k) + m.help_w_in[:, :new_ins]
        help_w_2 = help_w_2.data
        
        
        #help_w_1 = m.help_w_1.data
        #help_w_2 = m.help_w_2.data

        #outs, ins, k, k = weight.size()

        bn = m.bn.weight.data.reshape(outs, 1,1,1)
        bn_1 = m.bn_1.weight.data.reshape(outs, 1,1,1)
        bn_2 = m.bn_2.weight.data.reshape(outs, 1,1,1)
        
        #merged_w = weight + alpha_vert * help_w_1 + alpha_hori * help_w_2
        #print(bn.size(), ' ', weight.size())
        #print(bn * weight)
        merged_w = (bn * weight) + alpha_vert * (bn_1 * help_w_1) + alpha_hori * (bn_2 * help_w_2) # 将BN层的gamma系数也考虑进通道重要性
        merged_w *= m.mask.data # 前向传播的hard pruned mask  

        #outs, ins, k, k = weight.size()
        
        length = outs * ins * k * k
        
        codebook = np.ones(length)
        hard_codebook = np.ones(length)

        codebook_bn = np.ones(outs)
        
        
        assert len( merged_w.size())==4
            
        filter_pruned_num = int(merged_w.size()[0]*(1-compress_rate))
        filter_hard_pruned_num = int(merged_w.size()[0]*(1-compress_rate)*self.lambda_hard)
        
        weight_vec = merged_w.view(merged_w.size()[0],-1)
        
        # weight_vec (out_channels, -1)
        # 2-norm
        # dim=1 对 第二维进行reduced
        
        norm2 = torch.norm(weight_vec,2,1)
        norm2_np = norm2.cpu().numpy()
        filter_index = norm2_np.argsort()[:filter_pruned_num]
        filter_index_hard = norm2_np.argsort()[:filter_hard_pruned_num]
        
#            norm1_sort = np.sort(norm1_np)
#            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
        
        kernel_length = merged_w.size()[1] *merged_w.size()[2] *merged_w.size()[3]
        
        for x in range(0,len(filter_index)):
            codebook [filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0
           
        for x in range(0,len(filter_index_hard)):
            codebook_bn[filter_index_hard[x]] = 0 
            hard_codebook [filter_index_hard[x] *kernel_length : (filter_index_hard[x]+1) *kernel_length] = 0

        return codebook, hard_codebook, codebook_bn
    
    def convert2tensor(self,x):
        x = torch.FloatTensor(x)
        return x
    
                    
     
    def init_rate(self, comp_rate):
        
        idx = 0 # comp_rate idx
        
        for index, m in enumerate(self.model.modules()):
            
            if isinstance(m, nn.Conv2d) or isinstance(m, layers.MaskedConv2d):
                #print(index, str(type(m)))
                
                self.compress_rate[index] = comp_rate
                #self.compress_rate[index] = rate_per_layer_list[idx]
                idx += 1
                self.mask_index.append(index)   
                    
    
    def init_mask(self, rate):
        
        self.init_rate(rate)
        
        
        for index, m in enumerate(self.model.modules()):
                
            if isinstance(m, nn.Conv2d) or isinstance(m, layers.MaskedConv2d):
                
                self.mat[index], self.mat_hard[index], self.mat_hard_bn[index] = \
                    self.get_filter_codebook_v1(m, 
                                             self.compress_rate[index])
                    
                
                self.mat[index] = self.convert2tensor(self.mat[index])
                self.mat_hard[index] = self.convert2tensor(self.mat_hard[index])
                self.mat_hard_bn[index] = self.convert2tensor(self.mat_hard_bn[index])
                
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
                    self.mat_hard[index] = self.mat_hard[index].cuda()
                    self.mat_hard_bn[index] = self.mat_hard_bn[index].cuda()
        
    def do_mask(self, mask_0_decay):
        
        for index, m in enumerate(self.model.modules()):
                
            if isinstance(m, nn.Conv2d) or isinstance(m, layers.MaskedConv2d):
                #print(index, str(type(m)))
                
                outs, ins, k, k = m.weight.data.size()
                aimed_size = (outs, ins, k, k)
                length = outs * ins * k * k
                
                weight = m.weight.data.view(length)
                weight_b = weight * ( self.mat[index] +  mask_0_decay*( 1 - self.mat[index] )* self.mat_hard[index]  )
                m.weight.data = weight_b.view(aimed_size)
                
                
                #help_w_1 = m.help_w_1.data.view(length)
                #help_w_1_b = help_w_1 * ( self.mat[index] +  mask_0_decay*( 1 - self.mat[index] )* self.mat_hard[index]  )
                #m.help_w_1.data = help_w_1_b.view(aimed_size)
                
                #help_w_2 = m.help_w_2.data.view(length)
                #help_w_2_b = help_w_2 * ( self.mat[index] +  mask_0_decay*( 1 - self.mat[index] )* self.mat_hard[index]  )
                #m.help_w_2.data = help_w_2_b.view(aimed_size)

                m.mask.data = self.mat_hard[index].view(aimed_size) # hard mask
                m.mask_bn.data = self.mat_hard_bn[index].view(1, outs, 1, 1) # hard mask bn
    
if __name__ == '__main__':
    
    #mask_0_decay_list = [0.0]
    mask_0_decay_list = [0.0, 1.0]
   
    lambda_hard_list = [1.0]
    
    used_seed = {}
    for t in range(1):# 实验次数
  
        
        for lambda_hard in lambda_hard_list:
       
            manualSeed = random.randint(1, 10000)
            #manualSeed = 4669
            while used_seed.__contains__(manualSeed):
                manualSeed = random.randint(1, 10000)
            used_seed[manualSeed] = 1 # 标记为使用过的种
            
            
            random.seed(manualSeed)
            torch.manual_seed(manualSeed)
            if args.use_cuda:
                torch.cuda.manual_seed_all(manualSeed)
            
            for i in range(len(mask_0_decay_list)):
                print("i == ", i, " mask_0_decay[i] == ", mask_0_decay_list[i])
                main(mask_0_decay_list[i], manualSeed, lambda_hard)
                #break
