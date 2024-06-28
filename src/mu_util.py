# machine unlearning functions
import os
import time
import sys
import pdb
import json
import random
import copy
from itertools import cycle

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter, accuracy


def seedEverything(seed):
    random.seed(seed)             # Python random module
    np.random.seed(seed)          # Numpy module
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    # _, term_width = os.popen('stty size', 'r').read().split()
    # term_width = int(term_width)
    TOTAL_BAR_LENGTH = 65.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def run_epoch(epoch, net, data_loader, criterion, optimizer=None, args=None, mode='train', return_labels=False):
    """Run an epoch, either for training or testing
    
    Args:
        epoch (int): Current epoch number.
        net (torch.nn.Module): Neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        opt: Command line arguments.
        mode (str): 'train' for training, 'test' for testing.
        quiet (bool, optional): If True, suppress print statements. Default is False.
    
    Returns:
        AverageMeter: Metrics calculated during the epoch.
    """
    net.train() if mode == 'train' else net.eval()
    DEVICE = next(net.parameters()).device
    
    metrics = AverageMeter()

    labels = []
    for batch_idx, (input, target) in enumerate(data_loader):
        target = torch.from_numpy(np.asarray(target).astype('long'))
        input, target = input.to(DEVICE), target.to(DEVICE)

        # Forward pass
        output = net(input)
        loss = criterion(output, target)
        
        if mode == 'train':
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate metrics
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metrics.update(n=input.size(0), losses=loss.item(), top1=acc1.item(), top5=acc5.item())
        if not args.quiet:
            progress_bar(batch_idx, len(data_loader), 'Loss: %.3f | Acc: %.3f%% (N=%d)'
                     % (metrics.avg['losses'], metrics.avg['top1'], metrics.count['top1']))
        # if not quiet and batch_idx % opt.print_freq == 0:
        #     print(f'[{epoch}][{batch_idx}/{len(data_loader)}] {mode} metrics val:' + json.dumps(metrics.val))
        #     print(f'[{epoch}][{batch_idx}/{len(data_loader)}] {mode} metrics avg:' + json.dumps(metrics.avg))
        if return_labels:
            labels.extend(output.max(dim=1)[1].cpu().numpy().tolist())
    if return_labels:
        return metrics, labels
    return metrics


def finetune(net, retain_loader, forget_loader, val_loader, criterion, optimizer, scheduler, args, readouts=None, all_readouts=None):
        """Simple unlearning by finetuning.
        
        Args:
            net (torch.nn.Module): Neural network model.
            retain_loader (torch.utils.data.DataLoader): DataLoader for the retained data.
            forget_loader (torch.utils.data.DataLoader): DataLoader for the data to forget.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            args (argparse.Namespace): Command line arguments.
            readouts (dict, optional): Dictionary to store the results. Default is None.
            all_readouts (function, optional): Function to print statistics / evaluations for all methods. Default is None.
        """
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr_unlearn, momentum=0.9, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_unlearn)
        name_parts = [
                f"{args.unlearn_method}",
                # f"_forgetClass_{args.forget_class}" if args.forget_class else "",
                f"_retainRatio_{args.retain_ratio}" if args.retain_ratio != 1 else "",
                f"_lr_{str(args.lr_unlearn)}",
                "_schedule" if args.schedule_unlearn else "",
                "_augment" if args.augment_ft else "",
            ]
        args.name_unlearn = "".join(name_parts)
        for epoch in range(args.epochs_unlearn):
            metrics = run_epoch(epoch, net, retain_loader, criterion, optimizer, args=args, mode="train")
            if epoch < 10 or (epoch+1) % 10 == 0:
                torch.save(net.state_dict(), args.ckpt_root/f"{args.name_unlearn}_epoch_{epoch+1}.pt")
            if args.schedule_unlearn:
                scheduler.step()
            if all_readouts:
                name = f'Finetune_{epoch+1}'
                readouts[name] = all_readouts(net, name, args.seed)


def run_epoch_negrad(epoch, net, retain_loader, forget_loader, alpha, criterion, optimizer, args, mode='train'):
    """Run an epoch, either for training or testing
    
    Args:
        epoch (int): Current epoch number.
        net (torch.nn.Module): Neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        opt: Command line arguments.
        mode (str): 'train' for training, 'test' for testing.
        quiet (bool, optional): If True, suppress print statements. Default is False.
    
    Returns:
        AverageMeter: Metrics calculated during the epoch.
    """
    net.train() if mode == 'train' else net.eval()
    DEVICE = next(net.parameters()).device
    
    metrics = AverageMeter()

    for batch_idx, ((input, target), (input_forget, target_forget)) in enumerate(zip(retain_loader, cycle(forget_loader))):
        target = torch.from_numpy(np.asarray(target).astype('long'))
        target_forget = torch.from_numpy(np.asarray(target_forget).astype('long'))
        input, target = input.to(DEVICE), target.to(DEVICE)
        input_forget, target_forget = input_forget.to(DEVICE), target_forget.to(DEVICE)

        # Forward pass
        output = net(input)
        output_forget = net(input_forget)
        loss_retain = criterion(output, target)
        loss_forget = criterion(output_forget, target_forget)
        loss = alpha*loss_retain - (1-alpha)*loss_forget

        if mode == 'train':
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate metrics
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metrics.update(n=input.size(0), losses=loss.item(), top1=acc1.item(), top5=acc5.item())
        if not args.quiet:
            progress_bar(batch_idx, len(retain_loader), 'Loss: %.3f | Acc: %.3f%% (N=%d)'
                     % (metrics.avg['losses'], metrics.avg['top1'], metrics.count['top1']))
    return metrics


def negrad(net, retain_loader, forget_loader, val_loader, criterion, optimizer, scheduler, args, readouts=None, all_readouts=None):
        """Unlearning by NegGrad.
        
        Args:
            net (torch.nn.Module): Neural network model.
            retain_loader (torch.utils.data.DataLoader): DataLoader for the retained data.
            forget_loader (torch.utils.data.DataLoader): DataLoader for the data to forget.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            args (argparse.Namespace): Command line arguments.
            readouts (dict, optional): Dictionary to store the results. Default is None.
            all_readouts (function, optional): Function to print statistics / evaluations for all methods. Default is None.
        """
        name_parts = [
                f"{args.unlearn_method}",
                # f"_forgetClass_{args.forget_class}" if args.forget_class else "",
                f"_retainRatio_{args.retain_ratio}" if args.retain_ratio != 1 else "",
                f"_lr_{str(args.lr_unlearn)}",
                f"_alpha_{str(args.alpha)}",
                "_schedule" if args.schedule_unlearn else "",
                "_augment" if args.augment_ft else "",
            ]
        args.name_unlearn = "".join(name_parts)
        for epoch in range(args.epochs_unlearn):
            metrics = run_epoch_negrad(epoch, net, retain_loader, forget_loader, args.alpha, criterion, optimizer, args=args, mode="train")
            if epoch < 10 or (epoch+1) % 10 == 0:
                torch.save(net.state_dict(), args.ckpt_root/f"{args.name_unlearn}_epoch_{epoch+1}.pt")
            if args.schedule_unlearn:
                scheduler.step()
            if all_readouts:
                name = f'NegGrad_{epoch+1}'
                readouts[name] = all_readouts(net, name, args.seed)


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def sgda_adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.sgda_learning_rate
    if steps > 0:
        new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr


def param_dist(model, swa_model, p):
    # This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


def run_epoch_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    metrics = AverageMeter()
    
    for batch_idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target = data
        
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
                index = index.cuda()

        # ===================forward=====================
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        else:
            raise NotImplementedError(opt.distill)

        if split == "minimize":
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, opt.smoothing)
        # reg = param_dist(model_s, swa_model, opt.smoothing)
        # loss = loss + reg
        # print(f"""loss_cls: {loss_cls}, loss_div: {loss_div}, loss_kd: {loss_kd}, reg: {reg}""")

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

        # ===================meters=====================
        if not opt.quiet:
            if split == "minimize":
                acc1, _ = accuracy(logit_s, target, topk=(1,1))
                metrics.update(n=input.size(0), losses=loss.item(), top1=acc1.item())
                progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (N=%d)'
                        % (metrics.avg['losses'], metrics.avg['top1'], metrics.count['top1']))
            elif split == "maximize":
                # kd_losses.update(loss.item(), input.size(0))
                metrics.update(n=input.size(0), kd_losses=loss.item())
                progress_bar(batch_idx, len(train_loader), 'KD_loss: %.3f | Acc: %.3f%% (N=%d)'
                        % (metrics.avg['kd_losses'], metrics.avg['top1'], metrics.count['top1']))
        
    return metrics
    

def scrub(net, retain_loader, forget_loader, val_loader, args, readouts=None, all_readouts=None):
    """Unlearning by SCRUB.
        
    Args:
        net (torch.nn.Module): Neural network model.
        retain_loader (torch.utils.data.DataLoader): DataLoader for the retained data.
        forget_loader (torch.utils.data.DataLoader): DataLoader for the data to forget.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        args (argparse.Namespace): Command line arguments.
        readouts (dict, optional): Dictionary to store the results. Default is None.
        all_readouts (function, optional): Function to print statistics / evaluations for all methods. Default is None.
    """
    # use the same setting as in the original paper
    args.msteps = 5
    args.sstart = 10
    args.lr_decay_epochs = [3,5,9]
    args.lr_decay_rate = 0.1
    args.kd_T = 4
    args.distill = 'kd'
    
    args.gamma = 1
    args.alpha = 0.5
    args.beta = 0
    args.smoothing = 0.5
    args.clip = 0.2

    # change the batch-szie of the forget_loader and retain_loader
    if args.dataset == 'imagenet':
        forget_loader = torch.utils.data.DataLoader(forget_loader.dataset, batch_size=args.batch_size/4)
        retain_loader = torch.utils.data.DataLoader(retain_loader.dataset, batch_size=args.batch_size)
    else:
        forget_loader = torch.utils.data.DataLoader(forget_loader.dataset, batch_size=16)
        retain_loader = torch.utils.data.DataLoader(retain_loader.dataset, batch_size=64)

    name_parts = [
            f"{args.unlearn_method}",
            # f"_forgetClass_{args.forget_class}" if args.forget_class else "",
            f"_retainRatio_{args.retain_ratio}" if args.retain_ratio != 1 else "",
            f"_lr_{str(args.lr_unlearn)}",
            f"_alpha_{str(args.alpha)}",
            f"_gamma_{str(args.gamma)}",
            "_schedule" if args.schedule_unlearn else "",
            "_augment" if args.augment_ft else "",
        ]
    args.name_unlearn = "".join(name_parts)
    teacher = copy.deepcopy(net)
    student = copy.deepcopy(net)
    model_t = copy.deepcopy(teacher)
    model_s = copy.deepcopy(student)
    
    #this is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    #For SGDA smoothing
    beta = 0.1
    def avg_fn(averaged_model_parameter, model_parameter, num_averaged): 
        return (1 - beta) * averaged_model_parameter + beta * model_parameter
    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)
    # swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=lambda x, y, z: x * (1 - 0.1) + y * 0.1)

    module_list = nn.ModuleList([model_s, model_t])
    trainable_list = nn.ModuleList([model_s])

    # Set up criterion and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)
    criterion_list = nn.ModuleList([criterion_cls, criterion_div, criterion_kd])

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        swa_model.cuda()

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(trainable_list.parameters(), lr=args.lr_unlearn, momentum=0.9, weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(trainable_list.parameters(), lr=args.lr_unlearn, weight_decay=args.wd)
    elif args.opt == 'adamw':
        optimizer = torch.optim.AdamW(trainable_list.parameters(), lr=args.lr_unlearn, weight_decay=args.wd)

    for epoch in tqdm(range(1, args.epochs_unlearn + 1)):

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)
        print(f"Epoch: {epoch}\t lr: {lr}")
        
        # max-steps
        if epoch <= args.msteps:
            metrics = run_epoch_distill(epoch, forget_loader, module_list, swa_model, criterion_list, optimizer, args, "maximize")

        metrics = run_epoch_distill(epoch, retain_loader, module_list, swa_model, criterion_list, optimizer, args, "minimize")

        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)

        torch.save(model_s.state_dict(), args.ckpt_root/f"{args.name_unlearn}_epoch_{epoch}.pt")
        name = f'SCRUB_{epoch}'
        readouts[name] = all_readouts(model_s, name, args.seed)


def run_epoch_align(epoch, net, retain_loader, forget_loader, align_loader, criterion, optimizer, args, mode='train'):
    """Run an epoch, either for training or testing
    
    Args:
        epoch (int): Current epoch number.
        net (torch.nn.Module): Neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        opt: Command line arguments.
        mode (str): 'train' for training, 'test' for testing.
        quiet (bool, optional): If True, suppress print statements. Default is False.
    
    Returns:
        AverageMeter: Metrics calculated during the epoch.
    """
    net.train() if mode == 'train' else net.eval()
    DEVICE = next(net.parameters()).device
    criterion_align = nn.MSELoss()
    metrics = AverageMeter()

    for batch_idx, ((input, target), (input_forget, target_forget), (logit_align, target_align)
                    ) in enumerate(zip(retain_loader, cycle(forget_loader), cycle(align_loader))):
        # pdb.set_trace()
        target = torch.from_numpy(np.asarray(target).astype('long'))
        target_forget = torch.from_numpy(np.asarray(target_forget).astype('long'))
        target_align = torch.from_numpy(np.asarray(target_align).astype('long'))
        input, target = input.to(DEVICE), target.to(DEVICE)
        input_forget, target_forget = input_forget.to(DEVICE), target_forget.to(DEVICE)
        logit_align, target_align = logit_align.to(DEVICE), target_align.to(DEVICE)
        logit_align = logit_align.squeeze()

        # Forward pass
        output = net(input)
        logit_forget = net(input_forget)
        loss_retain = criterion(output, target)
        loss_align = criterion_align(logit_align, logit_forget)
        loss = args.alpha*loss_retain + (1-args.alpha)*loss_align

        if mode == 'train':
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate metrics
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metrics.update(n=input.size(0), losses=loss.item(), loss_retain=loss_retain.item(), loss_align=loss_align.item(),
                       top1=acc1.item(), top5=acc5.item())
        if not args.quiet:
            progress_bar(batch_idx, len(retain_loader), 'Loss: %.3f | Loss_retain: %.3f | Loss_align: %.3f | Acc: %.3f%% (N=%d)'
                     % (metrics.avg['losses'], metrics.avg['loss_retain'], metrics.avg['loss_align'], metrics.avg['top1'], metrics.count['top1']))
    return metrics
