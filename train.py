from __future__ import division

import os
import random
import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from data.coco import COCODataset
from data.voc0712 import VOCDetection
from data.transform import Augmentation, BaseTransform

from utils.misc import detection_collate
from utils.com_paras_flops import FLOPs_and_Params
from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator

from models.build import build_yolo
from models.matcher import gt_creator


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # 基本参数
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')

    # 模型参数
    parser.add_argument('-v', '--version', default='yolo',
                        help='yolo')

    # 训练配置
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--max_epoch', type=int, default=150,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[90, 120], type=int,
                        help='lr epoch to decay')

    # 优化器参数
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')

    # 数据集参数
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)
    
    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 是否使用多尺度训练
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416

    # 构建dataset类和dataloader类
    dataset, num_classes, evaluator = build_dataset(args, device, train_size, val_size)
    
    # 构建dataloader类
    dataloader = build_dataloader(args, dataset)

    # 构建我们的模型
    model = build_yolo(args, device, train_size, num_classes, trainable=True)
    model.to(device).train()

    # 计算模型的FLOPs和参数量
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    model_copy.set_grid(val_size)
    FLOPs_and_Params(model=model_copy, 
                        img_size=val_size, 
                        device=device)
    del model_copy

    # 使用 tensorboard 可视化训练过程
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # 构建训练优化器
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )

    max_epoch = args.max_epoch                 # 最大训练轮次
    lr_epoch = args.lr_epoch
    epoch_size = len(dataloader)  # 每一训练轮次的迭代次数

    # 开始训练
    best_map = -1.
    t0 = time.time()
    for epoch in range(args.start_epoch, max_epoch):

        # 使用阶梯学习率衰减策略
        if epoch in lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets) in enumerate(dataloader):
            # 使用warm-up策略来调整早期的学习率
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr) 

            # 多尺度训练
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # 随机选择一个新的尺寸
                train_size = random.randint(10, 19) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # 插值
                images = F.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = gt_creator(
                input_size=train_size,
                stride=model.stride, 
                label_lists=targets
                )
            
            # to device
            images = images.to(device)          
            targets = targets.to(device)
            
            # 前向推理和计算损失
            conf_loss, cls_loss, bbox_loss, total_loss = model(images, targets=targets)

            # 反向传播
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', bbox_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), 
                            cls_loss.item(), 
                            bbox_loss.item(), 
                            total_loss.item(), 
                            train_size, 
                            t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if epoch  % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

            cur_map = evaluator.map
            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print('Saving state, epoch:', epoch + 1)
                weight_name = '{}_epoch_{}_{:.1f}.pth'.format(args.version, epoch + 1, best_map*100)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save(model.state_dict(), checkpoint_path)                      


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_dataset(args, device, train_size, val_size):
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    train_transform = Augmentation(train_size, pixel_mean, pixel_std)
    val_transform = BaseTransform(val_size, pixel_mean, pixel_std)
    
    # 构建dataset类和dataloader类
    if args.dataset == 'voc':
        data_root = os.path.join(args.root, 'VOCdevkit')
        # 加载voc数据集
        num_classes = 20
        dataset = VOCDetection(
            root=data_root,
            transform=train_transform
            )

        evaluator = VOCAPIEvaluator(
            data_root=data_root,
            img_size=val_size,
            device=device,
            transform=val_transform
            )

    elif args.dataset == 'coco':
        # 加载COCO数据集
        data_root = os.path.join(args.root, 'COCO')
        num_classes = 80
        dataset = COCODataset(
            data_dir=data_root,
            img_size=train_size,
            transform=train_transform
            )

        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=val_size,
            device=device,
            transform=val_transform
            )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")


    return dataset, num_classes, evaluator


def build_dataloader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )
    
    return dataloader


if __name__ == '__main__':
    train()
