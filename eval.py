import os
import torch
import argparse

from data.voc0712 import VOCDetection
from data.coco import COCODataset
from data.transform import BaseTransform

from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator

from utils.misc import load_weight
from models.build import build_yolo


parser = argparse.ArgumentParser(description='YOLO Detector Evaluation')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val, coco-test.')
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')

parser.add_argument('-v', '--version', default='yolo',
                    help='yolo.')

parser.add_argument('--weight', type=str, default=None, 
                    help='Trained state_dict file path to open')
parser.add_argument('-size', '--input_size', default=416, type=int,
                    help='input_size')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Use cuda')

args = parser.parse_args()



def voc_test(model, device, input_size, val_transform):
    data_root = os.path.join(args.root, 'VOCdevkit')
    evaluator = VOCAPIEvaluator(
        data_root=data_root,
        img_size=input_size,
        device=device,
        transform=val_transform,
        display=True
        )

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, device, input_size, val_transform, test=False):
    data_root = os.path.join(args.root, 'COCO')
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=input_size,
            device=device,
            testset=True,
            transform=val_transform
            )

    else:
        # eval
        evaluator = COCOAPIEvaluator(
            data_dir=data_root,
            img_size=input_size,
            device=device,
            testset=False,
            transform=val_transform
            )

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 构建模型
    model = build_yolo(args, device, args.input_size, num_classes, trainable=False)

    # 加载已训练好的模型权重
    model = load_weight(model, args.weight)
    model.to(device).eval()
    
    val_transform = BaseTransform(args.input_size)

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, device, args.input_size, val_transform)
        elif args.dataset == 'coco-val':
            coco_test(model, device, args.input_size, val_transform, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, device, args.input_size, val_transform, test=True)
