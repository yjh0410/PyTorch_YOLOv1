import torch
import torch.nn as nn
from utils import Conv, SPP
from backbone import resnet18
import numpy as np
import tools

class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5):
        super(myYOLO, self).__init__()
        self.device = device                           # cuda或者是cpu
        self.num_classes = num_classes                 # 类别的数量
        self.trainable = trainable                     # 训练的标记
        self.conf_thresh = conf_thresh                 # 得分阈值
        self.nms_thresh = nms_thresh                   # NMS阈值
        self.stride = 32                               # 网络的最大步长
        self.grid_cell = self.create_grid(input_size)  # 网格坐标矩阵
        self.input_size = input_size                   # 输入图像大小
        
        # backbone: resnet18
        self.backbone = resnet18(pretrained=True)
        c5 = 512

        # neck: SPP
        self.neck = nn.Sequential(
            SPP(),
            Conv(c5*4, c5, k=1),
        )

        # detection head
        self.convsets = nn.Sequential(
            Conv(c5, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1)
        )

        # pred
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)
    

    def create_grid(self, input_size):
        """ 
            用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        """
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy


    def set_grid(self, input_size):
        """
            用于重置G矩阵。
        """
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)


    def decode_boxes(self, pred):
        """
            将txtytwth转换为常用的x1y1x2y2形式。
        """
        output = torch.zeros_like(pred)
        # 得到所有bbox 的中心点坐标和宽高
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2
        
        return output


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        

        keep = []                                             
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    def forward(self, x, target=None):
        # backbone主干网络
        c5 = self.backbone(x)

        # neck网络
        p5 = self.neck(c5)

        # detection head网络
        p5 = self.convsets(p5)

        # 预测层
        pred = self.pred(p5)

        # 对pred 的size做一些view调整，便于后续的处理
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        pred = pred.view(p5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测  
        # [B, H*W, 1]
        conf_pred = pred[:, :, :1]
        # [B, H*W, num_cls]
        cls_pred = pred[:, :, 1 : 1 + self.num_classes]
        # [B, H*W, 4]
        txtytwth_pred = pred[:, :, 1 + self.num_classes:]

        # train
        if self.trainable:
            conf_loss, cls_loss, bbox_loss, total_loss = tools.loss(pred_conf=conf_pred, 
                                                                    pred_cls=cls_pred,
                                                                    pred_txtytwth=txtytwth_pred,
                                                                    label=target
                                                                    )

            return conf_loss, cls_loss, bbox_loss, total_loss            
        # test
        else:
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W, 1] -> [H*W, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W, 4] -> [H*W, 4], 并做归一化处理
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W, 1] -> [H*W, num_class]，得分=<类别置信度>乘以<objectness置信度>
                scores = (torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred)
                
                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()
                
                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds
