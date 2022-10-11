# PyTorch_YOLOv1
这个YOLOv1项目是配合我在知乎专栏上连载的《YOLO入门教程》而创建的：

https://zhuanlan.zhihu.com/c_1364967262269693952

感兴趣的小伙伴可以配合着上面的专栏来一起学习，入门目标检测。

另外，这个项目在小batch size 的情况，如batch size=8，可能会出现nan的问题，经过其他伙伴的调试，
在batch size=8时，可以把学习率lr跳到2e-4，兴许就可以稳定炼丹啦！ 我自己训练的时候，batch size
设置为16或32，比较大，所以训练稳定。

当然，这里也诚挚推荐我的另一个YOLO项目，训练更加稳定，性能更好呦

https://github.com/yjh0410/PyTorch_YOLO-Family


## 网络结构

- Backbone: ResNet-18
- Neck: SPP

## 训练所使用的tricks

- [x] 多尺度训练 (multi-scale)

## 数据集

### VOC2007与VOC2012数据集

读者可以从下面的百度网盘链接来下载VOC2007和VOC2012数据集

链接：https://pan.baidu.com/s/1qClcQXSXjP8FEnsP_RrZjg 

提取码：zrcj 

读者会获得 ```VOCdevkit.zip```压缩包, 分别包含 ```VOCdevkit/VOC2007``` 和 ```VOCdevkit/VOC2012```两个文件夹，分别是VOC2007数据集和VOC2012数据集.

### COCO 2017 数据集

运行 ```sh data/scripts/COCO2017.sh```，将会获得 COCO train2017, val2017, test2017三个数据集.

## 实验结果

VOC2007 test 测试集

| Model             |  Input size    |   mAP   | Weight|
|-------------------|----------------|---------|-------|
| YOLOv1            |  320×320       |   64.8  |   -   |
| YOLOv1            |  416×416       |   69.2  |   -   |
| YOLOv1            |  512×512       |   71.8  |   -   |
| YOLOv1            |  608×608       |   73.3  |   [github]()   |


COCO val 验证集

| Model             |  Input size    |   AP    |   AP50    | Weight|
|-------------------|----------------|---------|-----------|-------|
| YOLOv1            |  320×320       |   13.7  |   29.6    |   -   |
| YOLOv1            |  416×416       |   16.4  |   34.7    |   -   |
| YOLOv1            |  512×512       |   18.1  |   37.9    |   -   |
| YOLOv1            |  608×608       |   18.6  |   39.0    |   [github]()   |

大家可以点击表格中的[github]()来下载模型权重文件。

抱歉，正确的权重文件丢失了，暂时没有找到，很难受……