
# YOLOv3

![image-20220317213604982](README.assets/image-20220317213604982.png)

## 改进点

* 最大的改进就是网络结构,使其更适合小目标检测。
* 特征做的更细致，融入多持续特征图信息来预测不同规格物体。
* 先验框更丰富了，3中scale，每种3个规格，一共9种
* softmax改进，预测多标签任务

多scale

![image-20220318005512785](README.assets/image-20220318005512785.png)

Relu , leakRelu
