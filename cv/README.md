# 计算机视觉

计算机视觉应用可以分为：**图片识别(Image Classification)、目标检测(Object Detection)、语义分割(Semantic Segmentation) 、视频理解(Video Understanding)和图片生成(Image Generation)** 等。下面对其进行详细介绍。

**图片识别(Image Classification)** 是常见的分类问题。神经网络的输入为图片数据，输出值为当前样本属于每个类别的概率，通常选取概率值最大的类别作为样本的预测类别。图片识别是最早成功应用深度学习的任务之一，经典的网络模型有 VGG 系列、Inception 系列、ResNet 系列等。

![img](pics/cv1.png)

**目标 检测(Object Detection)**是指通过算法自动检测出图片中常见物体的大致位置，通常用边界框(Bounding box)表示，并分类出边界框中物体的类别信息。常见的目标检测算法有CNN，Fast RCNN，Faster RCNN，Mask RCNN，SSD,YOLO 系列等。

![img](pics/cv2.png)

**语义分割(Semantic Segmentation)**是通过算法自动分割并识别出图片中的内容，可以将语义分割理解为每个像素点的分类问题，分析每个像素点属于物体的类别。常见的语义分割模型有 FCN，U-net，SegNet，DeepLab 系列等。

![img](pics/cv3.png)

**视频理解(Video Understanding)** 随着深度学习在 2D 图片的相关任务上取得较好的效，具有时间维度信息的 3D 视频理解任务受到越来越多的关注。常见的视频理解任务有视频分类，行为检测，视频主体抽取等。常用的模型有 C3D，TSN，DOVF，TS_LSTM等。

![GAN](pics/cv4.png)

**图片生成(Image Generation)** 通过学习真实图片的分布，并从学习到的分布中采样而获得逼真度较高的生成图片。目前主要的生成模型有 VAE 系列，GAN 系列等。其中 GAN 系列算法近年来取得了巨大的进展，最新 GAN 模型产生的图片样本达到了肉眼难辨真伪的效果。