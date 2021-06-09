@[toc]

论文全称为：Schlegl et al. - 2017 - Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery

<br>

# 异常检测的基本概念
异常检测即检查某一类数据是否存在异常，例如病人是否患有某种疾病。一般情况下，模型只会使用正常数据进行训练

<br>

# 主要工作

有监督的异常检测模型往往需要大量标注的训练数据，但是在医疗领域，标注数据需要耗费大量的人力，数据标注成本高，因此，论文作者在DCGAN的基础上，通过精心设计的loss函数，构建了名为AnoGAN的架构，首次利用无监督的GAN进行异常检测。

关于GAN，可以查看我之前的博客[深度学习——生成对抗网络（GAN）](https://blog.csdn.net/dhaiuda/article/details/102751214)

<br>

# AnoGAN介绍

<br>

## motivation
DCGAN将隐空间中的点映射为一张图片，通过使用正常图片训练DCGAN，训练完毕后，DCGAN隐空间中的点生成的应该都是正常图像，那我们能不能将一张名为A的图片映射为隐空间中的某个点呢？这有点困难，那我们能不能在隐空间中查找到一个点，该点生成的图片与图片A在视觉上最为相近呢？这便是论文的出发点。

<br>

## AnoGAN异常检测原理

DCGAN训练完毕后，冻结DCGAN的参数，隐空间为待训练参数。随机在隐空间中选择一个点$z_y$，生成器生成的图像记为$G(z_y)$，测试图像记为x，通过使用反向传播，利用某些优化算法，例如SGD，经过若干轮优化，可以在隐空间中找到一个点，记为$z_y'$，使得loss最小，此时生成器生成的图像$G(z_y')$与图像A最为相近。由于生成器只能生成正常图片，异常图片与正常图片的差距本来就比较大，比较图像$G(z_y')$与图像x的差距，如果差距大于某个数值，则可认为图像A出现异常。

由此可以抛出两个问题

 1. 如何设计loss
 2. 如何比较比较图像$G(z_y')$与图像x的差距

<br>

## loss的定义
loss函数分为两部分：$Residual\ Loss$与$Discrimination\ Loss$。

$Residual\ Loss$定义为
$$R(z_y)=\sum|x-G(z_y')|$$
论文并没有具体解释如何计算这个loss，从形式上看，这个loss可以度量生成图片与测试图片的差距，如果生成图片与测试图片一致，则为0

$Discrimination\ Loss$定义为
$$D(z_y)=\sum|f(x)-f(G(z_y'))|$$
论文并没有直接使用鉴别器的输出作为函数$f$，而是选择鉴别器中间某一层的输出作为函数$f$，作者将鉴别器看成是一个特征提取器，$Discrimination\ Loss$反映了鉴别器对两张照片提取的特征之间的差异

最终定义的loss函数为
$$A(z_y)=(1-\lambda)R(z_y)+\lambda D(z_y)$$

$\lambda$取值为0到1，为超参数，该loss从像素以及鉴别器提取的图像特征两个维度比较了两张图片的差异。我们可以直接使用该loss度量图片差异，如果经过若干轮优化后，loss仍然大于某个阈值，则可认为该图片存在异常。

以上便是上一节两个疑问的解答。

<br>

# 实验
论文选用的数据集为视网膜的光学相干断层扫描图像，给定含有$M$张正常图像的数据集$K$，每张图片上采集$K$张大小为$c*c$的图片，构成训练数据，测试时，从测试图片采取一定数量的图片，输入到AnoGAN中，利用loss判断是否为异常图片，若为异常图片，则用红点标明原图该块区域，这是一个二分类问题，因此可以使用召回率、准确率等指标对模型性能进行评估。

实验具体解答了以下几个问题

 1. AnoGAN可以生成质量不错的图像吗？
 2. AnoGAN可以检测异常吗？
 3. 比较了使用不同loss或模型的测试结果的差异。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026191818252.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

上图有三个大模块，每个模块有四行，所有模块的第一行为真实图片，第二行为生成的图片，第三行为异常检测的结果，第四行应该是将异常区域分割出来的结果。

比对一、二模块的第一行与第二行，可见生成图像与真实图像具有一定的相似性，可简单说明AnoGAN可以生成质量不错的图像。

比对一、二、三模块的三、四行，可见AnoGAN可以可以检测异常。

从效果图来看，AnoGAN效果不错

接着论文设计了三个使用不同loss或模型的baseline，比对了不同loss对DCGAN异常检测的影响

 1. aCAE：GAN的另一实现方式
 2. $P_D$：使用鉴别器的输出作为$Loss$。
 3. $GAN_R$：设文献Semantic image
inpainting with perceptual and contextual losses.提出的鉴别器损失函数$\hat D(z_y)$，则 $GAN_R$使用的损失函数为$A(z_y)=(1-\lambda)R(z_y)+\lambda \hat D(z_y)$

结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026193211525.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

AnoGAN的效果不错。论文也比较了单纯使用$R(z_y)$、$D(z_y)$、$\hat D(z_y)$作为$loss$的结果，对应的ROC以及AUC如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026193726305.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

单纯使用$R(z_y)$或是$D(z_y)$，也可以带来不错的效果。论文中仍有一些其他图表，就不在此进行总结啦~

<br>

# 个人理解
这篇论文的重点

 1. 如何定义损失函数，以找到距离最近的隐空间的点（重要）。损失函数定义的好，找到的点越好，准确率自然越高，这个损失函数的设计大概也是不断试错出来的
 2. 如何确认图片是否异常（重要）。用于判断图像是否异常的指标要选择好，这个指标的设计大概也需要不断试错
 
 总体来说，整篇论文给我感觉非常不错

