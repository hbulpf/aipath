# 风格迁移

**风格迁移（style transfer）**最近两年非常火，可谓是深度学习领域很有创意的研究成果。它主要是通过神经网络，将一幅艺术风格画（style image）和一张普通的照片（content image）巧妙地融合，形成一张非常有意思的图片。

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/645zwawgmv.png?imageView2/2/w/1620)

因为新颖而有趣，自然成为了大家研究的焦点。目前已经有许多基于风格迁移的应用诞生了，如移动端风格画应用Prisma，手Q中也集成了不少的风格画滤镜：

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/izxjwewx8i.png?imageView2/2/w/1620)

本文将对风格迁移[[1](http://melonteam.com/https:/arxiv.org/pdf/1508.06576.pdf)]的实现原理进行下简单介绍，然后介绍下它的快速版，即fast-style- transfer[[2](http://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)]。

## 1. 风格迁移开山之作

2015年，Gatys等人发表了文章[1]《[A Neural Algorithm of Artistic Style](http://melonteam.com/https:/arxiv.org/pdf/1508.06576.pdf)》，首次使用深度学习进行艺术画风格学习。把风格图像Xs的绘画风格融入到内容图像Xc，得到一幅新的图像Xn。则新的图像Xn：即要保持内容图像Xc的原始图像内容（内容画是一部汽车，融合后应仍是一部汽车，不能变成摩托车），又要保持风格图像Xs的特有风格（比如纹理、色调、笔触等）。

### 1.1 内容损失（Content Loss）

在CNN网络中，一般认为较低层的特征描述了图像的具体视觉特征（即纹理、颜色等），较高层的特征则是较为抽象的图像内容描述。 所以要比较两幅图像的**内容相似性**，可以比较两幅图像在CNN网络中`高层特征`的相似性（欧式距离）。

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/680qnkm5d8.png?imageView2/2/w/1620)

### 1.2 风格损失（Style Loss）

而要比较两幅图像的**风格相似性**，则可以比较它们在CNN网络中较`低层特征`的相似性。不过值得注意的是，不能像内容相似性计算一样，简单的采用欧式距离度量，因为低层特征包含较多的图像局部特征（即空间信息过于显著），比如两幅风格相似但内容完全不同的图像，若直接计算它们的欧式距离，则可能会产生较大的误差，认为它们风格不相似。论文中使用了`Gram矩阵`，用于计算不同响应层之间的联系，即在保留低层特征的同时去除图像内容的影响，只比较风格的相似性。

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/ggmq1jjj29.png?imageView2/2/w/1620)

那么风格的相似性计算可以用如下公式表示：

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/9iamytyr42.png?imageView2/2/w/1620)

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/9i19v6bgw4.png?imageView2/2/w/1620)

### 1.3 总损失（Total Loss）

这样对两幅图像进行“内容+风格”的相似度评价，可以采用如下的损失函数：

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/uhknc8oyye.png?imageView2/2/w/1620)

### 1.4 训练过程

文章使用了著名的VGG19网络[3]来进行训练（包含16个卷积层和5个池化层，但实际训练中未使用任何全连接层，并使用平均池化average- pooling替代最大池化max-pooling）。

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/1l4kf4gnhu.png?imageView2/2/w/1620)

**内容层和风格层的选择：**将`内容图像`和`风格图像`分别输入到VGG19网络中，并将网络各个层的特征图（feature map）进行可视化（重构）。

内容重构五组对比实验：

```javascript
1. conv1_1 (a)
2. conv2_1 (b)
3. conv3_1 (c)
4. conv4_1 (d)
5. conv5_1 (e)
```

复制

风格重构五组对比实验：

```javascript
1. conv1_1 (a)
2. conv1_1 and conv2_1 (b) 
3. conv1_1, conv2_1 and conv3_1 (c)
4. conv1_1, conv2_1, conv3_1 and conv4_1 (d)
5. conv1_1, conv2_1, conv3_1, conv4_1 and conv5_1 (e)
```

复制

通过实验发现：对于内容重构，(d)和(e)较好地保留了图像的高阶内容（high-level content）而丢弃了过于细节的像素信息；对于风格重构，(e)则较好地描述了艺术画的风格。如下图红色方框标记：

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/kmnwobcbnz.png?imageView2/2/w/1620)

在实际实验中，内容层和风格层选择如下：

**内容层：conv4_2**

**风格层：conv1\*1\*, \*conv2_1, conv3\*__1_, _conv4_1, conv5_1**

**训练过程：**以白噪声图像作为输入(x)到VGG19网络，conv4_2层的响应与原始内容图像计算出内容损失（Content Loss），“conv1_1, conv2_1, conv3_1, conv4_1, conv5_1”这5层的响应分别与风格图像计算出风格损失，然后它们相加得到总的风格损失（Style Loss），最后Content Loss + Style Loss = Total Loss得到总的损失。采用梯度下降的优化方法求解Total Loss函数的最小值，不断更新x，最终得到一幅“合成画”。

### 1.5 总结

1. 每次训练迭代，更新的参数并非VGG19网络本身，而是随机初始化的输入x；
2. 由于输入x是随机初始化的，最终得到的“合成画”会有差异；
3. 每生成一幅“合成画”，都要重新训练一次，速度较慢，难以做到实时。

## 2. 快速风格迁移

2016年Johnson等人提出了一种更为快速的风格迁移方法[2]《[Perceptual losses for real-time style transfer and super- resolution](http://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)》。

### 2.1 网络结构

它们设计了一个变换网络（Image Transform Net），并用VGG16网络作为损失网络（Loss Net）。输入图像经由变换网络后，会得到一个输出，此输出与风格图像、内容图像分别输入到VGG16损失网络，类似于[1]的思路，使用VGG16不同层的响应结果计算出内容损失和风格损失，最终求得总损失。然后使用梯度下降的优化方法不断更新变换网络的参数。 

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/w2kty29edn.png?imageView2/2/w/1620)

**内容层：relu3_3**

**风格层：relu1\*2\*, \*relu2_2, relu3_3\*, _relu4_3**

其中变换网络（Image Transform Net）的具体结构如下图所示： 

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/85ln9yin8i.png?imageView2/2/w/1620)

### 2.2 跑个实验

Johnson等人将论文的代码实现在[github](http://melonteam.com/https:/github.com/jcjohnson/fast-neural-style)上进行了开源，包括了论文的复现版本，以及将“Batch-Normalization ”改进为“Instance Normalization”[[4](http://melonteam.com/https:/arxiv.org/pdf/1607.08022.pdf)]的版本。咱们可以按照他的说明，训练一个自己的风格化网络。我这里训练了一个“中国风”网络，运行效果如下： 

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/wsu6tl6g0i.png?imageView2/2/w/1620)

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/vapwv6zpp7.png?imageView2/2/w/1620)

### 2.3 总结

1. 网络训练一次即可，不像Gatys等人[1]的方法需要每次重新训练网络；
2. 可以实现实时的风格化滤镜：在Titan X [GPU](https://cloud.tencent.com/product/gpu?from=10680)上处理一张512x512的图片可以达到20FPS。下图为fast-style-transfer与Gatys等人[1]方法的运行速度比较，包括了不同的图像大小，以及Gatys方法不同的迭代次数。

![img](https://ask.qcloudimg.com/http-save/yehe-1069763/owxt2k7q2z.png?imageView2/2/w/1620)

## 3. 参考资料

1. Gatys L A, Ecker A S, Bethge M. A neural algorithm of artistic style[J]. arXiv preprint arXiv:1508.06576, 2015.
2. Johnson J, Alahi A, Fei-Fei L. Perceptual losses for real-time style transfer and super-resolution[C]//European Conference on Computer Vision. Springer International Publishing, 2016: 694-711.
3. Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.
4. Ulyanov D, Vedaldi A, Lempitsky V. Instance normalization: The missing ingredient for fast stylization[J]. arXiv preprint arXiv:1607.08022, 2016.
5. [Fast Style Transfer(快速风格转移)](http://closure11.com/fast-style-transfer快速风格转移/)
6. [图像风格迁移(Neural Style)简史](http://melonteam.com/https:/zhuanlan.zhihu.com/p/26746283)