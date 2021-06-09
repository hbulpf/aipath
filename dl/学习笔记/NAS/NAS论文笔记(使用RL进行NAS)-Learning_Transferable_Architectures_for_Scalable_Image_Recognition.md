@[toc]



# 论文主要工作
论文提出了一种使用强化学习自动搜索图像识别网络的方法，主要工作如下

 1. 定义了一个名为NASNet Search Space的搜索空间，使得在小数据集上训练得到的网络架构模块（注意是模块，不是整个网络架构），可以迁移到大数据集上网络架构的构建。
 2. 对[Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578?twitter=@bigdata)提出的方法进行了改进，提出了一种新的正则化方法——ScheduledDropPath（本文不会总结）

由于计算资源的限制，本文不会过多解释如何使用强化学习进行NAS search

<br>

# 定义神经网络搜索空间

在介绍NASNet Search Space灵感来源之前，先了解一下NASNet Search Space

## NASNet Search Space
NASNet Search Space将神经网络结构看成由一系列的模块（又叫Cell）堆叠而成，组成神经网络的Cell个数是提前固定的，神经网络的大体架构被预定义好，论文定义了两种类型的Cell——Normal Cell以及Reduction Cell，论文在CIFAR10和ImageNet上预定义的神经网络结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190704105308927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
NASNet Search Space将神经网络的层数（深度）固定，只搜索Cell的结构。该搜索空间包含的神经网络具有相同的深度，但是具有不同的Cell结构。而在AutoML的某些方法中，神经网络结构的搜索空间包含的神经网络具有不同的深度，每层的操作（例如卷积、池化）也不同，假设该搜索空间为Temp Search Space，这么做的优点有两个：

 1. 相比于Temp Search Space， NASNet Search Space的大小更小，在相同搜索次数的前提下，NASNet  Search Space可以更好的被探索，举个例子，假设搜索次数为10次，NASNet Search Space包含的神经网络结构数目为100，Temp Search Space包含的神经网络结构数目为1000，那么NASNet Search Space被探索了10%，而Temp Search Space仅被探索了1%，但是这么做有个问题，虽然NASNet Search Space更小，但是对于某个数据集来说，NASNet Search Space搜索到的网络架构足够优秀吗？
 2. 探索到的Cell具有一定的泛化能力，可能同样适用于其他数据集上的识别任务。

论文最终通过实验验证了上述两个优点，从而确认了NASNet Search Space的优势

 1. 对于CIFAR10数据集，NASNet Search Space包含的网络架构足够优秀
 2. 探索到的Cell结构具有一定的泛化能力，能适应ImageNet数据集

Normal Cell以及Reduction Cell的介绍可以查看[AutoML论文笔记（aging evolution）：Regularized Evolution for Image Classifier Architecture Search](https://blog.csdn.net/dhaiuda/article/details/93337707)，实践表明定义两类Cell更加有益，Reduction Cell输出的特征图是Normal cell的两倍

<br>

 ## NASNet Search Space灵感来源

大多数的CNN网络架构都是通过堆叠一系列相同模块（相当于cell）来实现的，例如通过堆叠Separable Convolution+BN+Relu的Xception，这意味着我们或许可以利用强化学习来发现一种或多种cell结构，通过堆叠发现的cell结构，来进行不同的图像识别任务

<br>

## 算法的超参数
 1. 一个stack中Normal Cell的个数N
 2. 卷积操作中过滤器的个数F

<br>

# 使用RL进行NAS搜索
定义完了搜索空间，接着便是搜索搜索空间，本论文使用RL（强化学习）进行搜索，具体的实现方法请查看[Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578?twitter=@bigdata)

大致的搜索方法如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710100857899.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

RNN产生一种网络结构，将该网络结构训练至收敛，其在验证集上的准确率用于更新RNN，以让其产生更好的网络架构，RNN的权重通过策略梯度进行更新，在本论文中，RNN将用于生成Cell结构，具体流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710101605477.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
初始状态时，将含有两个隐藏状态（特征图），接着，RNN采用五个步骤，用于生成一个Cell结构

 1. 从隐藏状态集中（含有经过不同处理的特征图集合，例如卷积、池化）选择一个隐藏状态
 2. 选择第二个隐藏状态
 3. 选择一个op操作（例如卷积）应用于第一个隐藏状态
 4. 选择另一个op操作（例如池化）应用于第二个隐藏状态
 5. 选择一个combine操作应用于3、4步骤的结果，具体可以是add，或是concat，产生的隐藏状态进入到隐藏状态集中，应用于下一阶段生成新的隐藏状态

重复上述5个步骤B次后（论文将B设置为5），所有未使用过的隐藏状态将在深度方向进行concat，做为Cell的输出，举个例子：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710102954106.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
注意到一个Cell初始时有两个特征图输入，其实是一个残差结构，例如Normal Cell之间的连接如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710103802239.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
网络架构中第一个Cell的两个特征图输入均为输入图像，可供RNN选择的op操作如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710103617694.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

可供RNN选择的combine操作有add或是深度方向的concat

论文通过上述搜索方法，利用一个RNN，搜索Normal Cell与Reduction Cell的结构

<br>

# 实验过程与结果
使用上述方法，利用500个GPU训练后，在CIFAR-10上得到的Cell结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710151620248.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

<br>

## 在CIFAR-10上的识别结果
CIFAR-10上使用的网络架构之前已经给出，具体结果如下表：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710152026188.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
x@y表示N=x，F=y，关于F与N的解释，请查看超参数一节，可以看到，NASNet-A（7@2304）+cutout具有最优秀的准确率，足以可以对于CIFRA-10数据集来说，NASNet Search Space是一个良好定义的搜索空间

<br>

## 在ImageNet上的识别结果
通过在CIFAR-10上发现的Cell结构直接应用于预定义好的ImageNet网络架构，并重头训练，得到的结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710153118360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
可以看到，NASNet-A(6@4032)具有最优秀的top-1和top-5准确率，可见学得的Cell具有一定的泛化能力

<br>

## 具有计算资源限制的识别结果
论文同时将发现的Cell结构应用于具有计算资源限制的场景中，例如手机上，在ImageNet数据集上的结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190710153522486.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
可以看到，NASNet-A(4@1056)具有最高的准确率，可见学得Cell结构非常的优秀

值得注意的是，在CIFAR-10数据集上，在NASNet Search Space上使用随机搜索得到的网络结构也具有较高的准确率，可见对于CIFAR-10数据集，NASNet Search Space具有良好的定义

