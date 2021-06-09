@[toc]

# 什么是增量学习
增量学习是指一个学习系统能不断地从新样本中学习新的知识，并能保存大部分以前已经学习到的知识，就和人类学习知识一样，学习完新知识后，我们仍然记得旧知识

[自己复现的代码](https://github.com/zhuoyunli/iCaRL)

<br>

# 增量学习存在的问题

<br>

## 灾难性遗忘（catastrophic forgetting）
神经网络在新数据上训练后，权重发生变化，导致神经网络遗忘旧数据上学习到的规律，类似于背英语单词，背了新的忘了旧的

<br>

## 克服灾难性遗忘的策略

<br>

### 策略一
冻结当前神经网络的部分权重，并且增加神经网络的深度，让神经网络能够保留学习新知识的能力。

以下为个人对策略一的理解
浅层神经网络会保留旧数据上学习的知识，而深层神经网络则在旧知识的基础上继续学习新知识，假设利用残差结构连接深层网络的输出与浅层网络的输出，对于旧知识，只需要让深层网络的输出近似于0即可完全保留旧知识，这和残差结构的出发点类似。

<br>

### 策略二
利用旧数据与新数据对网络进行训练。

以下为个人对策略二的理解
策略二更像是一种微调，如果仅仅利用新数据训练网络，那么网络的确很有可能遗忘旧知识，提供旧数据有助于网络在学习新知识的前提下抵抗遗忘。

<br>

# 论文主要工作
针对图像分类网络的增量学习，即让网络在保留旧类别区分能力的基础上，区分新类别，提出了一种名为iCaRL的训练策略，该算法可以让分类器（例如softmax分类器）与特征提取器（例如CNN）同时学习。

论文提出的训练策略并不针对某个CNN框架，因此可以自由选择CNN框架。另外，CNN框架的不同并不能解决灾难性遗忘。

<br>

# 网络架构细节
可以自由选择CNN架构，CNN架构之后接一个激活函数为$sigmoid$的全连接层$L$，作为分类器，全连接层的输入（特征向量），均进行了[$L2$归一化](https://www.jianshu.com/p/1092578cdc1c)，L2归一化有助于网络收敛。使用多标签分类损失函数，多标签分类损失函数其实就是极大似然估计，具体可查看[我是链接]()。

<br>

# 算法介绍

<br>

## 名词解释
- $examplar$：可以看成一个容器，一个$examplar$保存了某一类的部分训练数据
- $K$：K即为存储训练数据的上限，假设目前有$m$类，那么每一个$examplar$可以存储的训练数据为$\frac{K}{m}$。


<br>

## 总体流程
iCaRL算法的总体流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191025091002233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
每个函数的具体解释如下：
$updaterepresentation$函数：用于更新CNN与全连接层分类器的参数。
$reduceexemplarset$函数：用于减少旧类的$examplar$的容量。
$constructexemplarset$函数：用于构造新类的$examplar$。

整个算法的流程总结一下，即为：

 1. 添加新类别数据后，训练CNN与全连接层分类器
 2. 减少旧类别的$examplar$的容量，去除部分训练数据
 3. 构造新类别的$examplar$，存储部分新类别的训练数据

$examplar$存储数据会用于模型的训练，帮助模型抵抗遗忘

<br>

## 步骤一：模型训练
由$updaterepresentation$函数实现
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026093640954.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
该函数需要的参数为

 1. 新类别数据
 2. $examplar$集合
 3. CNN与全连接层分类器

具体步骤如下：
 1. 合并$examplar$集合以及新类别数据集，构造训练数据集
 2. 存储训练前，模型对于训练数据的预测结果（迷之操作，$g(x_i)$表示CNN+全连接层分类器的输出向量，$g_y(x_i)$表示输出向量的第$y$维的值）
 3. 训练模型

损失函数使用多标签分类损失函数，分为分类以及蒸馏两部分，按步骤来说

 1. 从旧模型上得到新数据的分类标签，该类标签可以提供一种更强的监督信号，表明与新旧类别相似的类别
 2. 新模型对应的one-hot标签的旧类别部分替换为步骤一的结果

对于旧类别而言，步骤一的标签可以提供更强的监督信号，来弥补数据不足导致的监督信号缺失。

<br>

## 步骤二：$examplar$管理
训练完模型后，需要对$examplar$集合进行调整，$examplar$集合的作用有两个

 1. 帮助模型抵抗遗忘（上面已经介绍）
 2. 预测分类（在模型预测模块介绍）

$examplar$的管理由$reduceexemplarset$函数与$constructexemplarset$函数实现。

<br>

### $constructexemplarset$函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026100223297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
采样的标准：每次选择离训练数据类别中心最接近的图像样本，这和之后的分类有关。
按照采样的先后顺序构建新类别的$examplar$（即$P$），在$P$中越靠前位置的样本组成的集合，离类别中心越接近

<br>

### $reduceexemplarset$函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026100716288.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
$examplar$中的样本要遵循的原则——样本集合的中心尽可能的接近类别集合，依据进入$examplar$的顺序，靠前位置样本组成的集合，比靠后位置样本组成的集合，离类别中心更接近，因此，每次都去除尾部一定数量的样本，让$examplar$的大小不超过$\frac{K}{m}$，$K$为存储空间上限，$m$为当前类别数。

<br>

## 模型预测
常见的神经网络分类模型，都是分类器（例如softmax分类器）与特征提取器（例如CNN）一起学习，但是iCaRL将分类器与特征提取器分开，只有特征提取器进行学习，分类器则选择了不需训练的最近邻分类器，这点有点反直觉，分类过程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026101247922.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
由于使用最近邻分类器，所以$examplar$中，A类别的样本集合中心应该尽可能接近A类别训练数据的中心，这也是构建$examplar$的出发点。

<br>

# 实验

<br>

## 与其他方法的比较
论文选取了三个baseline

 1. Finetuning：在之前学习的基础上添加一个新的类别分支，利用新数据微调网络
 2. Fixed representation：第一次增量学习训练完毕后，冻结特征提取器的权重，只会训练分类器新分支的权重，新分支训练完毕后，冻结新分支的权重
 3. Learning without Forgetting模型

iCaRL与上述三个baseline都使用相同的CNN（ResNet），一次增量学习即让学习器多学习$N$类，论文比对了N=2,5,10,20,50的情况，论文在CIFAR100以及ImageNet数据集上比对了这几个方法

CIFAR-100数据集的结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026182307810.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
在ImageNet数据集的结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026183418238.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

学习次数的越多，网络的性能表现越糟糕，这在一定程度上表明模型存在遗忘现象，从上图可知，iCaRL训练的网络，性能最号，但是无法判断相比于baseline，iCaRL是否可以更好的抵抗遗忘。

为了进一步显示模型是否出现遗忘，作者还比较了iCaRL与三个baseline在CIFAR-100上的混淆矩阵（一次增量学习多学习10类），结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026183729841.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
iCaRL存在一个明显的对角线，分类性能最好，LwF.MC偏向于预测新的类别，这在一定程度上说明模型出现了遗忘，fixed representation偏向于预测旧的类别，这很好理解，因为特征提取器一次增量学习后就被固定了，对于新类别，很难提取出足够区分度的特征，finetuning
的遗忘现象最为严重。

<br>

## 其他实验

iCaRL采取了三个策略

 1. 使用最近邻分类器
 2. 使用$examplar$集合以及新类别数据训练模型
 3. 使用蒸馏loss

为了探究上述三个策略在抵抗遗忘方面的作用，论文设计了三个比对实验

 1. $hybrid1$：使用策略2、3，使用全连接层分类器
 2. $hybrid2$：使用策略1、2，不使用蒸馏loss
 3.  $hybrid3$：仅使用策略2
 4. $iCaRL$：使用策略1、2、3

多次增量学习后，模型的平均准确率如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191026185249663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
$iCaRL$ vc $hybrid1$：可以康出使用最近邻分类器更具有优势
$iCaRL$ vc $hybrid2$：当N取值较小(例如2)时，蒸馏loss似乎无法有效提高模型准确率，当N取值较大(N>2)时，蒸馏loss有助于抵抗遗忘
$hybrid3$ vs LwF.MC：使用$examplar$集合与新数据一起训练模型一定程度上有助于模型抵抗遗忘



<br>

# 个人理解

本论文的分类准确率提升来源于两个方面

 1. 使用新数据与部分旧数据微调网络
 2. 使用更为鲁棒的分类算法——最近邻

从论文结果中可以看到，使用最近邻进行分类的架构比使用全连接层进行分类的架构准确率提升了几个百分点，个人认为这属于分类器的鲁棒性带来的性能提升。

对新数据进行训练时，特征提取器（CNN）的输出可能与旧数据的输出发生非常大的改变，如果分类器对于输入的扰动过于敏感，可能会导致旧数据的遗忘，而最近邻算法鲁棒性恰好非常优越。

论文开源代码地址[iCaRL](https://github.com/srebuffi/iCaRL)
