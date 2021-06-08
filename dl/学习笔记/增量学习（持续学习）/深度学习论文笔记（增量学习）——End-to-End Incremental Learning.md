@[toc]

# 主要工作
论文提出了一种算法，以解决增量学习中的灾难性遗忘问题，与iCaRL将特征提取器的学习与分类器分开不同，本论文提出的算法通过引入新定义的loss以及finetuning过程，在有效抵抗灾难性遗忘的前提下，允许特征提取器与分类器同时学习。

本论文提出的方法需要[$examplar$](https://blog.csdn.net/dhaiuda/article/details/102736082#_52)

<br>

# 算法介绍

<br>

## 总体流程
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101100635705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
总体分为四个流程

 1. 构建训练数据
 2. 模型训练
 3. finetuning
 4. 管理 $examplar$

<br>

## 步骤一：构建训练数据
训练数据由新类别数据与[examplar]((https://blog.csdn.net/dhaiuda/article/details/102736082#_52))构成。

设有$n$个旧类别，$m$个新类别，每个训练数据都有两个标签，第$i$个训练数据的标签为

 1. 使用onehot编码的$1*(m+n)$的向量$p_i$
 2. 设旧模型为$F_{t-1}$，训练数据为$x$，$q_i=F_{t-1}(x)$，$q_i$为一个$1*n$维的向量

<br>

## 步骤二：模型训练
模型可以选用常见的CNN网络，例如ResNet32等，按照国际惯例，这一节会介绍distillation loss，作为一篇被顶会接收的论文，自然不能免俗

<br>

### loss函数介绍

符号约定
|符号名|含义  |
|--|--|
| $N$ | 有$N$个训练数据 |
|$p_i$|含义查看上一节|
|$q_i$|含义查看上一节|
|$\hat q_i$|新模型旧类别分支的输出，为一个$1*n$的向量|
|$n$|旧类别分支|
|$m$|新类别分支|
|$o_i$|新模型对于第$i$个数据的输出，为一个$(n+m)*1$的向量|

**Classification loss**即交叉熵，如下：

$$L_C(w)=-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^{n+m}p_{ij}*softmax(o_{ij})$$

其中
$$softmax(o_{ij})=\frac{e^{o_{ij}}}{\sum_{j=1}^{n+m}e^{o_{ij}}}$$

<br>

**distillation loss**的形式如下

$$L_D(w)=-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{n}pdist_{ij}qdist_{ij}$$

其中
$$
pdist_{ij}=\frac{e^{\frac{\hat q_{ij}}{t}}}{\sum_{j=1}^{n}e^{\frac{\hat q_{ij}}{t}}}\\
qdist_{ij}=\frac{e^{\frac{q_{ij}}{t}}}{\sum_{j=1}^{n}e^{\frac{q_{ij}}{t}}}
$$

$L_D(w)$即让模型尽可能的记住旧类别的输出分布。t是一个超参数，在本论文中，$t=2$


---
**个人疑问**

distillation loss的作用是让模型记住以往学习到的规律，相当于侧面引入了旧数据集，从而抵抗类别遗忘。

直觉上来说，distillation loss应该只对旧类别数据进行计算，但是**新类别数据的旧类别分支输出仍用于计算distillation loss**，论文对此给出的解释是“To reinforce the old knowledge”

我认为这种做法的出发点为：**旧模型对于新类别数据的输出（经softmax处理），也是一种旧知识，也需要防止遗忘，因此，新模型对于新类别数据的旧类别输出（经softmax处理），与旧模型对于新类别数据的输出（经softmax处理）也要尽可能一致**

---

## 步骤三：finetuning
使用**herding selection**算法，从新类别数据中抽取部分数据，构成与旧类别examplar大小相等的数据集，此时各类别数据之间类别平衡，利用该数据集，在小学习率下对模型进行微调，选用的loss函数应该是交叉熵。

步骤二使用类别不平衡的数据训练模型，会导致分类器出现分类偏好，finetuning可以在一定程度上矫正分类器的分类偏好

<br>

## 步骤四：管理$examplar$
论文给出了两类方法

 1. Fixed number of samples：没有内存上限，每个类别都有$M$个数据
 2. Fixed memory size：内存上限为$K$

使用**herding selection**算法选择新类别数据，构成新类别的$examplar$

<br>

# 实验
论文训练模型使用了数据增强，具体方式如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102161627635.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
每个实验都进行了五次训练，取平均准确率
实验过程没有太多有趣的地方，在此不做过多说明
<br>

## Fixed memory size
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102162008828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102162056210.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102162233121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

<br>

## Fixed number of samples
在CIFAR100上的结果如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102162422674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
img/cls表示每个examplar中图片的个数

<br>

# Ablation studies
首先是选择数据构建examplar的方法，论文比对了三类方法

 1. herding selection：平均准确率63.6%
 2. random selection：平均准确率63.1%
 3. histogram selection：平均准确率59.1%

上述三个选择方法的解释如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102163359200.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
接下来论文比对了算法各部分对准确率提升的贡献
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102163502728.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
上述模型的解释如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102163523998.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

# 个人理解
灾难性遗忘的本质是类别不平衡，模型在学习旧类别时，所使用的数据是充分的，引入知识蒸馏loss，就是尽可能保留旧数据上学习到的规律，在训练时，相当于侧面引入了旧数据。

论文在distillation loss的基础上又引入了类别平衡条件下的finetuning，相当于进一步抵抗增量学习下类别不平衡的导致的分类器偏好问题，由此取得模型性能的提升。


