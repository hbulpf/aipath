@[toc]

# 主要工作
类别不平衡导致增量学习出现灾难性遗忘，本论文设计了一种loss函数，以抵抗类别不平衡造成的负面影响。

**本论文提出的算法需要[$examplar$](https://blog.csdn.net/dhaiuda/article/details/102736082#_52)**

<br>

# 算法介绍

论文将类别不平衡对增量学习的影响分为三个部分

 1. Imbalanced Magnitudes：新类别权重向量的模大于旧类别，如上节所示
 2. Deviation：出现灾难性遗忘
 3. Ambiguities：新类别的权重向量与旧类别相似，模型容易将旧类别数据划分为新类别

为了解决上述问题，论文通过如下三个步骤来构建最终的loss函数，以消除类别不平衡造成的影响。

 1. Cosine Normalization（抵抗Imbalanced Magnitudes）
 2. Less-Forget Constraint（抵抗Deviation）
 3. Inter-Class Separation（抵抗Ambiguities）

对应关系如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101141718738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
<br>

## 符号约定
|符号名|含义  |
|--|--|
| $f(x)$ | 特征提取器的输出 |
|$\overline{f}(x)$|特征提取器的输出L2归一化后的结果|
|$\theta_i$|全连接层分类器中，第$i$类对应的1*n维权重向量|
|$\overline{\theta}_i$|全连接层分类器中，第$i$类对应的1*n维权重向量L2归一化后的结果|
|$b_i$|全连接层分类器中，第$i$类对应的偏置|
|$p_i(x)$|第$i$类的概率|
|$<\theta_i,f(x)>$|$<\theta_i,f(x)>=\theta_i*f(x)$|

<br>

## Cosine Normalization

在CIFAR100上使用iCaRL，分类器权重的L2范式以及偏置（$b$）值可视化的结果如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101090339137.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
从上图**至少**可知，类别不平衡会导致分类器出现两个问题

 1. 新类别权重向量的L2范式大于旧类别权重向量
 2. 新类别的偏置（参数$b$）基本大于0，旧类别的偏置（参数$b$）基本小于0

上述两个问题可能导致分类器出现分类偏好

---
**个人疑问**
实验一：在Large Scale Incremental Learning一文中，去除掉分类器的偏置项（参数$b$）后，分类器的准确率有所上升
实验二：去除上述两个影响后，分类器的准确率有所提升（请查看[Ablation Study](https://blog.csdn.net/dhaiuda/article/details/102850853)部分）。
**上述两个实验，都是给出准确率，但是抵抗分类偏好，不应该给出混淆矩阵吗？**

**回答**
一个简单的步骤，例如去除偏置项、L2归一化只是在一定程度上抵抗分类偏好，其混淆矩阵仍可能显示分类器有分类偏好。采取某些步骤后，模型的准确率**大幅**上升，意味着误分为新类别的数据被分类器正确分类，在一定程度上说明该步骤可以**抵抗**分类偏好

---

为了解决上述两个问题，论文做了两个工作

 1. 对每个类别的权重向量使用L2归一化，这样所有类别的权重向量的L2范式均为1
 2. 去除偏置

如果将特征提取器的输出也进行L2归一化，经过softmax层处理后的结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191101143101384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)


$\eta$是一个可学习参数，其存在对于分类而言意义不大（所有值都放大或是缩小相同倍数，大小关系不变），论文对其解释是用来控制softmax分布的[峰度](https://www.cnblogs.com/shadow1/p/10914798.html)，可能与优化有关，个人认为这个参数没有深入了解的必要，因此不在此做过多解释

**为什么要对特征提取器的输出进行L2归一化呢？**
此时特征提取器的输出向量与类权重向量都位于一个高维球体内部，但论文并没有解释这样做有什么好处，由于特征提取器进行L2正则化有助于模型收敛，这里这么做可能是为了加速模型收敛

<br>

## Less-Forget Constraint
按国际惯例，一篇增量学习论文必然会对loss函数进行魔改，本论文自然不能免俗

论文冻结了全连接层分类器旧类别分支的权重向量，定义的知识蒸馏loss如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110114543229.png)
$\overline{f}^*(x)$与$\overline{f}(x)$表示增量学习前后特征提取器L2归一化后的输出，由于进行了L2归一化，$\overline{f}^*(x)$与$\overline{f}(x)$的模为1，当上式取值为0时，意味着两个向量的夹角为0，则有$\overline{f}^*(x)=\overline{f}(x)$，由于全连接层旧类别分支的权重向量被冻结，此时对于旧类别数据，增量学习前后模型的输出一致（新类别分支的输出会为0）。

作者认为全连接层分类器中的权重在一定程度上反映了类与类之间的关系，因此一个
自然的想法就是固定旧类别分支的权重向量（从而保留类与类之间的关系），让训练后的特征提取器尽可能与训练前的一致，从而抵抗灾难性遗忘。

<br>

## Inter-Class Separation
为了预防模型将新旧类别混淆，论文定义了如下loss函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/201911011515016.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
选出新类别中，输出（$<\overline\theta_i,\overline f(x)>$）值与旧类别输出值最接近的$K$个分支，计算其差距，只要差距大于$m$，损失函数的值即为0，对于旧类别数据，随着优化的进行，旧类别分支的输出与新类别分支的输出差距会逐渐拉大，从而防止将旧类别数据划分为新类别数据

需注意，旧类别的权重向量是固定的，上式中，$\overline\theta(x)$是固定的

<br>

## 损失函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110208375376.png)
$L_{ce}(x)$即为交叉熵损失函数，$N$表示训练数据，$N_o$表示训练数据中的旧类别数据，$\lambda$是是一个自适应参数，其取值为
$$\lambda=\lambda_{base}\sqrt\frac{|C_n|}{|C_o|}\tag{式1}$$
$|C_o|、|C_n|$表示旧类别与新类别的数目，$\lambda_{base}$是一个自定义大小的参数

**疑问**
由于每次需要学习的新类别数目是固定的，即$|C_n|$固定，$|C_o|$不断提高，会导致$\lambda$下降，即**distillation loss**在损失函数中的占比下降，这有点奇怪，随着增量学习步骤的增多，**distillation loss**在损失函数中的占比应该增加才对。

<br>

# 实验

|baseline| 解释 |
|----------|--|
|   [iCaRL-CNN](https://blog.csdn.net/dhaiuda/article/details/102736082)   | 用examplar+distillation loss训练CNN |
|[iCaRL-NME]((https://blog.csdn.net/dhaiuda/article/details/102736082)  )|用examplar+distillation loss训练CNN，分类器采用nearest- mean-of-exemplars（最近邻）|
|Ours-CNN|examplar+上述损失函数训练CNN|
|Ours-NME|examplar+上述损失函数训练CNN，分类器采用nearest- mean-of-exemplars（最近邻）|
|joint-CNN|用全部数据训练CNN|

**CIFAR100、ImageNet-Subset、ImageNet-Full上的结果**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102091156159.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
比较有意思的是Ours-CNN与Ours-NME差距不大，两者只是采用的分类器不同，NME并不会出现分类偏好的情况，这在一定程度上说明，使用论文提出的损失函数进行增量学习，可以让分类器抵抗分类偏好

按国际惯例，应该给出混淆矩阵进一步说明抵抗分类偏好，如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102091601693.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
<br>

# Ablation Study
符号约定
 1. CN：Cosine Normalization
 2. LS：Less-Forget Constraint
 3. IS：Inter-Class Separation
 4. AW：自适应参数，即式1

每进行完一次增量学习，都会使用类别平衡的数据（examplar+新类别部分数据）对模型进行finetuning（这个操作可以查看End-to-End Incremental Learning）

---
**CN、LS、IS的影响**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102092225657.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
上图可以看出损失函数每个部分对于准确率提升的效果，说明三者缺一不可，上图中的Ours-CNN使用了AW，未使用CBF，其他模型都使用了CBF，可以看出，CBF对于模型的准确率的影响不大，说明应用本论文提出的方法，分类器分类偏好已经被较好解决

---

**AW的影响**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191102092516645.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
**所有实验数据都是进行多次实验取平均**

<br>

# 个人理解

为什么增量学习的CNN比非增量学习的CNN准确率低？
答案是灾难性遗忘，但是造成灾难性遗忘的核心原因，个人觉得还是类别不平衡，类别不平衡会导致分类器出现分类偏好（更偏向于新类别，因为新类别的训练数据多），因此，目前阅读过的大部分论文都是针对分类器入手。

想要提高增量学习的分类准确率，首要解决的是类别不平衡问题带来的负面影响，

**但是即使类别不平衡问题可以较好的解决，模型的分类准确率为什么无法达到非增量学习分类模型的准确率呢？**
