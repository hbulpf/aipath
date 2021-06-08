@[toc]

# 主要工作
目前来说，NAS的搜索方法包含有强化学习和进化算法，论文提出了一种新方法，利用SMBO策略，即通过不断增强网络的复杂度（深度）来探索整个搜索空间，同时利用surrogate函数（其实就是surrogate-based）来指引探索过程，从而发现性能优异的网络。

与之前基于NASNet Search Space的强化学习与进化算法相比，该算法所需的计算量少八倍，在CIFAR-10上达到同等精度所需训练的模型数量少五倍，同时搜索到的网络结构在ImageNet和CIFAR-10上达到了SOTA水准。

吐槽：是，是，是显卡燃烧的味道！本论文的作者们应该都是显卡战士，我试着跑了一下NAS相关的算法，真的好慢！！！用的单GPU.....，没钱.......，也可能是因为在自定义的100\*100的数据集上跑的缘故，32*32大小的CIFAR-10简直是天使，相信训练速度一定会加快。

从主要工作出发，可以生出两个疑问

 1. 搜索空间如何定义？
 2. surrogate函数是啥模型，如何训练？

<br>

# 搜索空间定义
对NASNet Search Space进行一定的更改，就得到了本论文定义的搜索空间。

关于NASNet Search Space，可以查看我之前写过的两篇博客，在此不对其进行介绍
[NAS论文笔记(使用RL进行NAS)：Learning Transferable Architectures for Scalable Image Recognition](https://blog.csdn.net/dhaiuda/article/details/94598175#_12)
[NAS论文笔记（aging evolution）：Regularized Evolution for Image Classifier Architecture Search](https://blog.csdn.net/dhaiuda/article/details/93337707#_14)

NASNet Search Space中的Cell含有固定数目的pairwise，而本论文定义的搜索空间中Cell的pairwise数目是不固定的，并且是递增式的，直至增长至上限$B$。举个例子，含有一个pairwise的Cell，含有两个pairwise的Cell，含有三个pairwise的Cell.......含有$B$个pairwise的Cell，整个搜索空间非常的大，作者使用了一种层次性的搜索算法，具体如下：

在第$b$轮循环中，含有$K$个候选Cell，每个Cell含有$b$个$pairwise$，这$K$个候选Cell对应的网络将被训练和评估，接着这$K$个候选Cell将派生出含有$b+1$个$pairwise$的Cell，接着使用surrogate函数选出最有可能表现优异的$K$个候选细胞，训练对应的网络并进行评估，不断重复上述过程，整个流程可以用下图来描述

![在这里插入图片描述](https://img-blog.csdnimg.cn/201910171010330.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

与NASNet Search Space不同的是，本论文没有区分NormalCell与ReductionCell，只是将NormalCell的stride设置为2来模拟ReductionCell，这会在一定程度上减少搜索空间的大小。

<br>

# 预测模型
预测模型用于预测某个cell组成的网络的性能，在[NAS论文笔记——Peephole: Predicting Network Performance Before Training](https://blog.csdn.net/dhaiuda/article/details/102585029)一文中，我解释了一下为什么可以使用LSTM、RNN等时序模型预测网络性能。这篇论文同样使用了序列模型，具体来说，需要解决的问题主要是**如何对神经网络进行编码**。

论文用一个四元组来$<I_1,I_2,O_1,O_2>$表示一个pairwise，其中$I_1、I_2$表示选择的特征图，$O_1、O_2$表示对特征图进行的操作，$I_1,I_2,O_1,O_2$都是one-hot编码，接着对one-hot使用$embedding$，获得一个$D$维向量，$I_1,I_2$共用一个$Embeded$，$O_1,O_2$共用一个$Embeded$，则可将一个四元组$<I_1,I_2,O_1,O_2>$转变为一个$4D$维的向量，以此作为LSTM+MLP、RNN的输入，其实LSTM+MLP的组合和[NAS论文笔记——Peephole: Predicting Network Performance Before Training](https://blog.csdn.net/dhaiuda/article/details/102585029)中的模型很像。

论文同时尝试了MLP预测网络性能，假设一个Cell有$B$个pairwise，依据上述方法，我们可以得到$B$个$4D$维的向量，接着对每一维上的数据进行平均，得到一个$4D$维的向量，作为MLP的输入。

<br>

## 训练预测模型
<br>

### 数据集准备
设b为Cell中pairwise的个数，b的取值为1~5，对于含有b（b>1）个pairwise的Cell，随机抽取10000个模型进行训练，b=1的所有Cell都会训练，在CIFAR10上一共训练了接近40000个模型。

<br>

### 模型训练
采用下列算法训练模型，K的取值为256
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191017183136761.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

具体来说，模型训练的步骤如下：

 1. 从含有b个pairwise的cell数据集中随机抽取256个数据训练模型
 2. 训练好的模型会在训练数据（256个含有b个pairwise）以及测试数据（10000个含有b+1个pairwise）上进行测试

<br>

### 训练结果

#### MLP ensemble
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191017183953293.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
论文给出了在训练数据以及测试数据上的线性相关系数，论文并没有解释使用了什么线性相关系数（例如皮尔逊相关系数等），可以看到，MLP-ensemble在训练数据集上的线性相关性还是不错的，但是在测试数据集上，线性相关性并不是很高，但是随着训练的进行，MLP-ensemble在测试数据集上的线性相关性逐渐上升。

<br>

#### RNN与MLP的对比
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191017185048294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

不论是否使用ensemble，RNN在训练数据上的表现都更为亮眼，而MLP在测试数据上的表现更为亮眼，这在一定程度上说明，RNN可能出现了过拟合。

<br>

# 实验
本部分不会过多细说实验的细节（例如优化参数等），只给出自认为比较有意思的结果

<br>

## 运行速度比较

在NASNet Search Space上使用强化学习的算法与PNAS（就是上面所述的算法）的性能对比结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191017185836561.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
$B$表示PNAS中cell最多含有的pairwise个数，TOP表示选出N（N=1,5,25）个模型，Accuracy表示选出的N个模型的测试准确率平均值，实验从两种算法达到相同测试准确率平均值所需训练的模型数以及计算开销来比较两者的性能，PNAS的性能提升为表格中的Speedup(#models)以及Speedup(#examples)。

可以看到，PNAS能在短时间内得到较好的结果，但是这个结果不够严谨，没有把训练预测模型所需训练的模型数（约为40000个）计算在内，如果计算在内，可以看到PNAS运行所需的模型数将远比NAS多，这样的计算开销，普通实验室根本无法承担起。

论文对于计算性能的开销，即Speedup(#examples)的计算方式，是通过两个算法达到相同平均测试准确率所需的SGD步骤得出的，虽然数值很惊艳，但是同样没把训练预测模型的40000模型算进去。

以上结果不免让人有些失望，这篇论文提出的算法，任然需要耗费大量的计算资源。


<br>

## 发现的Cell在CIFAR10与ImageNet上的表现
具体表现如下，就不总结了，注意标为红色的语句。

**CIFAR10**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191017191104291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

**ImageNet**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191017191202655.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

