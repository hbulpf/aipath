@[toc]

# 主要工作
提出一种无需examplar的增量模型——FearNet


<br>

# motivation
对于人类大脑而言，近期记忆存储在海马复合体中，而长期记忆存储在内侧前额叶皮层，在睡眠阶段，近期记忆会整合转变为长期记忆中，从海马复合体中转移至内测前额叶皮层。

受大脑记忆方式启发，FearNet采用了一种双记忆系统，利用PNN（[概率神经网络](https://blog.csdn.net/guoyunlei/article/details/76209647)）存储短期记忆，可理解为存储最近几次增量学习的知识。利用AutoEncoder存储长期记忆，可理解为存储前几次增量学习的知识。在”睡眠阶段“，PNN中存储的知识会”转移“至AutoEncoder中，PNN中存储的所有类别信息被清空。

<br>

# method
采用一个预训练好的ResNet网络作为特征提取器，将图片映射为Feature Embeded。这么做其实有点打擦边球了，增量应该连着特征提取器一起增量才对，不过本文工作的新意主要体现在旧知识的存储上。

FearNet的结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191126145201219.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
<br>

## mPFC network
AutoEncoder，用于存储长期记忆，Encoder将feature embeded映射为大小为m的向量，m为类别个数。Decoder将Encoder的输出映射回feature embeded。Encoder用于分类，Decoder用于存储记忆信息。Encoder与Decoder层数相同。

mPFC network训练的loss如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191126150458230.png)
$L_{class}$为标准的分类损失函数，例如交叉熵，$L_{recon}$定义如下：
$$L_{recon}=\sum_{j=0}^M\sum_{i=0}^{H_j-1}\lambda_j||h_{encoder,(i,j)}-h_{decoder,(i,j)}||^2$$
$M$为Encoder的层数，$H_j$为第$j$层隐藏层神经元的个数，$h_{encoder,(i,j)}$为Encoder第$j$层第$i$个神经元的输出，$h_{decoder,(i,j)}$同理，$\lambda_j$为超参数，$\lambda_0$取值最大，深度越深，$\lambda_j$取值越小。

<br>

## HC network
PNN，用于存储短期记忆，PNN存储有最近几次增量学习训练数据的feature embeded，其分类依据如下（其实PNN有点冷启动算法的味道）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191126150313578.png)
$x$表示训练数据的feature embeded，$u_{k,j}$表示k类别第j个样本的feature embeded。

<br>

## BLA
由于HC network与mPFC network均具备分类的功能，假设目前有$M$类，HC network可以预测$m$到$M$类，mPFC network可以预测0到$m-1$类，两个网络预测的类别不重合，对于测试数据的feature embeded，由BLA决定采用哪个网络的输出作为最终输出。

BLA的输出取值为0~1之间，表明取mPFC作为最终输出的概率，设$A(x)$表示BLA的输出，x表示测试图片的feature embeded，预测公式如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112615212859.png)

<br>

## Consolidation
该阶段用于训练mPFC网络，分为首次训练与增量训练，首次训练，利用训练数据训练AutoEncoder，训练完毕后，对每一个类别，利用训练数据分别计算其Encoder输出的均值与方差。

增量训练，依据每个类别的均值与方差，从对应的混合高斯分布采样，将其输入到decoder中得到与旧类别feature embeded类似的pseudo-examples，与HC中存储的feature embeded构成训练集微调mPFC网络，训练完毕后，利用训练数据重新计算其混合高斯分布的均值与方差。此时mPFC可以生成每个类别（包括HC中存储的类别）的feature embeded，因此，清空HC network中的类别信息，相当于短期记忆转换为长期记忆。

<br>

## Training/Recall
训练阶段只训练BLA，mPFC network与HC network是固定的，从每个类别的混合高斯分布中采样，利用Decoder，将其转变为feature embeded，并标记为1，将HC中的feature embeded标记为0，训练BLA，BLA的损失函数论文没说，可以是交叉熵等分类损失函数，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191126161308298.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

<br>

## 整体算法
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191126161344344.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
每隔K个极端，进行一次Consolidation，即睡眠阶段

实验部分并没有太有意思的地方，在此不做过多总结，有兴趣可以浏览原文。





