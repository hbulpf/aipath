@[toc]

# 论文主要工作
提出了一种层次化的神经网络搜索空间，并通过进化算法进行搜索，在CIFAR-10上的top-1错误率为3.6%，将发现的网络架构迁移至ImageNet后，得到的top-1准确率为20.3%，如果在定义的搜索空间中使用随机搜索，则在CIFAR-10和ImageNet上的top-1准确率分别下降了0.3%和0.1%，由此证明了搜索空间的定义是良好的

<br>

# 神经网络搜索空间定义

<br>

## 扁平架构
在介绍分层架构前，先介绍扁平架构，分层架构由扁平架构组成

扁平架构就是一张计算图，这张计算图有单一的源节点和终点结点，图中的每一个结点都是特征图，有向边表示一些基础操作，例如池化、卷积操作等，计算图接收一个特征图作为输入，对其进行一系列的基础操作后进行输出，一个结点可以由多个有向边指向，则该顶点的特征图等于所有有向边对应的特征图在depth方向进行concat后得到，由此可知，一个计算图中卷积、池化前后特征图的大小是不变的，计算图的结构需要提前定义

扁平架构可以用下列符号表示：

 1. 基础操作集$o=${$o_1,o_2,.....$}
 2. 邻接矩阵G，$G_{ij}$=k表示存在结点$i$指向结点$j$的有向边，该边代表的操作是$o_k$
 3. 在上述两个定义的基础上，可以认为网络架构就是选择$o$中的操作，并将其填到$G$中，即$arch=assemble(G,o)$，$assemble$表示填充操作

<br>

## 分层架构
分层架构分为多层，每一层都是一个或多个计算图，每个计算图的定义与扁平结构一致，但是有向边表示的不一定是基础操作，假设分层架构有$L$层，则第$l$层有向边的基础操作集由第$l-1（l>1）$层的计算图组成，第一层的基础操作集需要自定义，每一层的计算图的个数以及结构需要自定义，第$L$层只有一个计算图，即为最终的分层架构，最终的分层架构是自低而上构建的，举个例子：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071309495917.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
分层架构有两层，第二层只有一种计算图，即为最终的分层架构

**第一层**

 1. 定义了三种计算图，分别是上图中的红蓝绿色调的计算图
 2. 第一层的基础操作集为1*1conv，3*3 conv，3*3 max-pooling

**第二层**

 1. $o_1^{(3)}$即为最终的分层架构
 2. 第二层的基础操作集为第一层的计算图组成，也就是红蓝绿色调的计算图

<br>

## 第一层的基础操作集

论文在定义了六种第一层基础操作，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713101345959.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
所有操作的步幅数都是1，并且需要保证操作前后特征图的空间分辨率不变，过滤器的个数为C，需要提前指定，卷积操作后都会应用BN+ReLU，除了上述操作外，论文还定义了none操作，表示没有边存在

<br>

## 搜索空间
本论文与之前看过的大部分论文一样，都是提前定义好了网络结构，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713102016208.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
分层架构即上图中的Cell，Cell的可能形式即搜索空间中神经网络架构的可能形式

<br>

# 进化算法

<br>

## 种群初始化
具体步骤如下：

 1. 定义分层架构的层数，定义每层计算图的形状，定义第一层基础操作集
 2. 初始化一批种群个体，个体中的每个cell都是直接映射，即输入等于输出
 3. 对种群中的个体进行大规模的变异操作

<br>

## 变异操作
具体步骤如下：

 1. 随机选择一层
 2. 随机选择该层的一个计算图
 3. 随机选择该计算图的一个顶点i
 4. 随机算则该计算图顶点i的后继结点j
 5. 从基础操作集中随机选择一个操作替换现有操作，如果结点i与结点j不存在操作，此时相当于添加一条由结点i指向结点j的有向边

上述步骤可以实现添加一条有向边，去除一条有向边，更改一条有向边对应的操作等

<br>

## 选择
采用锦标赛选择算法，每次选出当前种群中5%的个体，选择适应度（在验证集上的准确率）最高的个体，对其进行变异操作后产生新个体，新个体在训练一定轮数并计算适应度后放回种群中，论文采取的锦标赛算法不会kill掉任何个体，随着算法的运行，种群的规模会不断增大

<br>

# 随机搜索
随机搜索即去除掉锦标赛算法后的进化算法，即随机选择个体，随机进行变异操作

<br>

# 算法的超参数

 1. 初始种群个数N
 2. 分层架构的层数L
 3. 每一层基础操作集中操作的个数
 4. 每一层的计算图架构

<br>

# 实验结果
具体的实验参数请查看论文。
论文横向比较了扁平架构、分层架构、随机搜索在在CIFAR-10上发现的网络架构在CIFAR-10和ImageNet的准确率，如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713144324944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
即使是随机搜索，搜索到的架构表现能力也非常好，由此可见神经网络搜索空间的定义是良好的

接下来，论文比对了CIFAR-10上发现的表现能力最好的架构与手工设计或是其他NAS算法发现的网络架构在CIFAR-10测试集上的准确率：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713144849896.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
可见发现的网络架构表现优异

论文还比较了在CIFAR-10上发现的网络架构在ImageNet验证集上与手工设计的网络架构，结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713145011997.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
最后论文给出了发现的分层架构（cell）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019071314584521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190713150235164.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
