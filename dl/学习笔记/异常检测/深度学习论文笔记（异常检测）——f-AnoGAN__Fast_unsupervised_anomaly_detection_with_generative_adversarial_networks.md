@[toc]

# 主要工作
提出了一种Encoder，可以快速将图片映射到隐空间中的某个点，接着利用WGAN进行异常检测。

在[深度学习论文笔记（异常检测）—— Generative Adversarial Networks to Guide Marker Discovery](https://blog.csdn.net/dhaiuda/article/details/102736763)一文中，我总结了AnoGAN，其通过不断迭代优化，在隐空间中寻找某个点，该点生成的图片与测试图片最为相近，接着利用DCGAN进行异常检测，由于需要迭代优化，势必会耗费大量时间，而f-AnoGAN通过引入Encoder，解决了这个问题。

**[代码地址](https://github.com/tSchlegl/f-AnoGAN)**

<br>

# 算法介绍

**有监督异常检测存在的问题**

 1. 需要耗费大量人力与时间对数据进行标注，在医疗领域，数据标注的代价更高，并且数据量比较少，而有监督学习往往需要耗费大量数据。
 2. 有监督学习只能处理训练样例中存在的情况。
 
 针对有监督的问题，论文提出了使用无监督的GAN进行医疗数据的异常检测，其具体机制为：使用正常数据训练GAN，生成器$G$只能生成正常数据，如果能在隐空间中找到一点$Z$，$G(Z)$与测试图像最为相近，$G(Z)$为正常图像，如果两者的差距大于某个值，就可判断测试图像为异常图像。鉴别器$D$本质是一个二分类模型，可以鉴别出真实图像与生成器生成图像之间的细微差别，而异常图像本身与正常图像差别较大，鉴别器会将异常图像分为非正常图像。可以看到，鉴别器与生成器都可以单独用于异常检测，和AnoGAN一样，论文将两者进行了结合。

模型分为两个阶段，如下图
阶段一：训练WGAN
阶段二：训练Encoder

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110315372775.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

## 阶段一：训练WGAN

此处总结数据准备

设有$N$张正常医疗图片构成的集合$R$，$I_n \in R$（$n=1、2、3...N$），从$I_n$中随机截取$K$张大小为$c*c$的图片构成训练数据集

设标记好的数据集为$J$，按上述方式采集大小为$c*c$的图像$y_m$，同时获得对应的大小为$c*c$的掩码图像$a_m$（为像素为1表示异常，为0表示无异常），$<y_m,a_m>$构成了一个测试数据，不断重复上述方式构成测试数据集

在视网膜光学相干断层扫描图像数据集上构建训练与测试数据集的流程如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103183358850.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)


<br>

## 阶段二：训练Encoder

**论文没有给出Encoder的结构，应该是一个卷积神经网络，具体可以查看代码部分**

WGAN训练完毕后，不在改变，由生成器充当decoder，与Encoder一起构成了auto-encoder结构，Encoder负责将训练图片（查看上一节数据准备部分）映射为隐空间中的点$Z$，生成器将$Z$映射为图片。

Encoder存在三种训练方式，如下图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103160036963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

### 训练方式一：$izi$
具体步骤

 1. Encoder将图片x映射为隐空间中的点$\hat z$
 2. 生成器将$\hat z$映射为图片$G(\hat z)$
 3. 损失函数为MSE：$$L_{izi}(x)=\frac{1}{n}||x-G(\hat z)||^2$$ $n$为像素的个数


<br>

### 训练方式二：$ziz$
具体步骤

 1. 在隐空间中随机选取一个点$z$，生成器将$z$映射为图片$G(z)$
 2. Encoder将$G(z)$映射为隐空间中的点$\hat z$
 3. 损失函数为MSE：$$L_{ziz}(z)=\frac{1}{d}||z-\hat z)||^2$$ d为隐空间的维数

<br>

### 训练方式三：$izi_f$

具体步骤

 1. Encoder将图片x映射为隐空间中的点$\hat z$
 2. 生成器将$\hat z$映射为图片$G(\hat z)$
 3. 将$G(\hat z)$与$x$输入到鉴别器中，得到$L_D=\frac{1}{n_d}||f(x)-f(G(\hat z))||^2$，f(x)为鉴别器中间某一层的特征图，该特征图被认为含有输入图像的统计信息，$L_D$用于比较图像之间统计信息的差异，$n_d$为特征图的维数（个人理解为特征图像素个数）
 4. 损失函数为$$L_{izi_f}(x)=\frac{1}{n}||x-G(\hat z)||^2+\lambda \frac{1}{n_d}||f(x)-f(G(\hat z))||^2$$$\lambda$为超参数

$f-AnoGAN$将 $izi_f$ 作为Encoder的训练方式

<br>

## 异常检测
异常检测其实是一个二分类问题，我们需要设计一个异常分数公式用于计算异常分数，异常分数高于某个值，即可认为出现异常，f-AnoGAN将$L_{izi_f}(x)$作为异常分数公式，$L_{izi_f}(x)$从像素差异与图片之间的统计学差异角度比较了两张图片之间的差距。

假设$x$为异常图片，由于生成器只能生成正常图片，鉴别器能鉴别图片是否符合正常图片分布，则$G(\hat z)$与$x$、$f(G(\hat z))$与$f(x)$之间的差异势必比较大。

$L_{izi}$对应AnoGAN中的$Residual Loss$，$L_D$则对应$Discrimination Loss$，AnoGAN中统计过正常与异常图片在$Residual Loss$与$Discrimination Loss$上的取值差异，如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103173813534.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

x轴表示$L_{izi}$与$L_D$的值，y轴表示频率，可以看出

 1. 异常图片的$L_{izi}$与$L_D$普遍大于正常图片
 2. 正常图片与异常图片在$L_{izi}$与$L_D$上的取值分布重叠部分小，说明$L_{izi}$与$L_D$对于正常图片与异常图片的区分度高

因此，$L_{izi_f}(x)$可用于计算异常得分


<br>

# 实验

**问题一：隐空间是否平滑连续？**
如果隐空间不够平滑连续，只有部分隐空间中的点能生成真实度较高的图片，为了验证隐空间是连续的，论文进行了两个实验

实验一：随机选择隐空间中的两个点，两点之间做一条位于高维度空间的直线，生成这条直线上的点对应的图片
实验二：依据真实图片在隐空间中选择两个点（应该是使用了Encoder），两点之间做一条位于高维度空间的直线，生成这条直线上的点对应的图片

两个实验的结果如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103201759766.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

可以看到，图像之间的变化非常自然，由此可见隐空间还是比较平滑的，如果隐空间是剧烈抖动的，那么图像之间的渐变效果应该会非常明显

---

**问题二：f-AnoGAN的预测准确率如何？**
异常检测本质上是一个二分类问题，可以使用准确率召回率等指标进行模型评估

论文比对的baseline有

 1. $AE$
 2. $AdvAE$
 3. $ALI$
 4. $A_D$：使用WGAN的鉴别器输出作为异常分数，由于WGAN的输出比较的是生成图片与真实图片的Wasserstein距离，因此不能直接作为异常分数，设测试图片为$x$，则异常分数定义如下$$A_D=\hat m_{x}-D(x)$$
 随机选择32000张测试图片，统计对应的鉴别器输出，计算平均值，即为$\hat m_x$
 6. $iterative$：使用WCAN的AnoGAN

结果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103202758698.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
f-AnoGAN的评价指标均为最高，表现相当优异

AUC如下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103203158567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

---

**问题三：f-AnoGAN异常检测的效果如何？**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103203042140.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

---

**问题四：不同Encoder训练策略的比较**

首先是异常检测的视觉效果
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110320342041.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

接着是各项指标
AUC如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/201911032035029.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191103203519325.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

可看出，$izi_f$策略的训练结果最佳
