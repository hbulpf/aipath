@[toc]

# 主要工作
提出了一种具有注意力机制的前馈卷积神经网络——Convolutional Block Attention Module(CBAM)




# method

注意力机制是人类视觉所特有的大脑信号处理机制。人类视觉通过快速扫描全局图像，获得需要重点关注的目标区域，也就是一般所说的注意力焦点，而后对这一区域投入更多注意力资源，以获取更多所需要关注目标的细节信息，而抑制其他无用信息[摘自[深度学习中的注意力机制](https://blog.csdn.net/qq_40027052/article/details/78421155)]，作者希望CNN也能获得此类能力，实际上，通过grad-CAM对CNN可视化，优秀的网络结构往往能正确定位图中目标所在区域，也即优秀的网络本身就具有注意力机制，作者希望通过强化这一概念，让网络性能更加优异，并且对于噪声输入更加健壮

CNN的卷积操作从channel与spatial两个维度提取特征，因此，论文从channel与spatial两个维度提取具有意义的注意力特征，motivation如下：

 1. 由于每个feature map相当于捕获了原图中的某一个特征，channel attention有助于筛选出有意义的特征，即告诉CNN原图哪一部分特征具有意义（what）
 2. 由于feature map中一个像素代表原图中某个区域的某种特征，spatial attention相当于告诉网络应该注意原图中哪个区域的特征（where）

CBAM将某一层的特征图抽取出来，接着进行channel attention与spatial attention的提取后，与原特征图进行结合作为下一层卷积层的输入，具体流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117084152272.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

## channel attention module
总体流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117084259446.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
对输入的特征图使用全局平均池化与全局最大池化，分别输入到MLP中，将结果进行element-wise add，经过激活函数后输出channel attention module，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019111708454319.png)
$\delta$表示sigmoid激活函数，设$F$的大小为$C*H*W$，$W_0$为$\frac{C}{r}*C$的矩阵，$W_1$为$C*\frac{C}{r}$的矩阵，$M_c(F)$大小为$C$，即$F$的channel个数。


## spatial attention module
总体流程如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117085128209.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
沿着通道方向对特征图$F'$施加全局平均池化与全局最大池化，将$C*H*W$的特征图转变为$2*H*W$的特征图，什么是通道方向的全局池化呢？若特征图的大小为$C*H*W$，则池化层的大小为$C*1*1$，即可得到$1*H*W$的特征图。$2*H*W$的特征图后接一个7卷积层，卷积大小通过实验后确定为7*7，得到$1*H*W$的特征图，经过激活函数后输出spatial attention module，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117085801887.png)

## 如何结合spatial attention module与channel attention module
对原图施加channel attention module，即在通道方向将channel attention module广播为$C*H*W$大小的特征图后，与原特征图进行element-wise multiplication。
对原图施加spatial attention module，即将$1*H*W$的spatial attention module与$C*H*W$大小的原特征图集合中的每一张特征图进行element-wise multiplication。

我们有三个策略：

 1. 先对原特征图施加channel attention module 后 spatial attention module
 2. 先对原特征图施加spatial attention module 后 channel attention module
 3. 分别对原特征图施加spatial attention module 与 channel attention module，将两者进行element-wise add后用sigmoid函数激活后输出
 
 经过试验，发现第一个策略效果最佳，试验结果如下：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117090539862.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
策略一的图示如下，无需改变原神经网络原有的参数（由于Input Feature与Refined Feature大小一致）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117090646816.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
数学表示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117091339377.png)

# 实验
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117090831748.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
 数值上看，提升不大，个人认为无注意力机制的网络本身具有较好的focus目标的能力，因此从分类准确率上看不太出区别，但是使用grad-CAM可视化后，区别就出来了，如下：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20191117091026797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

颜色越深，表示神经网络越注意该区域，可以看到，含有注意力机制的网络注意到的目标相关区域更广，并且softmax输出的值也更大，这些特性是无法从分类准确率看出来的。
