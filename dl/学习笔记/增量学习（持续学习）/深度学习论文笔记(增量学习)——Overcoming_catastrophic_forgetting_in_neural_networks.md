@[toc]

越来越懒，看的文献越来越多，做的总结越来越少，大概要写十几篇总结，寒假不知道写得完不.......

# 主要工作
该文章由deepmind在2016年出品，其将抵抗灾难性遗忘的工作分为三步
1、选择对于旧任务而言，较为重要的权重
2、在第一步的基础上，对权重的重要性做一个排序
3、在学习新任务时，尽量使步骤二的权重不发生太大改变，即在损失函数中添加了一个正则项，重要性大的神经元权重变化大，则释加的惩罚也大

上述算法简称为EWC（elastic weight consolidation）

本文数学公式较多

<br>

# motivation
在任务A上训练完毕后，得到模型，接着在任务B上训练，此时权重会发生很大改变，导致模型在任务A上的准确率下降，这便是灾难性遗忘（其实灾难性遗忘感觉像是从迁移学习中衍生出来的一个领域）

大脑的某些突触对于习得的知识巩固尤为重要，大脑会降低此类突触的可塑性，从而记忆过去习得的知识，以此为出发点，作者尝试在人工神经网络中识别对旧任务而言较为重要的神经元，并降低其权重在之后的任务训练中的改变程度

值得注意的是，识别出较为重要的神经元后，需要更进一步的给出各个神经元对于旧任务而言的重要性排序

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020011409484868.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

如上图，首先假设task A与task B存在一个公共解，这个解应该是存在的（我们将task A与task B合在一起训练，得到的解便是公共解），如果我们不加限制，就和蓝箭头一样，新训练的模型在task A上的准确率不足，如果我们对每个权重施加L2惩罚，会出现绿箭头的情况，因为所有权重都不能发生太大改变，模型无法充分学习task B，EWC的目标即为红箭头，此类做法和knowledge distillation的本质基本一致，只是做法不同

<br>

# method

给定数据集$D$，我们的目标是找出一个最有可能出现的解$\theta$，即目标为
$$log\ P(\theta|D)\tag{1.0}$$

此类目标和我们常用的极大似然估计不一致，其实这么理解也是可行的，对1.0进行变化，则有

$$\begin{aligned}
log\ P(\theta|D)&=log \frac{P(\theta D)}{P(D)}\\
&=log \frac{P(D|\theta)P(\theta)}{P(D)}\\
&=log\ P(D|\theta)+log\ P(\theta)-log\ P(D) \tag{1.1}
\end{aligned}
$$

假设$D$由task A与task B的数据集$D_A、D_B$组成，则有

$$\begin{aligned}
log\ P(\theta|D)=&log\ P(\theta|D_AD_B)\\
=&log\ \frac{P(\theta D_A D_B)}{P(D_AD_B)}\\
=&log\ \frac{P(\theta D_B|D_A)P(D_A)}{P(D_B|D_A)P(D_A)}\\
=&log\ \frac{P(\theta D_B|D_A)}{P(D_B|D_A)}\\
=&log\ \frac{P(D_B|\theta D_A)P(\theta|D_A)}{P(D_B|D_A)}\\
=&log\ P(D_B|\theta)+\log P(\theta|D_A)-log P(D_B)
\end{aligned}\tag {1.2}
$$
由于$D_A$与$D_B$相互独立，则有
$$\begin{aligned}
P(D_B|\theta D_A)&=P(D_B|\theta)\\
P(D_B|D_A)&=P(D_B)
\end{aligned}
$$

我也不知道论文给出式1.1的意图，式1.2是全文的核心

$log\ P(D_B|\theta)$可用task B上的损失函数的负数代替，即$-L_B(\theta)$，对于单标签分类任务而言，交叉熵损失函数的形式即为$log\ P(D_B|\theta)$

假设我们的模型有n个参数，第$i$个参数记为$\theta_i$，n个参数相对于$D_A$条件独立，则有
$$\begin{aligned}
log\ P(\theta|D_A)&=log\ P(\theta_1\theta_2...\theta_n|D_A)\\
&=log\ P(\theta_1|D_A)+log\ P(\theta_2|D_A)+...+log\ P(\theta_n|D_A)
\end{aligned}
$$

$log\ P(\theta_i|D_A)$给出了给定数据集$D_A$，哪个参数值出现的概率，如果某个参数值出现的概率大，则可认为该参数对于task A而言较为重要，但是估计$log\ P(\theta_i|D_A)$的值是非常困难的，假设task A训练完毕后，对应的参数记为$\theta_{A}^*$，利用拉普拉斯近似，在使用mini batch的前提下，则可将$log\ P(\theta_i|D_A)$看成均值为$\theta_{A,i}^*$，方差为该参数对应的Fisher information的倒数（记为$F_i$）的高斯分布，式1.2可记为：

$$log\ P(\theta|D)=-L_B(\theta)-\frac{1}{2}\sum_{i=1}^nF_i(\theta_i-\theta_{A,i}^*)^2\tag{1.3}$$

对于高斯分布而言，有
$$
log\ \frac{1}{\sqrt{2\pi}}e^{-\frac{(w-u)^2}{2\delta^2}}=log\ \frac{1}{\sqrt{2\pi}}-\frac{(w-u)^2}{2\delta^2}
$$
常数对优化结果没有影响，所以式1.3省去了常数，优化目标为
$$
\begin{aligned}
\max_{\theta}log\ P(\theta|D)=&\max_{\theta}(-L_B(\theta)-\frac{1}{2}\sum_{i=1}^nF_i(\theta_i-\theta_{A,i}^*)^2)
\end{aligned}
$$

即
$$
\min_{\theta}(L_B(\theta)+\frac{1}{2}\sum_{i=1}^nF_i(\theta_i-\theta_{A,i}^*)^2))
$$

如果把第二项看成是正则项，添加控制系数$\lambda$，则优化目标为
$$
\min_{\theta}(L_B(\theta)+\frac{\lambda}{2}\sum_{i=1}^nF_i(\theta_i-\theta_{A,i}^*)^2))\tag{1.4}
$$

<br>

## 什么是拉普拉斯近似
问题：给定数据集$X$，求模型的参数$W$出现的概率，即$P(W|X)$，此处讲解一维拉普拉斯近似

拉普拉斯近似的定义：假设分布$P(X)=\frac{1}{z}P(Y)$，$z$为常数，拉普拉斯近似的目的在于求解一个高斯分布$P(Y)$来近似$P(X)$

参考文献：https://www.cnblogs.com/hapjin/p/8848480.html

求解：

依据条件概率公式，我们有
$$
P(W|X)=\frac{P(X|W)P(W)}{P(X)}
$$

由于我们无法直接求解$P(W|X)$，而$P(X)$可以看成是常量，假设参数$W$符合均匀分布，因此，我们可以通过求解$P(X|W)$间接求得$P(W|X)$，此处满足拉普拉斯近似的定义，设$P(X|W)$服从均值为$u$，方差为$\delta$的高斯分布，$X$已知，$W$为变量，则有

$$
log\ P(X|W)=log\ \frac{1}{\sqrt{2\pi}}-\frac{(w-u)^2}{2\delta^2} \tag{2.0}
$$


设$w^*$使得$P(X|W)$的一阶偏导为0，二阶偏导小于0，即$P(X|W)$在该点具有极大值。$log\ P(X|W)$的一阶偏导为0，由式2.0的形式可知，$w^*$一定存在，设

$$
f'(W)=\frac{\partial log\ P(X|W)}{\partial W}\\
f''(W)=\frac{\partial^2 log\ P(X|W)}{\partial W^2}
$$

则有$f'(w^*)=0$，在$W^*$处对$log\ P(X|W)$进行二阶泰勒展开得：

$$
\begin{aligned}
log\ P(X|W) \approx & log\ P(X|W^*)+f'(w^*)+f''(W^*) \frac{(w-w^*)^2}{2}\\
\approx & log\ P(X|W^*)+f''(w^*) \frac{(w-w^*)^2}{2} \tag{2.1}
\end{aligned}
$$

比对2.0与2.1，可得
$$
\begin{aligned}
u&=w^*\\
\delta^2&=-\frac{1}{f''(w^*)}
\end{aligned}
$$

式2.0得到求解，由此可得$P(W|X)$

回到EWC，在task A上依据极大似然估计求得参数$\theta_A^*$满足
$$
\begin{aligned}
log\ P(D_A|\theta_A^*)&=log\ \frac{P(D_A\theta_A^*)}{P(\theta_A^*)}\\
&=log\ P(D_A\theta_{A,1}^*\theta_{A,2}^*...\theta_{A,n}^*)-log\ P(\theta_{A,1}^*\theta_{A,2}^*...\theta_{A,n}^*)\\
&=log\ P(\theta_{A,1}^*\theta_{A,2}^*...\theta_{A,n}^*|D_A)+log\ P(D_A)-log\ P(\theta_{A,1}^*\theta_{A,2}^*...\theta_{A,n}^*)\tag{2.2}
\end{aligned}
$$

各参数相互独立且对$D_A$满足条件独立，式2.2可变为：
$$
log\ P(\theta_{A,1}^*|D_A)+log\ P(\theta_{A,2}^*|D_A)+...+log\ P(\theta_{A,n}^*|D_A)+log\ P(D_A)-\sum_{i=1}^nlog\ P(\theta_{A,i}^*)\tag{2.3}
$$

假设各参数满足均匀分布，则$log\ P(D_A)-\sum_{i=1}^nlog\ P(\theta_{A,i}^*)$为常数，依据式2.3，$log\ P(\theta_{A,i}^*|D_A)$为极大值，因此task A上优化的参数可作为拉普拉斯近似的均值。

<br>

## 什么是Fisher information
参考文献：https://wiseodd.github.io/techblog/2018/03/11/fisher-information/（需翻墙）

不敲了，好累，直接贴图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200118202312452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200118202403783.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200118202409497.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
最重要的是最后一个结论，

$$
F=-E_P(x|\theta)[H_{log\ P(x|\theta)}]
$$

对于一维函数而言，$H_{log\ P(x|\theta)}=(log\ P(x|\theta))''$，假设我们有$x_1,x_2,...,x_n$，n个采样点，则$$F=-E_P(x|\theta)[H_{log\ P(x|\theta)}]=-\frac{1}{n}\sum_{i=1}^{n}(log\ P(x_i|\theta))''$$

在mini-batch的前提下，假设有$m$个样本，记为$x_1,x_2,...,x_m$，则EWC的目标函数为
$$
\begin{aligned}
L_B(\theta)-\frac{\lambda}{2}\sum_{i=1}^n\sum_{j=1}^m(log\ P(x_j|\theta_i))''(\theta_i-\theta_{A,i}^*)^2=L_B(\theta)+\frac{\lambda}{2}\sum_{i=1}^nF_i(\theta_i-\theta_{A,i}^*)^2
\end{aligned}
$$
常数$\frac{1}{m}$可以从$\lambda$中抽取，之所以要强制转成Fisher information，是因为Fisher information是一个一阶导数，计算资源可以得到节约，Fisher information计算公式如下：

$$F_j=\frac{1}{m}\sum_{i=1}^m(\frac{\partial log\ P(x_i|\theta_j)}{\partial \theta_j})^2$$

论文很短，但是公式推导省略了一堆，后面的公式较为混乱，有错误亦或是不懂可在评论区指出

