@[toc]

# 问题 

Bradley发现反向传播的梯度从输出层往输入层逐层递减，通过研究以线性函数为激活函数的深度网络，Bradley发现从输出层往输入层方向，反向传播梯度的方差逐级递减，可以理解为出现了梯度消失，为了解决梯度消失，Xavier Glorot, Yoshua Bengio提出了xavier初始化

<br>

# 解决方案

<br>

## 问题突破口

从Bradley的研究出发，可以得出反向传播梯度的方差逐级递减将导致梯度消失，一个很自然的想法便是，能否让每层反向传播梯度的方差一致？这便是Xavier初始化的出发点

<br>

## Xavier初始化

<br>

### 问题假设
- 反向传播同一层神经元的梯度独立同分布
- 同一层神经元的权重独立同分布
- 同一层神经元的输入与权重相互独立
- 初始时，激活函数的输出近似y=x，对于sigmoid、tanh函数而言，即初始时刻，激活函数的输入位于0附近
- 前向传播同一层神经元的输入独立同分布

除去最后两点，其余假设是我自己加进去的，论文中并未说明

<br>

### 符号表
|符号|含义  |
|--|--|
| $cost$ | 损失函数值  |
|$Z^i$| 第$i$层输入|
|$W^i$|第$i$层权重|
|$b^i$|第$i$层偏移|
|$s^i$|第$i$层输出|
|$n^i$|第$i$层神经元个数|

若$x$表示$\begin{bmatrix}
x_1\\
x_2\\
....\\
x_n
\end{bmatrix}$，第i层的激活函数向量$f^i(x)$表示为$\begin{bmatrix}
f(x_1)\\
f(x_2)\\
....\\
f(x_n)
\end{bmatrix}$，$f(x)$为激活函数，$(f^i(x))'$表示为$\begin{bmatrix}
\frac{\partial{f(x_1)}}{{\partial x_1}}\\
\frac{\partial{f(x_2)}}{{\partial x_2}}\\
....\\
\frac{\partial{f(x_n)}}{{\partial x_n}}
\end{bmatrix}$

<br>

### 推导

<br>

#### 前向传播
基于假设，对于前向传播，我们有
$$Z^{i+1}=f(W^{i+1}Z^i+b^i)=W^{i+1}Z^i+b^i$$
$z^{i+1}_j$表示向量$Z^{i+1}$第$j$个维度的值，$w^{i+1}_{dj}$表示矩阵$W^{i+1}$第$i$行第$j$列的值，由于前向传播同一层神经元的输入同分布，即有
$$
Var(Z^{i+1})=Var(Z^{i+1}_1)=Var(Z^{i+1}_2)=.....=Var(Z^{i+1}_{n^i})
$$

而$Z^{i+1}_1=w^{i+1}_{11}z_1^i+w^{i+1}_{12}z_2^i+......+w^{i+1}_{1{n^i}}z_{n^i}^i$，由于同一层神经元的输入与权重相互独立，同一层神经元的权重同分布，则有
$$
\begin{aligned}
Var(Z^{i+1})&=Var(Z_1^{i+1})\\
&=Var(w^{i+1}_{11}z_1^i)+Var(w^{i+1}_{12}z_2^i)+......+Var(w^{i+1}_{1{n^i}}z_{n^i}^i)\\
&=Var(w^{i+1}_{11})Var(z_1^i)+Var(w^{i+1}_{12})Var(z_2^i)+......+Var(w^{i+1}_{1{n^i}})Var(z_{n^i}^i)\\
&=n_{i}Var(W^{i+1})Var(Z^i)
\end{aligned}
$$

因此，我们有
$$
\begin{aligned}
Var(Z^{i+1})=n_{i}Var(W^{i+1})Var(Z^i)
\end{aligned}\tag{式1}
$$

<br>

#### 反向传播

反向传播时，由于激活函数的输出近似y=x，每层的梯度为
$$
\begin{aligned}
\frac{\partial cost}{\partial Z^i}
&=(W^{i+1})^T*\frac{\partial cost}{\partial Z^{i+1}}  ☉ (f^{i}(S_{in}^i))'\\
&=(W^{i+1})^T*\frac{\partial cost}{\partial Z^{i+1}} 
\end{aligned}
$$
由于反向传播同一层神经元的梯度同分布并且同一层神经元的权重同分布，和前向传播一样，我们有
$$
\begin{aligned}
Var(\frac{\partial cost}{\partial Z^i})=n^{i+1}Var(W^{i+1})Var(\frac{\partial cost}{\partial Z^{i+1}} )
\end{aligned}\tag{式2}
$$

<br>

#### Xavier初始化
对于前向传播，为了让信息从底层网络流向高层网络，我们往往希望
$$Var(Z^{i+1})=Var(Z^i)$$
为什么要保证底层网络信息能很好的流向高层网络呢？论文并没有做出解答，但是实践表明这么做往往是有利的。

对于反向传播，为了不让梯度的方差逐渐变小，我们往往希望
$$Var(\frac{\partial cost}{\partial Z^i})=Var(\frac{\partial cost}{\partial Z^{i+1}} )$$
因此对于式1、2，我们希望
$$
\begin{aligned}
n^{i+1}Var(W^{i+1})&=1\\
n^{i}Var(W^{i+1})&=1
\end{aligned}
$$
一般情况下，权重的方差无法同时满足上述两个等式，由于相邻两层的神经元个数一般相差不大，因此我们有
$$
Var(W^{i+1})=\frac{2}{n^i+n^{i+1}}
$$
满足上述方差条件的分布有很多，xavier建议使用均匀分布，因此，权重初始化分布为$W$ ~ $U[-\frac{\sqrt6}{n^i+n^{i+1}}，\frac{\sqrt6}{n^i+n^{i+1}}]$

<br>

# 实验结果
论文对xavier初始化与标准初始化$W$ ~ $U[-\frac{1}{\sqrt {n}}，\frac{1}{\sqrt {n}}]$(n为前一层神经元的个数)进行了对比，对于标准初始化，假设每层的神经元个数均为n，可知
$$
Var(W^{i+1})=\frac{1}{3n}
$$
将其代入式2，可得
$$\begin{aligned}
Var(\frac{\partial cost}{\partial Z^i})=\frac{1}{3}Var(\frac{\partial cost}{\partial Z^{i+1}} )
\end{aligned}\tag{式2}$$
因此，使用标准初始化的BP神经网络必然会出现梯度消失，对于使用标准初始化的前向传播而言，则有
$$
\begin{aligned}
Var(Z^{i+1})=\frac{1}{3}Var(Z^i)
\end{aligned}\tag{式1}
$$
因此，从输入层到输出层，使用标准初始化的BP神经网络的激活函数方差逐渐下降

为了验证上述分析，论文给出了训练的初始阶段，两种初始化方法对应的激活函数值、反向传播梯度值的归一化直方图，如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190818185536163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190818185604423.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
可见反向传播梯度以及激活函数值基本满足标准正态分布，而方差越小，标准正态分布就越瘦高，上述情况验证了理论分析，由于反向传播梯度的方差一致，也就不会出现梯度消失的情况

<br>

# Xavier初始化的思考
Xavier是否完全解决了梯度消失的问题？显然不是，随着参数的跟新，Xavier的假设将不再成立，权重与输入值将不再独立，激活函数也将不会呈现线性状态，只能说在训练初期，Xavier可以很好的抵抗梯度消失
