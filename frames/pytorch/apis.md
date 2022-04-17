# Pytorch常用API汇总

张量创建
torch.tensor()
从data创建tensor

data: 数据, 可以是list, numpy
dtype : 数据类型，默认与data的一致
device : 所在设备, cuda/cpu
requires_grad：是否需要梯度
pin_memory：是否存于锁页内存
torch.zeros()
依size创建全0张量

size: 张量的形状, 如(3, 3)、(3, 224,224)
out : 输出的张量layout : 内存中布局形式, 有
strided,sparse_coo等
device : 所在设备, gpu/cpu
requires_grad：是否需要梯度
torch.zeros_like()
依input形状创建全0张量

intput: 创建与input同形状的全0张量
dtype : 数据类型
layout : 内存中布局形式
torch.ones()
torch.ones_like()
依input形状创建全1张量

size: 张量的形状, 如(3, 3)、(3, 224,224)
dtype : 数据类型
layout : 内存中布局形式
device : 所在设备, gpu/cpu
requires_grad：是否需要梯度
torch.full()
torch.full_like()
依input形状创建指定数据的张量

size: 张量的形状, 如(3, 3)
fill_value : 张量的值
torch.arange()
创建等差的1维张量，数值区间为[start, end)

start: 数列起始值
end : 数列“结束值”
step: 数列公差，默认为1
torch.linspace()
创建均分的1维张量，数值区间为[start, end]

start: 数列起始值
end : 数列结束值
steps: 数列长度
torch.logspace()
创建对数均分的1维张量，长度为steps, 底为base

start: 数列起始值
end : 数列结束值
steps: 数列长度
base : 对数函数的底，默认为10
torch.eye()
创建单位对角矩阵（ 2维张量），默认为方阵

n: 矩阵行数
m : 矩阵列数
torch.normal()
生成正态分布（高斯分布）

mean : 均值
std : 标准差
torch.randn()
torch.randn_like()
生成标准正态分布

size : 张量的形状
torch.rand()
torch.rand_like()
在区间[0, 1)上，生成均匀分布

torch.randint()
torch.randint_like()
区间[low, high)生成整数均匀分布

size : 张量的形状
在区间[0, 1)上，生成均匀分布

torch.randperm()
生成生成从0到n-1的随机排列

n : 张量的长度
torch.bernoulli()
以input为概率，生成伯努力分布（0-1分布，两点分布）

input : 概率值
张量操作
torch.topk()
找出前k大的数据，及其索引序号

input：张量
k：决定选取k个值
dim：索引维度
返回：

Tensor：前k大的值
LongTensor：前k大的值所在的位置
torch.cat()
将张量按维度dim进行拼接

tensors: 张量序列
dim : 要拼接的维度
torch.stack()
在新创建的维度dim上进行拼接

tensors:张量序列
dim :要拼接的维度
torch.chunk()
将张量按维度dim进行平均切分，返回值为张量列表，若不能整除，最后一份张量小于其他张量

input: 要切分的张量
chunks : 要切分的份数
dim : 要切分的维度
torch.split()
将张量按维度dim进行切分，返回值为张量列表

tensor: 要切分的张量
split_size_or_sections : 为int时，表示每一份的长度；为list时，按list元素切分
dim : 要切分的维度
torch.index_select()
在维度dim上，按index索引数据，返回值为依index索引数据拼接的张量

input: 要索引的张量
dim: 要索引的维度
index : 要索引数据的序号
torch.masked_select()
按mask中的True进行索引，返回值为一维张量

input: 要索引的张量
mask: 与input同形状的布尔类型张量
torch.reshape()
变换张量形状，当张量在内存中是连续时，新张量与input共享数据内存

input: 要变换的张量
shape: 新张量的形状
torch.transpose()
交换张量的两个维度

input: 要变换的张量
dim0: 要交换的维度
dim1: 要交换的维度
torch.t()
2维张量转置，对矩阵而言，等价于torch.transpose(input, 0, 1)

torch.squeeze()
压缩长度为1的维度（轴）

dim: 若为None，移除所有长度为1的轴；若指定维度，当且仅当该轴长度为1时，可以被移除；
torch.unsqueeze()
依据dim扩展维度

dim: 扩展的维度
张量数学运算
torch.add()
torch.addcdiv()
torch.addcmul()
torch.sub()
torch.div()
torch.mul()
torch.log(input, out=None)
torch.log10(input, out=None)
torch.log2(input, out=None)
torch.exp(input, out=None)
torch.pow()
torch.abs(input, out=None)
torch.acos(input, out=None)
torch.cosh(input, out=None)
torch.cos(input, out=None)
torch.asin(input, out=None)
torch.atan(input, out=None)
torch.atan2(input, other, out=None)
自动求导系统
梯度不自动清零
依赖于叶子结点的结点，requires_grad默认为True
叶子结点不可执行in-place
torch.autograd.backward
自动求取梯度

tensors: 用于求导的张量，如 loss
retain_graph : 保存计算图
create_graph : 创建导数计算图，用于高阶求导
grad_tensors：多梯度权重
torch.autograd.grad
求取梯度

outputs: 用于求导的张量，如 loss
inputs : 需要梯度的张量
create_graph : 创建导数计算图，用于高阶求导
retain_graph : 保存计算图
grad_outputs：多梯度权重
数据导入
Epoch: 所有训练样本都已输入到模型中，称为一个Epoch
Iteration：一批样本输入到模型中，称之为一个Iteration
Batchsize：批大小，决定一个Epoch有多少个Iteration
torch.utils.data.DataLoader
构建可迭代的数据装载器

dataset: Dataset类，决定数据从哪读取及如何读取
batchsize : 批大小
num_works: 是否多进程读取数据
shuffle: 每个epoch是否乱序
drop_last：当样本数不能被batchsize整除时，是否舍弃最后一批数据
torch.utils.data.Dataset
Dataset抽象类，所有自定义的Dataset需要继承它，并且复写_getitem__()

getitem :接收一个索引，返回一个样本

数据预处理
transforms.Normalize
逐channel的对图像进行标准化：output = (input - mean) / std

mean：各通道的均值
std：各通道的标准差
inplace：是否原地操作
裁剪
transforms.CenterCrop
从图像中心裁剪图片

size：所需裁剪图片尺寸
transforms.RandomCrop
从图片中随机裁剪出尺寸为size的图片

size：所需裁剪图片尺寸
padding：设置填充大小当为a时，上下左右均填充a个像素；当为(a, b)时，上下填充b个像素，左右填充a个像素；当为(a, b, c, d)时，左，上，右，下分别填充a, b, c, d
pad_if_need：若图像小于设定size，则填充
padding_mode：填充模式，有4种模式
1、constant：像素值由fill设定
2、edge：像素值由图像边缘像素决定
3、reflect：镜像填充，最后一个像素不镜像，eg：[1,2,3,4] → [3,2,1,2,3,4,3,2]
4、symmetric：镜像填充，最后一个像素镜像，eg：[1,2,3,4] → [2,1,1,2,3,4,4,3]
fill：constant时，设置填充的像素值
transforms.RandomResizedCrop
随机大小、长宽比裁剪图片

size：所需裁剪图片尺寸
scale：随机裁剪面积比例, 默认(0.08, 1)
ratio：随机长宽比，默认(3/4, 4/3)
interpolation：插值方法
FiveCrop and TenCrop
在图像的上下左右以及中心裁剪出尺寸为size的5张图片，TenCrop对这5张图片进行水平或者垂直镜像获得10张图片

size：所需裁剪图片尺寸
vertical_flip：是否垂直翻转
旋转和翻转
transforms.RandomHorizontalFlip
transforms.RandomVerticalFlip
transforms.RandomRotation
图像变换
transforms.Pad
transforms.ColorJitter
transforms.Grayscale
transforms.RandomGrayscale
transforms.RandomAffine
transforms.LinearTransformation
transforms.RandomErasing
transforms.Lambda
transforms.Resize
transforms.Totensor
transforms.Normalize
transform的操作
transforms.RandomChoice
transforms.RandomApply
transforms.RandomOrder
模型容器
nn.Sequential
nn.module的容器，用于按顺序包装一组网络层

顺序性：各网络层之间严格按照顺序构建
自带forward()：自带的forward里，通过for循环依次执行前向传播运算
nn.ModuleList
nn.module的容器，用于包装一组网络层，以迭代方式调用网络层

append()：在ModuleList后面添加网络层
extend()：拼接两个ModuleList
insert()：指定在ModuleList中位置插入网络层
nn.ModuleDict
nn.module的容器，用于包装一组网络层，以索引方式调用网络层

clear()：清空ModuleDict
items()：返回可迭代的键值对(key-value pairs)
keys()：返回字典的键(key)
values()：返回字典的值(value)
pop()：返回一对键值，并从字典中删除
容器总结
nn.Sequential：顺序性，各网络层之间严格按顺序执行，常用于block构建
nn.ModuleList：迭代性，常用于大量重复网构建，通过for循环实现重复构建
nn.ModuleDict：索引性，常用于可选择的网络层
模型构建
nn.Conv2d
对多个二维信号进行二维卷积

in_channels：输入通道数
out_channels：输出通道数，等价于卷积核个数
kernel_size：卷积核尺寸
stride：步长
padding ：填充个数
dilation：空洞卷积大小
groups：分组卷积设置
bias：偏置
nn.ConvTranspose2d
转置卷积实现上采样

in_channels：输入通道数
out_channels：输出通道数，等价于卷积核个数
kernel_size：卷积核尺寸
stride：步长
padding ：填充个数
dilation：空洞卷积大小
groups：分组卷积设置
bias：偏置
nn.MaxPool2d
对二维信号（图像）进行最大值池化

kernel_size：池化核尺寸
stride：步长
padding ：填充个数
dilation：池化核间隔大小
ceil_mode：尺寸向上取整
return_indices：记录池化像素索引
nn.AvgPool2d
对二维信号（图像）进行平均值池化

kernel_size：池化核尺寸
stride：步长
padding ：填充个数
ceil_mode：尺寸向上取整
count_include_pad：填充值用于计算
divisor_override ：除法因子
nn.MaxUnpool2d
：对二维信号（图像）进行最大值池化
上采样

kernel_size：池化核尺寸
stride：步长
padding ：填充个数
nn.Linear
对一维信号（向量）进行线性组合

in_features：输入结点数
out_features：输出结点数
bias ：是否需要偏置
nn.Sigmoid
nn.ReLU
nn.tanh
权值初始化
Xavier均匀分布
torch.nn.init.xavier_uniform_(tensor, gain=1)

Xavier正态分布
torch.nn.init.xavier_normal_(tensor, gain=1)

Kaiming均匀分布
torch.nn.init.kaiming_uniform_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)

Kaiming正态分布
torch.nn.init.kaiming_normal_(tensor, a=0, mode=‘fan_in’, nonlinearity=‘leaky_relu’)

均匀分布
torch.nn.init.uniform_(tensor, a=0, b=1)

正态分布
torch.nn.init.normal_(tensor, mean=0, std=1)

常数分布
torch.nn.init.constant_(tensor, val)

正交矩阵初始化
torch.nn.init.orthogonal_(tensor, gain=1)

单位矩阵初始化
torch.nn.init.eye_(tensor)

稀疏矩阵初始化
torch.nn.init.sparse_(tensor, sparsity, std=0.01)

损失函数
nn.CrossEntropyLoss()
l o s s i = w y i ( − x i + l o g ( ∑ j e x j ) ) loss_i=w_{y_i}(-x_i+log(\sum_je^{x_{j}}))
loss 
i

 =w 
y 
i

 

 (−x 
i

 +log( 
j
∑

 e 
x 
j


 ))

将nn.LogSoftmax()与nn.NLLLoss()结合，进行交叉熵计算

weight：各类别的loss设置权值，类别即标签，是向量形式，要对每个样本都设置权重，默认均为1；
ignore_index：忽略某个类别；
reduction：计算模式，可分为none:逐个元素计算，输出为各个input的loss；sum:none模式所有元素求和，返回标量；mean:none模式所有元素加权平均，返回标量，若设置了weight，则mean计算时，分母为weight的和；
nn.NLLLoss
l o s s i = − w y i ∗ x i loss_i=-w_{y_i}*x_i
loss 
i

 =−w 
y 
i

 

 ∗x 
i

 

实现负对数似然函数中的负号功能

weight
ignore_index
reduction
nn.BCELoss
l o s s i = − w y i [ y i ∗ l o g ( x i ) + ( 1 − y i ) ∗ l o g ( 1 − x i ) ] loss_i=-w_{y_i}[y_i*log(x_i)+(1-y_i)*log(1-x_i)]
loss 
i

 =−w 
y 
i

 

 [y 
i

 ∗log(x 
i

 )+(1−y 
i

 )∗log(1−x 
i

 )]

二分类交叉熵，输入值取值需要在[0,1]上，因为要符合概率取值，所以使用时，可以把输出值输入Sigmoid()后再计算loss。

weight
ignore_index
reduction
nn.BCEWithLogitsLoss
l o s s i = − w y i [ y n ∗ l o g ( S i g m o i d ( x i ) + ( 1 − y i ) ∗ l o g ( 1 − S i g m o i d ( x i ) ] loss_i=-w_{y_i}[y_n*log(Sigmoid(x_i)+(1-y_i)*log(1-Sigmoid(x_i)]
loss 
i

 =−w 
y 
i

 

 [y 
n

 ∗log(Sigmoid(x 
i

 )+(1−y 
i

 )∗log(1−Sigmoid(x 
i

 )]

结合Sigmoid与二分类交叉熵(因为有时不希望最后一层是Sigmoid，但是BCELoss又需要，故设置一个集成函数)

pos_weight：正样本的权值，顾名思义，标签为正数的权值；
weight
ignore_index
reduction
nn.L1Loss
L o s s i = ∣ x i − y i ∣ Loss_i=|x_i-y_i|
Loss 
i

 =∣x 
i

 −y 
i

 ∣

计算inputs与target之差的绝对值。

reduction
nn.MSELoss
L o s s i = ( x i − y i ) 2 Loss_i=(x_i-y_i)^2
Loss 
i

 =(x 
i

 −y 
i

 ) 
2


计算inputs与target之差的平方

reduction
nn.SmoothL1Loss
L o s s i = 1 n ∑ i z i z i = { 0.5 ∗ ( x i − y i ) 2   i f   ∣ x i − y i ∣ < 1 ∣ x i − y i ∣ − 0.5   o t h e r w i s e \end{aligned}


Loss 
i

 = 
n
1


i
∑

 z 
i


z 
i

 = 
{ 
0.5∗(x 
i

 −y 
i

 ) 
2

∣x 
i

 −y 
i

 ∣−0.5 


if ∣x 
i

 −y 
i

 ∣<1
otherwise

 

 

 

平滑的L1Loss，减低离群点带来的影响

reduction
nn.PoissonNLLLoss
l o s s i = { e x i − y i ∗ x i   l o g _ i n p u t = T r u e x i − y i ∗ l o g ( x i + e p s )   l o g _ i n p u t = F a l s e
loss 
i

 ={ 
e 
x 
i


 −y 
i

 ∗x 
i


x 
i

 −y 
i

 ∗log(x 
i

 +eps) 


log_input=True
log_input=False

 

 

泊松分布的负对数似然函数损失函数

log_input：输入是否为对数形式，决定计算公式；
full：计算所有loss，默认为False；
eps：修正项：e p s = 1 0 − 8 eps=10^{-8}eps=10 
−8
 ，避免log(input)为nan；
nn.KLDivLoss
D K L ( P ∣ ∣ Q ) = E x ∼ p [ l o g ( P ( x ) Q ( x ) ) ] = E x ∼ p [ l o g ( P ( x ) ) − l o g ( Q ( x ) ) ] = ∑ i = 1 N ( l o g ( P ( x i ) ) − l o g ( Q ( x i ) ) ) 其 中 P 为 真 实 分 布 ， 即 标 签 ； Q 为 网 络 输 出 ， 即 使 Q 逼 近 P 。 L o s s i = y i ∗ ( l o g ( y i ) − x i ) \\ 其中P为真实分布，即标签；Q为网络输出，即使Q逼近P。\\ Loss_i=y_i*(log(y_i)-x_i)
D 
KL

 (P∣∣Q)


=E 
x∼p

 [log( 
Q(x)
P(x)

 )]
=E 
x∼p

 [log(P(x))−log(Q(x))]
= 
i=1
∑
N

 (log(P(x 
i

 ))−log(Q(x 
i

 )))


其中P为真实分布，即标签；Q为网络输出，即使Q逼近P。
Loss 
i

 =y 
i

 ∗(log(y 
i

 )−x 
i

 )

计算KL散度，即相对熵。因为KLDivLoss计算时直接使用x i x_ix 
i

 而KL散度公式使用的是log，故使用时需要提前将输入计算log-probabilities，可以通过torch.log(x i x_ix 
i

 )实现。

reduction：额外有batchmean参数，以batchsize为维度计算mean的值；
nn.MarginRankingLoss
L o s s i = m a x ( 0 , − y i ∗ ( x 1 i − x 2 i ) + m a r g i n ) Loss_i=max(0,-y_i*(x_{1_i}-x_{2_i})+margin)
Loss 
i

 =max(0,−y 
i

 ∗(x 
1 
i

 

 −x 
2 
i

 

 )+margin)

计算两个向量之间的相似度，用于排序任务，该方法计算两组数据之间的差异，返回一个n*n的loss矩阵。如公式，当y=1时，希望x 1 x_1x 
1

 比x 2 x_2x 
2

 大，则不产生loss，当y=-1时相反。

margin：边界值，x1与x2的差异值，默认为0；
reduction
nn.MultiLabelMarginLoss
L o s s ( x , y ) = ∑ i j m a x ( 0 , 1 − ( x y j − x i ) ) x . s i z e x y j 指 标 签 在 的 神 经 元 的 值 x i 指 标 签 不 在 的 神 经 元 的 值 Loss(x,y)=\sum_{ij}\frac{max(0,1-(x_{y_j}-x_i))}{x.size}\\ x_{y_j}指标签在的神经元的值\\ x_i指标签不在的神经元的值
Loss(x,y)= 
ij
∑


x.size
max(0,1−(x 
y 
j

 

 −x 
i

 ))


x 
y 
j

 

 指标签在的神经元的值
x 
i

 指标签不在的神经元的值

多标签边界损失函数。多标签不是多分类，多分类对应二分类，多标签指的是一张图对应多个类别，如一张图包含云，草地，树等等。
标签使用类别的名称，如0，1，2，3等

reduction
nn.SoftMarginLoss
L o s s ( x , y ) = ∑ i l o g ( 1 + e − y i ∗ x i ) x . n e l e m e n t ( ) x . n e l e m e n t ( ) 为 张 量 中 元 素 个 数 Loss(x,y)=\sum_i\frac{log(1+e^{-y_i*x_i})}{x.nelement()}\\ x.nelement()为张量中元素个数
Loss(x,y)= 
i
∑


x.nelement()
log(1+e 
−y 
i

 ∗x 
i


 )


x.nelement()为张量中元素个数

用于计算二分类的logistic损失。

reduction
nn.MultiLabelSoftMarginLoss
L o s s ( x , y ) = − 1 C ∗ ∑ i y i ∗ l o g ( 1 1 + e − x i ) + ( 1 − y i ) ∗ l o g ( e − x i 1 + e − x i ) Loss(x,y)=-\frac{1}{C}*\sum_iy_i*log(\frac{1}{1+e^{-x_i}})+(1-y_i)*log(\frac{e^{-x_i}}{1+e^{-x_i}})
Loss(x,y)=− 
C
1

 ∗ 
i
∑

 y 
i

 ∗log( 
1+e 
−x 
i

 

1

 )+(1−y 
i

 )∗log( 
1+e 
−x 
i

 

e 
−x 
i

 


 )

SoftMarginLoss的多标签版本。此时标签为0101形式

weight
reduction
nn.MultiMarginLoss
L o s s ( x , y ) = ∑ i m a x ( 0 , m a r g i n − x y + x i ) p x . s i z e ( ) Loss(x,y)=\frac{\sum_imax(0,margin-x_y+x_i)^p}{x.size()}
Loss(x,y)= 
x.size()
∑ 
i

 max(0,margin−x 
y

 +x 
i

 ) 
p




计算多分类的折页损失。x y x_yx 
y

 为标签所在位置的值，x i x_ix 
i

 为非标签位置的值。

p：可选1或2
weight
margin
reduction
nn.TripletMarginLoss
L ( a , p , n ) = m a x { d ( a i , p i ) − d ( a i , n i ) + m a r g i n , 0 } d ( x i , y i ) = ∣ ∣ x i − x i ∣ ∣ p a : a n c h o r ;   n : n e g a t i v e ;   p : p o s i t i v e a n c h o r 越 靠 近 p o s i t i v e 则 l o s s 越 小 L(a,p,n)=max\{d(a_i,p_i)-d(a_i,n_i)+margin,0\}\\ d(x_i,y_i)=||x_i-x_i||_p\\ a:anchor;\ n:negative;\ p:positive\\ anchor越靠近positive则loss越小
L(a,p,n)=max{d(a 
i

 ,p 
i

 )−d(a 
i

 ,n 
i

 )+margin,0}
d(x 
i

 ,y 
i

 )=∣∣x 
i

 −x 
i

 ∣∣ 
p


a:anchor; n:negative; p:positive
anchor越靠近positive则loss越小

计算三元组损失，人脸验证中常用。

p：范数的阶
margin：
reduction：
nn.HingeEmbeddingLoss
L o s s i = { x i   i f   y i = 1 m a x ( 0 , Δ − x i )   i f   y i = − 1
Loss 
i

 ={ 
x 
i


max(0,Δ−x 
i

 ) 


if y 
i

 =1
if y 
i

 =−1

 

 

计算两个输入的相似性，常用于非线性embedding和半监督学习，注意输入x应为两个输入之差的绝对值。

margin
reduction
nn.CosineEmbeddingLoss
L o s s ( x , y ) = { 1 − c o s ( x 1 , x 2 )   i f   y = 1 m a x ( 0 , c o s ( x 1 , x 2 ) − m a r g i n )   i f   y = − 1
Loss(x,y)={ 
1−cos(x 
1

 ,x 
2

 ) 
max(0,cos(x 
1

 ,x 
2

 )−margin) 


if y=1
if y=−1

 

 

采用余弦相似度计算连个输入的相似性，为了更加注重两个输入方向上的差异，而不是距离长度的差异。

margin
reduction
nn.CTCLoss
计算CTC损失，解决时序类数据的分类
Connectionlist Temporal Classification

优化器
Optimizer
defaults：优化器超参数
state：参数的缓存，如momentum缓存
param_groups：管理的参数组
_step_count：记录更新次数，学习率调整中使用
.zero_grad()
清空所管理参数的梯度(张量梯度不自动清零，而是累加)

.step()
执行一步更新，更新了一次权值参数

.add_param_group()
添加参数组

.state_dick() or .load_state_dick
获取优化器当前状态，加载状态信息字典，用于模型断点续训练

optim.SGD
随机梯度下降法

params：管理的参数组
lr：初始学习率
monentum：动量系数，β
weight_decay：L2正则化系数
nesterov：是否采用NAG
其他优化器
optim.Adagrad：自适应学习率梯度下降法
optim.RMSprop：Adagrad的改进
optim.Adadelta：Adagrad的改进
optim.Adam：RMSprop结合Momentum(常用)
optim.Adamax：Adam增加学习率上限
optim.SparseAdam：稀疏版的Adam
optim.ASGD：随机平均梯度下降
optim.Rprop：弹性反向传播
optim.LBFGS：BFGS的改进
学习率调整策略
学习率初始化方式

设置较小数：0.01，0.001，0.0001等
搜索最大学习率：《Cyclical Learning Rate for Training Neural Networks》
class _LRScheduler
optimizer：关联的优化器
last_epoch：记录的epoch数
base_lrs：记录初始学习率
.step()
更新下一个epoch的学习率

.get_lr()
虚函数，用于orrived，计算下一个epoch的学习率

StepLR
l r = l r ∗ g a m m a lr=lr*gamma
lr=lr∗gamma

等间隔调整学习率

step_size：调整间隔数，单位为epoch
gamma：调整系数
MultiStepLR
按给定间隔调整学习率

milestones：给定的调整时刻
gamma
ExponentiaILR
l r = l r ∗ g a m m a e p o c h lr=lr*gamma^{epoch}
lr=lr∗gamma 
epoch


按照指数衰减调整学习率

gamma：指数的底
CosineAnnealingLr
l r t = l r m i n + 1 2 ( l r m a x − l r m i n ) ( 1 + c o s ( T c u r T m a x ∗ π ) ) lr_t=lr_{min}+\frac{1}{2}(lr_{max}-lr_{min})(1+cos(\frac{T_{cur}}{T_{max}}*\pi))
lr 
t

 =lr 
min

 + 
2
1

 (lr 
max

 −lr 
min

 )(1+cos( 
T 
max


T 
cur

 

 ∗π))

余弦周期调整学习率

T_max：下降周期，即余弦函数半周期
eta_min：学习率下限
ReduceLROnPlateau
监控指标，当指标不再变化则调整

mode：min/max两种模式，不下降就调整还是不上升就调整
factor：调整系数与gamma作用相同
patience：“耐心”，接受几次不变化，要连续，默认值为10
cooldown：“冷却”，停止监控一段时间
verbose：是否打印标记
min_lr：学习率下限
eps：学习率衰减最小值
LambdaLR
自定义调整策略

lr_lambda：一个函数或者一个list，即自定义的调整策略
可视化
TensorBoard
SummaryWriter
提供创建event file的高级接口，event file即需要可视化的文件

log_dir：event file输出文件夹
comment：不指定log_dir时，文件夹后缀
filename_suffix：文件名后缀
.add_scalar()
记录标量

tag：图像的标签名，图的唯一标识
scalar_value：要记录的标量
global_step：x轴
.add_scalars()
main_tag：该图的标签，对应tag
tag_scalar_dict：key是变量的tag，value是变量的值
add_histogram()
统计直方图与多分位数折线图

tag：图像的标签名，图的唯一标识
values：要统计的参数
global_step：y轴
bins：取直方图的bins，默认是"tensorflow"
.add_image()
记录图像

tag：图像的标签名，图的唯一标识
img_tensor：图像数据，注意尺度
global_step：x轴
dataformats：数据形式，CHW，HWC，HW
add_graph()
可视化模型计算图

model：模型，必须是 nn.Module
input_to_model：输出给模型的数据
verbose：是否打印计算图结构信息
torchvision.utils.make_grid
制作网格图像

tensor：图像数据, BCH*W形式
nrow：行数（列数自动计算）
padding：图像间距（像素单位）
normalize：是否将像素值标准化
range：标准化范围
scale_each：是否单张图维度标准化
pad_value：padding的像素值
torchsummary
查看模型信息，便于调试

model：pytorch模型
input_size：模型输入size
batch_size：batch size
device：“cuda” or “cpu”
Hook
不改变主体，实现额外的功能。因为pytorch是动态图机制，计算时有些数据会被丢弃，使用Hook可以在计算过程中外挂一个函数，实现额外的功能。

Tensor.register_hook
注册一个反向传播hook函数，仅一个输入参数，为张量的梯度

Module.register_forward_hook
注册module的前向传播hook函数

module: 当前网络层
input：当前网络层输入数据
output：当前网络层输出数据
Module.register_forward_pre_hook
注册module前向传播前的hook函数

module: 当前网络层
input：当前网络层输入数据
Module.register_backward_hook
注册module反向传播的hook函数

module: 当前网络层
grad_input：当前网络层输入梯度数据
grad_output：当前网络层输出梯度数据
CAM
CAM与Grad-CAM，Grad-CAM为CAM的改进版

正则化
Batch Normalization
_BatchNorm
pytorch中的Batch Normalization实现

num_features：一个样本特征数量(最重要)
eps：分母修正项
momentum：指数加权平均估计当前mean/var
affine：是否需要affine transform，默认为True
track_running_stats：是训练状态，还是测试状态
以下三个方法具体的实现都继承于_BatchNorm

nn.BatchNorm1d
nn.BatchNorm2d
nn.BatchNorm3d
running_mean：均值
running_var：方差
weight：affine transform中的gamma
bias： affine transform中的beta
均值和方差根据训练与测试的不同，拥有不同的计算式。
在训练时：不止考虑当前时刻，还会考虑之前的结果，计算式如下：
m e a n r u n n i n g = ( 1 − m o m e n t u m ) ∗ m e a n p r e + m o m e n t u r m ∗ m e a n t v a r r u n n i n g = ( 1 − m o m e n t u m ) ∗ v a r p r e + m o m e n t u m ∗ v a r t mean_{running} = (1 - momentum) * mean_{pre} + momenturm * mean_t\\ var_{running} = (1 - momentum) * var_{pre} + momentum * var_t
mean 
running

 =(1−momentum)∗mean 
pre

 +momenturm∗mean 
t


var 
running

 =(1−momentum)∗var 
pre

 +momentum∗var 
t

 

Layer Normalization
起因：BN不适用于变长的网络，如RNN
思路：逐层计算均值和方差

不再有m e a n r u n n i n g mean_{running}mean 
running

 和v a r r u n n i n g var_{running}var 
running


gamma和beta为逐元素的
nn.LayerNorm
normalized_shape：该层特征形状
eps：分母修正项
elementwise_affine：是否需要affine
transform
instance Normalization
起因：BN在图像生成（Image Generation）中不适用
思路：逐Instance（channel）计算均值和方差

nn.InstanceNorm
num_features：一个样本特征数量（最重要）
eps：分母修正项
momentum：指数加权平均估计当前mean/var
affine：是否需要affine transform
track_running_stats：是训练状态，还是测试状
态
Group Normalization
起因：小batch样本中，BN估计的值不准
思路：数据不够，通道来凑
应用场景：大模型（小batch size）任务

不再有running_mean和running_var
gamma和beta为逐通道（channel）的
nn.GroupNorm
num_groups：分组数
num_channels：通道数（特征数）
eps：分母修正项
affine：是否需要affine transform
Dropout
注意：数据尺度变化：测试时，所有权重乘以1-drop_prob

nn.Dropout
p：被舍弃概率，失活概率
注意：在Pytorch中为了测试更加方便，在训练时权重已经预先乘以1 1 − p \frac{1}{1-p} 
1−p
1

 ，故如果使用Pytorch中的nn.Dropout，则测试时，不需要再乘上（1-p）

模型保存与加载
torch.save
obj：对象
f：输出路径
法1: 保存整个Module
torch.save(net, path)
法2: 保存模型参数(推荐，占用空间小，加载快)
state_dict = net.state_dict()
torch.save(state_dict , path)

torch.load
f：文件路径
map_location
GPU运算
torch.cuda
torch.cuda.device_count()：计算当前可见可用gpu数
torch.cuda.get_device_name()：获取gpu名称
torch.cuda.manual_seed()：为当前gpu设置随机种子
torch.cuda.manual_seed_all()：为所有可见可用gpu设置随机种子
torch.cuda.set_device()：设置主gpu为哪一个物理gpu（不推荐）
os.environ.setdefault(“CUDA_VISIBLE_DEVICES”, “2, 3”)（推荐）

## 参考

1. [Pytorch常用API汇总](https://blog.csdn.net/qq_49134563/article/details/108200828)