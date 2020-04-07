
@[toc]
# pytorch使用CNN总结

## 安装pytorch
如果cuda版本不对，是无法使用gpu的。
请查看[官网](https://pytorch.org/get-started/locally/)，获得本系统与cuda版本一致的命令后进行安装，如下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191107105506719.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
判断cuda是否可用

```python
torch.cuda.is_available()
```
判断pytorch所需cuda版本

```python
torch.version.cuda
```

判断实际运行时使用的 cuda 目录（把cuda移动到这个目录）

```python
   >>> import torch
   >>> import torch.utils
   >>> import torch.utils.cpp_extension
   >>> torch.utils.cpp_extension.CUDA_HOME
```

更多pytorch关于cuda的部分请查阅[嗨呀库，点我](https://www.cnblogs.com/yhjoker/p/10972795.html)

<br>

## 如何定义数据

- 数据的输入也是采用tensor的形式



### 定义常量

- 将数据转换为tensor的方式和numpy转为数组的方式很像

  > ```python
  > import torch
  > 
  > #将list转为tensor
  > torch.FloatTensor(list)
  > 
  > #将tensor转换为numpy的数组
  > tensor.numpy()
  > 
  > #将numpy转换为tensor
  > Torch.from_numpy(np_data)
  > ```

- pytorch中的tensor运算和numpy基本一致

<br>

### 定义变量

- 变量类所处的包

  > ```python
  > from torch.autograd import Varivale
  > ```

- 定义变量

  > ```python
  > #requires_grad表示是否用于反向传播计算，即要不要计算梯度
  > Variable(tensor,requires_grad=True)
  > ```

- 变量用于构建动态计算图

- 计算变量的梯度

  > ```python
  > tensor = torch.FloatTensor([[1,2],[3,4]])
  > variable = Variable(tensor, requires_grad=True)
  > v_out = torch.mean(variable*variable)
  > #反向传播梯度计算
  > v_out.backward()
  > #获得梯度
  > print(variable.grad)
  > #获得Varibale中的数据，得到的类型为tensor
  > print(variable.data)
  > ```

<br>

## 如何定义模型



### 快速搭建

> ```python
> import torch
> net=torch.nn.Sequential(
>     torch.nn.Linear(1, 10),
>     torch.nn.ReLU(),
>     torch.nn.Linear(10, 1)
> )
> ```

<br>

### 个性化搭建

创建一个继承了torch.nn.Module的类，在\__init__函数中定义所有的层属性，所有的层属性必须是类的直接属性，如果不是，需要使用setattr进行设置，在forward(x)函数中定义层与层之间的连接关系，其实就是定义前向传播

> ```python
> import torch
> import torch.nn.functional as F
> 
> class Net(torch.nn.Module):
>     def __init__(self, n_feature, n_hidden, n_output):
>         super(Net, self).__init__()  
>         self.hidden = torch.nn.Linear(n_feature, n_hidden)
>         self.out = torch.nn.Linear(n_hidden, n_output) 
> 
>     def forward(self, x):
>         x = F.relu(self.hidden(x))
>         x = self.out(x)
>         return x
> ```



上面例子中的层都直接定义为了类的直接属性，从而加入到了计算图中，下面这个例子演示了如何不通过类属性将层加入到计算图中

> ```python
> import torch
> import torch.nn.functional as F
> 
> class Net(torch.nn.Module):
>     def __init__(self):
>      		fc = nn.Linear(input_size, 10)
>      		#等价于self.fc1=fc
>      		setattr(self, 'fc1', fc)
> ```


<br>


## 如何定义激活函数

- 激活函数所在包

  > ```python
  > torch.nn.functional
  > ```

- 常用激活函数

  > ```python
  > import torch.nn.functional as F
  > F.relu(variable)
  > F.sigmoid(varibale)
  > F.tanh(variable)
  > ```


<br>


## 如何定义优化算法

- 优化算法所在的包

  > ```python
  > torch.optim
  > ```

- 常用的优化算法

  > ```python
  > import torch.optim as F
  > #输入参数为模型以及对应的优化算法的参数，例如学习率，动量值等
  > F.SGD(model.parameters(), lr=LR)
  > F.SGD(model.parameters(), lr=LR, momentum=0.8)
  > F.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
  > F.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
  > ```


<br>


## 如何定义loss函数

- 损失函数所在的包

  > ```python
  > torch.nn.functional
  > ```

- 常用损失函数

  [快，快点我，我等不及了](<https://pytorch.org/docs/stable/nn.html#loss-functions>)

  和keras把正则化放在层中，以此实现损失函数的正则化不同，pytorch定义了含有正则项的损失函数


<br>

## 如何训练模型

- 一般流程

  > - 定义模型
  > - 定义优化算法
  > - 定义损失函数
  > - 给模型喂数据
  > - 计算损失函数的值
  > - 清空上一步残余更新参数值
  > - 误差反向传播，计算梯度
  > - 应用梯度，进行参数更新

- 代码示例

  > ```python
  > # 模型定义省去
  > # 定义优化算法
  > optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
  > # 定义损失函数
  > loss_func = torch.nn.MSELoss()
  > 
  > for t in range(100):
  > 		# 给模型喂数据
  >     prediction = net(x)
  >     # 计算损失函数的值
  >     loss = loss_func(prediction, y)
  > 		# 清空上一步残余更新参数值
  >     optimizer.zero_grad()
  >     # 误差反向传播
  >     loss.backward()
  >     # 参数更新
  >     optimizer.step()
  > ```


<br>


## 如何保存与读取模型
<br>

### 保存

> ```python
> import torch
> # 定义模型，省略
> net1=.....
> # 保存整个模型
> torch.save(net1,'net1.pkl')
> # 保存网络中的参数
> torch.save(net1.state_dict(),'net_params.pkl')
> ```

<br>

### 读取

> ```python
> import torch
> # 加载模型
> net1=torch.load('net1.pkl')
> 
> # 定义与net1.pkl相同的模型结构
> net2=.....
> # 加载模型参数
> net2.load_state_dict(torch.load('net_params.pkl'))
> 
> 使用GPU训练的模型，现在使用cpu进行测试
> torch.load('G_38_4.893.pth',map_location='cpu')
> ```

<br>

## 如何进行批训练

DataLoader可用于批量迭代数据，常用于批训练，具体步骤如下

- 将数据转换为tensor
- 将数据放入DataLoader中
- 训练数据



DataLoader位于的包

> torch.utils.data



代码如下

> ```python
> import torch
> import torch.utils.data as Data
> torch.manual_seed(1)    # reproducible
> 
> BATCH_SIZE = 5      # 批训练的数据个数
> 
> x = torch.linspace(1, 10, 10)       # x data (torch tensor)
> y = torch.linspace(10, 1, 10)       # y data (torch tensor)
> 
> # 先转换成 torch 能识别的 Dataset
> torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
> 
> # 把 dataset 放入 DataLoader
> loader = Data.DataLoader(
>     dataset=torch_dataset,      # torch TensorDataset format
>     batch_size=BATCH_SIZE,      # mini batch size
>     shuffle=True,               # 要不要打乱数据 (打乱比较好)
>     num_workers=2,              # 多线程来读数据
> )
> 
> for epoch in range(3):   # 训练所有!整套!数据 3 次
>     for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
>         # 模型训练
> 
>         # 打印日志
>         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
>               batch_x.numpy(), '| batch y: ', batch_y.numpy())
> ```


<br>


## 如何在单GPU上训练模型

- 判断GPU是否可以使用

  > 若torch.cuda.is_available()返回为true，则可以使用GPU

- 将tensor存储到显存

  > ```python
  > ten1 = torch.FloatTensor(2)
  > ten1_cuda = ten1.cuda()
  > ```

- 将Variable存储到显存

  > 有两种创建方式
  >
  > ```python
  > 方式一：利用CPU上的张量创建变量，在将变量迁移至显存
  > ten1 = torch.FloatTensor(2)
  > V1_cpu = autograd.Variable(ten1)
  > V1 = V1_cpu.cuda()
  > 
  > 方式二：利用GPU上的张量创建变量，此时变量直接位于显存
  > ten1_cuda = ten1.cuda()
  > V2 = autograd.Variable(ten1_cuda)
  > ```

- 模型迁移

  > ```python
  > # 定义模型，具体过程略
  > model=....
  > # 转移模型到GPU,此时模型上的所有操作都会在GPU中进行
  > model.cuda()
  > ```


<br>


## pytorch中常用的CNN层

[二维卷积](<https://pytorch.org/docs/stable/nn.html#conv2d>) 与keras这么多封装好的卷积不同，pytorch只有普通的卷积，有点贫穷…..

[全连接层](<https://pytorch.org/docs/stable/nn.html#linear-layers>)

[dropout层](<https://pytorch.org/docs/stable/nn.html#dropout-layers>)

[池化层](<https://pytorch.org/docs/stable/nn.html#pooling-layers>)

[二维BN](<https://pytorch.org/docs/stable/nn.html#batchnorm2d>)


<br>

## pytorch自定义损失函数

Pytorch自定义损失函数的方法有很多，此处只讲解自定义函数方式，以MSE损失函数为例
> ```python
> # 所有的数学操作都必须使用tensor操作完成
> def my_MSE_loss(x,y):
> 	return torch.mean(torch.pow(x-y,2))
> ```
以使用自定义的MSE损失函数训练线性回归为例

```python
import torch
import torch.optim as opt
import torch.utils.data as Data

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.model=torch.nn.Linear(5,1)

    def forward(self,x):
        return self.model(x)

# 自定义损失函数
def my_MSE_loss(x,y):
    return torch.mean(torch.pow(x-y,2))

model=LinearModel()
loss_fuc=torch.nn.MSELoss()
optimizer=opt.SGD(model.parameters(),lr=0.01)

# 定义参数
x=torch.randn((10,5))
y=torch.mm(x,torch.Tensor([[1],[2],[3],[4],[5]]))+5+0.3*torch.rand(10)

# 定义批训练器
dataTensorSet=Data.TensorDataset(x,y)
loader=Data.DataLoader(dataset=dataTensorSet,
                                shuffle=True,
                                batch_size=5)

# 模型训练
for epoch in range(500):
    for step,(batch_x,batch_y) in enumerate(loader):
        predict=model(batch_x)
        #loss=loss_fuc(predict,batch_y)
        #使用自定义的损失函数
        loss=my_MSE_loss(predict,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("epoch=%s,step=%s,loss=%4f"%(epoch,step,loss.data.numpy()))
        
#打印训练参数
print(model.state_dict())
```

<br>

## pytorch对图片进行预处理
在进行训练时，我们通常需要对图片进行某些预处理操作。例如resize，对图片进行数据增强等，pytorch中通过transforms类实现，具体实现如下：

```python
import torchvision.transforms as transforms
import cv2

img_path=""
# 定义一系列的预处理操作，ToTensor表示归一化操作
PicTransform=transforms.Compose([transforms.Resize((32,32)),
                   transforms.ToTensor()])
img = cv2.imread(img_path)
PicTransform(img)
```

关于transforms定义的更多预处理方法，请查看[快，快点我，我等不了了](https://pytorch.org/docs/stable/torchvision/transforms.html)

<br>

## pytorch批量读取图片
pytorch通过ImageFolder来批量读取图片，并且还可以对图片进行预处理操作，以猫狗识别的CNN为例：

```python
import torch.nn as nn
# 对输入图像进行操作的类，例如图像的resize、图像的归一化等
import torchvision.transforms as transforms
# 批量读取数据的包
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import torch.optim as opt

root='/Users/菜到怀疑人生/Desktop/数据集/猫狗识别数据集'

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model=nn.Sequential(nn.Conv2d(3,6,5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(6,16,5),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2),
                    )
        self.classifier=nn.Sequential(nn.Linear(400,800),
                                      nn.ReLU(),
                                      nn.Linear(800,120),
                                      nn.ReLU(),
                                      nn.Linear(120,2))

    def forward(self, x):
        x=self.model(x)
        print(x.shape)
        x=x.view(-1,400)
        x=self.classifier(x)
        return x


# 定义图像预处理器
PicTransform=transforms.Compose([transforms.Resize((32,32)),
                   transforms.ToTensor()])

# 用于批量读取图片，并对图像进行预处理
train_data=ImageFolder(root,PicTransform)
train_loader=Data.DataLoader(dataset=train_data,
                             shuffle=True,
                             batch_size=8)

model=Model()
print(model)

loss_fuc=nn.CrossEntropyLoss()
optimizer=opt.SGD(model.parameters(),lr=0.01)

for epoch in range(10):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        predict=model(batch_x)
        loss=loss_fuc(predict,batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch=%s,step=%s,loss=%s"%(epoch,step,loss.data.numpy()))
```


<br>

## pytorch的反卷积操作ConvTranspose2d
网络上解释清楚ConvTranspose2d概念的文章没见到，而且有些人写错了，估计也是百度搜索引擎的原因，通过阅读文档，现对ConvTranspose2d的使用进行一个总结

函数原型

```python
torch.nn.ConvTranspose2d(in_channels,
						 out_channels, 
						 kernel_size, 
						 stride=1,
					 	 padding=0, 
					 	 output_padding=0, 
					 	 groups=1, 
					 	 bias=True, 
					 	 dilation=1, 
					 	 padding_mode='zeros')
```

这里对经常使用的参数进行总结

- in_channels：输入特征图的个数
- out_channels：输出特征图的个数
- kernel_size：卷积核大小
- stride：对输入特征图进行填充
	> 假设输入特征图为的大小为$H*W$，stride设置为$S$，对于宽度$W$来说，每两列之间插入$W*(S-1)$维的零矩阵，此时特征图的宽变为$$W*(S-1)+W=(W-1)*S+1$$对于高度$H$，同理。插入完毕后，特征图的大小为$$[(H-1)*S+1]*[(W-1)*S+1]$$
- dilation:
	>应该是空洞卷积，其实和stride对于特征图的处理类似，只是dilation对于卷积核进行处理，可以查看[Dilated convolution animations](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)进行了解
	如果参数dilation的值为$d$，卷积核大小为$k*k$，可将卷积核大小看成$$[(k-1)*d+1]*[(k-1)*d+1]$$
- padding
	> pytorch的padding与平常意义的不同，设参数dilation的值为$d$，卷积核大小为$k*k$，padding数目为$p$，在特征图周边padding的数目为$$d*(k-1)-p$$
- outputpadding：对输出特征图进行的padding数，设为$O$

基于上述定义，同时结合卷积后特征图大小的公式，具体查看[卷积过后特征图的大小](https://blog.csdn.net/dhaiuda/article/details/102560896#_114)，沿用上述符号，可得pytorch的反卷积操作后特征图大小$H_{out}*W_{out}$为：

$$
\begin{aligned}
H_{out}=&(H-1)*S+1+2*(d*(k-1)-p)-[(k-1)*d+1]+1\\
=&(H-1)*S+d*(k-1)-2P+1+O\\
W_{out}=&(W-1)*S+1+2*(d*(k-1)-p)-[(k-1)*d+1]+1\\
=&(W-1)*S+d*(k-1)-2P+1+O
\end{aligned}
$$

其实就是pytorch官网上写的计算公式[ConvTranspose2d](https://pytorch.org/docs/stable/nn.html?highlight=convtranspose2d#torch.nn.ConvTranspose2d)

<br>

## 模型的简单可视化
https://www.aiuai.cn/aifarm467.html


<br>

## 自动求导机制
pytorch基于计算图，所有的tensor以及tensor操作都会成为计算图中的一个部分。

autograd机制基于计算图，计算图中的结点分为叶结点和非叶结点，均为tensor，非叶结点需要由其他结点通过运算得到，如下图，图源自[pytorch的计算图](https://zhuanlan.zhihu.com/p/33378444)：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203085118135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)
a、b、c即叶结点，f、g、y由其他结点通过运算得到，为非叶结点。

<br>

### backward

tensor类型存在一个函数backward，该函数可以实现自动求导机制，但是pytorch的求导只能基于标量进行，backward的gradient参数指明对tensor的哪个标量进行求导，默认参数为None，即$[1]$，默认情况下，调用backward的tensor，shape必须为1，官网的定义如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191203091232866.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RoYWl1ZGE=,size_16,color_FFFFFF,t_70)


举个例子，假如有一个计算图的输出$a$的shape为$[3]$，即有三个标量，设为$$y_1=f(x)、y_2=f(x)、y_3=f(x)$$，如果我想对这三个标量同时求导，代码如下

```python
b=torch.tensor([1,1,1])
a.backward(b)
```

上述代码的功能类似于
$$b[0]\frac{\partial y_1}{\partial x}、b[1]\frac{\partial y_2}{\partial x}、b[2]\frac{\partial y_3}{\partial x}$$

梯度传播到叶子节点后，会分别累加，类似于
$$b[0]\frac{\partial y_1}{\partial x}+b[1]\frac{\partial y_2}{\partial x}+b[2]\frac{\partial y_3}{\partial x}$$

所以每次调用时，需要依据自己的需求清空计算图中的梯度，对于继承了nn.Module的类而言，可以调用zero_grad()函数清空计算图中的所有梯度。

一次forward对应一次backward，每次调用完backward后，计算图都会被释放，如果需要针对一个计算图多次调用backward，则将backward函数的retain_graph置为true即可，多次调用会出现问题，如下代码所示，代码来源[pytorch的计算图](https://zhuanlan.zhihu.com/p/33378444)

```python
net = nn.Linear(3, 4)
input = Variable(torch.randn(2, 3), requires_grad=True)
output = net(input)
loss = torch.sum(output)

loss.backward()
loss.backward()

RuntimeError: Trying to backward through the graph a second time, but the buffers have already been freed.
```

```python
net = nn.Linear(3, 4)
input = Variable(torch.randn(2, 3), requires_grad=True)
output = net(input)
loss = torch.sum(output)
loss.backward(retain_graph=True) # 添加retain_graph=True标识，让计算图不被立即释放
loss.backward()
```

<br>

### tensor中的梯度
调用完backward后，我想知道每个变量具体的梯度，怎么办？tensor类型中存在grad属性，可以查看该节点的梯度值，只有叶子节点的grad属性会存储梯度，如果想要查看非叶子节点的梯度，就需要register_hook函数了，具体如下：

```python
class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # 获得各层信息，name为层的名字，module为层的具体操作
        for name, module in self.model._modules.items():
            # 在这里进行了前向传播
            x = module(x)
            # 如果是目标层
            if name in self.target_layers:
                # 添加一个钩子，这个钩子在该层的梯度被计算时会被执行，此处特指记录该层的梯度
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x
```
上述代码是grad-cam实现代码的一部分

可以按下列代码清空某个节点的梯度

```python
# input为一个tensor
input.grad.data.zero_()
```

如果一个tensor的grad属性为None

 1. 压根没求梯度
 2. 该tensor是计算图中的非叶结点
 3. 该tensor本身不需要求导

针对第三点，我们引出tensor中的相应标志位，与梯度相关的标志位主要有requires_grad，只有requires_grad为true的tensor会被求导，从而有梯度，requires_grad具有传递性，例如

```python
a=torch.tensor([1,2,3],requires_grad=True)
b=torch.tensor([1,2,3],requires_grad=False)
c=torch.sum(a,b,dim=1)
```

则c的requires_grad默认也为true，这样梯度才能从c传递到a。

与梯度相关的函数主要有detach()函数，假设x为一个tensor，x.detach()会创建一个与x相同，但是requires_grad为false的节点x’，用于替换计算图中x的位置，此时梯度回传到x‘后，便不会在向后回传

<br>

## PyTorch的上采样

[PyTorch学习笔记(10)——上采样和PixelShuffle](https://blog.csdn.net/g11d111/article/details/82855946)

<br>

## Pytorch中的学习率衰减方法
[Pytorch中的学习率衰减方法](https://www.jianshu.com/p/9643cba47655)

<br>

## Pytorch多GPU训练模型
[pytorch使用记录（三） 多GPU训练](https://blog.csdn.net/daydayjump/article/details/81158777)
[pytorch多GPU训练以及多线程加载数据](https://blog.csdn.net/daniaokuye/article/details/79133351)

上述两个链接请看完，第二个链接有踩坑记录

<br>

## 如何让Pytorch运行速度更快（个人经验）

- 使用dataloader时，指定多线程读取数据，即num_worker参数
- 使用多GPU训练模型
- 对于数据的操作尽可能使用dataloader，通过指定num_worker，往往比自己操作数据快得多
- 对于python而言，尽可能少的使用for循环，多使用矩阵运算
