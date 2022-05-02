# MindSpore

中文名昇思,[官网](https://mindspore.cn/)

## 模型训练一般步骤

在模型训练过程中，一般分为四个步骤: 

* 定义神经网络。
* 构建数据集。
* 定义超参、损失函数及优化器。
* 输入训练轮次和数据集进行训练。

### 导入模块并传入数据集

```
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore import nn, Tensor, Model
from mindspore import dtype as mstype

DATA_DIR = "./datasets/cifar-10-batches-bin/train"
```

### 定义神经网络
```
class Net(nn.Cell):

    def __init__(self, num_class=10, num_channel=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

 
    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
 
net = Net()
epochs = 5
batch_size = 64
learning_rate = 1e-3
```

### 构建数据集
```
sampler = ds.SequentialSampler(num_samples=128)
dataset = ds.Cifar10Dataset(DATA_DIR, sampler=sampler)
```
 

### 数据类型转换

```
type_cast_op_image = C.TypeCast(mstype.float32)
type_cast_op_label = C.TypeCast(mstype.int32)
HWC2CHW = CV.HWC2CHW()
dataset = dataset.map(operations=[type_cast_op_image, HWC2CHW], input_columns="image")
dataset = dataset.map(operations=type_cast_op_label, input_columns="label")
dataset = dataset.batch(batch_size)
```

### 定义超参、损失函数及优化器
```
optim = nn.SGD(params=net.trainable_params(), learning_rate=learning_rate)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
```

### 输入训练轮次和数据集进行训练
```
model = Model(net, loss_fn=loss, optimizer=optim)
model.train(epoch=epochs, train_dataset=dataset)
```
