#!/usr/bin/env python
# coding: utf-8

# [![在线运行](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_modelarts.png)](https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svcjIuMC90dXRvcmlhbHMvemhfY24vYmVnaW5uZXIvbWluZHNwb3JlX3F1aWNrX3N0YXJ0LmlweW5i=&imageid=b8671c1e-c439-4ae2-b9c6-69b46db134ae)&emsp;[![下载Notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_notebook.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r2.0/tutorials/zh_cn/beginner/mindspore_quick_start.ipynb)&emsp;[![下载样例代码](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_download_code.png)](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/r2.0/tutorials/zh_cn/beginner/mindspore_quick_start.py)&emsp;[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/tutorials/source_zh_cn/beginner/quick_start.ipynb)
# 
# [基本介绍](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/introduction.html) || **快速入门** || [张量 Tensor](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/tensor.html) || [数据集 Dataset](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/dataset.html) || [数据变换 Transforms](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/transforms.html) || [网络构建](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/model.html) || [函数式自动微分](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/autograd.html) || [模型训练](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/train.html) || [保存与加载](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/save_load.html)

# # 快速入门
# 
# 本节通过MindSpore的API来快速实现一个简单的深度学习模型。若想要深入了解MindSpore的使用方法，请参阅各节最后提供的参考链接。

# In[1]:


import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset


# ## 处理数据集
# 
# MindSpore提供基于Pipeline的[数据引擎](https://www.mindspore.cn/docs/zh-CN/r2.0/design/data_engine.html)，通过[数据集（Dataset）](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/dataset.html)和[数据变换（Transforms）](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/transforms.html)实现高效的数据预处理。在本教程中，我们使用Mnist数据集，自动下载完成后，使用`mindspore.dataset`提供的数据变换进行预处理。
# 
# > 本章节中的示例代码依赖`download`，可使用命令`pip install download`安装。如本文档以Notebook运行时，完成安装后需要重启kernel才能执行后续代码。

# In[2]:


# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)


# 数据下载完成后，获得数据集对象。

# In[3]:


train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')


# 打印数据集中包含的数据列名，用于dataset的预处理。

# In[4]:


print(train_dataset.get_col_names())


# MindSpore的dataset使用数据处理流水线（Data Processing Pipeline），需指定map、batch、shuffle等操作。这里我们使用map对图像数据及标签进行变换处理，然后将处理好的数据集打包为大小为64的batch。

# In[5]:


def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset


# In[6]:


# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)


# 使用`create_tuple_iterator`或`create_dict_iterator`对数据集进行迭代。

# In[7]:


for image, label in test_dataset.create_tuple_iterator():
    print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
    print(f"Shape of label: {label.shape} {label.dtype}")
    break


# In[8]:


for data in test_dataset.create_dict_iterator():
    print(f"Shape of image [N, C, H, W]: {data['image'].shape} {data['image'].dtype}")
    print(f"Shape of label: {data['label'].shape} {data['label'].dtype}")
    break


# 更多细节详见[数据集 Dataset](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/dataset.html)与[数据变换 Transforms](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/transforms.html)。

# ## 网络构建
# 
# `mindspore.nn`类是构建所有网络的基类，也是网络的基本单元。当用户需要自定义网络时，可以继承`nn.Cell`类，并重写`__init__`方法和`construct`方法。`__init__`包含所有网络层的定义，`construct`中包含数据（[Tensor](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/tensor.html)）的变换过程（即[计算图](https://www.mindspore.cn/tutorials/zh-CN/r2.0/advanced/compute_graph.html)的构造过程）。

# In[9]:


# Define model
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512),
            nn.ReLU(),
            nn.Dense(512, 512),
            nn.ReLU(),
            nn.Dense(512, 10)
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network()
print(model)


# 更多细节详见[网络构建](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/model.html)。

# ## 模型训练

# In[10]:


# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)


# 在模型训练中，一个完整的训练过程（step）需要实现以下三步：
# 
# 1. **正向计算**：模型预测结果（logits），并与正确标签（label）求预测损失（loss）。
# 2. **反向传播**：利用自动微分机制，自动求模型参数（parameters）对于loss的梯度（gradients）。
# 3. **参数优化**：将梯度更新到参数上。

# MindSpore使用函数式自动微分机制，因此针对上述步骤需要实现：
# 
# 1. 正向计算函数定义。
# 2. 通过函数变换获得梯度计算函数。
# 3. 训练函数定义，执行正向计算、反向传播和参数优化。

# In[11]:


# Define forward function
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


# 除训练外，我们定义测试函数，用来评估模型的性能。

# In[12]:


def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练过程需多次迭代数据集，一次完整的迭代称为一轮（epoch）。在每一轮，遍历训练集进行训练，结束后使用测试集进行预测。打印每一轮的loss值和预测准确率（Accuracy），可以看到loss在不断下降，Accuracy在不断提高。

# In[13]:


epochs = 3
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset)
    test(model, test_dataset, loss_fn)
print("Done!")


# 更多细节详见[模型训练](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/train.html)。

# ## 保存模型
# 
# 模型训练完成后，需要将其参数进行保存。

# In[14]:


# Save checkpoint
mindspore.save_checkpoint(model, "model.ckpt")
print("Saved Model to model.ckpt")


# ## 加载模型

# 加载保存的权重分为两步：
# 
# 1. 重新实例化模型对象，构造模型。
# 2. 加载模型参数，并将其加载至模型上。

# In[15]:


# Instantiate a random initialized model
model = Network()
# Load checkpoint and load parameter to model
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)


# > `param_not_load`是未被加载的参数列表，为空时代表所有参数均加载成功。

# 加载后的模型可以直接用于预测推理。

# In[16]:


model.set_train(False)
for data, label in test_dataset:
    pred = model(data)
    predicted = pred.argmax(1)
    print(f'Predicted: "{predicted[:10]}", Actual: "{label[:10]}"')
    break


# 更多细节详见[保存与加载](https://www.mindspore.cn/tutorials/zh-CN/r2.0/beginner/save_load.html)。
