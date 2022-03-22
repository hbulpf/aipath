#  学习PyTorch

## 数据操作

入门

```
import torch
x = torch.arange(12)
x.shape
x.numel()
X = x.reshape(3, 4)
Y = x.reshape(-1,4)
torch.ones((2, 3, 4))
torch.randn(3, 4)
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

运算符

```
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
torch.exp(x)
# 联结元素
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
X == Y
#对张量元素求和
X.sum()
```

广播机制: 张量形状不同，我们可以通过调用 *广播机制*（broadcasting mechanism）来执行按元素操作

```
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a + b
```

索引和切片

```
#读取
X, X[-1], X[1:3]
#写入
X[1, 2] = 9
X
```

节省内存: 使用切片表示法将操作的结果分配给先前分配的数组

```
Z = torch.zeros_like(Y)
print('id(Z):', id(Z),Z)
Z[:] = X + Y
print('id(Z):', id(Z), Z)
```

> 使用`X[:] = X + Y`或`X += Y`来减少操作的内存开销

转换为其他Python对象: 将深度学习框架定义的张量转换为NumPy张量（`ndarray`）很容易，反之也同样容易。 torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。

```
A = X.numpy()
B = torch.tensor(A)
type(A), type(B),A,B,id(A),id(B)
```

要将大小为1的张量转换为Python标量，我们可以调用`item`函数或Python的内置函数。

```
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

## 数据预处理

### 读取数据集

```
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

```
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

### 处理缺失值

通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。

```
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

`pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

```
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

### 转换为张量格式

```
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

### 删除缺失值最多的列

方法1

```
nan_numer = data.isnull().sum(axis=0)
nan_max_id = nan_numer.idxmax()
data = data.drop([nan_max_id], axis=1)
```

