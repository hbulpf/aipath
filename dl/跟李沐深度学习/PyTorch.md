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

