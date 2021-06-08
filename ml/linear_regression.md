# 一元线性回归

给定数据集$ D={(x_1,y_1),(x_2,y_2),\dots,(x_m,y_m)},y_i \in R$. 线性回归(linear regression)试图学得一个线性模型以尽可能地预测实值输出标记，即：
$$f(x_i)=wx_i+b,使得f(x_i) \simeq y_i$$如果$w$和$b$确定了，那么整个模型也就定义好了。但是，如何确定$w$和$b$呢？
显然，如何确定$w$和$b$，与预测值$f(x)$与真实值$y$之间的误差密切相关。倘若，$w$和$b$很准确地确定下来，意味着$f(x)$与$y$地误差将会非常小。我们可以从这个角度，求出$w$和$b$地值。均方误差是回归任务中常用的性能度量，我们在这里也使用均方误差来描述$f(x)$和$y$之间的差别。由于均方误差是一个凸函数，凸函数的最小值必然存在，那么，$f(x)$和$y$的均方误差取最小值时所对应的$w$和$b$就是所求值。
综上所述，求$w$和$b$的的值的过程，实际上是求均方误差极值点的问题，即：
$$
MSE(w,b) = {\underset {w,b}{\operatorname {arg\,min} }}\sum_{i=1}^m(f(x_i)-y_i)^2={\underset {w,b}{\operatorname {arg\,min} }}\sum_{i=1}^m(wx_i+b-y_i)^2(3.3)
$$
均方误差有非常好的几何意义，它对应了常用的欧几里得距离或简称“欧式距离”(Euclidean distance).基于均方误差最小化来进行模型求解的方法成为“**最小二乘法**”(least square method).在线性回归中，最小二乘法就是试图找到一条直线，使所有样本到直线上的欧式距离之和最小。
先将$MSE(w,b)$对$b$求偏导数：

$
\frac{\partial MSE(w,b)}{\partial b} = \sum_{i=1}^m2(wx_i+b-y_i)
$

$
\qquad\qquad\,\,=2\left(w\sum_{i=1}^mx_i+mb-\sum_{i=1}^my_i \right)(3.4)
$
令公式3.4等于0，有：

$
w\sum_{i=1}^mx_i+mb-\sum_{i=1}^my_i=0
$

$
mb = \sum_{i=1}^my_i-w\sum_{i=1}^mx_i
$

$
b = \frac{1}{m}\sum_{i=1}^my_i-w\frac{1}{m}\sum_{i=1}^mx_i
$

$
b = \overline{y}-w\overline{x}(3.5)
$

再把$MSE(w,b)$对$w$求偏导数：

$
\frac{\partial MSE(w,b)}{\partial w} =\sum_{i=1}^m2(wx_i+b-y_i)x_i
$

$
\qquad\qquad\,\,=\sum_{i=1}^n2(wx_1^2+bx_i-y_ix_i)
$

$
\qquad\qquad\,\,=2\left(w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i\right)(3.6)
$

令上式等于0，把式3.5代入有：

$
w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-\overline{y}+w\overline{x})x_i=0
$

$
w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_ix_i-\overline{y}x_i)-w\sum_{i=1}^m\overline{x}x_i=0
$

$
w\sum_{i=1}^m(x_i^2-\overline{x}x_i) = \sum_{i=1}^m(y_ix_i-\overline{y}x_i)
$

$
w = \frac{\sum_{i=1}^m(y_ix_i-\overline{y}x_i)}{\sum_{i=1}^m(x_i^2-\overline{x}x_i)}
$

$
\quad =\frac{\sum_{i=1}^m(y_ix_i-\overline{y}x_i-y_i\overline{x}+\overline{x}\,\overline{y})}{\sum_{i=1}^m(x_i^2-\overline{x}x_i-x_i\overline{x}+\overline{x}\,\overline{x})}
$(因为$
\sum_{i=1}^my_i\overline{x}=m\frac{1}{m}\sum_{i=1}^my_i\overline{x}=m\overline{y}\,\overline{x}=\sum_{i=1}^m\overline{y}\,\overline{x}
$，$
\sum_{i=1}^mx_i\overline{x}=m\frac{1}{m}\sum_{i=1}^mx_i\overline{x}=m\overline{x}\,\overline{x}=\sum_{i=1}^m\overline{x}\,\overline{x}
$)


$
\quad = \frac{\sum_{i=1}^m(y_i-\overline{y})(x_i-\overline{x})}{\sum_{i=1}^m(x_i-\overline{x})^2}
$(3.7)（因为$
(y_i-\overline{y})(x_i-\overline{x})=y_ix_i-y_i\overline{x}-\overline{y}x_i+\overline{y}\,\overline{x}
$）

综上所述，给出给定训练数据集$ D={(x_1,y_1),(x_2,y_2),\dots,(x_m,y_m)},y_i \in R$，可以根据式3.7计算出$w$的值，然后再根据式3.5计算出$b$的值，至此，线性回归模型$f(x)$就确定下来了。

