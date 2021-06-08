**使用 pytorch**

### 1. 在 python3.6 下使用 pytorch
贴入以下代码即可
```
export PATH="/opt/anaconda3/bin:$PATH"
source activate py36 
```

### 2. 在 python3.6 下安装pytorch

#### 2.1 创建激活 python3.6 的环境
```
export PATH="/opt/anaconda3/bin:$PATH"
conda create -n py36 python=3.6  # 建立python3.6的虚拟环境，并将虚拟环境命名为py36
source activate py36  # 激活py36
```

#### 2.2 安装 pytorch
##### 2.2.1 conda 安装 pytorch
```
conda install -n py36 pytorch torchvision cudatoolkit=9.0 -c pytorch
```

##### 2.2.2 pip 安装 pytorch
```
pip install torch torchvision
```

### 3. 测试 pytorch
进入 python 环境后
```
import torch
import torchvision
print(torch.__version__);
```
不报错即可

### 参考
1. https://pytorch.org/