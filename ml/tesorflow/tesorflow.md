**python 3.6 下的 Tensorflow** 

# 1. 在 python3.6 下使用 Tensorflow
```
export PATH="/opt/anaconda3/bin:$PATH"
source activate py36 
```

# 2. 安装 Tensorflow过程
### 1. 获得python3.6
```
export PATH="/opt/anaconda3/bin:$PATH"
```
执行以上命令将获得Python3.7，为了使用Tensorflow你目前需要Python3.6版本，使用conda提供的虚拟环境来获得python3.6

```bash
conda create -n py36 python=3.6  # 建立python3.6的虚拟环境，并将虚拟环境命名为py36
source activate py36 # 激活py36
#
# To activate this environment, use
#
#     $  source activate py36
#
# To deactivate an active environment, use
#
#     $  source deactivate
```
### 2. 安装 tensorflow
#### 2.1 conda 方式安装

1. 安装gpu 1.13.1 版本
```
conda install -n py36 -y tensorflow-gpu==1.13.1
```
2. 安装cpu 1.13.1 版本
```
conda install -n py36 -y tensorflow==1.13.1
```


#### 2.2 pip 方式安装

1. 安装CPU版本tensorflow

```
pip install --upgrade tensorflow==1.13.1 #安装CPU版本tensorflow
```

2. 安装GPU版本tensorflow
```
pip install --upgrade tensorflow_gpu==1.13.1 #安装GPU版本tensorflow
```

如果使用pip的安装过程中出现网络error或者速度太慢，请参考配置：[pip设置阿里云的镜像源，速度超级快](https://segmentfault.com/a/1190000006111096)

### 3. 测试tensorflow
```
import tensorflow as tf
print(tf.__version__)
hello=tf.constant('hello world') 
sess = tf.Session() 
print(sess.run(hello))  
```
### 注意

#### 永久性激活 python3.6
在`~/.bash_profile`文件的末尾加上` source activate py36`，用户登录后将自动切换到(py36)环境  
```
cat <<EOM >>~/.bash_profile
#get python36
source activate py36
EOM
source ~/.bash_profile
```

#### Tensorflow兼容表
服务器 cuda 为9.0，兼容的 tensorflow 见：[Tensorflow GPU支持](https://www.tensorflow.org/install/source#tested_source_configurations)
