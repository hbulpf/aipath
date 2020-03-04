**使用docker方式启动 tensorflow:1.13.1**
>使用docker方式启动 tensorflow:1.13.1,每个容器是一个环境隔离，大家互不影响。如需其他版本的 tensorflow , 请联系 @lijiawei
 
# 1. 建立自己的tensorflow:1.13.1 
注意 
- `JUPYTER_PORT` 是启动的 jupyter notebook 的外网访问端口，为防止端口冲突，请改为为自己分配的端口，自己的端口为 `64000 + 自己的vnc端口号`。例如我的vnc端口为 11，所以下面我设置为 **64011**
- `PWD` 是自己的 tensorflow 数据存储目录
- `TENSORFOWA_NAME` 是自己的tensorflow系统名。例如，我的用户名为 lipengfei ，我的tensorflow系统名为 **tensorflow_lipengfei_py3**

修改完后根据需要，按照如下方式启动自己的tensorflow系统。

### 1.1 使用GPU方式启动
#### python3 使用 tensorflow:1.13.1-gpu-py3
```
PWD=~/tensorflow_dir
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
JUPYTER_PORT="64011"
docker run --restart=always --runtime=nvidia -d -p $JUPYTER_PORT:8888 -v $PWD:/tmp -w /tmp -m 8g --memory-swap 16g --name $TENSORFOWA_NAME tensorflow/tensorflow:1.13.1-gpu-py3 
docker exec -it $TENSORFOWA_NAME bash
```

退出后如需再次进入，在命令行中输入:
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker exec -it $TENSORFOWA_NAME bash
```

暂停自己的tensorflow系统
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker stop $TENSORFOWA_NAME bash
```

重启自己的tensorflow系统
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker start $TENSORFOWA_NAME bash
```

#### python2 使用 tensorflow:1.13.1-gpu 
```
PWD=~/tensorflow_dir
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
JUPYTER_PORT="64011"
docker run --restart=always --runtime=nvidia -d -p $JUPYTER_PORT:8888 -v $PWD:/tmp -w /tmp -m 8g --memory-swap 16g --name $TENSORFOWA_NAME tensorflow/tensorflow:1.13.1-gpu 
docker exec -it $TENSORFOWA_NAME bash
```
退出后如需再次进入，在命令行中输入:
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
docker exec -it $TENSORFOWA_NAME bash
```

重启自己的tensorflow系统
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
docker start $TENSORFOWA_NAME bash
```


### 1.2 使用CPU方式启动
#### python3 使用 tensorflow:1.13.1-py3
```
PWD=~/tensorflow_dir
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
JUPYTER_PORT="64011"
docker run --restart=always --runtime=nvidia -d -p $JUPYTER_PORT:8888 -v $PWD:/tmp -w /tmp -m 8g --memory-swap 16g -m 8G --memory-swap 16G --name $TENSORFOWA_NAME tensorflow/tensorflow:1.13.1-py3 
docker exec -it $TENSORFOWA_NAME bash
```

退出后如需再次进入，在命令行中输入:
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker exec -it $TENSORFOWA_NAME bash
```

暂停自己的tensorflow系统
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker stop $TENSORFOWA_NAME bash
```

重启自己的tensorflow系统
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker start $TENSORFOWA_NAME bash
```

#### python2 使用 tensorflow:1.13.1 
```
PWD=~/tensorflow_dir
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
JUPYTER_PORT="64011"
docker run --restart=always --runtime=nvidia -d -p $JUPYTER_PORT:8888 -v $PWD:/tmp -w /tmp -m 8g --memory-swap 16g --name $TENSORFOWA_NAME tensorflow/tensorflow:1.13.1
docker exec -it $TENSORFOWA_NAME bash
```
退出后如需再次进入，在命令行中输入:
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
docker exec -it $TENSORFOWA_NAME bash
```


暂停自己的tensorflow系统
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
docker stop $TENSORFOWA_NAME bash
```

重启自己的tensorflow系统
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
docker start $TENSORFOWA_NAME bash
```

### 1.3 远程访问tensorflow系统中的jupyter notebook
启动后,使用 **_ip:`JUPYTER_PORT`_** 即可远程访问 自己的tensorflow系统中的 jupyter notebook 。例如我的vnc端口为 11，
- 25服务器访问 : http://50125.hnbdata.cn:64011/
- 26服务器访问 : http://50126.hnbdata.cn:64011/

如需输入登录密码，进入自己的tensorflow系统,执行
```
jupyter notebook list
```
即可获得登录 token

# 2. 删除tensorflow系统 
如果不再使用tensorflow系统,可以删除自己的tensorflow系统。但数据将会一直保留在自己设置的 **tensorflow 数据存储目录** 会一直保留。

python3 删除tensorflow系统 
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker stop $TENSORFOWA_NAME && docker rm $TENSORFOWA_NAME
```

python2 删除tensorflow系统 
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
docker stop $TENSORFOWA_NAME && docker rm $TENSORFOWA_NAME
```

# 3. 保存当前tensorflow系统 环境
如长期保存自己的当前tensorflow系统环境的软件环境配置可以使用:

python3  
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py3"
docker commit \
--author "$(whoami)" \
--message "我修改了什么内容" \
$TENSORFOWA_NAME \
$TENSORFOWA_NAME:v2
```

python2  
```
TENSORFOWA_NAME="tensorflow_$(whoami)_py2"
docker commit \
--author "$(whoami)" \
--message "我修改了什么内容" \
$TENSORFOWA_NAME \
$TENSORFOWA_NAME:v2
```

>语法为: docker commit [选项] <容器ID或容器名> [<仓库名>[:<标签>]]

# 参考
1. nvidia/cuda hub . https://hub.docker.com/r/nvidia/cuda
1. NVIDIA Container Runtime for Docker . https://github.com/NVIDIA/nvidia-docker
1. Docker 安装TensorFlow . https://tensorflow.google.cn/install/docker
1. tensorflow hub . https://hub.docker.com/r/tensorflow/tensorflow/tags?page=5