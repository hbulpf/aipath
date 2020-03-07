# 安装
>以下说明适用于 CentOS 7，仅在 CentOS 7.5 上测试
## 一、安装 Nvidia 显卡驱动
1. 装前准备工作：运行脚本 [0_nvidia_pre.sh](./0_nvidia_pre.sh) 即可   `sh ./0_nvidia_pre.sh`
2. 安装 Nvidia 显卡驱动，运行脚本 [1_nvidia_install.sh](./1_nvidia_install.sh) 即可 `sh ./1_nvidia_install.sh`
```
lsmod | grep nouveau #检测开源驱动是否禁用成功，如果未输出任何内容，即为禁用成功
cd /501_raid_common/driver/  #nvidia显卡驱动在到 /501_raid_common/driver/ 目录下
chmod a+x NVIDIA-Linux-x86_64-384.183_forK20m.run #根据显卡型号安装驱动，这里安装 k20m 的驱动
./NVIDIA-Linux-x86_64-384.183_forK20m.run #25服务器安装 k20m 的驱动
# ./NVIDIA-Linux-x86_64-384.145_forK80.run #25服务器安装 k80 的驱动
nvidia-smi  #检测是否安装成功
# nvidia-uninstall #uninstall the NVIDIA Driver
# Logfile : /tmp/cuda_install_23258.log
```

## 二、安装 CUDA
### 安装 cuda tookit
如果没有 cuda-9.0 tookit ，请先[下载 cuda-9.0 tookit 包](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=rpmlocal)。
centos系统最好下载 rpm 形式存储在以下目录
```
cd /501_raid_common/driver/driver/cuda_tookit_9.0/rpm/
```

###### 1. 安装 cuda-9.0 tookit
```
yum install -y epel-release  #安装依赖
yum install -y --enablerepo=epel dkms #安装依赖

mkdir -p /usr/local/cuda-9.0 && ln -s /usr/local/cuda-9.0 /usr/local/cuda
cat <<EOM >>/etc/profile
export PATH="$PATH:/usr/local/cuda/bin"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64"
EOM
source /etc/profile

cd /501_raid_common/driver/driver/cuda_tookit_9.0/rpm/
rpm -i cuda-repo-rhel7-9-0-local-9.0.176-1.x86_64.rpm
yum clean all
yum install cuda
yum install -y cuda-repo-rhel7-9-0-local-cublas-performance-update-1.0-1.x86_64.rpm
yum install -y cuda-repo-rhel7-9-0-local-cublas-performance-update-2-1.0-1.x86_64.rpm
yum install -y cuda-repo-rhel7-9-0-local-cublas-performance-update-3-1.0-1.x86_64.rpm
yum install -y cuda-repo-rhel7-9-0-176-local-patch-4-1.0-1.x86_64.rpm
```

###### 2. 安装CUDA样例程序 
样例程序建议目录为   `/opt/cuda`
```
cuda-install-samples-9.0.sh /opt/cuda/
```
    该命令已经在系统环境变量中，/opt/cuda/为自定义目录；执行完该命令之后，如果成功，会在 /opt/cuda/ 中生成一个 NVIDIA_CUDA-9.0_Samples 目录 ,假设目录为 `/opt/cuda/NVIDIA_CUDA-9.0_Samples`

###### 3. 编译样例程序，校验CUDA安装

```
cd /opt/cuda/NVIDIA_CUDA-9.0_Samples 
make -j8
```

###### 4. 运行样例程序 

```
/opt/cuda/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release/deviceQuery
```
输出结果末端显示如下，即CUDA安装校验完成。 
```
......
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 8.0, NumDevs = 2, Device0 = Tesla M40, Device1 = Tesla M40 
Result = PASS 
```

## 三、安装 cudnn
cudnn 下载地址： [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download) (需注册nvidia账号并登录),最好下载 rpm 类型的(REHL和CentOS)。
```
cd /501_raid_common/driver/cudnn/
yum install -y libcudnn7-7.5.0.56-1.cuda9.0.x86_64.rpm
```

## 四、[安装tensorflow](../tesorflow/tesorflow.md)

>注: 安装完 cuda-9.0 tookit 后，显卡驱动可能会失效。使用 `nvidia-smi` 检测一下。如果失效，按照上面安装 Nvidia 显卡驱动的方式再重装显卡驱动即可。

---
# 卸载

### 卸载 cuda tookit
卸载cuda-toolkit
```
/usr/local/cuda/bin/uninstall_*** # ***换成自己的文件名
```
完全清理
```
rm -rf /usr/local/cuda*
dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 sudo dpkg --purge
```

### 卸载 显卡驱动
假如安装的是 NVIDIA-Linux-x86_64-396.44_forK20.run , 卸载运行如下命令：
```
sh NVIDIA-Linux-x86_64-396.44_forK20.run --uninstall  # NVIDIA-Linux-x86_64-396.44_forK20.run 换成自己的文件名
```


---
# 参考
1. TESLA DRIVER FOR LINUX X64 驱动下载 . https://www.nvidia.cn/Download/index.aspx?lang=cn
1. cuda-9.0 tookit下载地址：https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=CentOS&target_version=7&target_type=runfilelocal
1. NVIDIA CUDA Toolkit Release Notes 版本支持 . https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
1. GPU版Tensorflow安装 centos7 64位 . https://blog.csdn.net/wang2008start/article/details/71319970
1. 下载 cudnn . https://developer.nvidia.com/cudnn
1. Tensorflow GPU支持 . https://www.tensorflow.org/install/source#tested_source_configurations


------------
@504实验室<br>
友情贡献： @[`鹏飞`](https://github.com/RunAtWorld)   @[`加伟`](https://github.com/1846263444)   