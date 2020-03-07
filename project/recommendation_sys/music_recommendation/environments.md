# 音乐推荐系统环境搭建

##  <font color=red> 配置基本环境 </font>

1. 安装 Anaconda，并配置环境变量, 使用[清华大学镜像源下载Anaconda3-5.3.1](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) 后 [安装](../../../anaconda/install_anaconda.md) 
2. 增加 Anaconda 的镜像源，命令行输入：
    ```bash
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
    conda config --set show_channel_urls yes  #从channel中安装包时显示channel的url，这样就可以知道包的安装来源
    ```
3. [创建使用 python3.6](../../../anaconda/py37_To_py36.md)
    ```bash 
    conda create -n py36 python=3.6 #建立python3.6的虚拟环境，并将虚拟环境命名为py36
    ```
    
    激活python3.6版本，使用命令：
    ```
    source activate py36
    ```
    如需退出python3.6：`source deactivate`   
    windows 操作系统，上述命令不要输入 `source` 
4. 安装 Surprise 库，命令行输入：`conda -y install scikit-surprise=1.0.4`  
   如果不奏效，使用 `conda install -y --channel https://conda.anaconda.org/conda-forge scikit-surprise=1.0.4`
5. 升级 pip，命令行输入： `python -m pip install --upgrade pip`
6. 安装 gensim 库，命令行输入：`pip install gensim`  
   如果下载很慢，[使用清华大学镜像](../../../python/pip_mirrors.md): `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple gensim`
7. 安装tensorflow 库，命令行输入：`pip install tensorflow==1.4.0`  
   如果下载很慢，[使用清华大学镜像](../../../python/pip_mirrors.md): `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==1.4.0`
8. 安装 pyspark 库，命令行输入：`pip install pyspark`  
   如果下载很慢，[使用清华大学镜像](../../../python/pip_mirrors.md): `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyspark`

> 注意，以后每次使用环境都要先激活 py36 的环境

## 快速配置
如果需要快速配置基本环境，可以在安装好 Anaconda 后，在本目录执行以下命令即可安装好到上面第8步:
```bash
conda env create -f environment.yml
```


## JDK、Hadoop、Spark 解压后需要配置环境变量

1. windows上环境变量设置：右键我的电脑==》属性==》高级系统设置==》环境变量==》系统变量
2. 配置 jdk 环境变量
    - 新建系统变量JAVA_HOME,设置值为"JDK解压路径"
    - 编辑系统变量PATH，添加 %JAVA_HOME%\bin;%JAVA_HOME%\jre\bin;
3. 配置 hadoop 环境变量
    - 新建系统变量HADOOP_HOME,设置值="hadoop解压路径"
    - 编辑系统变量PATH，添加 %HADOOP_HOME%\bin;
4. 配置 spark 环境变量：
    - 新建系统变量SPARK_HOME,设置值为"spark解压路径"
    - 编辑系统变量PATH，添加 %SPARK_HOME%\bin;
5. 验证安装，命令行输入如下命令，若可以显示出对应版本信息，则说明安装成功
    - java -version
    - hadoop version
    - pyspark --version