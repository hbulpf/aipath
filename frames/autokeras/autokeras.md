**python 3.6 下 的 autokeras** 
> autokeras 依赖于 pytorch

# 1. 使用 autokeras
贴入以下代码即可
```
export PATH="/opt/anaconda3/bin:$PATH"
source activate py36 
```
# 2. 安装 autokeras 0.4.0
1. 激活 py36 环境
    ```
    export PATH="/opt/anaconda3/bin:$PATH"
    source activate py36 
    ```

2. 安装tensorflow-gpu
    ```
    conda install -n py36 -y tensorflow-gpu==1.13.1
    ```

3. 安装Keras
    ```
    conda install -y -n py36 keras==2.2.4
    ```

4. 安装
    推荐方式
    ```
    pip install autokeras==0.4.0
    ```

    其他方式
    ```
    git clone https://github.com/keras-team/autokeras.git
    cd autokeras
    git checkout -b 0.4.0 --track 0.4.0
    python setup.py install
    ```

5. 测试
    ```
    from keras.datasets import mnist
    from autokeras import ImageClassifier
    from autokeras.constant import Constant

    if __name__ == '__main__':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))
        clf = ImageClassifier(verbose=True, augment=False)
        clf.fit(x_train, y_train, time_limit=30 * 60)
        clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
        y = clf.evaluate(x_test, y_test)

        print(y * 100)
    ```

# 3. 常见问题
1. 导入 `autokeras` 报缺少 `GLIBCXX_3.4.20`

    具体错误：
    `ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.20' not found (required by /opt/anaconda3/envs/py36/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so)`

    **解决：**
    
    在系统中查找是否有大于等于 `libstdc++.so.6.0.20` 版本的 c++ 库
    ```
    find / -name "libstdc++.so.6"
    ```
    如果有，复制到 `/usr/lib64`下并修改 `libstdc++.so.6` 的链接到新版本的 c++ 库。
    
    已经在 `/501_raid_common/Centos7Software/lib.bak/lib64/libstdc++.so.6.0.25` 下面备份了新版本的 c++ 库。操作如下：
    ```
    cp /501_raid_common/Centos7Software/lib.bak/lib64/libstdc++.so.6.0.25 /usr/lib64
    mv /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so.6.bak
    ln -s /usr/lib64/libstdc++.so.6.0.25 /usr/lib64/libstdc++.so.6
    ```

# 参考
1. Auto-Keras . https://autokeras.com/
2. keras-team/autokeras . https://github.com/keras-team/autokeras/tree/master/autokeras
