# GPU选型

## 英伟达Tesla系列

### NVIDIA Tesla GPU系列P4、T4、P40以及V100

NVIDIA Tesla系列GPUP4、T4、P40以及V100性能规格参数对比表，[阿里云GPU云服务器](https://www.aliyun.com/product/ecs/gpu?source=5176.11533457&userCode=r3yteowb&type=copy)提供的实例GN4（Nvidia M40）、GN5（Nvidia P100）、GN5i（Nvidia P4）及GN6（Nvidia V100），也会基于NVIDIA Tesla GPU系列。

| 云服务器吧          | Tesla T4：世界领先的推理加速器 | Tesla V100：通用数据中心 GPU           | 适用于超高效、外扩型服务器的 | Tesla P4 适用于推理吞吐量服务器的 Tesla P40 |
| :------------------ | :----------------------------- | :------------------------------------- | :--------------------------- | :------------------------------------------ |
| 单精度性能 (FP32)   | 8.1 TFLOPS                     | 14 TFLOPS (PCIe) 15.7 teraflops (SXM2) | 5.5 TFLOPS                   | 12 TFLOPS                                   |
| 半精度性能 (FP16)   | 65 TFLOPS                      | 112 TFLOPS (PCIe)125 TFLOPS (SXM2)     | —                            | —                                           |
| 整数运算能力 (INT8) | 130 TOPS                       | —                                      | 22 TOPS*                     | 47 TOPS*                                    |
| 整数运算能力 (INT4) | 260 TOPS                       | —                                      | —                            | —                                           |
| GPU 显存            | 16GB                           | 32/16GB HBM2                           | 8GB                          | 24GB                                        |
| 显存带宽            | 320GB/秒                       | 900GB/秒                               | 192GB/秒                     | 346GB/秒                                    |
| 系统接口/外形规格   | PCI Express 半高外形           | PCI Express 双插槽全高外形 SXM2/NVLink | PCI Express 半高外形         | PCI Express 双插槽全高外形                  |
| 功率                | 70 W                           | 250 W (PCIe) 300 W (SXM2)              | 50 W/75 W                    | 250 W                                       |
| 硬件加速视频引擎    | 1 个解码引擎，2 个编码引擎     | —                                      | 1 个解码引擎，2 个编码引擎   | 1 个解码引擎，2 个编码引擎                  |

### 关于NVIDIA TESLA系列GPU详细介绍如下：

#### NVIDIA TESLA V100

NVIDIA Tesla V100采用NVIDIA Volta架构，非常适合为要求极为苛刻的双精度计算工作流程提供加速，并且还是从P100升级的理想路径。该GPU的渲染性能比Tesla P100提升了高达80%，借此可缩短设计周期和上市时间。

Tesla V100的每个GPU均可提供125 teraflops的推理性能，配有8块Tesla V100的单个服务器可实现1 petaflop的计算性能。

#### NVIDIA TESLA P40

The Tesla P40能够提供高达2倍的专业图形性能。Tesla P40能够对组织中每个vGPU虚拟化加速图形和计算（NVIDIA CUDA® 和 OpenCL）工作负载。支持多种行业标准的2U服务器。

Tesla P40可提供出色的推理性能、INT8精度和24GB板载内存。

#### NVIDIA TESLA T4

NVIDIA Tesla T4的帧缓存高达P4的2倍，性能高达M60的2倍，对于利用NVIDIA Quadro vDWS软件开启高端3D设计和工程工作流程的用户而言，不失为一种理想的解决方案。凭借单插槽、半高外形特性以及低至70瓦的功耗，Tesla T4堪称为每个服务器节点实现最大GPU密度的绝佳之选。

#### NVIDIA TESLA P4

Tesla P4可加快任何外扩型服务器的运行速度，能效高达CPU的60倍。

![img](gpu_selection.assets/bc97af6850f574d9.jpg)



## 英伟达消费级系列

![preview](gpu_selection.assets/v2-3ec9e8e5256385f2558cbc0567663f26_r.jpg)


## 参考

1. [NVIDIA Tesla GPU系列P4、T4、P40以及V100参数性能对比](https://developer.aliyun.com/article/753454)