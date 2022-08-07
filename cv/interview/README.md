# 面试

## CV

**1 基础**

我一般上来就会给候选人做几个小计算，随便给个输入给个卷积，算输出尺寸、参数量、计算量。

大概有50%的候选人没法都算出来，在这里面少部分人算不对输出尺寸，一半人算不对参数量，大部分人不会算计算量。其他的感受野BN之类的随便问问也就大概知道对方基础怎么样了。

很多候选人都是简历天花乱坠，基础都不太牢靠。



**2 诚实性**

这个很重要，如果不诚实，别的都免谈。

我一般就问项目里最基础的细节，比如模型大小、数据量、线上/实验室指标。

再比如说有很多人说有部署经验，用过trt、mnn这些，那就问用的推理引擎版本是什么，模型转换的步骤是什么。

如果觉得这些还判断不出来，就问对方在简历里写的熟悉的xxx模型，看是不是真熟悉。



**3 对数据/模型/训练技巧的理解**

这里其实是在考察分析问题和解决问题的能力。

我一般会问对方在项目里遇到的各种困难，以及项目经历中各种可以[cue](https://www.zhihu.com/search?q=cue&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2511367385})的点。

这一趴我期待候选人有分析数据，尤其是分析bad case并且能做针对性解决的能力；对模型选型、魔改的方式、训练的手段给出理由和思考，在这个过程中可以看出对方是不是在背博客上的答案。



**4 代码能力**

必考，但我也不出太难，只是看看对方平时写不写代码，是不是只会扒github。[leetcode](https://www.zhihu.com/search?q=leetcode&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2511367385})我从来不考。

我一般就让候选人随便给我写个卷积模型，真是随便写，就这样能难倒一多半的候选人，我也不知道为什么。

最近面的几个人，让他们实现一个深度可分离卷积，或者随便写个模型，我都是把架子给他们搭好，比如这样：

```python
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        pass
    
    def forward(self, x):
        pass

    
def test():
    pass


test()
    
```

如果不给搭架子，就从白板儿写，那大部分更是写不出来（惨

所以只填里面的nn.Conv2d和forward，再写个test case。就这，没有一个人能顺顺当当填出来的。

## 深度学习面试

对于面试者，可以帮助你在面试前查漏补缺；对于面试官，可以帮助你综合判断候选人的深度学习水平。

1. 理论基础
2. 项目经验
3. 编程能力
4. 行业认知

## 一、理论基础

### 基础篇（通用型）

**1）了解前向传播反向传播，以及链式求导**；例如给一个两层的MLP和简单的二维向量，能推导出 forward propagation，再用 chain rule 推导出 back propagation；

**2）了解常见的损失函数，激活函数和优化算法，明白它们的区别优劣**，例如 [relu](https://www.zhihu.com/search?q=relu&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640}) 相比 sigmoid 的优点在哪，Adam对于SGD做了哪些改进，交叉熵公式是什么；

**3）了解常见的评价指标以及它们的区别**，例如 accuracy，precision，recall，F1，AUC，混淆矩阵 各自的公式；

4）有哪些**防止过拟合**的策略，至少3个以上；

**5）了解[梯度下降](https://www.zhihu.com/search?q=梯度下降&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})算法原理**，准确指出梯度下降、随机梯度下降、批量梯度下降的区别；

6）如何合理的设置训练**batch**，是不是越大越好？

**7）了解基础的神经网络**，例如 MLP、CNN、RNN(GRU, LSTM)、Attention，基本各个子领域都会用到；

**8）明白各个神经网络的优缺点是什么**，例如 Attention相比CNN做了哪些改进，RNN为什么容易造成梯度消失&爆炸；

9）有哪些**缓解数据分布不均匀**的方法，至少3个以上；

10）了解基本的**数据冷启动和数据增强**策略；

以上列举的部分知识点，如果短时间内没答上超过1个，[基础功](https://www.zhihu.com/search?q=基础功&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})可能还需要加强。

**这里推荐一本《[深度学习500问](https://www.zhihu.com/search?q=深度学习500问&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})》，书中按章节罗列了500多个深度学习高频问题，是一本不错的复盘工具书。**

**有需要的读者在我的公众号【NLP情报局】后台回复“500”即可领取。**

### 进阶篇（以NLP为例）

**1）熟悉语言模型发展史**（word2vec->ELMO->GPT->BERT->...），以及改进的优点；

**2）熟悉[word2vec](https://www.zhihu.com/search?q=word2vec&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})**，例如两种模型结构和优化技巧，与[glove](https://www.zhihu.com/search?q=glove&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})的差异；

3）熟悉**BERT模型**，例如2个预训练任务，3个Embedding层，主要缺点，如何做英文分词(BPE)等；

4）熟悉**文本预处理**方法和常见的**特征提取策略**；

5）能解释**文本CNN**使用平均/最大池化时，反向传播参数梯度如何更新；

6）能手推**RNN**计算公式以及参数量；

7）熟悉**Transformer**有哪些核心模块，**Attention**计算公式；

8）熟悉底层的**分类、匹配任务**，并掌握一些经典的神经网络模型；

9）如果做过实体抽取任务，至少熟悉CRF的原理与损失函数，与HMM的区别等；

实际面试中，面试官不会完全从答案是否正确来判断面试者的深度学习水平，有些问题甚至没有标准答案。

和面试官及时交流，反馈自己的思考过程也十分重要。

## 二、项目经验

这一小节主要针对手上还没有顶会论文的同学，项目经历将是面试中的考查重点，往往会占据至少一半的面试时间。

实验室项目、实习经历、算法比赛等都可以看作是项目。由于每个人的研究方向有差异，这一小节主要归纳一些共性问题。

1）这个项目的背景以及最大亮点是什么？

2）自己负责了哪些具体任务？

3）项目中碰到最大的挑战是什么，最后如何克服的？

4）除了采用深度学习，有没有尝试其他解决方案，例如传统的机器学习算法？

5）项目最终的评测结果如何？是否部署上线？

通过上面几个问题，面试官基本可以摸清面试者在项目中的角色，是核心骨干还是浑水摸鱼，有创新价值还是仅仅跑了一遍开源代码。

如果我是面试官，会非常希望同学有基础的深度学习项目经验，最好熟悉完整的算法开发流程、GPU训练和服务器调用指令，这样可以帮助公司减少很多培训成本。

## 三、编程能力

面试中手撕代码是近几年技术岗的必备环节，甚至是面试通过与否的硬门槛之一，可以快速考查编程基本功。之前网上流传，pony马化腾每周还会在leetcode上刷一道编程题！

而考查的类型大致可以分为以下3种：

**1）1-2道[数据结构](https://www.zhihu.com/search?q=数据结构&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})算法题**，类似于[leetcode](https://www.zhihu.com/search?q=leetcode&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})；难度一般为medium，偶尔穿插easy和hard，考查的数据结构以二叉树、动态规划、数组、字符串、二分查找等为主；

**2）用深度学习框架（TF、Keras、Pytorch等）实现一个简单的任务**，要求定义类、损失函数、优化器、前向传播等，考查对框架的熟悉程度；

**3）用Python或深度学习框架手推一个[机器学习](https://www.zhihu.com/search?q=机器学习&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2021166640})算法**，例如KNN、KMeans、LR等。

## 四、行业认知

经过3个环节的考查，面试官已经基本掌握了你的深度学习水平。如果前面答的都不错，恭喜你离offer仅一步之遥了。

在某些公司的总监或boss面中，一般不会再问技术细节，但是会对**全局认知能力**再次考核。面试者是仅仅站在程序员的角度思考问题，还是会有更高层次的全局观、更长远的见解和规划。

而这种宏观认知，只需要询问几个问题：

1）该领域未来发展方向有哪些，会在哪发力？

2）目前的瓶颈或痛点在哪？

3）针对这些难点有没有好的思路？

面试就像是老师和学生们围坐在一起面对面探讨问题，大家依次发表自己的想法，最终老师会选出思路敏捷清晰、专业基础扎实的同学当课代表（发offer）。