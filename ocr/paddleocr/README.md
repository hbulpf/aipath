# PaddleOcr
在线尝试ppHub: https://www.paddlepaddle.org.cn/hub/scene/ocr

## 形态

PP-OCR

[PP-OCRv2](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/ppocr_introduction.md)

## 使用

[安装paddle](https://www.paddlepaddle.org.cn/)

```
python -m pip install paddlepaddle==2.3.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 应用场景



## 部署方式

> [在线API](https://ai.baidu.com/ai-doc/OCR/Ek3h7xypm)、[HTTP SDK](https://ai.baidu.com/ai-doc/OCR/vkibizxjr)、[离线SDK](https://ai.baidu.com/tech/ocr_sdk)、[私有化部署、一体机](https://ai.baidu.com/tech/ocr_private)等多种部署方式

1. [PaddleHub](https://gitee.com/paddlepaddle/PaddleHub)-[快速开始](https://www.paddlepaddle.org.cn/hubdetail?name=ch_pp-ocrv3&en_category=TextRecognition): 便捷地获取PaddlePaddle生态下的预训练模型，完成模型的管理和一键预测。配合使用Fine-tune API，可以基于大规模预训练模型快速完成迁移学习，让预训练模型能更好地服务于用户特定场景的应用
2. [边缘端侧SDK部署](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)

![](imgs/deployment.png)

# 参考

### 参考资料
1. [PaddleOcr gitee仓库](https://gitee.com/paddlepaddle/PaddleOCR)
2. [课程: AI快车道2020-PaddleOCR](https://aistudio.baidu.com/aistudio/education/group/info/1519)

### 参考课程
1. [动手学OCR·十讲](https://aistudio.baidu.com/aistudio/education/group/info/25207)

### 示例
1. [在线测试: OCR超轻量中英文识别](https://www.paddlepaddle.org.cn/hub/scene/ocr)
2. [PaddleHub一键OCR中文识别（超轻量8.1M模型，火爆）](https://aistudio.baidu.com/aistudio/projectdetail/507159)
3. [项目:PaddleOCR 文字检测](https://aistudio.baidu.com/aistudio/projectdetail/2006263?channelType=0&channel=0)


### 📖 官方文档教程

- [运行环境准备](./environment.md)
- [PP-OCR文本检测识别🔥](./ppocr_introduction.md)
    - [快速开始](./quickstart.md)
    - [模型库](./models_list.md)
    - [模型训练](./training.md)
        - [文本检测](./detection.md)
        - [文本识别](./recognition.md)
        - [文本方向分类器](./angle_class.md)
    - 模型压缩
        - [模型量化](./deploy/slim/quantization/README.md)
        - [模型裁剪](./deploy/slim/prune/README.md)
        - [知识蒸馏](./knowledge_distillation.md)
    - [推理部署](./deploy/README_ch.md)
        - [基于Python预测引擎推理](./inference_ppocr.md)
        - [基于C++预测引擎推理](./deploy/cpp_infer/readme_ch.md)
        - [服务化部署](./deploy/pdserving/README_CN.md)
        - [端侧部署](./deploy/lite/readme.md)
        - [Paddle2ONNX模型转化与预测](./deploy/paddle2onnx/readme.md)
        - [云上飞桨部署工具](./deploy/paddlecloud/README.md)
        - [Benchmark](./benchmark.md)
- [PP-Structure文档分析🔥](./ppstructure/README_ch.md)
    - [快速开始](./ppstructure/docs/quickstart.md)
    - [模型库](./ppstructure/docs/models_list.md)
    - [模型训练](./training.md)
        - [版面分析](./ppstructure/layout/README_ch.md)
        - [表格识别](./ppstructure/table/README_ch.md)
        - [关键信息提取](./ppstructure/kie/README_ch.md)
    - [推理部署](./deploy/README_ch.md)
        - [基于Python预测引擎推理](./ppstructure/docs/inference.md)
        - [基于C++预测引擎推理](./deploy/cpp_infer/readme_ch.md)
        - [服务化部署](./deploy/hubserving/readme.md)
- [前沿算法与模型🚀](./algorithm_overview.md)
    - [文本检测算法](./algorithm_overview.md)
    - [文本识别算法](./algorithm_overview.md)
    - [端到端OCR算法](./algorithm_overview.md)
    - [表格识别算法](./algorithm_overview.md)
    - [关键信息抽取算法](./algorithm_overview.md)
    - [使用PaddleOCR架构添加新算法](./add_new_algorithm.md)
- [场景应用](./applications)
- 数据标注与合成
    - [半自动标注工具PPOCRLabel](./PPOCRLabel/README_ch.md)
    - [数据合成工具Style-Text](./StyleText/README_ch.md)
    - [其它数据标注工具](./data_annotation.md)
    - [其它数据合成工具](./data_synthesis.md)
- 数据集
    - [通用中英文OCR数据集](doc/doc_ch/dataset/datasets.md)
    - [手写中文OCR数据集](doc/doc_ch/dataset/handwritten_datasets.md)
    - [垂类多语言OCR数据集](doc/doc_ch/dataset/vertical_and_multilingual_datasets.md)
    - [版面分析数据集](doc/doc_ch/dataset/layout_datasets.md)
    - [表格识别数据集](doc/doc_ch/dataset/table_datasets.md)
    - [关键信息提取数据集](doc/doc_ch/dataset/kie_datasets.md)
- [代码组织结构](./tree.md)
- [效果展示](#效果展示)
- [《动手学OCR》电子书📚](./ocr_book.md)
- FAQ
    - [通用问题](./FAQ.md)
    - [PaddleOCR实战问题](./FAQ.md)
- [参考文献](./reference.md)