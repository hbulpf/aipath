# FiftyOne

[FiftyOne](https://voxel51.com/docs/fiftyone/index.html)  是用于构建高质量数据集和计算机视觉模型的开源工具。

FiftyOne提供对象检测评估分析的可视化工具，不仅可以计算 AP，还可以轻松地 可视化单个样本和对象级别的结果， 查看精确召回曲线，并绘制**交互式混淆矩阵**等，一开始以为FiftyOne只用于目标检测中，看完官网后发现强大的不行，分类、目标检测、语义分割、关键点检测等预测结果均可以可视化，是一个优化模型的强大工具。

FiftyOne核心能力包括：

1. 导入数据集进行操作，轻松管理数据。
2. 评估模型。
3. 使嵌入数据和模型可视化。
4. 查找标注错误。
5. 管理数据集去除冗余图像。

FiftyOne工具包含三个组件：Python库、App、Brain。

1. 提供的Python接口可轻松以多种常见格式加载数据集，并提供以自定义格式加载数据集。
2. App是一个图形用户界面，可快速直观了解数据集。
3.Brain是一个强大的机器学习驱动功能库，可提供对数据集的洞察并推荐修改数据集的方法，从而提高模型的性能。

测试代码

```
import fiftyone as fo
import fiftyone.zoo as foz
 
# reference: https://voxel51.com/docs/fiftyone/tutorials/evaluate_detections.html
 
datasets = foz.list_zoo_datasets()
print("available datasets:", datasets)
 
dataset = foz.load_zoo_dataset("coco-2017", split="validation", dataset_name="evaluate-detections-tutorial")
dataset.persistent = True
session = fo.launch_app(dataset)
 
# print some information about the dataset
print("dataset info:", dataset)
 
# print a ground truth detection
sample = dataset.first()
print("ground truth:", sample.ground_truth.detections[0])
 
session.wait()
```

