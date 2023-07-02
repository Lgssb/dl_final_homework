# Task 2
本次实验使用CeiT作为 CNN + Transformer 的网络模型
使用 ResNet18 作为CNN对比模型

### Train

需先在`./configs/default.yaml`文件的最后一行指定数据增强模式，可供选择的参数有：normal、cutmix、cutout、mixup. 

默认为normal，即无任何数据增强。

之后在终端运行下一行程序。

```
python train.py -c configs/default.yaml --name train
```

### test

下行程序中的checkpoint可替换成以下四种，分别代表使用四种不同数据增强模式所训练的模型：

- ./output/normal_checkpoint.pyt
- ./output/cutmix_checkpoint.pyt
- ./output/cutout_checkpoint.pyt
- ./output/mixup_checkpoint.pyt

之后在终端运行下一行程序。

```python
python test.py -c configs/defaul.yaml --name test -p checkpoint
```

