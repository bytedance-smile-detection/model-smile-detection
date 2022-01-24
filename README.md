# The Lenet model

# **Language / Packages used**

1. python 3.5
2. OpenCV 3.4.4
3. keras 2.2.4
4. Imutils
5. NumPy

# Dataset

使用的数据集为 **SMILES**

数据集中有 13,165 张图像，其中每张图像的尺寸为 64x64x1（灰度）。并且数据集中的图像在面部周围被紧密裁剪

![Untitled](<The%20selected%20model%20(temporary)%2004d8f369a97e40af8cda015f62195017/Untitled.png>)

[GitHub - hromi/SMILEsmileD: open source smile detector haarcascade and associated positive & negative image datasets](https://github.com/hromi/SMILEsmileD)

# Model

该项目使用的模型为 **Lenet 架构**

/pipeline/nn/conv/lenet.py

LeNet 的体系结构如下表所示。激活层没有在表中显示，它应该是每一层之后的一个。这个项目使用了激活函数

| Layer Type  | Output Size  | Filter Size / Stride |
| ----------- | ------------ | -------------------- |
| Input Image | 28 x 28 x 1  |                      |
| CONV        | 28 x 28 x 20 | 5 x 5, K = 20        |
| POOL        | 14 x 14 x 20 | 2 x 2                |
| CONV        | 14 x 14 x 50 | 3 x 3, K = 50        |
| POOL        | 7 x 7 x 50   | 2 x 2                |
| FC          | 500          |                      |
| softmax     | 2            |                      |

# Training result

下图为训练集和验证集的损失和准确度图。从图中我们可以看出，第 6 个 epoch 之后的验证损失开始停滞。超过第 15 个时期的进一步训练可能会导致过度拟合

![training_loss_and_accuracy_plot.png](<The%20selected%20model%20(temporary)%2004d8f369a97e40af8cda015f62195017/training_loss_and_accuracy_plot.png>)

下图说明了对该神经网络的评估，它在验证集上获得了大约 92% 的分类准确率

![evaluation.png](<The%20selected%20model%20(temporary)%2004d8f369a97e40af8cda015f62195017/evaluation.png>)

# The Problems

## problem 1

在试图运行 train_model.py 的时候，发现控制台报错，具体如下

![Untitled](<The%20selected%20model%20(temporary)%2004d8f369a97e40af8cda015f62195017/Untitled%201.png>)

在 stackoverflow 上找到了解决办法，修改代码如下

```python
# account for skew in the labeled data
classTotals = labels.sum(axis=0)
weight = classTotals / classTotals.max()
classWeight = {i: weight[i] for i in range(len(weight))}
```

[on colab - class_weight is causing a ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()](https://stackoverflow.com/questions/61261907/on-colab-class-weight-is-causing-a-valueerror-the-truth-value-of-an-array-wit)

## problem 2

报错如下，应该是 keras 版本问题

![Untitled](<The%20selected%20model%20(temporary)%2004d8f369a97e40af8cda015f62195017/Untitled%202.png>)

用 accuracy 替换 acc，val\__accuracy 替换 val_\_acc 即可

```python
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
```

# Convert model

按照官方文档说明，在 tran_model.py 中加入如下代码即可

```python
tfjs.converters.save_keras_model(model, './output')
```

# Run command

```python
# train_model.py
python train_model.py -d="./dataset" -m="./output/" -p="./output"

# detect_smile.py
python detect_smile.py -c="./haarcascade_frontalface_default.xml" -m="./output/lenet.hdf5"

# ./server/app.js
node server/app.js

# ./tfjs/model.js
node tfjs/model.js
```

# Github link

[https://github.com/meng1994412/Smile_Detection](https://github.com/meng1994412/Smile_Detection)
