# 摘要

本报告使用基于所提供的音乐数据集，多种机器学习方法进行分类训练和评估，并在一组200首歌曲的测试集上进行分类。主要进行了常规的预处理，如归一化等操作对数据集进行处理，并使用sklearn工具包内函数进行分类训练，使用k折交叉验证评估模型性能。在此基础上，本研究还引入了更高级的预处理手段，对模型参数进行了调整，以提升模型的准确率。本研究还使用深度学习方法进行训练和预测，并于经典机器学习方法进行比较。最后，将预测结果上传到网站，并于其他同学的准确率进行比较。

This report employs various machine learning methods for classification training and evaluation using a supplied music dataset, and conducts classification on a test set of 200 songs. Standard preprocessing techniques such as normalization are applied to the dataset, and the sklearn toolkit functions are utilized for classification training, with model performance evaluated using k-fold cross-validation. Additionally, this study introduces advanced preprocessing techniques and tunes model parameters to enhance accuracy. Deep learning methods are also employed for training and prediction, and a comparison is drawn with classical machine learning approaches. Finally, the predictive results are uploaded to a website and compared with the accuracy of other classmates' models.







# 1 数据处理

## 1.1 数据集与测试集的加载

数据集加载

使用pandas包中函数对课程提供的数据集与测试集进行读取：

数据划分

已知数据集的大小是[750*14]，其中每一行是一首歌曲的数据，前13列为特征值，最后一列为标签。所以需要对数据集进行划分。首先对列进行操作，划分特征与标签；其次对行进行操作，随机划分20%的数据为测试集，其余为训练集。



## 1.2 数据预处理

### 1.2.1 归一化处理

由于数据集中每项特征的尺度不同，所以需要进行归一化处理。这样有助于将不同特征的值映射到相似的尺度，以改善机器学习模型的性能。

采用Scikit-Learn 中的一个数据预处理工具`MinMaxScaler` 将特征缩放到[0, 1]之间。

### 1.2.2 离群样本检测与剔除

1. - 离群样本检测

   - 识别数据集中与大多数样本不同或明显偏离常规模式的常值或离群值。

     用于实现数据清洗，以去除噪声和异常值，以提高机器学习模型的性能。

2. - 离群样本剔除
   - 在离群样本检测的基础上，将识别到的异常值从数据集中移除。

### 1.2.3 SVD处理

本研究采用svd进行进一步的数据集处理，在某些情况下，数据集可能包含噪声或异常值，这些噪声可能对模型产生不良影响。SVD可以用于检测和剔除数据中的噪声或异常值。异常值通常对应于较小的奇异值，因此可以通过排除与较小奇异值对应的特征向量来剔除这些异常值。对奇异值进行排序，选取奇异值中的前几行，重组特征矩阵（数据集）。







# 2 机器学习方法

## 2.1 KNN

K-Nearest Neighbors，简称KNN 是一种用于分类和回归的监督学习算法。它的基本思想是：如果一个样本在特征空间中的K个最近邻居中的大多数属于某一个类别，那么该样本也属于这个类别，即采用投票法来进行分类。

KNN算法在实际应用中通常用于小型数据集、样本分布均匀的情况、以及需要解释性强的场景。

通过选取合适的k值，得到训练结果，如下图所示。

使用pca将特征压缩到三个维度，观测预测结果的标签在三维空间中的分布，如下图所示。

## 2.2 Random Forest

随机森林（Random Forest）是一种集成学习算法，用于解决分类和回归问题。它通过组合多个决策树来改善模型的性能和泛化能力。

首先，从原始训练集中随机选择一定数量的样本（有放回抽样）以及一定数量的特征（通常是特征总数的平方根）来构建决策树。通常使用CART（Classification and Regression Trees）算法
然后重复上述步骤构建多棵决策树，形成一个随机森林。对于分类问题，采用多数投票法确定样本的类别。
Random Forest 在sklearn中最重要的参数是“n_estimators ” 表示森林中树的数量。通常增加树的数量可以提高性能，但也会增加计算成本。以及“random_state” (default=None)，表示随机数生成器的种子，用于控制每次训练的随机性，使结果可重复。
通过调参，得到训练结果，如下图所示。
使用pca将特征压缩到三个维度，观测预测结果的标签在三维空间中的分布，如下图所示。

## 2.3 Logistic regression
逻辑回归（Logistic Regression）是一种经典的机器学习算法，主要用于二分类问题。它通常是机器学习和统计分析中的基准算法之一，特别适用于处理线性可分问题和高维数据。当数据的关系大致是线性的时候，逻辑回归通常能够提供良好的性能。

## 2.4 Voting









# 3 深度学习方法



# 4 结论

经过对模型的测试与分析，本研究得出以下结论，

传统机器学习方法中，KNN和Random Forest 在测试集上表现较好，经过SVD预处理之后，准确率能达到85%；而Logistic regression分类准确率较差，只能到80%左右。Voting的准确率较高，在83%左右，但训练时间较长。

而深度学习方法，在训练集上可以轻松到达100%准确率，然而很容易出现过拟合情况，且经过svd处理后，分类效果还有下降。

基于以上测试结果的综合考量，本研究最终选取random forest。



