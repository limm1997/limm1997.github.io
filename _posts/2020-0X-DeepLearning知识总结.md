---
layout:     post                    # 使用的布局
title:      Deep Learning               # 标题 
subtitle:   #副标题
date:       2019-01-04              # 时间
author:     Doublefierce                      # 作者
header-img: img/bg-post.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - DL
---

![img](https://pic2.zhimg.com/80/v2-0db16c79b4a9ad7ec688347567e39632_720w.jpeg)

### 损失函数、交叉熵、Softmax

- 损失函数

一文读懂机器学习常用损失函数（Loss Function）
https://www.cnblogs.com/guoyaohua/p/9217206.html
学点基本功：机器学习常用损失函数小结
https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/101875924
机器学习中的损失函数 （着重比较：hinge loss vs softmax loss）
https://blog.csdn.net/u010976453/article/details/78488279
目标函数，损失函数和代价函数
https://blog.csdn.net/qq_28448117/article/details/79199835

- 交叉熵

交叉熵（Cross-Entropy）
https://blog.csdn.net/rtygbwwwerr/article/details/50778098
一文搞懂交叉熵在机器学习中的使用，透彻理解交叉熵背后的直觉
https://blog.csdn.net/tsyccnh/article/details/79163834
交叉熵代价函数(损失函数)及其求导推导
https://blog.csdn.net/jasonzzj/article/details/52017438

- Softmax

详解softmax函数以及相关求导过程
https://blog.csdn.net/pql925/article/details/81010836
斯坦福CS231n assignment1：softmax损失函数求导
https://www.jianshu.com/p/6e405cecd609



### 激活函数

激活函数实现的是一对一的变换，即用相同的函数对输入向量的每个分量进行映射，得到输出向量，输入和输出向量的维数相同。
理解神经网络的激活函数
https://mp.weixin.qq.com/s/ix5RJcGQ7SMGeU5Yz4z2dg
常用激活函数（激励函数）理解与总结
https://www.jianshu.com/writer#/notebooks/38075299/notes/57457007/preview



### 初始化、标准化、Fine-tune

- 初始化

深度学习中的参数初始化

https://blog.csdn.net/mzpmzk/article/details/79839047?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task

- 标准化

数据的标准化是将数据按比例缩放，使之落入一个小的特定空间。在某些比较和评价的指标处理中经常会用到，去除数据的单位限制，将其转化为无量纲的纯数值，便于不同单位或量级的指标能够进行比较和加权。
目前数据标准化方法有很多，归结起来可以分为直线型方法(如极值法、标准差法)、折线型法（如三折线型法）、曲线型法（如半正态性分布）。不同的标准化方法，对系统的评价结果会产生不同的影响，然而不幸的是，在数据标准化方法的选择上，还没有通用的法则可以遵循。其中最典型的就是数据的归一化处理，即将数据统一映射到 [0，1] 区间上。

数据标准化/归一化normalization

https://blog.csdn.net/pipisorry/article/details/52247379

Scikit-learn：数据预处理Preprocessing data

https://blog.csdn.net/pipisorry/article/details/52247679

- fine-tune

什么是fine-tuning？

https://blog.csdn.net/weixin_42137700/article/details/82107208



### 偏差、方差、过拟合

- 偏差bias

偏差是指预测结果与真实值之间的差异，排除噪声的影响，偏差更多的是针对某个模型输出的样本误差，偏差是模型无法准确表达数据关系导致，比如模型过于简单，非线性的数据关系采用线性模型建模

- 方差Variance

模型方差不是针对某一个模型输出样本进行判定，而是指多个(次)模型输出的结果之间的离散差异，注意这里写的是多个模型或者多次模型，即不同模型或同一模型不同时间的输出结果方差较大，方差是由训练集的数据不够导致，一方面量 (数据量) 不够，有限的数据集过度训练导致模型复杂，另一方面质(样本质量)不行，测试集中的数据分布未在训练集中，导致每次抽样训练模型时，每次模型参数不同，输出的结果都无法准确的预测出正确结果。

偏差和方差有什么区别？

https://www.zhihu.com/question/20448464/answer/765401873

- 过拟合

理解过拟合

https://mp.weixin.qq.com/s/-kKGlqmpCiW_lyZShusazw

机器学习中用来防止过拟合的方法有哪些？

https://www.zhihu.com/question/59201590/answer/167392763



### 正则化、dropout

- 正则化

监督机器学习问题无非就是“minimizeyour error while regularizing your parameters”，也就是在规则化参数的同时最小化误差。最小化误差是为了让我们的模型拟合我们的训练数据，而规则化参数是防止我们的模型过分拟合我们的训练数据。多么简约的哲学啊！因为参数太多，会导致我们的模型复杂度上升，容易过拟合，也就是我们的训练误差会很小。但训练误差小并不是我们的最终目标，我们的目标是希望模型的测试误差小，也就是能准确的预测新的样本。所以，我们需要保证模型“简单”的基础上最小化训练误差，这样得到的参数才具有好的泛化性能（也就是测试误差也小），而模型“简单”就是通过规则函数来实现的。另外，规则项的使用还可以约束我们的模型的特性。这样就可以将人对这个模型的先验知识融入到模型的学习当中，强行地让学习到的模型具有人想要的特性，例如稀疏、低秩、平滑等等。要知道，有时候人的先验是非常重要的。前人的经验会让你少走很多弯路，这就是为什么我们平时学习最好找个大牛带带的原因。对机器学习也是一样，如果被我们人稍微点拨一下，它肯定能更快的学习相应的任务。只是由于人和机器的交流目前还没有那么直接的方法，目前这个媒介只能由规则项来担当了。

L0、L1、L2范数

http://blog.csdn.net/zouxy09/article/details/24971995
核范数与规则项参数选择
https://blog.csdn.net/zouxy09/article/details/24972869
什么是正则化
https://charlesliuyx.github.io/2017/10/03/%E3%80%90%E7%9B%B4%E8%A7%82%E8%AF%A6%E8%A7%A3%E3%80%91%E4%BB%80%E4%B9%88%E6%98%AF%E6%AD%A3%E5%88%99%E5%8C%96/

- dropout

开篇明义，dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。

理解dropout

https://blog.csdn.net/stdcoutzyx/article/details/49022443



### 梯度、反向传播算法

- 梯度

理解梯度下降法

https://zhuanlan.zhihu.com/p/36902908

gradient checking（梯度检验）

https://blog.csdn.net/u012328159/article/details/80232585

- 反向传播算法

反向传播算法

https://www.jiqizhixin.com/graph/technologies/7332347c-8073-4783-bfc1-1698a6257db3

神经网络反向传播时的梯度到底怎么求？

https://blog.csdn.net/magic_anthony/article/details/77531552

神经网络求导：computational graph 中涉及向量的求导问题 ？

https://www.zhihu.com/question/47024992/answer/103962301



### 优化算法

深入理解优化器Optimizer算法（BGD、SGD、MBGD、Momentum、NAG、Adagrad、Adadelta、RMSprop、Adam）
https://www.cnblogs.com/guoyaohua/p/8780548.html
优化方法总结：SGD，Momentum，AdaGrad，RMSProp，Adam
https://blog.csdn.net/u010089444/article/details/76725843



### 超参数调试

深度学习网络调参技巧

https://zhuanlan.zhihu.com/p/24720954?utm_source=wechat_session&utm_medium=social&utm_oi=637252858509135872

深度学习中的的超参数调节

https://zhuanlan.zhihu.com/p/41785031

深度学习中的batch size 以及learning rate参数理解

https://blog.csdn.net/lemonaha/article/details/72773056



### Batch Normalization

什么是批标准化 (Batch Normalization)
https://zhuanlan.zhihu.com/p/24810318
深入理解Batch Normalization批标准化
https://www.cnblogs.com/guoyaohua/p/8724433.html
关于Batch Normalization的另一种理解
https://blog.csdn.net/AIchipmunk/article/details/54234646



### 训练集、验证集、测试集

训练集、验证集和测试集

https://zhuanlan.zhihu.com/p/48976706

验证集与测试集的区别

https://blog.csdn.net/jmh1996/article/details/79838917



### Autoencoder

一文看懂AutoEncoder模型演进图谱

https://zhuanlan.zhihu.com/p/68903857

RBM（限制波尔兹曼机）

https://blog.csdn.net/mytestmy/article/details/9150213

稀疏自编码器（Sparse Autoencoder）

https://blog.csdn.net/u010278305/article/details/46881443

VAE、GAN及其变种

https://blog.csdn.net/heyc861221/article/details/80130968

变分自编码器（Variational Auto-Encoder，VAE）

https://blog.csdn.net/antkillerfarm/article/details/80648805



### 其它

深度学习算法与编程

https://blog.csdn.net/oBrightLamp/article/details/85067981#MSELoss_28

Deep Learning（深度学习）学习笔记整理系列

https://blog.csdn.net/zouxy09/article/details/8775360