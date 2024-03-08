UFLDL教程
=======

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
**说明：**本教程将阐述无监督特征学习和深度学习的主要观点。通过学习，你也将实现多个功能学习/深度学习算法，能看到它们为你工作，并学习如何应用/适应这些想法到新问题上。

本教程假定机器学习的基本知识（特别是熟悉的监督学习，逻辑回归，梯度下降的想法），如果你不熟悉这些想法，我们建议你去这里

[机器学习课程](http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning)，并先完成第II，III，IV章（到逻辑回归）。

**稀疏自编码器**

* [神经网络](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md "神经网络")
* [反向传导算法](%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95.md "反向传导算法")
* [梯度检验与高级优化](%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96.md "梯度检验与高级优化")
* [自编码算法与稀疏性](%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95%E4%B8%8E%E7%A8%80%E7%96%8F%E6%80%A7.md "自编码算法与稀疏性")
* [可视化自编码器训练结果](%E5%8F%AF%E8%A7%86%E5%8C%96%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9E%9C.md "可视化自编码器训练结果")
* [稀疏自编码器符号一览表](%E7%A8%80%E7%96%8F%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%AC%A6%E5%8F%B7%E4%B8%80%E8%A7%88%E8%A1%A8.md "稀疏自编码器符号一览表")
* [Exercise:Sparse Autoencoder](Exercise_Sparse_Autoencoder.md "Exercise:Sparse Autoencoder")

**矢量化编程实现**

* [矢量化编程](%E7%9F%A2%E9%87%8F%E5%8C%96%E7%BC%96%E7%A8%8B.md "矢量化编程")
* [逻辑回归的向量化实现样例](%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%9A%84%E5%90%91%E9%87%8F%E5%8C%96%E5%AE%9E%E7%8E%B0%E6%A0%B7%E4%BE%8B.md "逻辑回归的向量化实现样例")
* [神经网络向量化](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%90%91%E9%87%8F%E5%8C%96.md "神经网络向量化")
* [Exercise:Vectorization](Exercise_Vectorization.md "Exercise:Vectorization")

**预处理：主成分分析与白化**

* [主成分分析](%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90.md "主成分分析")
* [白化](%E7%99%BD%E5%8C%96.md "白化")
* [实现主成分分析和白化](%E5%AE%9E%E7%8E%B0%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90%E5%92%8C%E7%99%BD%E5%8C%96.md "实现主成分分析和白化")
* [Exercise:PCA in 2D](Exercise_PCA_in_2D.md "Exercise:PCA in 2D")
* [Exercise:PCA and Whitening](Exercise_PCA_and_Whitening.md "Exercise:PCA and Whitening")

**Softmax回归**

* [Softmax回归](Softmax%E5%9B%9E%E5%BD%92.md "Softmax回归")
* [Exercise:Softmax Regression](Exercise_Softmax_Regression.md "Exercise:Softmax Regression")

**自我学习与无监督特征学习**

* [自我学习](%E8%87%AA%E6%88%91%E5%AD%A6%E4%B9%A0.md "自我学习")
* [Exercise:Self-Taught Learning](Exercise_Self-Taught_Learning.md "Exercise:Self-Taught Learning")

**建立分类用深度网络**

* [从自我学习到深层网络](%E4%BB%8E%E8%87%AA%E6%88%91%E5%AD%A6%E4%B9%A0%E5%88%B0%E6%B7%B1%E5%B1%82%E7%BD%91%E7%BB%9C.md "从自我学习到深层网络")
* [深度网络概览](%E6%B7%B1%E5%BA%A6%E7%BD%91%E7%BB%9C%E6%A6%82%E8%A7%88.md "深度网络概览")
* [栈式自编码算法](%E6%A0%88%E5%BC%8F%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95.md "栈式自编码算法")
* [微调多层自编码算法](%E5%BE%AE%E8%B0%83%E5%A4%9A%E5%B1%82%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95.md "微调多层自编码算法")
* [Exercise: Implement deep networks for digit classification](Exercise__Implement_deep_networks_for_digit_classification.md "Exercise: Implement deep networks for digit classification")

**自编码线性解码器**

* [线性解码器](%E7%BA%BF%E6%80%A7%E8%A7%A3%E7%A0%81%E5%99%A8.md "线性解码器")
* [Exercise:Learning color features with Sparse Autoencoders](Exercise_Learning_color_features_with_Sparse_Autoencoders.md "Exercise:Learning color features with Sparse Autoencoders")

**处理大型图像**

* [卷积特征提取](%E5%8D%B7%E7%A7%AF%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96.md "卷积特征提取")
* [池化](%E6%B1%A0%E5%8C%96.md "池化")
* [Exercise:Convolution and Pooling](Exercise_Convolution_and_Pooling.md "Exercise:Convolution and Pooling")

---

**注意**: 这条线以上的章节是稳定的。下面的章节仍在建设中，如有变更，恕不另行通知。请随意浏览周围并欢迎提交反馈/建议。

**混杂的**

* [MATLAB Modules](MATLAB_Modules.md "MATLAB Modules")
* [Style Guide](Style_Guide.md "Style Guide")
* [Useful Links](Useful_Links.md "Useful Links")

**混杂的主题**

* [数据预处理](%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md "数据预处理")
* [用反向传导思想求导](%E7%94%A8%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E6%80%9D%E6%83%B3%E6%B1%82%E5%AF%BC.md "用反向传导思想求导")

**进阶主题**:

**稀疏编码**

* [稀疏编码](%E7%A8%80%E7%96%8F%E7%BC%96%E7%A0%81.md "稀疏编码")
* [稀疏编码自编码表达](%E7%A8%80%E7%96%8F%E7%BC%96%E7%A0%81%E8%87%AA%E7%BC%96%E7%A0%81%E8%A1%A8%E8%BE%BE.md "稀疏编码自编码表达")
* [Exercise:Sparse Coding](Exercise_Sparse_Coding.md "Exercise:Sparse Coding")

**独立成分分析样式建模**

* [独立成分分析](%E7%8B%AC%E7%AB%8B%E6%88%90%E5%88%86%E5%88%86%E6%9E%90.md "独立成分分析")
* [Exercise:Independent Component Analysis](Exercise_Independent_Component_Analysis.md "Exercise:Independent Component Analysis")

**其它**

* Convolutional training
* Restricted Boltzmann Machines
* Deep Belief Networks
* Denoising Autoencoders
* K-means
* Spatial pyramids / Multiscale
* Slow Feature Analysis
* Tiled Convolution Networks

---

英文原文作者: Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen

---

> * Language: [English](UFLDL_Tutorial.md "UFLDL Tutorial")
> * This page was last modified on 10 April 2013, at 00:26.
