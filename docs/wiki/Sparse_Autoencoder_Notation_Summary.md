Sparse Autoencoder Notation Summary
===================================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
Here is a summary of the symbols used in our derivation of the sparse autoencoder:

|  Symbol
 |  Meaning
 |
| --- | --- |
| \textstyle x |  Input features for a training example, \textstyle x \in \Re^{n}.
 |
| \textstyle y |  Output/target values. Here, \textstyle y can be vector valued. In the case of an autoencoder, \textstyle y=x.
 |
| \textstyle (x^{(i)}, y^{(i)}) |  The \textstyle i-th training example
 |
| \textstyle h_{W,b}(x) |  Output of our hypothesis on input \textstyle x, using parameters \textstyle W,b. This should be a vector of
the same dimension as the target value \textstyle y.
 |
| \textstyle W^{(l)}_{ij} |  The parameter associated with the connection between unit \textstyle j in layer \textstyle l, and
unit \textstyle i in layer \textstyle l+1.
 |
| \textstyle b^{(l)}_{i} |  The bias term associated with unit \textstyle i in layer \textstyle l+1. Can also be thought of as the parameter associated with the connection between the bias unit in layer \textstyle l and unit \textstyle i in layer \textstyle l+1.
 |
| \textstyle \theta |  Our parameter vector. It is useful to think of this as the result of taking the parameters \textstyle W,b and ``unrolling *them into a long column vector.* |
| \textstyle a^{(l)}_i |  Activation (output) of unit \textstyle i in layer \textstyle l of the network.
In addition, since layer \textstyle L_1 is the input layer, we also have \textstyle a^{(1)}_i = x_i.
 |
| \textstyle f(\cdot) |  The activation function. Throughout these notes, we used \textstyle f(z) = \tanh(z).
 |
| \textstyle z^{(l)}_i |  Total weighted sum of inputs to unit \textstyle i in layer \textstyle l. Thus, \textstyle a^{(l)}_i = f(z^{(l)}_i).
 |
| \textstyle \alpha |  Learning rate parameter
 |
| \textstyle s_l |  Number of units in layer \textstyle l (not counting the bias unit).
 |
| \textstyle n_l |  Number layers in the network. Layer \textstyle L_1 is usually the input layer, and layer \textstyle L_{n_l} the output layer.
 |
| \textstyle \lambda |  Weight decay parameter.
 |
| \textstyle \hat{x} |  For an autoencoder, its output; i.e., its reconstruction of the input \textstyle x. Same meaning as \textstyle h_{W,b}(x).
 |
| \textstyle \rho |  Sparsity parameter, which specifies our desired level of sparsity
 |
| \textstyle \hat\rho_i |  The average activation of hidden unit \textstyle i (in the sparse autoencoder).
 |
| \textstyle \beta |  Weight of the sparsity penalty term (in the sparse autoencoder objective).
 |

[Neural Networks](Neural_Networks.md "Neural Networks") | [Backpropagation Algorithm](Backpropagation_Algorithm.md "Backpropagation Algorithm") | [Gradient checking and advanced optimization](Gradient_checking_and_advanced_optimization.md "Gradient checking and advanced optimization") | [Autoencoders and Sparsity](Autoencoders_and_Sparsity.md "Autoencoders and Sparsity") | [Visualizing a Trained Autoencoder](Visualizing_a_Trained_Autoencoder.md "Visualizing a Trained Autoencoder") | **Sparse Autoencoder Notation Summary** | [Exercise:Sparse Autoencoder](Exercise_Sparse_Autoencoder.md "Exercise:Sparse Autoencoder")

---

> * Language: [中文](%E7%A8%80%E7%96%8F%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E7%AC%A6%E5%8F%B7%E4%B8%80%E8%A7%88%E8%A1%A8.md "稀疏自编码器符号一览表")
> * This page was last modified on 7 April 2013, at 12:45.

