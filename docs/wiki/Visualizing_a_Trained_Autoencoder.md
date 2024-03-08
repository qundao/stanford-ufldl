Visualizing a Trained Autoencoder
=================================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
Having trained a (sparse) autoencoder, we would now like to visualize the function
learned by the algorithm, to try to understand what it has learned.
Consider the case of training an autoencoder on ![\textstyle 10 \times 10](images/math/0/4/a/04aaf6cd0499a40a7c222ffdb85b55bb.png) images, so that ![\textstyle n = 100](images/math/5/4/8/548f3e32e47803886a1aacb25f80e82c.png).
Each hidden unit ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png) computes a function of the input:

![\begin{align}
a^{(2)}_i = f\left(\sum_{j=1}^{100} W^{(1)}_{ij} x_j  + b^{(1)}_i \right).
\end{align}](images/math/1/d/2/1d29407eddf5fc12ca94509c9a9f7979.png)

We will visualize the function computed by hidden unit ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)---which depends on the
parameters ![\textstyle W^{(1)}_{ij}](images/math/8/2/d/82d79561e2994ccba3e4fe2cc4d527e5.png) (ignoring
the bias term for now)---using a 2D image. In particular, we think of
![\textstyle a^{(2)}_i](images/math/e/1/4/e14f36d1b33f6ed0dc131a7ddd166004.png) as some non-linear feature of the input ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png).
We ask:
What input image ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) would cause
![\textstyle a^{(2)}_i](images/math/e/1/4/e14f36d1b33f6ed0dc131a7ddd166004.png) to be maximally activated?
(Less formally, what is the feature that hidden unit ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png) is looking for?)
For this question to have a non-trivial answer,
we must impose some constraints on ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png). If we suppose that
the input is
norm constrained by ![\textstyle ||x||^2 = \sum_{i=1}^{100} x_i^2 \leq 1](images/math/4/7/7/4777ad65a6cc46e9f07e4100cddf4161.png), then one can
show (try doing this yourself)
that the input which maximally activates hidden unit ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png) is given
by setting pixel ![\textstyle x_j](images/math/b/d/f/bdf5b20642553027712d5b5240b31cf3.png) (for all 100 pixels, ![\textstyle j=1,\ldots, 100](images/math/9/6/6/966104699d82737184a65294fddd8eea.png)) to

![\begin{align}
x_j = \frac{W^{(1)}_{ij}}{\sqrt{\sum_{j=1}^{100} (W^{(1)}_{ij})^2}}.
\end{align}](images/math/5/4/0/540c1290f18272da2c83610bd1c18380.png)

By displaying the image formed by these pixel intensity values, we can begin
to understand what feature hidden unit ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png) is looking for.

If we have an autoencoder with 100 hidden units (say), then we our
visualization will have 100 such images---one per hidden unit. By examining
these 100 images, we can try to understand what the ensemble of hidden units is
learning.

When we do this for a sparse autoencoder (trained with 100 hidden units on
10x10 pixel inputs1 we get the following result:

![ExampleSparseAutoencoderWeights.png](images/thumb/3/3e/ExampleSparseAutoencoderWeights.png/400px-ExampleSparseAutoencoderWeights.png) ![](/wayback-mooc/stanford-ufldl/wiki/skins/common/images/magnify-clip.png)
Each square in the figure above shows the (norm bounded) input image ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) that
maximally actives one of 100 hidden units. We see that the different hidden
units have learned to detect edges at different positions and orientations in
the image.

These features are, not surprisingly, useful for such tasks as object
recognition and other vision tasks. When applied to other input domains (such
as audio), this algorithm also learns useful representations/features for those
domains too.

---

1 *The learned features were obtained by training on **whitened** natural images. Whitening is a preprocessing step which removes redundancy in the input, by causing adjacent pixels to become less correlated.*

[Neural Networks](Neural_Networks.md "Neural Networks") | [Backpropagation Algorithm](Backpropagation_Algorithm.md "Backpropagation Algorithm") | [Gradient checking and advanced optimization](Gradient_checking_and_advanced_optimization.md "Gradient checking and advanced optimization") | [Autoencoders and Sparsity](Autoencoders_and_Sparsity.md "Autoencoders and Sparsity") | **Visualizing a Trained Autoencoder** | [Sparse Autoencoder Notation Summary](Sparse_Autoencoder_Notation_Summary.md "Sparse Autoencoder Notation Summary") | [Exercise:Sparse Autoencoder](Exercise_Sparse_Autoencoder.md "Exercise:Sparse Autoencoder")

---

> * Language: [中文](%E5%8F%AF%E8%A7%86%E5%8C%96%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9E%9C.md "可视化自编码器训练结果")
> * This page was last modified on 7 April 2013, at 12:49.

