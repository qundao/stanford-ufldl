Whitening
=========

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->

|  |
| --- |
| Contents* [1 Introduction](#Introduction)
* [2 2D example](#2D_example)
* [3 ZCA Whitening](#ZCA_Whitening)
* [4 Regularizaton](#Regularizaton)
 |

  Introduction
--------------

We have used PCA to reduce the dimension of the data. There is a closely related
preprocessing step called **whitening** (or, in some other literatures, **sphering**)
which is needed for some algorithms. If we are training on images,
the raw input is redundant, since adjacent pixel values
are highly correlated. The goal of whitening is to make the input less redundant; more formally,
our desiderata are that our learning algorithms sees a training input where (i) the features are less
correlated with each other, and (ii) the features all have the same variance.

  2D example
------------

We will first describe whitening using our previous 2D example. We will then 
describe how this can be combined with smoothing, and finally how to combine
this with PCA.

How can we make our input features uncorrelated with each other? We had
already done this when computing ![\textstyle x_{\rm rot}^{(i)} = U^Tx^{(i)}](images/math/c/d/0/cd047246fd68f6d52b2fd068e063c0ef.png). 
Repeating our previous figure, our plot for ![\textstyle x_{\rm rot}](images/math/1/7/0/170047e804738636731477291969d554.png) was:

![PCA-rotated.png](images/thumb/1/12/PCA-rotated.png/600px-PCA-rotated.png)

The covariance matrix of this data is given by:

![\begin{align}
\begin{bmatrix}
7.29 & 0  \\
0 & 0.69
\end{bmatrix}.
\end{align}](images/math/f/e/5/fe5ed797b9c818df5bc8bf5d5c001e02.png)

(Note: Technically, many of the
statements in this section about the "covariance" will be true only if the data
has zero mean. In the rest of this section, we will take this assumption as
implicit in our statements. However, even if the data's mean isn't exactly zero, 
the intuitions we're presenting here still hold true, and so this isn't something
that you should worry about.)

It is no accident that the diagonal values are ![\textstyle \lambda_1](images/math/e/1/3/e138a7c8755e6a4400dd6bb08974d139.png) and ![\textstyle \lambda_2](images/math/4/1/a/41ab4ee633f1ad3d25809270aedbe566.png). 
Further, 
the off-diagonal entries are zero; thus, 
![\textstyle x_{{\rm rot},1}](images/math/0/0/6/0066d1e2efa2f0019a3dfd3469862934.png) and ![\textstyle x_{{\rm rot},2}](images/math/3/f/2/3f2601aaa1d6e648c789bd9a831cc4eb.png) are uncorrelated, satisfying one of our desiderata 
for whitened data (that the features be less correlated).

To make each of our input features have unit variance, we can simply rescale
each feature ![\textstyle x_{{\rm rot},i}](images/math/d/1/5/d1527b3272bc5c1fe3fc308c7a21e689.png) by ![\textstyle 1/\sqrt{\lambda_i}](images/math/7/a/d/7ad8b4911f758fec9b3c6d0b4b61a82c.png). Concretely, we define
our whitened data ![\textstyle x_{{\rm PCAwhite}} \in \Re^n](images/math/9/6/9/9693d90272b2475c8369fa23df7267ed.png) as follows:

![\begin{align}
x_{{\rm PCAwhite},i} = \frac{x_{{\rm rot},i} }{\sqrt{\lambda_i}}.   
\end{align}](images/math/e/2/9/e296118ba2bdf453dbe38426359f2230.png)

Plotting ![\textstyle x_{{\rm PCAwhite}}](images/math/a/3/1/a3135c6f5975d0a74cd2d9082be9638a.png), we get:

![PCA-whitened.png](images/thumb/9/98/PCA-whitened.png/600px-PCA-whitened.png)

This data now has covariance equal to the identity matrix ![\textstyle I](images/math/5/4/f/54f708ffb9cc17b9820863a36120c90c.png). We say that
![\textstyle x_{{\rm PCAwhite}}](images/math/a/3/1/a3135c6f5975d0a74cd2d9082be9638a.png) is our **PCA whitened** version of the data: The 
different components of ![\textstyle x_{{\rm PCAwhite}}](images/math/a/3/1/a3135c6f5975d0a74cd2d9082be9638a.png) are uncorrelated and have
unit variance.

**Whitening combined with dimensionality reduction.** 
If you want to have data that is whitened and which is lower dimensional than
the original input, you can also optionally keep only the top ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) components of
![\textstyle x_{{\rm PCAwhite}}](images/math/a/3/1/a3135c6f5975d0a74cd2d9082be9638a.png). When we combine PCA whitening with regularization
(described later), the last few components of ![\textstyle x_{{\rm PCAwhite}}](images/math/a/3/1/a3135c6f5975d0a74cd2d9082be9638a.png) will be
nearly zero anyway, and thus can safely be dropped.

  ZCA Whitening
---------------

Finally, it turns out that this way of getting the 
data to have covariance identity ![\textstyle I](images/math/5/4/f/54f708ffb9cc17b9820863a36120c90c.png) isn't unique. 
Concretely, if 
![\textstyle R](images/math/f/e/e/fee54137ee7748e26642e71145effa05.png) is any orthogonal matrix, so that it satisfies ![\textstyle RR^T = R^TR = I](images/math/7/7/d/77d64d6a092c3f7adb9eae6eb4af41ff.png) (less formally,
if ![\textstyle R](images/math/f/e/e/fee54137ee7748e26642e71145effa05.png) is a rotation/reflection matrix),
then ![\textstyle R \,x_{\rm PCAwhite}](images/math/b/c/d/bcd43a98b71d807cddbdb7a3a33bbc1a.png) will also have identity covariance. 
In **ZCA whitening**,
we choose ![\textstyle R = U](images/math/b/6/1/b61977ba8ab2bacb0c31fa5575db43fd.png). We define

![\begin{align}
x_{\rm ZCAwhite} = U x_{\rm PCAwhite}
\end{align}](images/math/c/f/b/cfb1fa6b1049a5fdb2da4d7e88856751.png)

Plotting ![\textstyle x_{\rm ZCAwhite}](images/math/a/6/6/a668553308d25ae0f796a9f92c807931.png), we get:

![ZCA-whitened.png](images/thumb/a/a4/ZCA-whitened.png/600px-ZCA-whitened.png)

It can be shown that out of all possible choices for ![\textstyle R](images/math/f/e/e/fee54137ee7748e26642e71145effa05.png), 
this choice of rotation causes ![\textstyle x_{\rm ZCAwhite}](images/math/a/6/6/a668553308d25ae0f796a9f92c807931.png) to be as close as possible to the 
original input data ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png).

When using ZCA whitening (unlike PCA whitening), we usually keep all ![\textstyle n](images/math/0/c/5/0c59de0fa75c1baa1c024aabfa43b2e3.png) dimensions
of the data, and do not try to reduce its dimension.

  Regularizaton
---------------

When implementing PCA whitening or ZCA whitening in practice, sometimes some
of the eigenvalues ![\textstyle \lambda_i](images/math/2/3/5/23536ce45f0ee57fffa389163f8437bd.png) will be numerically close to 0, and thus the scaling
step where we divide by ![\sqrt{\lambda_i}](images/math/3/e/8/3e85dc0c50d11861f9d02bb43ab2d989.png) would involve dividing by a value close to zero; this 
may cause the data to blow up (take on large values) or otherwise be numerically unstable. In practice, we 
therefore implement this scaling step using 
a small amount of regularization, and add a small constant ![\textstyle \epsilon](images/math/a/8/e/a8eae7b5e90c024c40de690158e0e6b1.png) 
to the eigenvalues before taking their square root and inverse:

![\begin{align}
x_{{\rm PCAwhite},i} = \frac{x_{{\rm rot},i} }{\sqrt{\lambda_i + \epsilon}}.
\end{align}](images/math/6/7/b/67be9aaa628b437297c08a916d0d5307.png)

When ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) takes values around ![\textstyle [-1,1]](images/math/8/5/a/85a1c5a07f21a9eebbfb1dca380f8d38.png), a value of ![\textstyle \epsilon \approx 10^{-5}](images/math/c/d/d/cdd6f0cc52395a161edf391fad0ef2ef.png)
might be typical.

For the case of images, adding ![\textstyle \epsilon](images/math/a/8/e/a8eae7b5e90c024c40de690158e0e6b1.png) here also has the effect of slightly smoothing (or low-pass
filtering) the input image. This also has a desirable effect of removing aliasing artifacts
caused by the way pixels are laid out in an image, and can improve the features learned 
(details are beyond the scope of these notes).

ZCA whitening is a form of pre-processing of the data that maps it from ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) to
![\textstyle x_{\rm ZCAwhite}](images/math/a/6/6/a668553308d25ae0f796a9f92c807931.png). It turns out that this is also a rough model of how the
biological eye (the retina) processes images. Specifically, as your eye
perceives images, most adjacent "pixels" in your eye will perceive very
similar values, since adjacent parts of an image tend to be highly correlated
in intensity. It is thus wasteful for your eye to have to transmit every pixel
separately (via your optic nerve) to your brain. Instead, your retina performs
a decorrelation operation (this is done via retinal neurons that compute a function
called "on center, off surround/off center, on surround") which is similar to that
performed by ZCA. This results in a less redundant representation of the input
image, which is then transmitted to your brain.

[PCA](PCA.md "PCA") | **Whitening** | [Implementing PCA/Whitening](/wayback-mooc/stanford-ufldl/wiki/Implementing_PCA/Whitening "Implementing PCA/Whitening") | [Exercise:PCA in 2D](Exercise_PCA_in_2D.md "Exercise:PCA in 2D") | [Exercise:PCA and Whitening](Exercise_PCA_and_Whitening.md "Exercise:PCA and Whitening")

---

> * Language: [中文](%E7%99%BD%E5%8C%96.md "白化")
> * This page was last modified on 7 April 2013, at 13:20.

