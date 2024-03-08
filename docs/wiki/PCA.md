PCA
===

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->

|  |
| --- |
| Contents* [1 Introduction](#Introduction)
* [2 Example and Mathematical Background](#Example_and_Mathematical_Background)
* [3 Rotating the Data](#Rotating_the_Data)
* [4 Reducing the Data Dimension](#Reducing_the_Data_Dimension)
* [5 Recovering an Approximation of the Data](#Recovering_an_Approximation_of_the_Data)
* [6 Number of components to retain](#Number_of_components_to_retain)
* [7 PCA on Images](#PCA_on_Images)
* [8 References](#References)
 |

  Introduction
--------------

Principal Components Analysis (PCA) is a dimensionality reduction algorithm
that can be used to significantly speed up your unsupervised feature learning
algorithm. More importantly, understanding PCA will enable us to later
implement **whitening**, which is an important pre-processing step for many
algorithms.

Suppose you are training your algorithm on images. Then the input will be
somewhat redundant, because the values of adjacent pixels in an image are
highly correlated. Concretely, suppose we are training on 16x16 grayscale
image patches. Then ![\textstyle x \in \Re^{256}](images/math/3/e/c/3ec732c534e730334fbe728ae49c8fce.png) are 256 dimensional vectors, with one
feature ![\textstyle x_j](images/math/b/d/f/bdf5b20642553027712d5b5240b31cf3.png) corresponding to the intensity of each pixel. Because of the
correlation between adjacent pixels, PCA will allow us to approximate the input with
a much lower dimensional one, while incurring very little error.

  Example and Mathematical Background
-------------------------------------

For our running example, we will use a dataset 
![\textstyle \{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}](images/math/b/b/f/bbfa674fd83f37c2c66867d7e0cc264a.png) with 
![\textstyle n=2](images/math/b/1/9/b1993eef97e184af6b11db01e694445f.png) dimensional inputs, so that 
![\textstyle x^{(i)} \in \Re^2](images/math/1/b/a/1babb19c8b06f9a7bd624fa60f29d5fb.png).
Suppose we want to reduce the data 
from 2 dimensions to 1. (In practice, we might want to reduce data
from 256 to 50 dimensions, say; but using lower dimensional data in our example
allows us to visualize the algorithms better.) Here is our dataset:

![PCA-rawdata.png](images/thumb/b/ba/PCA-rawdata.png/600px-PCA-rawdata.png)

This data has already been pre-processed so that each of the features ![\textstyle x_1](images/math/f/a/7/fa7eebd32aa8c9cdae2b2aacbc324331.png) and ![\textstyle x_2](images/math/7/6/8/76879b7da23d4991dfcb03323403c152.png)
have about the same mean (zero) and variance.

For the purpose of illustration, we have also colored each of the points one of
three colors, depending on their ![\textstyle x_1](images/math/f/a/7/fa7eebd32aa8c9cdae2b2aacbc324331.png) value; these colors are not used by the
algorithm, and are for illustration only.

PCA will find a lower-dimensional subspace onto which to project our data. 
From visually examining the data, it appears that ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) is the principal direction of 
variation of the data, and ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png) the secondary direction of variation:

![PCA-u1.png](images/thumb/b/b4/PCA-u1.png/600px-PCA-u1.png)

I.e., the data varies much more in the direction ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) than ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png). 
To more formally find the directions ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) and ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png), we first compute the matrix ![\textstyle \Sigma](images/math/6/6/9/669ec82a71dede49eb73e539bc3423b6.png)
as follows:

![\begin{align}
\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)})(x^{(i)})^T. 
\end{align}](images/math/d/a/9/da9b50ec05dbe4ae513e4f52093b8342.png)

If ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) has zero mean, then ![\textstyle \Sigma](images/math/6/6/9/669ec82a71dede49eb73e539bc3423b6.png) is exactly the covariance matrix of ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png). (The symbol "![\textstyle \Sigma](images/math/6/6/9/669ec82a71dede49eb73e539bc3423b6.png)", pronounced "Sigma", is the standard notation for denoting the covariance matrix. Unfortunately it looks just like the summation symbol, as in ![\sum_{i=1}^n i](images/math/7/3/b/73b577d2b026ab8f8fb733953266427e.png); but these are two different things.)

It can then be shown that ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png)---the principal direction of variation of the data---is 
the top (principal) eigenvector of ![\textstyle \Sigma](images/math/6/6/9/669ec82a71dede49eb73e539bc3423b6.png), and ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png) is
the second eigenvector.

Note: If you are interested in seeing a more formal mathematical derivation/justification of this result, see the CS229 (Machine Learning) lecture notes on PCA (link at bottom of this page). You won't need to do so to follow along this course, however.

You can use standard numerical linear algebra software to find these eigenvectors (see Implementation Notes).
Concretely, let us compute the eigenvectors of ![\textstyle \Sigma](images/math/6/6/9/669ec82a71dede49eb73e539bc3423b6.png), and stack
the eigenvectors in columns to form the matrix ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png):

![\begin{align}
U = 
\begin{bmatrix} 
| & | & & |  \\
u_1 & u_2 & \cdots & u_n  \\
| & | & & | 
\end{bmatrix} 		
\end{align}](images/math/6/9/0/6906da1c5ac5f7f94a3b337447e69360.png)

Here, ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) is the principal eigenvector (corresponding to the largest eigenvalue),
![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png) is the second eigenvector, and so on. 
Also, let ![\textstyle \lambda_1, \lambda_2, \ldots, \lambda_n](images/math/d/2/b/d2b02582947d98e3be81be3d1e684f28.png) be the corresponding eigenvalues.

The vectors ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) and ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png) in our example form a new basis in which we 
can represent the data. Concretely, let ![\textstyle x \in \Re^2](images/math/b/2/6/b260df225bb49f3ff776b17a50cd20d3.png) be some training example. Then ![\textstyle u_1^Tx](images/math/7/c/0/7c0e7fb10fb6e75bad211b2f2070c24c.png)
is the length (magnitude) of the projection of ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) onto the vector ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png).

Similarly, ![\textstyle u_2^Tx](images/math/3/8/9/389b689de5736f95b05c3be9c373b95a.png) is the magnitude of ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) projected onto the vector ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png).

  Rotating the Data
-------------------

Thus, we can represent ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) in the ![\textstyle (u_1, u_2)](images/math/0/3/2/0329a7ca7eca352beded9f24406d34fe.png)-basis by computing

![\begin{align}
x_{\rm rot} = U^Tx = \begin{bmatrix} u_1^Tx \\ u_2^Tx \end{bmatrix} 
\end{align}](images/math/e/a/a/eaa1e40a68e966dd5a3d272dd6d091ed.png)

(The subscript "rot" comes from the observation that this corresponds to
a rotation (and possibly reflection) of the original data.)
Lets take the entire training set, and compute 
![\textstyle x_{\rm rot}^{(i)} = U^Tx^{(i)}](images/math/c/d/0/cd047246fd68f6d52b2fd068e063c0ef.png) for every ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png). Plotting this transformed data 
![\textstyle x_{\rm rot}](images/math/1/7/0/170047e804738636731477291969d554.png), we get:

![PCA-rotated.png](images/thumb/1/12/PCA-rotated.png/600px-PCA-rotated.png)

This is the training set rotated into the ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png),![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png) basis. In the general
case, ![\textstyle U^Tx](images/math/e/0/a/e0aec5d033ea89dc9bd9c83bc2b4edec.png) will be the training set rotated into the basis 
![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png),![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png), ...,![\textstyle u_n](images/math/0/b/e/0be80bb4e50881840b92fb8331ef2bbd.png).

One of the properties of ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) is that it is an "orthogonal" matrix, which means
that it satisfies ![\textstyle U^TU = UU^T = I](images/math/a/8/2/a825fd85c23ffa9b851fb64c9c816ad6.png). 
So if you ever need to go from the rotated vectors ![\textstyle x_{\rm rot}](images/math/1/7/0/170047e804738636731477291969d554.png) back to the 
original data ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png), you can compute

![\begin{align}
x = U x_{\rm rot}   ,
\end{align}](images/math/7/f/8/7f865e9a54f1151f48b8e6f433e50ea0.png)

because ![\textstyle U x_{\rm rot} =  UU^T x = x](images/math/a/5/f/a5fa6224542f5b2871447986260574d2.png).

  Reducing the Data Dimension
-----------------------------

We see that the principal direction of variation of the data is the first
dimension ![\textstyle x_{{\rm rot},1}](images/math/0/0/6/0066d1e2efa2f0019a3dfd3469862934.png) of this rotated data. Thus, if we want to
reduce this data to one dimension, we can set

![\begin{align}
\tilde{x}^{(i)} = x_{{\rm rot},1}^{(i)} = u_1^Tx^{(i)} \in \Re.
\end{align}](images/math/8/c/f/8cf51b2f3bf8c78ad1b03c27aa68f692.png)

More generally, if ![\textstyle x \in \Re^n](images/math/9/e/b/9ebd39996afb169318c1dd5fb1503b17.png) and we want to reduce it to 
a ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) dimensional representation ![\textstyle \tilde{x} \in \Re^k](images/math/2/1/3/21337248295f42f7fe18d9a9b3da57b1.png) (where ![\textstyle k < n](images/math/8/7/b/87b6508de7e0487479389cff2b5fa91a.png)), we would
take the first ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) components of ![\textstyle x_{\rm rot}](images/math/1/7/0/170047e804738636731477291969d554.png), which correspond to
the top ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) directions of variation.

Another way of explaining PCA is that ![\textstyle x_{\rm rot}](images/math/1/7/0/170047e804738636731477291969d554.png) is an ![\textstyle n](images/math/0/c/5/0c59de0fa75c1baa1c024aabfa43b2e3.png) dimensional
vector, where the first few components are likely to 
be large (e.g., in our example, we saw that ![\textstyle x_{{\rm rot},1}^{(i)} = u_1^Tx^{(i)}](images/math/8/0/e/80ebba0459d97a31a03e9de6b0957c31.png) takes
reasonably large values for most examples ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)), and
the later components are likely to be small (e.g., in our example, 
![\textstyle x_{{\rm rot},2}^{(i)} = u_2^Tx^{(i)}](images/math/4/6/8/468a726aaaea7f4aabbeb8a2e1966aae.png) was more likely to be small). What
PCA does it it 
drops the the later (smaller) components of ![\textstyle x_{\rm rot}](images/math/1/7/0/170047e804738636731477291969d554.png), and
just approximates them with 0's. Concretely, our definition of 
![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png) can also be arrived at by using an approximation to
![\textstyle x_{{\rm rot}}](images/math/7/7/4/774d8fa9b41f58dfc57cebb419e0de60.png) where 
all but the first
![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) components are zeros. In other words, we have:

![\begin{align}
\tilde{x} = 
\begin{bmatrix} 
x_{{\rm rot},1} \\
\vdots \\ 
x_{{\rm rot},k} \\
0 \\ 
\vdots \\ 
0 \\ 
\end{bmatrix}
\approx 
\begin{bmatrix} 
x_{{\rm rot},1} \\
\vdots \\ 
x_{{\rm rot},k} \\
x_{{\rm rot},k+1} \\
\vdots \\ 
x_{{\rm rot},n} 
\end{bmatrix}
= x_{\rm rot} 
\end{align}](images/math/5/e/8/5e8f3f68a933310015faa1eb439749f8.png)

In our example, this gives us the following plot of ![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png) (using ![\textstyle n=2, k=1](images/math/9/4/b/94b3c8bb8f57addfc319217446a14d56.png)):

![PCA-xtilde.png](images/thumb/2/27/PCA-xtilde.png/600px-PCA-xtilde.png)

However, since the final ![\textstyle n-k](images/math/7/4/2/742be0073915ce28ed208c2d5c83fc56.png) components of ![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png) as defined above would
always be zero, there is no need to keep these zeros around, and so we
define ![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png) as a ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png)-dimensional vector with just the first ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) (non-zero) components.

This also explains why we wanted to express our data in the ![\textstyle u_1, u_2, \ldots, u_n](images/math/d/5/2/d52832ed87962d3ece3043ddae3150a7.png) basis:
Deciding which components to keep becomes just keeping the top ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) components. When we
do this, we also say that we are "retaining the top ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) PCA (or principal) components."

  Recovering an Approximation of the Data
-----------------------------------------

Now, ![\textstyle \tilde{x} \in \Re^k](images/math/2/1/3/21337248295f42f7fe18d9a9b3da57b1.png) is a lower-dimensional, "compressed" representation
of the original ![\textstyle x \in \Re^n](images/math/9/e/b/9ebd39996afb169318c1dd5fb1503b17.png). Given ![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png), how can we recover an approximation ![\textstyle \hat{x}](images/math/2/9/0/29035749c12270bcc8de7e36bc459ece.png) to 
the original value of ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png)? From an [earlier section](#Rotating_the_Data), we know that ![\textstyle x = U x_{\rm rot}](images/math/f/a/a/faada910e82b90d1c221943616cc85ab.png). Further, 
we can think of ![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png) as an approximation to ![\textstyle x_{\rm rot}](images/math/1/7/0/170047e804738636731477291969d554.png), where we have
set the last ![\textstyle n-k](images/math/7/4/2/742be0073915ce28ed208c2d5c83fc56.png) components to zeros. Thus, given ![\textstyle \tilde{x} \in \Re^k](images/math/2/1/3/21337248295f42f7fe18d9a9b3da57b1.png), we can 
pad it out with ![\textstyle n-k](images/math/7/4/2/742be0073915ce28ed208c2d5c83fc56.png) zeros to get our approximation to ![\textstyle x_{\rm rot} \in \Re^n](images/math/f/c/5/fc52a57fe97de0666dc2857bde2df153.png). Finally, we pre-multiply
by ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) to get our approximation to ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png). Concretely, we get

![\begin{align}
\hat{x}  = U \begin{bmatrix} \tilde{x}_1 \\ \vdots \\ \tilde{x}_k \\ 0 \\ \vdots \\ 0 \end{bmatrix}  
= \sum_{i=1}^k u_i \tilde{x}_i.
\end{align}](images/math/0/a/0/0a07b56293a0b63ef434551e9ccda9ea.png)

The final equality above comes from the definition of ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) [given earlier](#Example_and_Mathematical_Background).
(In a practical implementation, we wouldn't actually zero pad ![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png) and then multiply
by ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png), since that would mean multiplying a lot of things by zeros; instead, we'd just 
multiply ![\textstyle \tilde{x} \in \Re^k](images/math/2/1/3/21337248295f42f7fe18d9a9b3da57b1.png) with the first ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) columns of ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) as in the final expression above.)
Applying this to our dataset, we get the following plot for ![\textstyle \hat{x}](images/math/2/9/0/29035749c12270bcc8de7e36bc459ece.png):

![PCA-xhat.png](images/thumb/5/52/PCA-xhat.png/600px-PCA-xhat.png)

We are thus using a 1 dimensional approximation to the original dataset.

If you are training an autoencoder or other unsupervised feature learning algorithm,
the running time of your algorithm will depend on the dimension of the input. If you feed ![\textstyle \tilde{x} \in \Re^k](images/math/2/1/3/21337248295f42f7fe18d9a9b3da57b1.png)
into your learning algorithm instead of ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png), then you'll be training on a lower-dimensional
input, and thus your algorithm might run significantly faster. For many datasets,
the lower dimensional ![\textstyle \tilde{x}](images/math/1/a/6/1a62e33dcf57261829692126a4dcd02d.png) representation can be an extremely good approximation 
to the original, and using PCA this way can significantly speed up your algorithm while
introducing very little approximation error.

  Number of components to retain
--------------------------------

How do we set ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png); i.e., how many PCA components should we retain? In our
simple 2 dimensional example, it seemed natural to retain 1 out of the 2
components, but for higher dimensional data, this decision is less trivial. If ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) is
too large, then we won't be compressing the data much; in the limit of ![\textstyle k=n](images/math/e/3/6/e36b85de9c58866d875f20cbf6fc5f5b.png),
then we're just using the original data (but rotated into a different basis).
Conversely, if ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) is too small, then we might be using a very bad
approximation to the data.

To decide how to set ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png), we will usually look at the **percentage of variance retained** 
for different values of ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png). Concretely, if ![\textstyle k=n](images/math/e/3/6/e36b85de9c58866d875f20cbf6fc5f5b.png), then we have
an exact approximation to the data, and we say that 100% of the variance is
retained. I.e., all of the variation of the original data is retained. 
Conversely, if ![\textstyle k=0](images/math/2/a/2/2a27a4874f5739de5d2947d12ac81d4b.png), then we are approximating all the data with the zero vector,
and thus 0% of the variance is retained.

More generally, let ![\textstyle \lambda_1, \lambda_2, \ldots, \lambda_n](images/math/d/2/b/d2b02582947d98e3be81be3d1e684f28.png) be the eigenvalues 
of ![\textstyle \Sigma](images/math/6/6/9/669ec82a71dede49eb73e539bc3423b6.png) (sorted in decreasing order), so that ![\textstyle \lambda_j](images/math/c/8/5/c851ef66a35ee95db0b63a592963ca77.png) is the eigenvalue
corresponding to the eigenvector ![\textstyle u_j](images/math/d/1/7/d175faaca44b996970abf70b700a94f1.png). Then if we retain ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) principal components, 
the percentage of variance retained is given by:

![\begin{align}
\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^n \lambda_j}.
\end{align}](images/math/6/0/b/60ba13aa4527ce9cc11772deaa1d5027.png)

In our simple 2D example above, ![\textstyle \lambda_1 = 7.29](images/math/6/b/f/6bf8708608604abff35895bb0ecf17f3.png), and ![\textstyle \lambda_2 = 0.69](images/math/5/7/9/5793e844fa46435301414cb62e5d7641.png). Thus,
by keeping only ![\textstyle k=1](images/math/9/7/7/97724a53ab7a652f75e945d2188850d9.png) principal components, we retained ![\textstyle 7.29/(7.29+0.69) = 0.913](images/math/8/5/9/859dd2ebffe06849e75ce9297f25d325.png),
or 91.3% of the variance.

A more formal definition of percentage of variance retained is beyond the scope
of these notes. However, it is possible to show that ![\textstyle \lambda_j =
\sum_{i=1}^m x_{{\rm rot},j}^2](images/math/9/7/e/97ecfffd8596d26deed9542b64cd6712.png). Thus, if ![\textstyle \lambda_j \approx 0](images/math/6/7/1/6716d88c3c1a368824d188c8b9b6b589.png), that shows that
![\textstyle x_{{\rm rot},j}](images/math/e/8/4/e84f84acac7b07e18a42a8e91b4433bc.png) is usually near 0 anyway, and we lose relatively little by
approximating it with a constant 0. This also explains why we retain the top principal
components (corresponding to the larger values of ![\textstyle \lambda_j](images/math/c/8/5/c851ef66a35ee95db0b63a592963ca77.png)) instead of the bottom
ones. The top principal components 
![\textstyle x_{{\rm rot},j}](images/math/e/8/4/e84f84acac7b07e18a42a8e91b4433bc.png) are the ones that're more variable and that take on larger values, 
and for which we would incur a greater approximation error if we were to set them to zero.

In the case of images, one common heuristic is to choose ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) so as to retain 99% of
the variance. In other words, we pick the smallest value of ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) that satisfies

![\begin{align}
\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^n \lambda_j} \geq 0.99. 
\end{align}](images/math/7/d/5/7d5f701649af052a671b7d195dccdd8f.png)

Depending on the application, if you are willing to incur some 
additional error, values in the 90-98% range are also sometimes used. When you
describe to others how you applied PCA, saying that you chose ![\textstyle k](images/math/b/0/0/b0066e761791cae480158b649e5f5a69.png) to retain 95% of
the variance will also be a much more easily interpretable description than saying
that you retained 120 (or whatever other number of) components.

  PCA on Images
---------------

For PCA to work, usually we want each of the features ![\textstyle x_1, x_2, \ldots, x_n](images/math/f/2/5/f25d5eb460ed8f894d9be2865a286908.png)
to have a similar range of values to the others (and to have a mean close to
zero). If you've used PCA on other applications before, you may therefore have
separately pre-processed each feature to have zero mean and unit variance, by
separately estimating the mean and variance of each feature ![\textstyle x_j](images/math/b/d/f/bdf5b20642553027712d5b5240b31cf3.png). However,
this isn't the pre-processing that we will apply to most types of images. Specifically,
suppose we are training our algorithm on **natural images**, so that ![\textstyle x_j](images/math/b/d/f/bdf5b20642553027712d5b5240b31cf3.png) is
the value of pixel ![\textstyle j](images/math/2/3/5/235c5146ab110558897640c34dad7d97.png). By "natural images," we informally mean the type of image that
a typical animal or person might see over their lifetime.

Note: Usually we use images of outdoor scenes with grass, trees, etc., and cut out small (say 16x16) image patches randomly from these to train the algorithm. But in practice most feature learning algorithms are extremely robust to the exact type of image it is trained on, so most images taken with a normal camera, so long as they aren't excessively blurry or have strange artifacts, should work.

When training on natural images, it makes little sense to estimate a separate mean and
variance for each pixel, because the statistics in one part
of the image should (theoretically) be the same as any other. 
This property of images is called **stationarity.**

In detail, in order for PCA to work well, informally we require that (i) The
features have approximately zero mean, and (ii) The different features have
similar variances to each other. With natural images, (ii) is already
satisfied even without variance normalization, and so we won't perform any 
variance normalization. 
(If you are training on audio data---say, on
spectrograms---or on text data---say, bag-of-word vectors---we will usually not perform
variance normalization either.) 
In fact, PCA is invariant to the scaling of
the data, and will return the same eigenvectors regardless of the scaling of
the input. More formally, if you multiply each feature vector ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) by some
positive number (thus scaling every feature in every training example by the
same number), PCA's output eigenvectors will not change.

So, we won't use variance normalization. The only normalization we need to
perform then is mean normalization, to ensure that the features have a mean
around zero. Depending on the application, very often we are not interested
in how bright the overall input image is. For example, in object recognition
tasks, the overall brightness of the image doesn't affect what objects
there are in the image. More formally, we are not interested in the
mean intensity value of an image patch; thus, we can subtract out this value,
as a form of mean normalization.

Concretely, if ![\textstyle x^{(i)} \in \Re^{n}](images/math/c/a/5/ca57b44909d158c3fdfaa849465dd4a2.png) are the (grayscale) intensity values of
a 16x16 image patch (![\textstyle n=256](images/math/6/c/0/6c07d223cfb098a75db66924dfcb7210.png)), we might normalize the intensity of each image
![\textstyle x^{(i)}](images/math/e/b/e/ebe8632b7c91a3dbbf9b590bea887a47.png) as follows:

![\mu^{(i)} := \frac{1}{n} \sum_{j=1}^n x^{(i)}_j](images/math/a/1/0/a104802ef43230cf0d364f378abd2c08.png)

![x^{(i)}_j := x^{(i)}_j - \mu^{(i)}](images/math/6/3/b/63bf04b76d7fffd53d851573573f5f7f.png), for all ![\textstyle j](images/math/2/3/5/235c5146ab110558897640c34dad7d97.png)

Note that the two steps above are done separately for each image ![\textstyle x^{(i)}](images/math/e/b/e/ebe8632b7c91a3dbbf9b590bea887a47.png),
and that ![\textstyle \mu^{(i)}](images/math/c/8/6/c862daa56646826c788aeb8ef0a5e4df.png) here is the mean intensity of the image ![\textstyle x^{(i)}](images/math/e/b/e/ebe8632b7c91a3dbbf9b590bea887a47.png). In particular,
this is not the same thing as estimating a mean value separately for each pixel ![\textstyle x_j](images/math/b/d/f/bdf5b20642553027712d5b5240b31cf3.png).

If you are training your algorithm on images other than natural images (for example, images of handwritten characters, or images of single isolated objects centered against a white background), other types of normalization might be worth considering, and the best choice may be application dependent. But when training on natural images, using the per-image mean normalization method as given in the equations above would be a reasonable default.

  References
------------

[http://cs229.stanford.edu](http://cs229.stanford.edu/)

**PCA** | [Whitening](Whitening.md "Whitening") | [Implementing PCA/Whitening](/wayback-mooc/stanford-ufldl/wiki/Implementing_PCA/Whitening "Implementing PCA/Whitening") | [Exercise:PCA in 2D](Exercise_PCA_in_2D.md "Exercise:PCA in 2D") | [Exercise:PCA and Whitening](Exercise_PCA_and_Whitening.md "Exercise:PCA and Whitening")

---

> * Language: [中文](%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90.md "主成分分析")
> * This page was last modified on 7 April 2013, at 13:18.

