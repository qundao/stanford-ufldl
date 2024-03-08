Sparse Coding
=============

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
  Sparse Coding
---------------

Sparse coding is a class of unsupervised methods for learning sets of over-complete bases to represent data efficiently. The aim of sparse coding is to find a set of basis vectors ![\mathbf{\phi}_i](images/math/9/6/f/96f401a31a42b4a238dbe0c5be68a746.png) such that we can represent an input vector ![\mathbf{x}](images/math/3/c/6/3c66d9170d4c3fb75456e1a9fc6ead37.png) as a linear combination of these basis vectors:

![\begin{align}
\mathbf{x} = \sum_{i=1}^k a_i \mathbf{\phi}_{i} 
\end{align}](images/math/9/5/7/95773d0fedcb4bc39aff6546ccd5af25.png)

While techniques such as Principal Component Analysis (PCA) allow us to learn a complete set of basis vectors efficiently, we wish to learn an **over-complete** set of basis vectors to represent input vectors ![\mathbf{x}\in\mathbb{R}^n](images/math/a/0/c/a0c529368bdcd396825fbe6bbbfb9fa8.png) (i.e. such that *k* > *n*). The advantage of having an over-complete basis is that our basis vectors are better able to capture structures and patterns inherent in the input data. However, with an over-complete basis, the coefficients *a**i* are no longer uniquely determined by the input vector ![\mathbf{x}](images/math/3/c/6/3c66d9170d4c3fb75456e1a9fc6ead37.png). Therefore, in sparse coding, we introduce the additional criterion of **sparsity** to resolve the degeneracy introduced by over-completeness.

Here, we define sparsity as having few non-zero components or having few components not close to zero. The requirement that our coefficients *a**i* be sparse means that given a input vector, we would like as few of our coefficients to be far from zero as possible. The choice of sparsity as a desired characteristic of our representation of the input data can be motivated by the observation that most sensory data such as natural images may be described as the superposition of a small number of atomic elements such as surfaces or edges. Other justifications such as comparisons to the properties of the primary visual cortex have also been advanced.

We define the sparse coding cost function on a set of *m* input vectors as

![\begin{align}
\text{minimize}_{a^{(j)}_i,\mathbf{\phi}_{i}} \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i)
\end{align}](images/math/f/1/1/f110901ddedcba59e339de5f16c547da.png)

where *S*(.) is a sparsity cost function which penalizes *a**i* for being far from zero. We can interpret the first term of the sparse coding objective as a reconstruction term which tries to force the algorithm to provide a good representation of ![\mathbf{x}](images/math/3/c/6/3c66d9170d4c3fb75456e1a9fc6ead37.png) and the second term as a sparsity penalty which forces our representation of ![\mathbf{x}](images/math/3/c/6/3c66d9170d4c3fb75456e1a9fc6ead37.png) to be sparse. The constant λ is a scaling constant to determine the relative importance of these two contributions.

Although the most direct measure of sparsity is the "*L*0" norm (![S(a_i) = \mathbf{1}(|a_i|>0)](images/math/9/2/0/9201129fb038db6903ec61196798181d.png)), it is non-differentiable and difficult to optimize in general. In practice, common choices for the sparsity cost *S*(.) are the *L*1 penalty ![S(a_i)=\left|a_i\right|_1 ](images/math/a/8/8/a884849a26a901395faa9eede9b00e81.png) and the log penalty ![S(a_i)=\log(1+a_i^2)](images/math/c/8/f/c8f980972ea11e452e9d5031c44f95f6.png).

In addition, it is also possible to make the sparsity penalty arbitrarily small by scaling down *a**i* and scaling ![\mathbf{\phi}_i](images/math/9/6/f/96f401a31a42b4a238dbe0c5be68a746.png) up by some large constant. To prevent this from happening, we will constrain ![\left|\left|\mathbf{\phi}\right|\right|^2](images/math/1/6/2/162a65a67f9ad82157da95a835185ede.png) to be less than some constant *C*. The full sparse coding cost function including our constraint on ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) is

![\begin{array}{rc}
\text{minimize}_{a^{(j)}_i,\mathbf{\phi}_{i}} & \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i) 
\\
\text{subject to}  &  \left|\left|\mathbf{\phi}_i\right|\right|^2 \leq C, \forall i = 1,...,k 
\\
\end{array}](images/math/a/9/3/a93c6a5d7e7a22c66e82490be078b2af.png)

  Probabilistic Interpretation [Based on Olshausen and Field 1996]
------------------------------------------------------------------

So far, we have considered sparse coding in the context of finding a sparse, over-complete set of basis vectors to span our input space. Alternatively, we may also approach sparse coding from a probabilistic perspective as a generative model.

Consider the problem of modelling natural images as the linear superposition of *k* independent source features ![\mathbf{\phi}_i](images/math/9/6/f/96f401a31a42b4a238dbe0c5be68a746.png) with some additive noise ν:

![\begin{align}
\mathbf{x} = \sum_{i=1}^k a_i \mathbf{\phi}_{i} + \nu(\mathbf{x})
\end{align}](images/math/4/d/a/4daf9370c4f4e65a8fb7ae213c59b996.png)

Our goal is to find a set of basis feature vectors ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) such that the distribution of images ![P(\mathbf{x}\mid\mathbf{\phi})](images/math/8/1/9/8198a76b9b1b28d7a299ad59b4aca55b.png) is as close as possible to the empirical distribution of our input data ![P^*(\mathbf{x})](images/math/a/f/c/afc77091b0831f8c4733ab0708062d63.png). One method of doing so is to minimize the KL divergence between ![P^*(\mathbf{x})](images/math/a/f/c/afc77091b0831f8c4733ab0708062d63.png) and ![P(\mathbf{x}\mid\mathbf{\phi})](images/math/8/1/9/8198a76b9b1b28d7a299ad59b4aca55b.png) where the KL divergence is defined as:

![\begin{align}
D(P^*(\mathbf{x})||P(\mathbf{x}\mid\mathbf{\phi})) = \int P^*(\mathbf{x}) \log \left(\frac{P^*(\mathbf{x})}{P(\mathbf{x}\mid\mathbf{\phi})}\right)d\mathbf{x}
\end{align}](images/math/7/b/3/7b39a1c36dc8d6463e4997495334c0f8.png)

Since the empirical distribution ![P^*(\mathbf{x})](images/math/a/f/c/afc77091b0831f8c4733ab0708062d63.png) is constant across our choice of ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png), this is equivalent to maximizing the log-likelihood of ![P(\mathbf{x}\mid\mathbf{\phi})](images/math/8/1/9/8198a76b9b1b28d7a299ad59b4aca55b.png).

Assuming ν is Gaussian white noise with variance σ2, we have that

![\begin{align}
P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi}) = \frac{1}{Z} \exp\left(- \frac{(\mathbf{x}-\sum^{k}_{i=1} a_i \mathbf{\phi}_{i})^2}{2\sigma^2}\right)
\end{align}](images/math/9/d/6/9d634e2a1b3457f439d442bf61f7381b.png)

In order to determine the distribution ![P(\mathbf{x}\mid\mathbf{\phi})](images/math/8/1/9/8198a76b9b1b28d7a299ad59b4aca55b.png), we also need to specify the prior distribution ![P(\mathbf{a})](images/math/4/9/b/49b4b770c52ed209b950c2fd00216bbf.png). Assuming the independence of our source features, we can factorize our prior probability as

![\begin{align}
P(\mathbf{a}) = \prod_{i=1}^{k} P(a_i)
\end{align}](images/math/d/8/9/d89ec802e2b5461efa8d0d2d84f9e829.png)

At this point, we would like to incorporate our sparsity assumption -- the assumption that any single image is likely to be the product of relatively few source features. Therefore, we would like the probability distribution of *a**i* to be peaked at zero and have high kurtosis. A convenient parameterization of the prior distribution is

![\begin{align}
P(a_i) = \frac{1}{Z}\exp(-\beta S(a_i))
\end{align}](images/math/8/5/0/850c6b42874fde83fef6001ba388d0b4.png)

Where *S*(*a**i*) is a function determining the shape of the prior distribution.

Having defined ![P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi})](images/math/d/0/2/d02802b0ba8bfd44edb2be30ee7607e5.png) and ![ P(\mathbf{a})](images/math/4/9/b/49b4b770c52ed209b950c2fd00216bbf.png), we can write the probability of the data ![\mathbf{x}](images/math/3/c/6/3c66d9170d4c3fb75456e1a9fc6ead37.png) under the model defined by ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) as

![\begin{align}
P(\mathbf{x} \mid \mathbf{\phi}) = \int P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi}) P(\mathbf{a}) d\mathbf{a}
\end{align}](images/math/6/b/7/6b7b96f043bd1d85571edc7ac556921e.png)

and our problem reduces to finding

![\begin{align}
\mathbf{\phi}^*=\text{argmax}_{\mathbf{\phi}} < \log(P(\mathbf{x} \mid \mathbf{\phi})) >
\end{align}](images/math/b/6/1/b61b290904ced2463333bdca70ba9a95.png)

Where  < . >  denotes expectation over our input data.

Unfortunately, the integral over ![\mathbf{a}](images/math/3/c/4/3c47f830945ee6b24984ab0ba188e10e.png) to obtain ![P(\mathbf{x} \mid \mathbf{\phi})](images/math/8/1/9/8198a76b9b1b28d7a299ad59b4aca55b.png) is generally intractable. We note though that if the distribution of ![P(\mathbf{x} \mid \mathbf{\phi})](images/math/8/1/9/8198a76b9b1b28d7a299ad59b4aca55b.png) is sufficiently peaked (w.r.t. ![\mathbf{a}](images/math/3/c/4/3c47f830945ee6b24984ab0ba188e10e.png)), we can approximate its integral with the maximum value of ![P(\mathbf{x} \mid \mathbf{\phi})](images/math/8/1/9/8198a76b9b1b28d7a299ad59b4aca55b.png) and obtain a approximate solution

![\begin{align}
\mathbf{\phi}^{*'}=\text{argmax}_{\mathbf{\phi}} < \max_{\mathbf{a}} \log(P(\mathbf{x} \mid \mathbf{\phi})) >
\end{align}](images/math/9/7/8/97822a58455d3c2c6d965597d0248d7d.png)

As before, we may increase the estimated probability by scaling down *a**i* and scaling up ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) (since *P*(*a**i*) peaks about zero) , we therefore impose a norm constraint on our features ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) to prevent this.

Finally, we can recover our original cost function by defining the energy function of this linear generative model

![\begin{array}{rl}
E\left( \mathbf{x} , \mathbf{a} \mid \mathbf{\phi} \right) & := -\log \left( P(\mathbf{x}\mid \mathbf{\phi},\mathbf{a}\right)P(\mathbf{a})) \\
 &= \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i) 
\end{array}](images/math/e/3/4/e34c091d504207038943443866f62ccc.png)

where λ = 2σ2β and irrelevant constants have been hidden. Since maximizing the log-likelihood is equivalent to minimizing the energy function, we recover the original optimization problem:

![\begin{align}
\mathbf{\phi}^{*},\mathbf{a}^{*}=\text{argmin}_{\mathbf{\phi},\mathbf{a}} \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i) 
\end{align}](images/math/b/c/1/bc124bd99a15b3035f82301dacf1993b.png)

Using a probabilistic approach, it can also be seen that the choices of the *L*1 penalty ![\left|a_i\right|_1 ](images/math/5/b/e/5beadeaa907c702956af765ff4080510.png) and the log penalty ![\log(1+a_i^2)](images/math/e/4/d/e4dd083f18a7b80eef831fcd53f6ce56.png) for *S*(.) correspond to the use of the Laplacian ![P(a_i) \propto \exp\left(-\beta|a_i|\right)](images/math/4/8/a/48a0ca02892923a1a279d84faa1f75c1.png) and the Cauchy prior ![P(a_i) \propto \frac{\beta}{1+a_i^2}](images/math/a/8/b/a8b02506e9e2267b363efcb139af11ad.png) respectively.

  Learning
----------

Learning a set of basis vectors ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) using sparse coding consists of performing two separate optimizations, the first being an optimization over coefficients *a**i* for each training example ![\mathbf{x}](images/math/3/c/6/3c66d9170d4c3fb75456e1a9fc6ead37.png) and the second an optimization over basis vectors ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) across many training examples at once.

Assuming an *L*1 sparsity penalty, learning ![a^{(j)}_i](images/math/a/a/5/aa52f3c4e4bbcf7defbe2a8b936bc78e.png) reduces to solving a *L*1 regularized least squares problem which is convex in ![a^{(j)}_i](images/math/a/a/5/aa52f3c4e4bbcf7defbe2a8b936bc78e.png) for which several techniques have been developed (convex optimization software such as CVX can also be used to perform L1 regularized least squares). Assuming a differentiable *S*(.) such as the log penalty, gradient-based methods such as conjugate gradient methods can also be used.

Learning a set of basis vectors with a *L*2 norm constraint also reduces to a least squares problem with quadratic constraints which is convex in ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png). Standard convex optimization software (e.g. CVX) or other iterative methods can be used to solve for ![\mathbf{\phi}](images/math/a/a/9/aa970cc66d8a8408dd1811b678a367b0.png) although significantly more efficient methods such as solving the Lagrange dual have also been developed.

As described above, a significant limitation of sparse coding is that even after a set of basis vectors have been learnt, in order to "encode" a new data example, optimization must be performed to obtain the required coefficients. This significant "runtime" cost means that sparse coding is computationally expensive to implement even at test time especially compared to typical feedforward architectures.

---

> * Language: [中文](%E7%A8%80%E7%96%8F%E7%BC%96%E7%A0%81.md "稀疏编码")
> * This page was last modified on 8 April 2013, at 04:28.

