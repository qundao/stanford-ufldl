Gradient checking and advanced optimization
===========================================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
Backpropagation is a notoriously difficult algorithm to debug and get right,
especially since many subtly buggy implementations of it—for example, one
that has an off-by-one error in the indices and that thus only trains some of
the layers of weights, or an implementation that omits the bias term—will
manage to learn something that can look surprisingly reasonable
(while performing less well than a correct implementation). Thus, even with a
buggy implementation, it may not at all be apparent that anything is amiss.
In this section, we describe a method for numerically checking the derivatives computed
by your code to make sure that your implementation is correct. Carrying out the
derivative checking procedure described here will significantly increase
your confidence in the correctness of your code.

Suppose we want to minimize ![\textstyle J(\theta)](images/math/c/e/0/ce027336c1cb3c0cd461406c81369ebf.png) as a function of ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png).
For this example, suppose ![\textstyle J : \Re \mapsto \Re](images/math/c/d/a/cda5857b15a23c03abfb2e42eb51b70c.png), so that ![\textstyle \theta \in \Re](images/math/d/c/7/dc7c1205b7193f92a71d1f4e7cb4e707.png).
In this 1-dimensional case, one iteration of gradient descent is given by

![\begin{align}
\theta := \theta - \alpha \frac{d}{d\theta}J(\theta).
\end{align}](images/math/a/8/c/a8c1af31e58f9f9f2c55c90b33deace1.png)

Suppose also that we have implemented some function ![\textstyle g(\theta)](images/math/e/9/f/e9fed70b38b2cfac3b42d1d21d46e449.png) that purportedly
computes ![\textstyle \frac{d}{d\theta}J(\theta)](images/math/0/9/6/09643c7c4bb96e039caf25737d835201.png), so that we implement gradient descent
using the update ![\textstyle \theta := \theta - \alpha g(\theta)](images/math/a/0/1/a01cdafbf71127043a4a5d2d097dfd80.png). How can we check if our implementation of
![\textstyle g](images/math/c/1/7/c172541f77a147fcf545237fefa03643.png) is correct?

Recall the mathematical definition of the derivative as

![\begin{align}
\frac{d}{d\theta}J(\theta) = \lim_{\epsilon \rightarrow 0}
\frac{J(\theta+ \epsilon) - J(\theta-\epsilon)}{2 \epsilon}.
\end{align}](images/math/a/2/3/a23bea0ab48ded7b9a979b68f6356613.png)

Thus, at any specific value of ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png), we can numerically approximate the derivative
as follows:

![\begin{align}
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}
\end{align}](images/math/4/8/a/48a000aed96c8595fcca2a45f48343ce.png)

In practice, we set EPSILON to a small constant, say around ![\textstyle 10^{-4}](images/math/c/f/d/cfd7bf1257600c6c7706c5597af1b94d.png).
(There's a large range of values of EPSILON that should work well, but
we don't set EPSILON to be "extremely" small, say ![\textstyle 10^{-20}](images/math/f/a/b/fab2be95b827b3db4ea4d2e27a3d5f99.png),
as that would lead to numerical roundoff errors.)

Thus, given a function ![\textstyle g(\theta)](images/math/e/9/f/e9fed70b38b2cfac3b42d1d21d46e449.png) that is supposedly computing
![\textstyle \frac{d}{d\theta}J(\theta)](images/math/0/9/6/09643c7c4bb96e039caf25737d835201.png), we can now numerically verify its correctness
by checking that

![\begin{align}
g(\theta) \approx
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}.
\end{align}](images/math/c/6/d/c6d06b4a25dab5ef468c72900872eda0.png)

The degree to which these two values should approximate each other
will depend on the details of ![\textstyle J](images/math/4/f/4/4f465a48d84668feb1081c49388cf9b4.png). But assuming ![\textstyle {\rm EPSILON} = 10^{-4}](images/math/8/7/5/875b9648ce24d3e6ed45c5fb1aef3833.png),
you'll usually find that the left- and right-hand sides of the above will agree
to at least 4 significant digits (and often many more).

Now, consider the case where ![\textstyle \theta \in \Re^n](images/math/a/8/e/a8e658b091c361cc9f85ea67d7689332.png) is a vector rather than a single real
number (so that we have ![\textstyle n](images/math/0/c/5/0c59de0fa75c1baa1c024aabfa43b2e3.png) parameters that we want to learn), and ![\textstyle J: \Re^n \mapsto \Re](images/math/3/9/f/39f1a609f6140108fb4f0ba2626e5d6a.png). In
our neural network example we used "![\textstyle J(W,b)](images/math/8/e/9/8e94ae776ae14b36b3af183726ababb9.png)," but one can imagine "unrolling"
the parameters ![\textstyle W,b](images/math/7/c/9/7c9aa03f5258ecf79556ba374d7eb2cd.png) into a long vector ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png). We now generalize our derivative
checking procedure to the case where ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png) may be a vector.

Suppose we have a function ![\textstyle g_i(\theta)](images/math/3/f/4/3f479459ba2e5ba889a1c2e36995ecc8.png) that purportedly computes
![\textstyle \frac{\partial}{\partial \theta_i} J(\theta)](images/math/3/e/2/3e2d8c5d93954b93d064c96a93f0a6d8.png); we'd like to check if ![\textstyle g_i](images/math/9/c/9/9c9d4fc87d716b87e446297d0ebb94f8.png)
is outputting correct derivative values. Let ![\textstyle \theta^{(i+)} = \theta +
{\rm EPSILON} \times \vec{e}_i](images/math/0/9/b/09b406ad4b7aa1c6933b9f26e957c1fb.png), where

![\begin{align}
\vec{e}_i = \begin{bmatrix}0 \\ 0 \\ \vdots \\ 1 \\ \vdots \\ 0\end{bmatrix}
\end{align}](images/math/7/d/7/7d7c568be5dc22311d9c60c7fa11457f.png)

is the ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)-th basis vector (a
vector of the same dimension as ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png), with a "1" in the ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)-th position
and "0"s everywhere else). So,
![\textstyle \theta^{(i+)}](images/math/a/e/5/ae5326f17ec53546152dd9f3cd06fe8a.png) is the same as ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png), except its ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)-th element has been incremented
by EPSILON. Similarly, let ![\textstyle \theta^{(i-)} = \theta - {\rm EPSILON} \times \vec{e}_i](images/math/a/a/0/aa0225fbe0ff42d79a568cfb2b10ecd7.png) be the
corresponding vector with the ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)-th element decreased by EPSILON.
We can now numerically verify ![\textstyle g_i(\theta)](images/math/3/f/4/3f479459ba2e5ba889a1c2e36995ecc8.png)'s correctness by checking, for each ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png),
that:

![\begin{align}
g_i(\theta) \approx
\frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2 \times {\rm EPSILON}}.
\end{align}](images/math/1/e/1/1e153c5e6de67d97bfaf25c7fe396495.png)

When implementing backpropagation to train a neural network, in a correct implementation
we will have that

![\begin{align}
\nabla_{W^{(l)}} J(W,b) &= \left( \frac{1}{m} \Delta W^{(l)} \right) + \lambda W^{(l)} \\
\nabla_{b^{(l)}} J(W,b) &= \frac{1}{m} \Delta b^{(l)}.
\end{align}](images/math/1/2/9/1297d5746b1a274d8ab855bb6e638bdb.png)

This result shows that the final block of psuedo-code in [Backpropagation Algorithm](Backpropagation_Algorithm.md "Backpropagation Algorithm") is indeed
implementing gradient descent.
To make sure your implementation of gradient descent is correct, it is
usually very helpful to use the method described above to
numerically compute the derivatives of ![\textstyle J(W,b)](images/math/8/e/9/8e94ae776ae14b36b3af183726ababb9.png), and thereby verify that
your computations of ![\textstyle \left(\frac{1}{m}\Delta W^{(l)} \right) + \lambda W](images/math/5/1/a/51abfba362dde73804e9d8dd913ceb00.png) and ![\textstyle \frac{1}{m}\Delta b^{(l)}](images/math/c/8/3/c83a6b2fce9939316356a4aa0c7e773b.png) are
indeed giving the derivatives you want.

Finally, so far our discussion has centered on using gradient descent to minimize ![\textstyle J(\theta)](images/math/c/e/0/ce027336c1cb3c0cd461406c81369ebf.png). If you have
implemented a function that computes ![\textstyle J(\theta)](images/math/c/e/0/ce027336c1cb3c0cd461406c81369ebf.png) and ![\textstyle \nabla_\theta J(\theta)](images/math/9/a/e/9ae0378bbaa18d11cdfbf3c76a612708.png), it turns out there are more
sophisticated algorithms than gradient descent for trying to minimize ![\textstyle J(\theta)](images/math/c/e/0/ce027336c1cb3c0cd461406c81369ebf.png). For example, one can envision
an algorithm that uses gradient descent, but automatically tunes the learning rate ![\textstyle \alpha](images/math/7/e/a/7eaa466003e48c1c96824a2edf3de038.png) so as to try to
use a step-size that causes ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png) to approach a local optimum as quickly as possible.
There are other algorithms that are even more
sophisticated than this; for example, there are algorithms that try to find an approximation to the
Hessian matrix, so that it can take more rapid steps towards a local optimum (similar to Newton's method). A full discussion of these
algorithms is beyond the scope of these notes, but one example is
the **L-BFGS** algorithm. (Another example is the **conjugate gradient** algorithm.) You will use one of
these algorithms in the programming exercise.
The main thing you need to provide to these advanced optimization algorithms is that for any ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png), you have to be able
to compute ![\textstyle J(\theta)](images/math/c/e/0/ce027336c1cb3c0cd461406c81369ebf.png) and ![\textstyle \nabla_\theta J(\theta)](images/math/9/a/e/9ae0378bbaa18d11cdfbf3c76a612708.png). These optimization algorithms will then do their own
internal tuning of the learning rate/step-size ![\textstyle \alpha](images/math/7/e/a/7eaa466003e48c1c96824a2edf3de038.png) (and compute its own approximation to the Hessian, etc.)
to automatically search for a value of ![\textstyle \theta](images/math/6/9/d/69d920fe8e1da0543eb63d1097f21754.png) that minimizes ![\textstyle J(\theta)](images/math/c/e/0/ce027336c1cb3c0cd461406c81369ebf.png). Algorithms
such as L-BFGS and conjugate gradient can often be much faster than gradient descent.

[Neural Networks](Neural_Networks.md "Neural Networks") | [Backpropagation Algorithm](Backpropagation_Algorithm.md "Backpropagation Algorithm") | **Gradient checking and advanced optimization** | [Autoencoders and Sparsity](Autoencoders_and_Sparsity.md "Autoencoders and Sparsity") | [Visualizing a Trained Autoencoder](Visualizing_a_Trained_Autoencoder.md "Visualizing a Trained Autoencoder") | [Sparse Autoencoder Notation Summary](Sparse_Autoencoder_Notation_Summary.md "Sparse Autoencoder Notation Summary") | [Exercise:Sparse Autoencoder](Exercise_Sparse_Autoencoder.md "Exercise:Sparse Autoencoder")

---

> * Language: [中文](%E6%A2%AF%E5%BA%A6%E6%A3%80%E9%AA%8C%E4%B8%8E%E9%AB%98%E7%BA%A7%E4%BC%98%E5%8C%96.md "梯度检验与高级优化")
> * This page was last modified on 7 April 2013, at 12:40.

