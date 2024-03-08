Autoencoders and Sparsity
=========================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
So far, we have described the application of neural networks to supervised learning, in which we have labeled
training examples. Now suppose we have only a set of unlabeled training examples ![\textstyle \{x^{(1)}, x^{(2)}, x^{(3)}, \ldots\}](images/math/b/a/4/ba46917dec3eaeec510bec377e100ed2.png),
where ![\textstyle x^{(i)} \in \Re^{n}](images/math/c/a/5/ca57b44909d158c3fdfaa849465dd4a2.png). An
**autoencoder** neural network is an unsupervised learning algorithm that applies backpropagation,
setting the target values to be equal to the inputs. I.e., it uses ![\textstyle y^{(i)} = x^{(i)}](images/math/1/8/e/18ea47f5937698844bad522f86679912.png).

Here is an autoencoder:

![Autoencoder636.png](images/thumb/f/f9/Autoencoder636.png/400px-Autoencoder636.png)
The autoencoder tries to learn a function ![\textstyle h_{W,b}(x) \approx x](images/math/4/3/a/43add74a1db8df97b8c6c18abeab16ec.png). In other
words, it is trying to learn an approximation to the identity function, so as
to output ![\textstyle \hat{x}](images/math/2/9/0/29035749c12270bcc8de7e36bc459ece.png) that is similar to ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png). The identity function seems a
particularly trivial function to be trying to learn; but by placing constraints
on the network, such as by limiting the number of hidden units, we can discover
interesting structure about the data. As a concrete example, suppose the
inputs ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png) are the pixel intensity values from a ![\textstyle 10 \times 10](images/math/0/4/a/04aaf6cd0499a40a7c222ffdb85b55bb.png) image (100
pixels) so ![\textstyle n=100](images/math/5/4/8/548f3e32e47803886a1aacb25f80e82c.png), and there are ![\textstyle s_2=50](images/math/7/e/6/7e62b2a6bbb0a0653e6a37f96f6e8c6c.png) hidden units in layer ![\textstyle L_2](images/math/c/f/7/cf7d186efd913f4fb9ceb939bf5135c4.png). Note that
we also have ![\textstyle y \in \Re^{100}](images/math/b/d/b/bdb1f202462a1911c48d37ae983f05a1.png). Since there are only 50 hidden units, the
network is forced to learn a *compressed* representation of the input.
I.e., given only the vector of hidden unit activations ![\textstyle a^{(2)} \in \Re^{50}](images/math/a/e/3/ae3fd01fde1fddeaa9588705a0c3de26.png),
it must try to **reconstruct** the 100-pixel input ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png). If the input were completely
random---say, each ![\textstyle x_i](images/math/0/f/d/0fd4cfa441e8ad71698b916a2ec0b9b4.png) comes from an IID Gaussian independent of the other
features---then this compression task would be very difficult. But if there is
structure in the data, for example, if some of the input features are correlated,
then this algorithm will be able to discover some of those correlations. In fact,
this simple autoencoder often ends up learning a low-dimensional representation very similar
to PCAs.

Our argument above relied on the number of hidden units ![\textstyle s_2](images/math/9/b/d/9bd7e6635a679f22fa8a38dd2a910942.png) being small. But
even when the number of hidden units is large (perhaps even greater than the
number of input pixels), we can still discover interesting structure, by
imposing other constraints on the network. In particular, if we impose a
**sparsity** constraint on the hidden units, then the autoencoder will still
discover interesting structure in the data, even if the number of hidden units
is large.

Informally, we will think of a neuron as being "active" (or as "firing") if
its output value is close to 1, or as being "inactive" if its output value is
close to 0. We would like to constrain the neurons to be inactive most of the
time. This discussion assumes a sigmoid activation function. If you are
using a tanh activation function, then we think of a neuron as being inactive
when it outputs values close to -1.

Recall that ![\textstyle a^{(2)}_j](images/math/4/a/2/4a21cfd212bf2151bef47bbbd8d935d4.png) denotes the activation of hidden unit ![\textstyle j](images/math/2/3/5/235c5146ab110558897640c34dad7d97.png) in the
autoencoder. However, this notation doesn't make explicit what was the input ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png)
that led to that activation. Thus, we will write ![\textstyle a^{(2)}_j(x)](images/math/4/b/3/4b3ea0e74395587b75475e7d1a648104.png) to denote the activation
of this hidden unit when the network is given a specific input ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png). Further, let

![\begin{align}
\hat\rho_j = \frac{1}{m} \sum_{i=1}^m \left[ a^{(2)}_j(x^{(i)}) \right]
\end{align}](images/math/8/7/2/8728009d101b17918c7ef40a6b1d34bb.png)

be the average activation of hidden unit ![\textstyle j](images/math/2/3/5/235c5146ab110558897640c34dad7d97.png) (averaged over the training set).
We would like to (approximately) enforce the constraint

![\begin{align}
\hat\rho_j = \rho,
\end{align}](images/math/1/8/9/189cc2ce8930608381f0aa234c009cb6.png)

where ![\textstyle \rho](images/math/7/c/9/7c941e85242c631f08f2c9aeec7e24bf.png) is a **sparsity parameter**, typically a small value close to zero
(say ![\textstyle \rho = 0.05](images/math/d/3/5/d35b301a869e4bb93c39abcc9354c7c1.png)). In other words, we would like the average activation
of each hidden neuron ![\textstyle j](images/math/2/3/5/235c5146ab110558897640c34dad7d97.png) to be close to 0.05 (say). To satisfy this
constraint, the hidden unit's activations must mostly be near 0.

To achieve this, we will add an extra penalty term to our optimization objective that
penalizes ![\textstyle \hat\rho_j](images/math/e/d/0/ed01b1fc91d4e40fe824b5644e4afe80.png) deviating significantly from ![\textstyle \rho](images/math/7/c/9/7c941e85242c631f08f2c9aeec7e24bf.png). Many choices of the penalty
term will give reasonable results. We will choose the following:

![\begin{align}
\sum_{j=1}^{s_2} \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}.
\end{align}](images/math/8/a/7/8a77066279d89dae4688d9bf4508e0a1.png)

Here, ![\textstyle s_2](images/math/9/b/d/9bd7e6635a679f22fa8a38dd2a910942.png) is the number of neurons in the hidden layer, and the index ![\textstyle j](images/math/2/3/5/235c5146ab110558897640c34dad7d97.png) is summing
over the hidden units in our network. If you are
familiar with the concept of KL divergence, this penalty term is based on
it, and can also be written

![\begin{align}
\sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}](images/math/0/d/1/0d16f0831c0cc8cb4a71b95388fe99ed.png)

where ![\textstyle {\rm KL}(\rho || \hat\rho_j)
 = \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}](images/math/0/b/3/0b34ac7d27ca8346693cd35e9f8cfb0c.png)
is the Kullback-Leibler (KL) divergence between
a Bernoulli random variable with mean ![\textstyle \rho](images/math/7/c/9/7c941e85242c631f08f2c9aeec7e24bf.png) and a Bernoulli random variable with mean ![\textstyle \hat\rho_j](images/math/e/d/0/ed01b1fc91d4e40fe824b5644e4afe80.png).
KL-divergence is a standard function for measuring how different two different
distributions are. (If you've not seen KL-divergence before, don't worry about
it; everything you need to know about it is contained in these notes.)

This penalty function has the property that ![\textstyle {\rm KL}(\rho || \hat\rho_j) = 0](images/math/0/f/8/0f8e718b0e1b53a6d05db3cde96b054b.png) if ![\textstyle \hat\rho_j = \rho](images/math/b/c/a/bcaf3a5f10de9ecad0501656219aa84d.png),
and otherwise it increases monotonically as ![\textstyle \hat\rho_j](images/math/e/d/0/ed01b1fc91d4e40fe824b5644e4afe80.png) diverges from ![\textstyle \rho](images/math/7/c/9/7c941e85242c631f08f2c9aeec7e24bf.png). For example, in the
figure below, we have set ![\textstyle \rho = 0.2](images/math/a/6/d/a6d55e7d6e75b4f436b65237fbd149cb.png), and plotted
![\textstyle {\rm KL}(\rho || \hat\rho_j)](images/math/a/6/5/a6597e94af97ff0d0b9b4c1502514653.png) for a range of values of ![\textstyle \hat\rho_j](images/math/e/d/0/ed01b1fc91d4e40fe824b5644e4afe80.png):

![KLPenaltyExample.png](images/thumb/4/48/KLPenaltyExample.png/400px-KLPenaltyExample.png)
We see that the KL-divergence reaches its minimum of 0 at
![\textstyle \hat\rho_j = \rho](images/math/b/c/a/bcaf3a5f10de9ecad0501656219aa84d.png), and blows up (it actually approaches ![\textstyle \infty](images/math/3/0/f/30f4c69377104600a42fc2cf6d55c31a.png)) as ![\textstyle \hat\rho_j](images/math/e/d/0/ed01b1fc91d4e40fe824b5644e4afe80.png)
approaches 0 or 1. Thus, minimizing
this penalty term has the effect of causing ![\textstyle \hat\rho_j](images/math/e/d/0/ed01b1fc91d4e40fe824b5644e4afe80.png) to be close to ![\textstyle \rho](images/math/7/c/9/7c941e85242c631f08f2c9aeec7e24bf.png).

Our overall cost function is now

![\begin{align}
J_{\rm sparse}(W,b) = J(W,b) + \beta \sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}](images/math/7/a/4/7a4ac86b3559db835f4357987252b088.png)

where ![\textstyle J(W,b)](images/math/8/e/9/8e94ae776ae14b36b3af183726ababb9.png) is as defined previously, and ![\textstyle \beta](images/math/c/0/3/c03f7273f858c35d0a482846d7cd54bf.png) controls the weight of
the sparsity penalty term. The term ![\textstyle \hat\rho_j](images/math/e/d/0/ed01b1fc91d4e40fe824b5644e4afe80.png) (implicitly) depends on ![\textstyle W,b](images/math/7/c/9/7c9aa03f5258ecf79556ba374d7eb2cd.png) also,
because it is the average activation of hidden unit ![\textstyle j](images/math/2/3/5/235c5146ab110558897640c34dad7d97.png), and the activation of a hidden
unit depends on the parameters ![\textstyle W,b](images/math/7/c/9/7c9aa03f5258ecf79556ba374d7eb2cd.png).

To incorporate the KL-divergence term into your derivative calculation, there is a simple-to-implement
trick involving only a small change to your code. Specifically, where previously for
the second layer (![\textstyle l=2](images/math/c/f/2/cf283abbcdb1c0f69cbff28e964776f5.png)), during backpropagation you would have computed

![\begin{align}
\delta^{(2)}_i = \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right) f'(z^{(2)}_i),
\end{align}](images/math/a/b/2/ab2e3ac6ec9172f9b2d9b8d3542158dc.png)

now instead compute

![\begin{align}
\delta^{(2)}_i =
  \left( \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right)
+ \beta \left( - \frac{\rho}{\hat\rho_i} + \frac{1-\rho}{1-\hat\rho_i} \right) \right) f'(z^{(2)}_i) .
\end{align}](images/math/b/e/f/bef8a29947bfb0d746c54a7d922874e8.png)

One subtlety is that you'll need to know ![\textstyle \hat\rho_i](images/math/3/9/0/39060ca518709da427114932253de53d.png) to compute this term. Thus, you'll need
to compute a forward pass on all the training examples first to compute the average
activations on the training set, before computing backpropagation on any example. If your
training set is small enough to fit comfortably in computer memory (this will be the case for the programming
assignment), you can compute forward passes on all your examples and keep the resulting activations
in memory and compute the ![\textstyle \hat\rho_i](images/math/3/9/0/39060ca518709da427114932253de53d.png)s. Then you can use your precomputed activations to
perform backpropagation on all your examples. If your data is too large to fit in memory, you
may have to scan through your examples computing a forward pass on each to accumulate (sum up) the
activations and compute ![\textstyle \hat\rho_i](images/math/3/9/0/39060ca518709da427114932253de53d.png) (discarding the result of each forward pass after you
have taken its activations ![\textstyle a^{(2)}_i](images/math/e/1/4/e14f36d1b33f6ed0dc131a7ddd166004.png) into account for computing ![\textstyle \hat\rho_i](images/math/3/9/0/39060ca518709da427114932253de53d.png)). Then after
having computed ![\textstyle \hat\rho_i](images/math/3/9/0/39060ca518709da427114932253de53d.png), you'd have to redo the forward pass for each example so that you
can do backpropagation on that example. In this latter case, you would end up computing a forward
pass twice on each example in your training set, making it computationally less efficient.

The full derivation showing that the algorithm above results in gradient descent is beyond the scope
of these notes. But if you implement the autoencoder using backpropagation modified this way,
you will be performing gradient descent exactly on the objective
![\textstyle J_{\rm sparse}(W,b)](images/math/b/1/2/b128c50f477dfa7b354111ed44f216a0.png). Using the derivative checking method, you will be able to verify
this for yourself as well.

[Neural Networks](Neural_Networks.md "Neural Networks") | [Backpropagation Algorithm](Backpropagation_Algorithm.md "Backpropagation Algorithm") | [Gradient checking and advanced optimization](Gradient_checking_and_advanced_optimization.md "Gradient checking and advanced optimization") | **Autoencoders and Sparsity** | [Visualizing a Trained Autoencoder](Visualizing_a_Trained_Autoencoder.md "Visualizing a Trained Autoencoder") | [Sparse Autoencoder Notation Summary](Sparse_Autoencoder_Notation_Summary.md "Sparse Autoencoder Notation Summary") | [Exercise:Sparse Autoencoder](Exercise_Sparse_Autoencoder.md "Exercise:Sparse Autoencoder")

---

> * Language: [中文](%E8%87%AA%E7%BC%96%E7%A0%81%E7%AE%97%E6%B3%95%E4%B8%8E%E7%A8%80%E7%96%8F%E6%80%A7.md "自编码算法与稀疏性")
> * This page was last modified on 7 April 2013, at 12:43.

