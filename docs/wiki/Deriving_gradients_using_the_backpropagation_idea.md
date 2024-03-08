Deriving gradients using the backpropagation idea
=================================================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->

|  |
| --- |
| Contents* [1 Introduction](#Introduction)
* [2 Examples](#Examples)
	+ [2.1 Example 1: Objective for weight matrix in sparse coding](#Example_1:_Objective_for_weight_matrix_in_sparse_coding)
	+ [2.2 Example 2: Smoothed topographic L1 sparsity penalty in sparse coding](#Example_2:_Smoothed_topographic_L1_sparsity_penalty_in_sparse_coding)
	+ [2.3 Example 3: ICA reconstruction cost](#Example_3:_ICA_reconstruction_cost)
 |

  Introduction
--------------

In the section on the  [backpropagation algorithm](Backpropagation_Algorithm.md "Backpropagation Algorithm"), you were briefly introduced to backpropagation as a means of deriving gradients for learning in the sparse autoencoder. It turns out that together with matrix calculus, this provides a powerful method and intuition for deriving gradients for more complex matrix functions (functions from matrices to the reals, or symbolically, from ![\mathbb{R}^{r \times c} \rightarrow \mathbb{R}](images/math/5/b/3/5b3a7630692b07263c08fac96c88c98e.png)).

First, recall the backpropagation idea, which we present in a modified form appropriate for our purposes below:

1. For each output unit *i* in layer *n**l* (the final layer), set
![
\delta^{(n_l)}_i
= \frac{\partial}{\partial z^{(n_l)}_i} \;\;
        J(z^{(n_l)})
](images/math/1/3/c/13cbf81577c102ed2e01d67f71723076.png)

where *J*(*z*) is our "objective function" (explained below).
- For ![l = n_l-1, n_l-2, n_l-3, \ldots, 2](images/math/9/8/8/988861db3f04c9f1150b482aca116daa.png)
For each node *i* in layer *l*, set
![
                 \delta^{(l)}_i = \left( \sum_{j=1}^{s_{l+1}} W^{(l)}_{ji} \delta^{(l+1)}_j \right) \bullet \frac{\partial}{\partial z^{(l)}_i} f^{(l)} (z^{(l)}_i)
](images/math/9/4/7/947031b9c9f1be0fc792bf2a1b98c27d.png)

- Compute the desired partial derivatives,
![
\begin{align}
\nabla_{W^{(l)}} J(W,b;x,y) &= \delta^{(l+1)} (a^{(l)})^T, \\
\end{align}
](images/math/5/a/3/5a34eec4ca6a8dd244ed4497cd78ad63.png)

Quick notation recap:

* *l* is the number of layers in the neural network
* *n**l* is the number of neurons in the *l*th layer
* ![W^{(l)}_{ji}](images/math/3/6/1/36184dd6c51daad9e5c9f1973933460e.png) is the weight from the *i*th unit in the *l*th layer to the *j*th unit in the (*l* + 1)th layer
* ![z^{(l)}_i](images/math/0/5/3/053932a35e5e7923d66bfd5cbc15b280.png) is the input to the *i*th unit in the *l*th layer
* ![a^{(l)}_i](images/math/2/f/1/2f12132475b24d761ca573173962be9b.png) is the activation of the *i*th unit in the *l*th layer
* ![A \bullet B](images/math/0/3/c/03caf6030df47b28250decb7a399c191.png) is the Hadamard or element-wise product, which for ![r \times c](images/math/f/5/b/f5b34ce727a51879b69d50dbb38cec68.png) matrices *A* and *B* yields the ![r \times c](images/math/f/5/b/f5b34ce727a51879b69d50dbb38cec68.png) matrix ![C = A \bullet B](images/math/d/b/f/dbf40e2ec518a8d773f3b648f9bd4b7d.png) such that ![C_{r, c} = A_{r, c} \cdot B_{r, c}](images/math/9/b/2/9b25139003c1d65c569180099b9e56a7.png)* *f*(*l*) is the activation function for units in the *l*th layer

Let's say we have a function *F* that takes a matrix *X* and yields a real number. We would like to use the backpropagation idea to compute the gradient with respect to *X* of *F*, that is ![\nabla_X F](images/math/c/8/a/c8a57f802f72156c4dbee1bd9fde338e.png). The general idea is to see the function *F* as a multi-layer neural network, and to derive the gradients using the backpropagation idea.

To do this, we will set our "objective function" to be the function *J*(*z*) that when applied to the outputs of the neurons in the last layer yields the value *F*(*X*). For the intermediate layers, we will also choose our activation functions *f*(*l*) to this end.

Using this method, we can easily compute derivatives with respect to the inputs *X*, as well as derivatives with respect to any of the weights in the network, as we shall see later.

  Examples
----------

To illustrate the use of the backpropagation idea to compute derivatives with respect to the inputs, we will use two functions from the section on  [sparse coding](Sparse_Coding__Autoencoder_Interpretation.md "Sparse Coding: Autoencoder Interpretation"), in examples 1 and 2. In example 3, we use a function from  [independent component analysis](Independent_Component_Analysis.md "Independent Component Analysis") to illustrate the use of this idea to compute derivates with respect to weights, and in this specific case, what to do in the case of tied or repeated weights.

###   Example 1: Objective for weight matrix in sparse coding

Recall for  [sparse coding](Sparse_Coding__Autoencoder_Interpretation.md "Sparse Coding: Autoencoder Interpretation"), the objective function for the weight matrix *A*, given the feature matrix *s*:

![F(A; s) = \lVert As - x \rVert_2^2 + \gamma \lVert A \rVert_2^2](images/math/d/8/a/d8a544d689b8b25c191b77b5010f2e98.png)

We would like to find the gradient of *F* with respect to *A*, or in symbols, ![\nabla_A F(A)](images/math/b/c/2/bc2d77b08b71888b46b4cc02b319a8d5.png). Since the objective function is a sum of two terms in *A*, the gradient is the sum of gradients of each of the individual terms. The gradient of the second term is trivial, so we will consider the gradient of the first term instead.

The first term, ![\lVert As - x \rVert_2^2](images/math/7/d/2/7d2f077de4b218982f04826f6f5a91aa.png), can be seen as an instantiation of neural network taking *s* as an input, and proceeding in four steps, as described and illustrated in the paragraph and diagram below:

1. Apply *A* as the weights from the first layer to the second layer.
- Subtract *x* from the activation of the second layer, which uses the identity activation function.
- Pass this unchanged to the third layer, via identity weights. Use the square function as the activation function for the third layer.
- Sum all the activations of the third layer.

![Backpropagation Method Example 1.png](images/thumb/b/bd/Backpropagation_Method_Example_1.png/400px-Backpropagation_Method_Example_1.png)

The weights and activation functions of this network are as follows:

| Layer | Weight | Activation function *f* |
| --- | --- | --- |
| 1 | *A* | *f*(*z**i*) = *z**i* (identity) |
| 2 | *I* (identity) | *f*(*z**i*) = *z**i* − *x**i* |
| 3 | N/A | f(z_i) = z_i^2 |

To have *J*(*z*(3)) = *F*(*x*), we can set ![J(z^{(3)}) = \sum_k J(z^{(3)}_k)](images/math/c/4/e/c4e22c48b65f68377d01e81d6312b145.png).

Once we see *F* as a neural network, the gradient ![\nabla_X F](images/math/c/8/a/c8a57f802f72156c4dbee1bd9fde338e.png) becomes easy to compute - applying backpropagation yields:

| Layer | Derivative of activation function *f*' | Delta | Input *z* to this layer |
| --- | --- | --- | --- |
| 3 | *f*'(*z**i*) = 2*z**i* | *f*'(*z**i*) = 2*z**i* | *A**s* − *x* |
| 2 | *f*'(*z**i*) = 1 | \left( I^T \delta^{(3)} \right) \bullet 1 | *A**s* |
| 1 | *f*'(*z**i*) = 1 | \left( A^T \delta^{(2)} \right) \bullet 1 | *s* |

Hence,

![
\begin{align}
\nabla_X F & = A^T I^T 2(As - x) \\
& = A^T 2(As - x)
\end{align}
](images/math/3/5/a/35a198eeea379f6e5fddd29fe4a6c2d7.png)

###   Example 2: Smoothed topographic L1 sparsity penalty in sparse coding

Recall the smoothed topographic L1 sparsity penalty on *s* in  [sparse coding](Sparse_Coding__Autoencoder_Interpretation.md "Sparse Coding: Autoencoder Interpretation"):

![\sum{ \sqrt{Vss^T + \epsilon} }](images/math/c/c/d/ccd5a0f991db6bdba852b147ee42d91b.png)

where *V* is the grouping matrix, *s* is the feature matrix and ε is a constant.

We would like to find ![\nabla_s \sum{ \sqrt{Vss^T + \epsilon} }](images/math/2/3/c/23c8b28a984cc20529d2eff361fbbe91.png). As above, let's see this term as an instantiation of a neural network:

![Backpropagation Method Example 2.png](images/thumb/5/57/Backpropagation_Method_Example_2.png/600px-Backpropagation_Method_Example_2.png)

The weights and activation functions of this network are as follows:

| Layer | Weight | Activation function *f* |
| --- | --- | --- |
| 1 | *I* | f(z_i) = z_i^2 |
| 2 | *V* | *f*(*z**i*) = *z**i* |
| 3 | *I* | *f*(*z**i*) = *z**i* + ε |
| 4 | N/A | f(z_i) = z_i^{\frac{1}{2}} |

To have *J*(*z*(4)) = *F*(*x*), we can set ![J(z^{(4)}) = \sum_k J(z^{(4)}_k)](images/math/5/c/c/5cc78742561e48008ea2fdc832873d87.png).

Once we see *F* as a neural network, the gradient ![\nabla_X F](images/math/c/8/a/c8a57f802f72156c4dbee1bd9fde338e.png) becomes easy to compute - applying backpropagation yields:

| Layer | Derivative of activation function *f*' | Delta | Input *z* to this layer |
| --- | --- | --- | --- |
| 4 | f'(z_i) = \frac{1}{2} z_i^{-\frac{1}{2}} | f'(z_i) = \frac{1}{2} z_i^{-\frac{1}{2}} | (*V**s**s**T* + ε) |
| 3 | *f*'(*z**i*) = 1 | \left( I^T \delta^{(4)} \right) \bullet 1 | *V**s**s**T* |
| 2 | *f*'(*z**i*) = 1 | \left( V^T \delta^{(3)} \right) \bullet 1 | *s**s**T* |
| 1 | *f*'(*z**i*) = 2*z**i* | \left( I^T \delta^{(2)} \right) \bullet 2s | *s* |

Hence,

![
\begin{align}
\nabla_X F & = I^T V^T I^T \frac{1}{2}(Vss^T + \epsilon)^{-\frac{1}{2}} \bullet 2s \\
& = V^T \frac{1}{2}(Vss^T + \epsilon)^{-\frac{1}{2}} \bullet 2s \\
& = V^T (Vss^T + \epsilon)^{-\frac{1}{2}} \bullet s
\end{align}
](images/math/c/0/1/c01e5b899a859c62c2a9de3d9e1bff34.png)

###   Example 3: ICA reconstruction cost

Recall the  [independent component analysis (ICA)](Independent_Component_Analysis.md "Independent Component Analysis") reconstruction cost term:
![\lVert W^TWx - x \rVert_2^2](images/math/c/9/8/c981b116dd26204d280f18b707c38a2c.png)
where *W* is the weight matrix and *x* is the input.

We would like to find ![\nabla_W \lVert W^TWx - x \rVert_2^2](images/math/c/1/0/c10b279f6aea106e455f113f8f3ab2c7.png) - the derivative of the term with respect to the **weight matrix**, rather than the **input** as in the earlier two examples. We will still proceed similarly though, seeing this term as an instantiation of a neural network:

![Backpropagation Method Example 3.png](images/thumb/9/9e/Backpropagation_Method_Example_3.png/400px-Backpropagation_Method_Example_3.png)

The weights and activation functions of this network are as follows:

| Layer | Weight | Activation function *f* |
| --- | --- | --- |
| 1 | *W* | *f*(*z**i*) = *z**i* |
| 2 | *W**T* | *f*(*z**i*) = *z**i* |
| 3 | *I* | *f*(*z**i*) = *z**i* − *x**i* |
| 4 | N/A | f(z_i) = z_i^2 |

To have *J*(*z*(4)) = *F*(*x*), we can set ![J(z^{(4)}) = \sum_k J(z^{(4)}_k)](images/math/5/c/c/5cc78742561e48008ea2fdc832873d87.png).

Now that we can see *F* as a neural network, we can try to compute the gradient ![\nabla_W F](images/math/e/7/3/e7379e93c2fe4b318c07026bd7adb4ab.png). However, we now face the difficulty that *W* appears twice in the network. Fortunately, it turns out that if *W* appears multiple times in the network, the gradient with respect to *W* is simply the sum of gradients for each instance of *W* in the network (you may wish to work out a formal proof of this fact to convince yourself). With this in mind, we will proceed to work out the deltas first:

| Layer | Derivative of activation function *f*' | Delta | Input *z* to this layer |
| --- | --- | --- | --- |
| 4 | *f*'(*z**i*) = 2*z**i* | *f*'(*z**i*) = 2*z**i* | (*W**T**W**x* − *x*) |
| 3 | *f*'(*z**i*) = 1 | \left( I^T \delta^{(4)} \right) \bullet 1 | *W**T**W**x* |
| 2 | *f*'(*z**i*) = 1 | \left( (W^T)^T \delta^{(3)} \right) \bullet 1 | *W**x* |
| 1 | *f*'(*z**i*) = 1 | \left( W^T \delta^{(2)} \right) \bullet 1 | *x* |

To find the gradients with respect to *W*, first we find the gradients with respect to each instance of *W* in the network.

With respect to *W**T*:

![
\begin{align}
\nabla_{W^T} F & = \delta^{(3)} a^{(2)T} \\
& = 2(W^TWx - x) (Wx)^T
\end{align}
](images/math/9/9/3/993726ae12c879a47221ef98d5278c7d.png)

With respect to *W*:

![
\begin{align}
\nabla_{W} F & = \delta^{(2)} a^{(1)T} \\
& = (W^T)(2(W^TWx -x)) x^T
\end{align}
](images/math/0/4/9/049ff65a0c5539792da624939122acb9.png)

Taking sums, noting that we need to transpose the gradient with respect to *W**T* to get the gradient with respect to *W*, yields the final gradient with respect to *W* (pardon the slight abuse of notation here):

![
\begin{align}
\nabla_{W} F & = \nabla_{W} F + (\nabla_{W^T} F)^T \\
& = (W^T)(2(W^TWx -x)) x^T + 2(Wx)(W^TWx - x)^T
\end{align}
](images/math/c/0/f/c0f24f7a4b6928641a9bc10318b6b85d.png)

---

> * Language: [中文](%E7%94%A8%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E6%80%9D%E6%83%B3%E6%B1%82%E5%AF%BC.md "用反向传导思想求导")
> * This page was last modified on 8 April 2013, at 04:26.

