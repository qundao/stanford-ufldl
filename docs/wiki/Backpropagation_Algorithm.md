Backpropagation Algorithm
=========================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
Suppose we have a fixed training set ![\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}](images/math/5/e/c/5ec89e9cf3712d45b80e93258352ea8f.png) of *m* training examples. We can train our neural network using batch gradient descent. In detail, for a single training example (*x*,*y*), we define the cost function with respect to that single example to be:

![
\begin{align}
J(W,b; x,y) = \frac{1}{2} \left\| h_{W,b}(x) - y \right\|^2.
\end{align}
](images/math/0/2/9/029cdd402b83ee43c7e9a900dccd675a.png)

This is a (one-half) squared-error cost function. Given a training set of *m* examples, we then define the overall cost function to be:

![
\begin{align}
J(W,b)
&= \left[ \frac{1}{m} \sum_{i=1}^m J(W,b;x^{(i)},y^{(i)}) \right]
                       + \frac{\lambda}{2} \sum_{l=1}^{n_l-1} \; \sum_{i=1}^{s_l} \; \sum_{j=1}^{s_{l+1}} \left( W^{(l)}_{ji} \right)^2
 \\
&= \left[ \frac{1}{m} \sum_{i=1}^m \left( \frac{1}{2} \left\| h_{W,b}(x^{(i)}) - y^{(i)} \right\|^2 \right) \right]
                       + \frac{\lambda}{2} \sum_{l=1}^{n_l-1} \; \sum_{i=1}^{s_l} \; \sum_{j=1}^{s_{l+1}} \left( W^{(l)}_{ji} \right)^2
\end{align}
](images/math/4/5/3/4539f5f00edca977011089b902670513.png)

The first term in the definition of *J*(*W*,*b*) is an average sum-of-squares error term. The second term is a regularization term (also called a **weight decay** term) that tends to decrease the magnitude of the weights, and helps prevent overfitting.

[Note: Usually weight decay is not applied to the bias terms ![b^{(l)}_i](images/math/6/e/a/6ea0ff7533b239d7ad97668ee35c259d.png), as reflected in our definition for *J*(*W*,*b*). Applying weight decay to the bias units usually makes only a small difference to the final network, however. If you've taken CS229 (Machine Learning) at Stanford or watched the course's videos on YouTube, you may also recognize this weight decay as essentially a variant of the Bayesian regularization method you saw there, where we placed a Gaussian prior on the parameters and did MAP (instead of maximum likelihood) estimation.]

The **weight decay parameter** λ controls the relative importance of the two terms. Note also the slightly overloaded notation: *J*(*W*,*b*;*x*,*y*) is the squared error cost with respect to a single example; *J*(*W*,*b*) is the overall cost function, which includes the weight decay term.

This cost function above is often used both for classification and for regression problems. For classification, we let *y* = 0 or 1 represent the two class labels (recall that the sigmoid activation function outputs values in [0,1]; if we were using a tanh activation function, we would instead use -1 and +1 to denote the labels). For regression problems, we first scale our outputs to ensure that they lie in the [0,1] range (or if we were using a tanh activation function, then the [ − 1,1] range).

Our goal is to minimize *J*(*W*,*b*) as a function of *W* and *b*. To train our neural network, we will initialize each parameter ![W^{(l)}_{ij}](images/math/9/1/8/9183f327132cdf5ca9876aa4038f6e2f.png) and each ![b^{(l)}_i](images/math/6/e/a/6ea0ff7533b239d7ad97668ee35c259d.png) to a small random value near zero (say according to a *N**o**r**m**a**l*(0,ε2) distribution for some small ε, say 0.01), and then apply an optimization algorithm such as batch gradient descent. Since *J*(*W*,*b*) is a non-convex function,
gradient descent is susceptible to local optima; however, in practice gradient descent
usually works fairly well. Finally, note that it is important to initialize
the parameters randomly, rather than to all 0's. If all the parameters start off
at identical values, then all the hidden layer units will end up learning the same
function of the input (more formally, ![W^{(1)}_{ij}](images/math/6/9/b/69b82501f76f6552dfe039cb8676511a.png) will be the same for all values of *i*, so that ![a^{(2)}_1 = a^{(2)}_2 = a^{(2)}_3 = \ldots](images/math/0/9/9/0995e9a2d04545cde7f01b9ac4250c01.png) for any input *x*). The random initialization serves the purpose of **symmetry breaking**.

One iteration of gradient descent updates the parameters *W*,*b* as follows:

![
\begin{align}
W_{ij}^{(l)} &= W_{ij}^{(l)} - \alpha \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) \\
b_{i}^{(l)} &= b_{i}^{(l)} - \alpha \frac{\partial}{\partial b_{i}^{(l)}} J(W,b)
\end{align}
](images/math/6/f/e/6fe7c74511cd6d49a4c9cb6de2afdc33.png)

where α is the learning rate. The key step is computing the partial derivatives above. We will now describe the **backpropagation** algorithm, which gives an
efficient way to compute these partial derivatives.

We will first describe how backpropagation can be used to compute ![\textstyle \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x, y)](images/math/5/f/b/5fb8e62e296ad365a076617b04d66d03.png) and ![\textstyle \frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x, y)](images/math/c/a/4/ca49d387f9ead91008f9688b3880e91b.png), the partial derivatives of the cost function *J*(*W*,*b*;*x*,*y*) defined with respect to a single example (*x*,*y*). Once we can compute these, we see that the derivative of the overall cost function *J*(*W*,*b*) can be computed as:

![
\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) &=
\left[ \frac{1}{m} \sum_{i=1}^m \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x^{(i)}, y^{(i)}) \right] + \lambda W_{ij}^{(l)} \\
\frac{\partial}{\partial b_{i}^{(l)}} J(W,b) &=
\frac{1}{m}\sum_{i=1}^m \frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x^{(i)}, y^{(i)})
\end{align}
](images/math/9/3/3/93367cceb154c392aa7f3e0f5684a495.png)

The two lines above differ slightly because weight decay is applied to *W* but not *b*.

The intuition behind the backpropagation algorithm is as follows. Given a training example (*x*,*y*), we will first run a "forward pass" to compute all the activations throughout the network, including the output value of the hypothesis *h**W*,*b*(*x*). Then, for each node *i* in layer *l*, we would like to compute an "error term" ![\delta^{(l)}_i](images/math/1/7/f/17f04626a30c825517a517e06870355c.png) that measures how much that node was "responsible" for any errors in our output. For an output node, we can directly measure the difference between the network's activation and the true target value, and use that to define ![\delta^{(n_l)}_i](images/math/a/c/9/ac95960f5ef00c208f5a2c730b5f6dcd.png) (where layer *n**l* is the output layer). How about hidden units? For those, we will compute ![\delta^{(l)}_i](images/math/1/7/f/17f04626a30c825517a517e06870355c.png) based on a weighted average of the error terms of the nodes that uses ![a^{(l)}_i](images/math/2/f/1/2f12132475b24d761ca573173962be9b.png) as an input. In detail, here is the backpropagation algorithm:

1. Perform a feedforward pass, computing the activations for layers *L*2, *L*3, and so on up to the output layer ![L_{n_l}](images/math/7/6/3/763f726de36c3e92b1ac9b84e9f7f778.png).
- For each output unit *i* in layer *n**l* (the output layer), set
![
\begin{align}
\delta^{(n_l)}_i
= \frac{\partial}{\partial z^{(n_l)}_i} \;\;
        \frac{1}{2} \left\|y - h_{W,b}(x)\right\|^2 = - (y_i - a^{(n_l)}_i) \cdot f'(z^{(n_l)}_i)
\end{align}
](images/math/5/7/a/57a203683fc9c009c41ff97c1e1f6f54.png)
- For ![l = n_l-1, n_l-2, n_l-3, \ldots, 2](images/math/9/8/8/988861db3f04c9f1150b482aca116daa.png)
For each node *i* in layer *l*, set
![
                 \delta^{(l)}_i = \left( \sum_{j=1}^{s_{l+1}} W^{(l)}_{ji} \delta^{(l+1)}_j \right) f'(z^{(l)}_i)
                 ](images/math/2/0/f/20f9979d6a46e7bca83f217bdfead4f0.png)

- Compute the desired partial derivatives, which are given as: 
![
\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x, y) &= a^{(l)}_j \delta_i^{(l+1)} \\
\frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x, y) &= \delta_i^{(l+1)}.
\end{align}
](images/math/2/1/d/21db5874b1c1c14bcb675e9961dac9cb.png)

Finally, we can also re-write the algorithm using matrix-vectorial notation. We will use "![\textstyle \bullet](images/math/9/9/3/9937b108a65d2d09961c23259e819e31.png)" to denote the element-wise product operator (denoted ".\*" in Matlab or Octave, and also called the Hadamard product), so that if ![\textstyle a = b \bullet c](images/math/b/1/3/b1362783e5c1d9d1e627ca2a91b04f28.png), then ![\textstyle a_i = b_ic_i](images/math/1/4/b/14b4e060883883de874d0ebf1ab758d3.png). Similar to how we extended the definition of ![\textstyle f(\cdot)](images/math/0/3/0/0303dd697c0e1b72185d7939f9870784.png) to apply element-wise to vectors, we also do the same for ![\textstyle f'(\cdot)](images/math/f/e/d/fedde117b610fc785ad71db67e618ab2.png) (so that ![\textstyle f'([z_1, z_2, z_3]) =
[f'(z_1),
f'(z_2),
f'(z_3)]](images/math/c/7/5/c7515c53b59e670ceee277e06c1229cb.png)).

The algorithm can then be written:

1. Perform a feedforward pass, computing the activations for layers ![\textstyle L_2](images/math/c/f/7/cf7d186efd913f4fb9ceb939bf5135c4.png), ![\textstyle L_3](images/math/d/9/b/d9b949d768ca8bab18830d9efc3fa441.png), up to the output layer ![\textstyle L_{n_l}](images/math/2/2/1/221a7296664022427d488fdb9b14b19b.png), using the equations defining the forward propagation steps
- For the output layer (layer ![\textstyle n_l](images/math/5/b/7/5b7a0657fdea25f29866c8e1d6e884ac.png)), set 
![\begin{align}
\delta^{(n_l)}
= - (y - a^{(n_l)}) \bullet f'(z^{(n_l)})
\end{align}](images/math/0/e/a/0ea6bda6255f544dca0bfa80d622f382.png)
- For ![\textstyle l = n_l-1, n_l-2, n_l-3, \ldots, 2](images/math/d/c/5/dc5396666d7679f1dae597dbc1a8ff5d.png)
Set
![\begin{align}
                 \delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \bullet f'(z^{(l)})
                 \end{align}](images/math/7/d/5/7d5660d4a911ecb84113c436f82b1109.png)

- Compute the desired partial derivatives: 
![\begin{align}
\nabla_{W^{(l)}} J(W,b;x,y) &= \delta^{(l+1)} (a^{(l)})^T, \\
\nabla_{b^{(l)}} J(W,b;x,y) &= \delta^{(l+1)}.
\end{align}](images/math/5/3/9/5391ac390a4e279ac8a543d4d5498ecc.png)

**Implementation note:** In steps 2 and 3 above, we need to compute ![\textstyle f'(z^{(l)}_i)](images/math/f/7/4/f745dea1a82d8cd64aa6b92466e3bbc5.png) for each value of ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png). Assuming ![\textstyle f(z)](images/math/5/d/1/5d1c55e9d6b297473de425651557d4fc.png) is the sigmoid activation function, we would already have ![\textstyle a^{(l)}_i](images/math/c/9/b/c9b144e0a6735fafb01b3615a2a0dc05.png) stored away from the forward pass through the network. Thus, using the expression that we worked out earlier for ![\textstyle f'(z)](images/math/a/5/f/a5f7d3f914f4e383ce51e4998592caee.png), 
we can compute this as ![\textstyle f'(z^{(l)}_i) = a^{(l)}_i (1- a^{(l)}_i)](images/math/d/4/d/d4d5e09ac8e035283671cc03d942f955.png).

Finally, we are ready to describe the full gradient descent algorithm. In the pseudo-code
below, ![\textstyle \Delta W^{(l)}](images/math/6/c/6/6c600894179e37800af01a5795be30b8.png) is a matrix (of the same dimension as ![\textstyle W^{(l)}](images/math/f/8/f/f8f8834256f511d88fec05e3b27c67b1.png)), and ![\textstyle \Delta b^{(l)}](images/math/e/5/8/e580f95036a0ccb35019a866cb10191f.png) is a vector (of the same dimension as ![\textstyle b^{(l)}](images/math/8/c/2/8c2936afffcaf9eeabf8837d501ddb9d.png)). Note that in this notation, 
"![\textstyle \Delta W^{(l)}](images/math/6/c/6/6c600894179e37800af01a5795be30b8.png)" is a matrix, and in particular it isn't "![\textstyle \Delta](images/math/5/2/9/529ca30eb74564461bc8e0e7d7864e95.png) times ![\textstyle W^{(l)}](images/math/f/8/f/f8f8834256f511d88fec05e3b27c67b1.png)." We implement one iteration of batch gradient descent as follows:

1. Set ![\textstyle \Delta W^{(l)} := 0](images/math/3/6/5/3650852a6b08d261b08a5f4f324fe3a0.png), ![\textstyle \Delta b^{(l)} := 0](images/math/7/5/b/75bf8778e859c31930f7629fe5eab821.png) (matrix/vector of zeros) for all ![\textstyle l](images/math/b/a/0/ba0593b3db2fa8535b077516f4b0d70b.png).
- For ![\textstyle i = 1](images/math/2/9/6/2964cb4e8851d521d24364f0d409a51d.png) to ![\textstyle m](images/math/2/5/e/25e97e8a905fc2cb05d76cd4872a8567.png),
	1. Use backpropagation to compute ![\textstyle \nabla_{W^{(l)}} J(W,b;x,y)](images/math/d/2/1/d21ff7e7308c9fd8c428fd926f671a39.png) and 
	![\textstyle \nabla_{b^{(l)}} J(W,b;x,y)](images/math/f/e/d/fed489077fe3753c894638d131c0b442.png).
	- Set ![\textstyle \Delta W^{(l)} := \Delta W^{(l)} + \nabla_{W^{(l)}} J(W,b;x,y)](images/math/5/0/b/50bd90d031437ba98debea738afad0a2.png). 
	- Set ![\textstyle \Delta b^{(l)} := \Delta b^{(l)} + \nabla_{b^{(l)}} J(W,b;x,y)](images/math/3/a/b/3abc7162b757ceac7bdb8f0c4555fe8e.png).- Update the parameters:
![\begin{align}
W^{(l)} &= W^{(l)} - \alpha \left[ \left(\frac{1}{m} \Delta W^{(l)} \right) + \lambda W^{(l)}\right] \\
b^{(l)} &= b^{(l)} - \alpha \left[\frac{1}{m} \Delta b^{(l)}\right]
\end{align}](images/math/0/f/7/0f7430e97ec4df1bfc56357d1485405f.png)

To train our neural network, we can now repeatedly take steps of gradient descent to reduce our cost function ![\textstyle J(W,b)](images/math/8/e/9/8e94ae776ae14b36b3af183726ababb9.png).

[Neural Networks](Neural_Networks.md "Neural Networks") | **Backpropagation Algorithm** | [Gradient checking and advanced optimization](Gradient_checking_and_advanced_optimization.md "Gradient checking and advanced optimization") | [Autoencoders and Sparsity](Autoencoders_and_Sparsity.md "Autoencoders and Sparsity") | [Visualizing a Trained Autoencoder](Visualizing_a_Trained_Autoencoder.md "Visualizing a Trained Autoencoder") | [Sparse Autoencoder Notation Summary](Sparse_Autoencoder_Notation_Summary.md "Sparse Autoencoder Notation Summary") | [Exercise:Sparse Autoencoder](Exercise_Sparse_Autoencoder.md "Exercise:Sparse Autoencoder")

---

> * Language: [中文](%E5%8F%8D%E5%90%91%E4%BC%A0%E5%AF%BC%E7%AE%97%E6%B3%95.md "反向传导算法")
> * This page was last modified on 7 April 2013, at 12:50.

