

### Autoencoders

---

So far, we have described the application of neural networks to supervised learning, in which we have labeled training examples. Now suppose we have only a set of unlabeled training examples $\textstyle \{x^{(1)}, x^{(2)}, x^{(3)}, \ldots\}$, where $\textstyle x^{(i)} \in \Re^{n}$. An **autoencoder** neural network is an unsupervised learning algorithm that applies backpropagation, setting the target values to be equal to the inputs. I.e., it uses $\textstyle y^{(i)} = x^{(i)}$.

Here is an autoencoder:

![](/wayback-mooc/stanford-ufldl/tutorial/images/Autoencoder636.png)

The autoencoder tries to learn a function $\textstyle h_{W,b}(x) \approx x$. In other words, it is trying to learn an approximation to the identity function, so as to output $\textstyle \hat{x}$ that is similar to $\textstyle x$. The identity function seems a particularly trivial function to be trying to learn; but by placing constraints on the network, such as by limiting the number of hidden units, we can discover interesting structure about the data. As a concrete example, suppose the inputs $\textstyle x$ are the pixel intensity values from a $\textstyle 10 \times 10$ image (100 pixels) so $\textstyle n=100$, and there are $\textstyle s_2=50$ hidden units in layer $\textstyle L_2$. Note that we also have $\textstyle y \in \Re^{100}$. Since there are only 50 hidden units, the network is forced to learn a ”compressed” representation of the input. I.e., given only the vector of hidden unit activations $\textstyle a^{(2)} \in \Re^{50}$, it must try to ”‘reconstruct”’ the 100-pixel input $\textstyle x$. If the input were completely random—say, each $\textstyle x_i$ comes from an IID Gaussian independent of the other features—then this compression task would be very difficult. But if there is structure in the data, for example, if some of the input features are correlated, then this algorithm will be able to discover some of those correlations. In fact, this simple autoencoder often ends up learning a low-dimensional representation very similar to PCAs.

Our argument above relied on the number of hidden units $\textstyle s_2$ being small. But even when the number of hidden units is large (perhaps even greater than the number of input pixels), we can still discover interesting structure, by imposing other constraints on the network. In particular, if we impose a ”‘sparsity”’ constraint on the hidden units, then the autoencoder will still discover interesting structure in the data, even if the number of hidden units is large.

Informally, we will think of a neuron as being “active” (or as “firing”) if its output value is close to 1, or as being “inactive” if its output value is close to 0. We would like to constrain the neurons to be inactive most of the time. This discussion assumes a sigmoid activation function. If you are using a tanh activation function, then we think of a neuron as being inactive when it outputs values close to -1.

Recall that $\textstyle a^{(2)}_j$ denotes the activation of hidden unit $\textstyle j$ in the autoencoder. However, this notation doesn’t make explicit what was the input $\textstyle x$ that led to that activation. Thus, we will write $\textstyle a^{(2)}_j(x)$ to denote the activation of this hidden unit when the network is given a specific input $\textstyle x$.

Further, let

$$
\begin{align}
\hat\rho_j = \frac{1}{m} \sum_{i=1}^m \left[ a^{(2)}_j(x^{(i)}) \right]
\end{align}
$$

be the average activation of hidden unit $\textstyle j$ (averaged over the training set). We would like to (approximately) enforce the constraint

$$
\begin{align}
\hat\rho_j = \rho,
\end{align}
$$

where $\textstyle \rho$ is a ”‘sparsity parameter”’, typically a small value close to zero (say $\textstyle \rho = 0.05$). In other words, we would like the average activation of each hidden neuron $\textstyle j$ to be close to 0.05 (say). To satisfy this constraint, the hidden unit’s activations must mostly be near 0.

To achieve this, we will add an extra penalty term to our optimization objective that penalizes $\textstyle \hat\rho_j$ deviating significantly from $\textstyle \rho$. Many choices of the penalty term will give reasonable results. We will choose the following:

$$
\begin{align}
\sum_{j=1}^{s_2} \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}.
\end{align}
$$

Here, $\textstyle s_2$ is the number of neurons in the hidden layer, and the index $\textstyle j$ is summing over the hidden units in our network. If you are familiar with the concept of KL divergence, this penalty term is based on it, and can also be written

$$
\begin{align}
\sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}
$$

where $\textstyle {\rm KL}(\rho || \hat\rho_j) = \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}$ is the Kullback-Leibler (KL) divergence between a Bernoulli random variable with mean $\textstyle \rho$ and a Bernoulli random variable with mean $\textstyle \hat\rho_j$. KL-divergence is a standard function for measuring how different two different distributions are. (If you’ve not seen KL-divergence before, don’t worry about it; everything you need to know about it is contained in these notes.)

This penalty function has the property that $\textstyle {\rm KL}(\rho || \hat\rho_j) = 0$ if $\textstyle \hat\rho_j = \rho$, and otherwise it increases monotonically as $\textstyle \hat\rho_j$ diverges from $\textstyle \rho$. For example, in the figure below, we have set $\textstyle \rho = 0.2$, and plotted $\textstyle {\rm KL}(\rho || \hat\rho_j)$ for a range of values of $\textstyle \hat\rho_j$:

![](/wayback-mooc/stanford-ufldl/tutorial/images/KLPenaltyExample.png)

We see that the KL-divergence reaches its minimum of 0 at

$\textstyle \hat\rho_j = \rho$
, and blows up (it actually

approaches $\textstyle \infty$) as $\textstyle \hat\rho_j$ approaches 0 or 1. Thus, minimizing this penalty term has the effect of causing $\textstyle \hat\rho_j$ to be close to $\textstyle \rho$.

Our overall cost function is now

$$
\begin{align}
J_{\rm sparse}(W,b) = J(W,b) + \beta \sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}
$$

where $\textstyle J(W,b)$ is as defined previously, and $\textstyle \beta$ controls the weight of the sparsity penalty term. The term $\textstyle \hat\rho_j$ (implicitly) depends on $\textstyle W,b$ also, because it is the average activation of hidden unit $\textstyle j$, and the activation of a hidden unit depends on the parameters $\textstyle W,b$.

To incorporate the KL-divergence term into your derivative calculation, there is a simple-to-implement trick involving only a small change to your code. Specifically, where previously for the second layer ($\textstyle l=2$), during backpropagation you would have computed

$$
\begin{align}
\delta^{(2)}_i = \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right) f'(z^{(2)}_i),
\end{align}
$$

now instead compute

$$
\begin{align}
\delta^{(2)}_i =
\left( \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right)
+ \beta \left( - \frac{\rho}{\hat\rho_i} + \frac{1-\rho}{1-\hat\rho_i} \right) \right) f'(z^{(2)}_i) .
\end{align}
$$

One subtlety is that you’ll need to know $\textstyle \hat\rho_i$ to compute this term. Thus, you’ll need to compute a forward pass on all the training examples first to compute the average activations on the training set, before computing backpropagation on any example. If your training set is small enough to fit comfortably in computer memory (this will be the case for the programming assignment), you can compute forward passes on all your examples and keep the resulting activations in memory and compute the $\textstyle \hat\rho_i$s. Then you can use your precomputed activations to perform backpropagation on all your examples. If your data is too large to fit in memory, you may have to scan through your examples computing a forward pass on each to accumulate (sum up) the activations and compute $\textstyle \hat\rho_i$ (discarding the result of each forward pass after you have taken its activations $\textstyle a^{(2)}_i$ into account for computing $\textstyle \hat\rho_i$). Then after having computed $\textstyle \hat\rho_i$, you’d have to redo the forward pass for each example so that you can do backpropagation on that example. In this latter case, you would end up computing a forward pass twice on each example in your training set, making it computationally less efficient.

The full derivation showing that the algorithm above results in gradient descent is beyond the scope of these notes. But if you implement the autoencoder using backpropagation modified this way, you will be performing gradient descent exactly on the objective $\textstyle J_{\rm sparse}(W,b)$. Using the derivative checking method, you will be able to verify this for yourself as well.

### Visualizing a Trained Autoencoder

Having trained a (sparse) autoencoder, we would now like to visualize the function learned by the algorithm, to try to understand what it has learned. Consider the case of training an autoencoder on $\textstyle 10 \times 10$ images, so that $\textstyle n = 100$. Each hidden unit $\textstyle i$ computes a function of the input:

$$
\begin{align}
a^{(2)}_i = f\left(\sum_{j=1}^{100} W^{(1)}_{ij} x_j + b^{(1)}_i \right).
\end{align}
$$

We will visualize the function computed by hidden unit $\textstyle i$—which depends on the parameters $\textstyle W^{(1)}_{ij}$ (ignoring the bias term for now)—using a 2D image. In particular, we think of $\textstyle a^{(2)}_i$ as some non-linear feature of the input $\textstyle x$. We ask: What input image $\textstyle x$ would cause $\textstyle a^{(2)}_i$ to be maximally activated? (Less formally, what is the feature that hidden unit $\textstyle i$ is looking for?) For this question to have a non-trivial answer, we must impose some constraints on $\textstyle x$. If we suppose that the input is norm constrained by $\textstyle ||x||^2 = \sum_{i=1}^{100} x_i^2 \leq 1$, then one can show (try doing this yourself) that the input which maximally activates hidden unit $\textstyle i$ is given by setting pixel $\textstyle x_j$ (for all 100 pixels, $\textstyle j=1,\ldots, 100$) to

$$
\begin{align}
x_j = \frac{W^{(1)}_{ij}}{\sqrt{\sum_{j=1}^{100} (W^{(1)}_{ij})^2}}.
\end{align}
$$

By displaying the image formed by these pixel intensity values, we can begin to understand what feature hidden unit $\textstyle i$ is looking for.

If we have an autoencoder with 100 hidden units (say), then we our visualization will have 100 such images—one per hidden unit. By examining these 100 images, we can try to understand what the ensemble of hidden units is learning.

When we do this for a sparse autoencoder (trained with 100 hidden units on 10x10 pixel inputs1 we get the following result:

![](/wayback-mooc/stanford-ufldl/tutorial/images/ExampleSparseAutoencoderWeights.png)

Each square in the figure above shows the (norm bounded) input image $\textstyle x$ that maximally actives one of 100 hidden units. We see that the different hidden units have learned to detect edges at different positions and orientations in the image.

These features are, not surprisingly, useful for such tasks as object recognition and other vision tasks. When applied to other input domains (such as audio), this algorithm also learns useful representations/features for those domains too.

---
