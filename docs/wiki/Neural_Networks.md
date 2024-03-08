Neural Networks
===============

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
Consider a supervised learning problem where we have access to labeled training
examples (*x*(*i*),*y*(*i*)). Neural networks give a way of defining a complex,
non-linear form of hypotheses *h**W*,*b*(*x*), with parameters *W*,*b* that we can
fit to our data.

To describe neural networks, we will begin by describing the simplest possible
neural network, one which comprises a single "neuron." We will use the following
diagram to denote a single neuron:

![SingleNeuron.png](images/thumb/3/3d/SingleNeuron.png/300px-SingleNeuron.png)
This "neuron" is a computational unit that takes as input *x*1,*x*2,*x*3 (and a +1 intercept term), and
outputs ![\textstyle h_{W,b}(x) = f(W^Tx) = f(\sum_{i=1}^3 W_{i}x_i +b)](images/math/8/9/f/89f1f9e549b908834d9fedca36d07bd4.png), where ![f : \Re \mapsto \Re](images/math/1/b/4/1b46053bca8c30163f849554243a6061.png) is
called the **activation function**. In these notes, we will choose
![f(\cdot)](images/math/a/1/0/a1044326f95cfbf46f9859c97cf280be.png) to be the sigmoid function:

![
f(z) = \frac{1}{1+\exp(-z)}.
](images/math/c/e/5/ce5df10952ab30aa868f44db2f77486b.png)

Thus, our single
neuron corresponds exactly to the input-output mapping defined by logistic regression.

Although these notes will use the sigmoid function, it is worth noting that
another common choice for *f* is the hyperbolic tangent, or tanh, function:

![
f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}},  
](images/math/a/9/0/a9025d0884453bd5898c9681e871b3fb.png)

Here are plots of the sigmoid and tanh functions:

![Sigmoid activation function.](images/thumb/c/ca/Sigmoid_Function.png/400px-Sigmoid_Function.png)
![Tanh activation function.](images/thumb/a/aa/Tanh_Function.png/400px-Tanh_Function.png)

The tanh(*z*) function is a rescaled version of the sigmoid, and its output range is
[ − 1,1] instead of [0,1].

Note that unlike some other venues (including the OpenClassroom videos, and parts of CS229), we are not using the convention
here of *x*0 = 1. Instead, the intercept term is handled separately by the parameter *b*.

Finally, one identity that'll be useful later: If *f*(*z*) = 1 / (1 + exp( − *z*)) is the sigmoid
function, then its derivative is given by *f*'(*z*) = *f*(*z*)(1 − *f*(*z*)).
(If *f* is the tanh function, then its derivative is given by
*f*'(*z*) = 1 − (*f*(*z*))2.) You can derive this yourself using the definition of
the sigmoid (or tanh) function.

  Neural Network model
----------------------

A neural network is put together by hooking together many of our simple
"neurons," so that the output of a neuron can be the input of another. For
example, here is a small neural network:

![Network331.png](images/thumb/9/99/Network331.png/400px-Network331.png)
In this figure, we have used circles to also denote the inputs to the network. The circles
labeled "+1" are called **bias units**, and correspond to the intercept term.
The leftmost layer of the network is called the **input layer**, and the
rightmost layer the **output layer** (which, in this example, has only one
node). The middle layer of nodes is called the **hidden layer**, because its
values are not observed in the training set. We also say that our example
neural network has 3 **input units** (not counting the bias unit), 3 
**hidden units**, and 1 **output unit**.

We will let *n**l*
denote the number of layers in our network; thus *n**l* = 3 in our example. We label layer *l* as
*L**l*, so layer *L*1 is the input layer, and layer ![L_{n_l}](images/math/7/6/3/763f726de36c3e92b1ac9b84e9f7f778.png) the output layer.
Our neural network has parameters (*W*,*b*) = (*W*(1),*b*(1),*W*(2),*b*(2)), where
we write
![W^{(l)}_{ij}](images/math/9/1/8/9183f327132cdf5ca9876aa4038f6e2f.png) to denote the parameter (or weight) associated with the connection
between unit *j* in layer *l*, and unit *i* in layer *l* + 1. (Note the order of the indices.)
Also, ![b^{(l)}_i](images/math/6/e/a/6ea0ff7533b239d7ad97668ee35c259d.png) is the bias associated with unit *i* in layer *l* + 1.
Thus, in our example, we have ![W^{(1)} \in \Re^{3\times 3}](images/math/f/1/b/f1b59a0d1b84461c5d4055909c08a4c9.png), and ![W^{(2)} \in \Re^{1\times 3}](images/math/5/5/2/552e1f3f4374b17f80228e0ecc8b9762.png).
Note that bias units don't have inputs or connections going into them, since they always output
the value +1. We also let *s**l* denote the number of nodes in layer *l* (not counting the bias unit).

We will write ![a^{(l)}_i](images/math/2/f/1/2f12132475b24d761ca573173962be9b.png) to denote the **activation** (meaning output value) of
unit *i* in layer *l*. For *l* = 1, we also use ![a^{(1)}_i = x_i](images/math/6/6/f/66f2ade33e4ad1fcfb34f814545193d7.png) to denote the *i*-th input.
Given a fixed setting of
the parameters *W*,*b*, our neural
network defines a hypothesis *h**W*,*b*(*x*) that outputs a real number. Specifically, the
computation that this neural network represents is given by:

![
\begin{align}
a_1^{(2)} &= f(W_{11}^{(1)}x_1 + W_{12}^{(1)} x_2 + W_{13}^{(1)} x_3 + b_1^{(1)})  \\
a_2^{(2)} &= f(W_{21}^{(1)}x_1 + W_{22}^{(1)} x_2 + W_{23}^{(1)} x_3 + b_2^{(1)})  \\
a_3^{(2)} &= f(W_{31}^{(1)}x_1 + W_{32}^{(1)} x_2 + W_{33}^{(1)} x_3 + b_3^{(1)})  \\
h_{W,b}(x) &= a_1^{(3)} =  f(W_{11}^{(2)}a_1^{(2)} + W_{12}^{(2)} a_2^{(2)} + W_{13}^{(2)} a_3^{(2)} + b_1^{(2)}) 
\end{align}
](images/math/f/d/e/fde22a388f607f526f03644c71a72f92.png)

In the sequel, we also let ![z^{(l)}_i](images/math/0/5/3/053932a35e5e7923d66bfd5cbc15b280.png) denote the total weighted sum of inputs to unit *i* in layer *l*,
including the bias term (e.g., ![\textstyle z_i^{(2)} = \sum_{j=1}^n W^{(1)}_{ij} x_j + b^{(1)}_i](images/math/a/a/e/aae7340fe1eb75c824b8abc107c3db27.png)), so that
![a^{(l)}_i = f(z^{(l)}_i)](images/math/4/9/0/49021bbf2ba72dad62e1e785a8f44d14.png).

Note that this easily lends itself to a more compact notation. Specifically, if we extend the
activation function ![f(\cdot)](images/math/a/1/0/a1044326f95cfbf46f9859c97cf280be.png)
to apply to vectors in an element-wise fashion (i.e.,
*f*([*z*1,*z*2,*z*3]) = [*f*(*z*1),*f*(*z*2),*f*(*z*3)]), then we can write
the equations above more
compactly as:

![\begin{align}
z^{(2)} &= W^{(1)} x + b^{(1)} \\
a^{(2)} &= f(z^{(2)}) \\
z^{(3)} &= W^{(2)} a^{(2)} + b^{(2)} \\
h_{W,b}(x) &= a^{(3)} = f(z^{(3)})
\end{align}](images/math/9/6/9/9690acc03c1e5133b0509257b532b4f7.png)

We call this step **forward propagation.** More generally, recalling that we also use *a*(1) = *x* to also denote the values from the input layer,
then given layer *l*'s activations *a*(*l*), we can compute layer *l* + 1's activations *a*(*l* + 1) as:

![\begin{align}
z^{(l+1)} &= W^{(l)} a^{(l)} + b^{(l)}   \\
a^{(l+1)} &= f(z^{(l+1)})
\end{align}](images/math/5/c/f/5cfcbbe6d55b6c882f56a85a57eafe6e.png)

By organizing our parameters in matrices and using matrix-vector operations, we can take
advantage of fast linear algebra routines to quickly perform calculations in our network.

We have so far focused on one example neural network, but one can also build neural
networks with other **architectures** (meaning patterns of connectivity between neurons), including ones with multiple hidden layers.
The most common choice is a ![\textstyle n_l](images/math/5/b/7/5b7a0657fdea25f29866c8e1d6e884ac.png)-layered network
where layer ![\textstyle 1](images/math/6/e/9/6e924e04b5c9d4c5be131609a038b821.png) is the input layer, layer ![\textstyle n_l](images/math/5/b/7/5b7a0657fdea25f29866c8e1d6e884ac.png) is the output layer, and each
layer ![\textstyle l](images/math/b/a/0/ba0593b3db2fa8535b077516f4b0d70b.png) is densely connected to layer ![\textstyle l+1](images/math/9/0/6/9068105ec8ebb97277c937bfa61b606d.png). In this setting, to compute the
output of the network, we can successively compute all the activations in layer
![\textstyle L_2](images/math/c/f/7/cf7d186efd913f4fb9ceb939bf5135c4.png), then layer ![\textstyle L_3](images/math/d/9/b/d9b949d768ca8bab18830d9efc3fa441.png), and so on, up to layer ![\textstyle L_{n_l}](images/math/2/2/1/221a7296664022427d488fdb9b14b19b.png), using the equations above that describe the forward propagation step. This is one
example of a **feedforward** neural network, since the connectivity graph
does not have any directed loops or cycles.

Neural networks can also have multiple output units. For example, here is a network
with two hidden layers layers *L*2 and *L*3 and two output units in layer *L*4:

![Network3322.png](images/thumb/4/40/Network3322.png/500px-Network3322.png)
To train this network, we would need training examples (*x*(*i*),*y*(*i*))
where ![y^{(i)} \in \Re^2](images/math/c/d/7/cd7718ae0161c845e716767f06285af0.png). This sort of network is useful if there're multiple
outputs that you're interested in predicting. (For example, in a medical
diagnosis application, the vector *x* might give the input features of a
patient, and the different outputs *y**i*'s might indicate presence or absence
of different diseases.)

**Neural Networks** | [Backpropagation Algorithm](Backpropagation_Algorithm.md "Backpropagation Algorithm") | [Gradient checking and advanced optimization](Gradient_checking_and_advanced_optimization.md "Gradient checking and advanced optimization") | [Autoencoders and Sparsity](Autoencoders_and_Sparsity.md "Autoencoders and Sparsity") | [Visualizing a Trained Autoencoder](Visualizing_a_Trained_Autoencoder.md "Visualizing a Trained Autoencoder") | [Sparse Autoencoder Notation Summary](Sparse_Autoencoder_Notation_Summary.md "Sparse Autoencoder Notation Summary") | [Exercise:Sparse Autoencoder](Exercise_Sparse_Autoencoder.md "Exercise:Sparse Autoencoder")

---

> * Language: [中文](%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md "神经网络")
> * This page was last modified on 6 April 2013, at 19:38.

