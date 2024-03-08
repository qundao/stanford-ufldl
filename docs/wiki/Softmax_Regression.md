Softmax Regression
==================

(Redirected from Softmax regression)
<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->

|  |
| --- |
| Contents* [1 Introduction](#Introduction)
* [2 Cost Function](#Cost_Function)
* [3 Properties of softmax regression parameterization](#Properties_of_softmax_regression_parameterization)
* [4 Weight Decay](#Weight_Decay)
* [5 Relationship to Logistic Regression](#Relationship_to_Logistic_Regression)
* [6 Softmax Regression vs. k Binary Classifiers](#Softmax_Regression_vs._k_Binary_Classifiers)
 |

  Introduction
--------------

In these notes, we describe the **Softmax regression** model. This model generalizes logistic regression to
classification problems where the class label *y* can take on more than two possible values.
This will be useful for such problems as MNIST digit classification, where the goal is to distinguish between 10 different
numerical digits. Softmax regression is a supervised learning algorithm, but we will later be
using it in conjuction with our deep learning/unsupervised feature learning methods.

Recall that in logistic regression, we had a training set
![\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}](images/math/5/e/c/5ec89e9cf3712d45b80e93258352ea8f.png)
of *m* labeled examples, where the input features are ![x^{(i)} \in \Re^{n+1}](images/math/1/2/3/123c8ca74aa217158129b671fc7e75a8.png). 
(In this set of notes, we will use the notational convention of letting the feature vectors *x* be
*n* + 1 dimensional, with *x*0 = 1 corresponding to the intercept term.) 
With logistic regression, we were in the binary classification setting, so the labels 
were ![y^{(i)} \in \{0,1\}](images/math/a/5/8/a589c252daed983404e6f9b3b1219954.png). Our hypothesis took the form:

![\begin{align}
h_\theta(x) = \frac{1}{1+\exp(-\theta^Tx)},
\end{align}](images/math/b/b/3/bb3791d463b832a88731b94f1d8e5279.png)

and the model parameters θ were trained to minimize
the cost function

![
\begin{align}
J(\theta) = -\frac{1}{m} \left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
\end{align}
](images/math/f/a/6/fa6565f1e7b91831e306ec404ccc1156.png)

In the softmax regression setting, we are interested in multi-class
classification (as opposed to only binary classification), and so the label
*y* can take on *k* different values, rather than only
two. Thus, in our training set
![\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}](images/math/5/e/c/5ec89e9cf3712d45b80e93258352ea8f.png),
we now have that ![y^{(i)} \in \{1, 2, \ldots, k\}](images/math/7/d/c/7dc095cfb7e3e1fc6bdbc358bd3e2888.png). (Note that
our convention will be to index the classes starting from 1, rather than from 0.) For example,
in the MNIST digit recognition task, we would have *k* = 10 different classes.

Given a test input *x*, we want our hypothesis to estimate
the probability that *p*(*y* = *j* | *x*) for each value of ![j = 1, \ldots, k](images/math/f/b/9/fb9db94a64ef15d63436fa6103921b86.png).
I.e., we want to estimate the probability of the class label taking
on each of the *k* different possible values. Thus, our hypothesis
will output a *k* dimensional vector (whose elements sum to 1) giving
us our *k* estimated probabilities. Concretely, our hypothesis
*h*θ(*x*) takes the form:

![
\begin{align}
h_\theta(x^{(i)}) =
\begin{bmatrix}
p(y^{(i)} = 1 | x^{(i)}; \theta) \\
p(y^{(i)} = 2 | x^{(i)}; \theta) \\
\vdots \\
p(y^{(i)} = k | x^{(i)}; \theta)
\end{bmatrix}
=
\frac{1}{ \sum_{j=1}^{k}{e^{ \theta_j^T x^{(i)} }} }
\begin{bmatrix}
e^{ \theta_1^T x^{(i)} } \\
e^{ \theta_2^T x^{(i)} } \\
\vdots \\
e^{ \theta_k^T x^{(i)} } \\
\end{bmatrix}
\end{align}
](images/math/a/1/b/a1b0d7b40fe624cd8a24354792223a9d.png)

Here ![\theta_1, \theta_2, \ldots, \theta_k \in \Re^{n+1}](images/math/f/d/9/fd93be6ab8e2b869691579202d7b4417.png) are the
parameters of our model. 
Notice that
the term ![\frac{1}{ \sum_{j=1}^{k}{e^{ \theta_j^T x^{(i)} }} } ](images/math/a/a/b/aab84964dbe1a2f77c9c91327ea0d6d6.png)
normalizes the distribution, so that it sums to one.

For convenience, we will also write 
θ to denote all the
parameters of our model. When you implement softmax regression, it is usually
convenient to represent θ as a *k*-by-(*n* + 1) matrix obtained by
stacking up ![\theta_1, \theta_2, \ldots, \theta_k](images/math/1/f/f/1ff687194349ee543cd4f1baa7bcaa58.png) in rows, so that

![
\theta = \begin{bmatrix}
\mbox{---} \theta_1^T \mbox{---} \\
\mbox{---} \theta_2^T \mbox{---} \\
\vdots \\
\mbox{---} \theta_k^T \mbox{---} \\
\end{bmatrix}
](images/math/a/b/4/ab4ba0d1df4b93696eec7d8bef86e9cd.png)

  Cost Function
---------------

We now describe the cost function that we'll use for softmax regression. In the equation below, ![1\{\cdot\}](images/math/f/e/1/fe105f91ad94aa996a47cb0ec3e2e2e8.png) is
the **indicator function,** so that 1{a true statement} = 1, and 1{a false statement} = 0.
For example, 1{2 + 2 = 4} evaluates to 1; whereas 1{1 + 1 = 5} evaluates to 0. Our cost function will be:

![
\begin{align}
J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{k}  1\left\{y^{(i)} = j\right\} \log \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)} }}\right]
\end{align}
](images/math/7/6/3/7634eb3b08dc003aa4591a95824d4fbd.png)

Notice that this generalizes the logistic regression cost function, which could also have been written:

![
\begin{align}
J(\theta) &= -\frac{1}{m} \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\
&= - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=0}^{1} 1\left\{y^{(i)} = j\right\} \log p(y^{(i)} = j | x^{(i)} ; \theta) \right]
\end{align}
](images/math/5/4/9/5491271f19161f8ea6a6b2a82c83fc3a.png)

The softmax cost function is similar, except that we now sum over the *k* different possible values
of the class label. Note also that in softmax regression, we have that
![
p(y^{(i)} = j | x^{(i)} ; \theta) = \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)}} }
](images/math/a/2/e/a2e69ec139cdd4828130c175d990d4e3.png).

There is no known closed-form way to solve for the minimum of *J*(θ), and thus as usual we'll resort to an iterative
optimization algorithm such as gradient descent or L-BFGS. Taking derivatives, one can show that the gradient is:

![
\begin{align}
\nabla_{\theta_j} J(\theta) = - \frac{1}{m} \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = j\}  - p(y^{(i)} = j | x^{(i)}; \theta) \right) \right]  }
\end{align}
](images/math/5/9/e/59ef406cef112eb75e54808b560587c9.png)

Recall the meaning of the "![\nabla_{\theta_j}](images/math/1/3/4/134b68f25065e33e0aee78aa8515824a.png)" notation. In particular, ![\nabla_{\theta_j} J(\theta)](images/math/e/d/8/ed816f8f2d99908bc1b6024da4cfa0bd.png)
is itself a vector, so that its *l*-th element is ![\frac{\partial J(\theta)}{\partial \theta_{jl}}](images/math/2/6/b/26bcda13f2b17ee8067d93022198e2c7.png)
the partial derivative of *J*(θ) with respect to the *l*-th element of θ*j*.

Armed with this formula for the derivative, one can then plug it into an algorithm such as gradient descent, and have it
minimize *J*(θ). For example, with the standard implementation of gradient descent, on each iteration
we would perform the update ![\theta_j := \theta_j - \alpha \nabla_{\theta_j} J(\theta)](images/math/e/f/6/ef6c858d8567880f0943848df352242f.png) (for each ![j=1,\ldots,k](images/math/f/b/9/fb9db94a64ef15d63436fa6103921b86.png)).

When implementing softmax regression, we will typically use a modified version of the cost function described above;
specifically, one that incorporates weight decay. We describe the motivation and details below.

  Properties of softmax regression parameterization
---------------------------------------------------

Softmax regression has an unusual property that it has a "redundant" set of parameters. To explain what this means, 
suppose we take each of our parameter vectors θ*j*, and subtract some fixed vector ψ
from it, so that every θ*j* is now replaced with θ*j* − ψ 
(for every ![j=1, \ldots, k](images/math/f/b/9/fb9db94a64ef15d63436fa6103921b86.png)). Our hypothesis
now estimates the class label probabilities as

![
\begin{align}
p(y^{(i)} = j | x^{(i)} ; \theta)
&= \frac{e^{(\theta_j-\psi)^T x^{(i)}}}{\sum_{l=1}^k e^{ (\theta_l-\psi)^T x^{(i)}}}  \\
&= \frac{e^{\theta_j^T x^{(i)}} e^{-\psi^Tx^{(i)}}}{\sum_{l=1}^k e^{\theta_l^T x^{(i)}} e^{-\psi^Tx^{(i)}}} \\
&= \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)}}}.
\end{align}
](images/math/d/8/0/d8076908fb40b49db821dc410b03700f.png)

In other words, subtracting ψ from every θ*j*
does not affect our hypothesis' predictions at all! This shows that softmax
regression's parameters are "redundant." More formally, we say that our
softmax model is **overparameterized,** meaning that for any hypothesis we might
fit to the data, there are multiple parameter settings that give rise to exactly
the same hypothesis function *h*θ mapping from inputs *x*
to the predictions.

Further, if the cost function *J*(θ) is minimized by some
setting of the parameters ![(\theta_1, \theta_2,\ldots, \theta_k)](images/math/6/1/e/61ead124771be0fc0903a00ed3dc5e56.png),
then it is also minimized by ![(\theta_1 - \psi, \theta_2 - \psi,\ldots,
\theta_k - \psi)](images/math/0/0/c/00cafcc498a2911c01b1ca5db6855869.png) for any value of ψ. Thus, the
minimizer of *J*(θ) is not unique. (Interestingly, 
*J*(θ) is still convex, and thus gradient descent will
not run into a local optima problems. But the Hessian is singular/non-invertible,
which causes a straightforward implementation of Newton's method to run into
numerical problems.)

Notice also that by setting ψ = θ1, one can always
replace θ1 with ![\theta_1 - \psi = \vec{0}](images/math/1/c/d/1cd054c99a1711d008e82fffd83b83e1.png) (the vector of all
0's), without affecting the hypothesis. Thus, one could "eliminate" the vector
of parameters θ1 (or any other θ*j*, for
any single value of *j*), without harming the representational power
of our hypothesis. Indeed, rather than optimizing over the *k*(*n* + 1)
parameters ![(\theta_1, \theta_2,\ldots, \theta_k)](images/math/6/1/e/61ead124771be0fc0903a00ed3dc5e56.png) (where
![\theta_j \in \Re^{n+1}](images/math/5/0/f/50f9d0b22f8fd210e7c611bb10869566.png)), one could instead set ![\theta_1 =
\vec{0}](images/math/d/a/9/da9190c2e350c287b0ab05acbb87f3aa.png) and optimize only with respect to the (*k* − 1)(*n* + 1)
remaining parameters, and this would work fine.

In practice, however, it is often cleaner and simpler to implement the version which keeps
all the parameters ![(\theta_1, \theta_2,\ldots, \theta_n)](images/math/1/b/6/1b6575b1c6961fafeef894594bfc69ec.png), without
arbitrarily setting one of them to zero. But we will
make one change to the cost function: Adding weight decay. This will take care of
the numerical problems associated with softmax regression's overparameterized representation.

  Weight Decay
--------------

We will modify the cost function by adding a weight decay term 
![\textstyle \frac{\lambda}{2} \sum_{i=1}^k \sum_{j=0}^{n} \theta_{ij}^2](images/math/7/6/6/766dfa7931741fa72672b9093205a850.png)
which penalizes large values of the parameters. Our cost function is now

![
\begin{align}
J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{k} 1\left\{y^{(i)} = j\right\} \log \frac{e^{\theta_j^T x^{(i)}}}{\sum_{l=1}^k e^{ \theta_l^T x^{(i)} }}  \right]
              + \frac{\lambda}{2} \sum_{i=1}^k \sum_{j=0}^n \theta_{ij}^2
\end{align}
](images/math/4/7/1/471592d82c7f51526bb3876c6b0f868d.png)

With this weight decay term (for any λ > 0), the cost function
*J*(θ) is now strictly convex, and is guaranteed to have a
unique solution. The Hessian is now invertible, and because *J*(θ) is 
convex, algorithms such as gradient descent, L-BFGS, etc. are guaranteed
to converge to the global minimum.

To apply an optimization algorithm, we also need the derivative of this
new definition of *J*(θ). One can show that the derivative is:
![
\begin{align}
\nabla_{\theta_j} J(\theta) = - \frac{1}{m} \sum_{i=1}^{m}{ \left[ x^{(i)} ( 1\{ y^{(i)} = j\}  - p(y^{(i)} = j | x^{(i)}; \theta) ) \right]  } + \lambda \theta_j
\end{align}
](images/math/3/a/f/3afb4b9181a3063ddc639099bc919197.png)

By minimizing *J*(θ) with respect to θ, we will have a working implementation of softmax regression.

  Relationship to Logistic Regression
-------------------------------------

In the special case where *k* = 2, one can show that softmax regression reduces to logistic regression.
This shows that softmax regression is a generalization of logistic regression. Concretely, when *k* = 2,
the softmax regression hypothesis outputs

![
\begin{align}
h_\theta(x) &=

\frac{1}{ e^{\theta_1^Tx}  + e^{ \theta_2^T x^{(i)} } }
\begin{bmatrix}
e^{ \theta_1^T x } \\
e^{ \theta_2^T x }
\end{bmatrix}
\end{align}
](images/math/e/3/2/e32efab7bff7353e04775b030af0dae9.png)

Taking advantage of the fact that this hypothesis
is overparameterized and setting ψ = θ1,
we can subtract θ1 from each of the two parameters, giving us

![
\begin{align}
h(x) &=

\frac{1}{ e^{\vec{0}^Tx}  + e^{ (\theta_2-\theta_1)^T x^{(i)} } }
\begin{bmatrix}
e^{ \vec{0}^T x } \\
e^{ (\theta_2-\theta_1)^T x }
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1 + e^{ (\theta_2-\theta_1)^T x^{(i)} } } \\
\frac{e^{ (\theta_2-\theta_1)^T x }}{ 1 + e^{ (\theta_2-\theta_1)^T x^{(i)} } }
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1  + e^{ (\theta_2-\theta_1)^T x^{(i)} } } \\
1 - \frac{1}{ 1  + e^{ (\theta_2-\theta_1)^T x^{(i)} } } \\
\end{bmatrix}
\end{align}
](images/math/b/8/1/b81d6e553283fadddbe29fe55226fb38.png)

Thus, replacing θ2 − θ1 with a single parameter vector θ', we find
that softmax regression predicts the probability of one of the classes as
![\frac{1}{ 1  + e^{ (\theta')^T x^{(i)} } }](images/math/4/0/2/4020e759c483c03f53e3f62695e31314.png),
and that of the other class as
![1 - \frac{1}{ 1 + e^{ (\theta')^T x^{(i)} } }](images/math/4/8/4/4849be3e71ce598369b419dc0b73633e.png),
same as logistic regression.

  Softmax Regression vs. k Binary Classifiers
---------------------------------------------

Suppose you are working on a music classification application, and there are
*k* types of music that you are trying to recognize. Should you use a
softmax classifier, or should you build *k* separate binary classifiers using
logistic regression?

This will depend on whether the four classes are *mutually exclusive.* For example,
if your four classes are classical, country, rock, and jazz, then assuming each
of your training examples is labeled with exactly one of these four class labels,
you should build a softmax classifier with *k* = 4.
(If there're also some examples that are none of the above four classes,
then you can set *k* = 5 in softmax regression, and also have a fifth, "none of the above," class.)

If however your categories are has\_vocals, dance, soundtrack, pop, then the
classes are not mutually exclusive; for example, there can be a piece of pop
music that comes from a soundtrack and in addition has vocals. In this case, it
would be more appropriate to build 4 binary logistic regression classifiers. 
This way, for each new musical piece, your algorithm can separately decide whether
it falls into each of the four categories.

Now, consider a computer vision example, where you're trying to classify images into
three different classes. (i) Suppose that your classes are indoor\_scene,
outdoor\_urban\_scene, and outdoor\_wilderness\_scene. Would you use sofmax regression
or three logistic regression classifiers? (ii) Now suppose your classes are
indoor\_scene, black\_and\_white\_image, and image\_has\_people. Would you use softmax
regression or multiple logistic regression classifiers?

In the first case, the classes are mutually exclusive, so a softmax regression
classifier would be appropriate. In the second case, it would be more appropriate to build
three separate logistic regression classifiers.

---

**Softmax Regression** | [Exercise:Softmax Regression](Exercise_Softmax_Regression.md "Exercise:Softmax Regression")

---

> * Language: [中文](Softmax%E5%9B%9E%E5%BD%92.md "Softmax回归")
> * This page was last modified on 7 April 2013, at 13:24.

