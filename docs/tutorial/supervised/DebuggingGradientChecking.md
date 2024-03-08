

### Debugging: Gradient Checking

---

So far we have worked with relatively simple algorithms where it is straight-forward to compute the objective function and its gradient with pen-and-paper, and then implement the necessary computations in MATLAB. For more complex models that we will see later (like the back-propagation method for neural networks), the gradient computation can be notoriously difficult to debug and get right. Sometimes a subtly buggy implementation will manage to learn something that can look surprisingly reasonable (while performing less well than a correct implementation). Thus, even with a buggy implementation, it may not at all be apparent that anything is amiss. In this section, we describe a method for numerically checking the derivatives computed by your code to make sure that your implementation is correct. Carrying out the derivative checking procedure described here will significantly increase your confidence in the correctness of your code.

Suppose we want to minimize $\textstyle J(\theta)$ as a function of $\textstyle \theta$. For this example, suppose $\textstyle J : \Re \mapsto \Re$, so that $\textstyle \theta \in \Re$. If we are using `minFunc` or some other optimization algorithm, then we usually have implemented some function $\textstyle g(\theta)$ that purportedly computes $\textstyle \frac{d}{d\theta}J(\theta)$.

How can we check if our implementation of $\textstyle g$ is correct?

Recall the mathematical definition of the derivative as:

$$
\begin{align}
\frac{d}{d\theta}J(\theta) = \lim_{\epsilon \rightarrow 0}
\frac{J(\theta+ \epsilon) - J(\theta-\epsilon)}{2 \epsilon}.
\end{align}
$$

Thus, at any specific value of $\textstyle \theta$, we can numerically approximate the derivative as follows:

$$
\begin{align}
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}
\end{align}
$$

In practice, we set ${\rm EPSILON}$ to a small constant, say around $\textstyle 10^{-4}$. (There’s a large range of values of ${\rm EPSILON}$ that should work well, but we don’t set ${\rm EPSILON}$ to be “extremely” small, say $\textstyle 10^{-20}$, as that would lead to numerical roundoff errors.)

Thus, given a function $\textstyle g(\theta)$ that is supposedly computing $\textstyle \frac{d}{d\theta}J(\theta)$, we can now numerically verify its correctness by checking that

$$
\begin{align}
g(\theta) \approx
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}.
\end{align}
$$

The degree to which these two values should approximate each other will depend on the details of $\textstyle J$. But assuming $\textstyle {\rm EPSILON} = 10^{-4}$, you’ll usually find that the left- and right-hand sides of the above will agree to at least 4 significant digits (and often many more).

Now, consider the case where $\textstyle \theta \in \Re^n$ is a vector rather than a single real number (so that we have $\textstyle n$ parameters that we want to learn), and $\textstyle J: \Re^n \mapsto \Re$. We now generalize our derivative checking procedure to the case where $\textstyle \theta$ may be a vector (as in our linear regression and logistic regression examples). If ever we are optimizing over several variables or over matrices, we can always pack these parameters into a long vector and use the same method here to check our derivatives. (This will often need to be done anyway if you want to use off-the-shelf optimization packages.)

Suppose we have a function $\textstyle g_i(\theta)$ that purportedly computes $\textstyle \frac{\partial}{\partial \theta_i} J(\theta)$; we’d like to check if $\textstyle g_i$ is outputting correct derivative values. Let $\textstyle \theta^{(i+)} = \theta + {\rm EPSILON} \times \vec{e}_i$, where

$$
\begin{align}
\vec{e}_i = \begin{bmatrix}0 \\ 0 \\ \vdots \\ 1 \\ \vdots \\ 0\end{bmatrix}
\end{align}
$$

is the $\textstyle i$-th basis vector (a vector of the same dimension as $\textstyle \theta$, with a “1” in the $\textstyle i$-th position and “0”s everywhere else). So, $\textstyle \theta^{(i+)}$ is the same as $\textstyle \theta$, except its $\textstyle i$-th element has been incremented by ${\rm EPSILON}$. Similarly, let $\textstyle \theta^{(i-)} = \theta - {\rm EPSILON} \times \vec{e}_i$ be the corresponding vector with the $\textstyle i$-th element decreased by ${\rm EPSILON}$.

We can now numerically verify $\textstyle g_i(\theta)$’s correctness by checking, for each $\textstyle i$, that:

$$
\begin{align}
g_i(\theta) \approx
\frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2 \times {\rm EPSILON}}.
\end{align}
$$

### Gradient checker code

As an exercise, try implementing the above method to check the gradient of your linear regression and logistic regression functions. Alternatively, you can use the provided `ex1/grad_check.m` file (which takes arguments similar to `minFunc`) and will check $\frac{\partial J(\theta)}{\partial \theta_i}$ for many random choices of $i$.
