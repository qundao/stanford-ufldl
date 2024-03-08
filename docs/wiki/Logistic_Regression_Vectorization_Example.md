Logistic Regression Vectorization Example
=========================================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
Consider training a logistic regression model using batch gradient ascent.
Suppose our hypothesis is

![\begin{align}
h_\theta(x) = \frac{1}{1+\exp(-\theta^Tx)},
\end{align}](images/math/b/b/3/bb3791d463b832a88731b94f1d8e5279.png)

where (following the notational convention from the OpenClassroom videos and from CS229) we let ![\textstyle x_0=1](images/math/c/5/8/c582053ce9cb63d69ae80acb53ded0d3.png), so that ![\textstyle x \in \Re^{n+1}](images/math/e/c/2/ec2c09e7951c093d21db55d95ffaa19e.png) 
and ![\textstyle \theta \in \Re^{n+1}](images/math/8/c/d/8cd47b42536a589ad69927f408921808.png), and ![\textstyle \theta_0](images/math/f/6/0/f6040edfd55be75383ff6ae2badc24f8.png) is our intercept term. We have a training set
![\textstyle \{(x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})\}](images/math/b/4/4/b449e6d375809abbc4097d2c55e9f8c0.png) of ![\textstyle m](images/math/2/5/e/25e97e8a905fc2cb05d76cd4872a8567.png) examples, and the batch gradient
ascent update rule is ![\textstyle \theta := \theta + \alpha \nabla_\theta \ell(\theta)](images/math/6/5/a/65a9cda07ee61ef59b4167897d4c5634.png), where ![\textstyle \ell(\theta)](images/math/a/1/f/a1fa0c7d5e58ae87f3231b8e381cf433.png)
is the log likelihood and ![\textstyle \nabla_\theta \ell(\theta)](images/math/8/b/5/8b52e48e33138f3366afb938605b7944.png) is its derivative.

[Note: Most of the notation below follows that defined in the OpenClassroom videos or in the class 
CS229: Machine Learning. For details, see either the [OpenClassroom videos](http://openclassroom.stanford.edu/MainFolder/CoursePage.php?course=MachineLearning) or Lecture Notes #1 of <http://cs229.stanford.edu/> .]

We thus need to compute the gradient:

![\begin{align}
\nabla_\theta \ell(\theta) = \sum_{i=1}^m \left(y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}_j.
\end{align}](images/math/b/9/e/b9e08cd04d5328fec470b92aa27dc8cc.png)

Suppose that the Matlab/Octave variable x is a matrix containing the training inputs, so that
x(:,i) is the ![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)-th training example ![\textstyle x^{(i)}](images/math/e/b/e/ebe8632b7c91a3dbbf9b590bea887a47.png), and x(j,i) is ![\textstyle x^{(i)}_j](images/math/3/1/8/318866dbcd6dd86e9d402d0f324fb8bd.png). 
Further, suppose the Matlab/Octave variable y is a *row* vector of the labels in the
training set, so that the variable y(i) is ![\textstyle y^{(i)} \in \{0,1\}](images/math/9/a/f/9af78a186bc4feb4ae23853de5556095.png). (Here we differ from the 
OpenClassroom/CS229 notation. Specifically, in the matrix-valued x we stack the training inputs in columns rather than in rows;
and y![\in \Re^{1\times m}](images/math/e/3/2/e32d32b1db225592d968799d331815f4.png) is a row vector rather than a column vector.)

Here's truly horrible, extremely slow, implementation of the gradient computation:

```
% Implementation 1
grad = zeros(n+1,1);
for i=1:m,
  h = sigmoid(theta'*x(:,i));
  temp = y(i) - h; 
  for j=1:n+1,
    grad(j) = grad(j) + temp * x(j,i); 
  end;
end;
```

The two nested for-loops makes this very slow. Here's a more typical implementation,
that partially vectorizes the algorithm and gets better performance:

```
% Implementation 2 
grad = zeros(n+1,1);
for i=1:m,
  grad = grad + (y(i) - sigmoid(theta'*x(:,i)))* x(:,i);
end;
```

However, it turns out to be possible to even further vectorize this. If we can get rid of the for-loop, we can significantly speed up the implementation. In particular, suppose b is a column vector, and A is a matrix. Consider the following ways of computing A \* b:

```
% Slow implementation of matrix-vector multiply
grad = zeros(n+1,1);
for i=1:m,
  grad = grad + b(i) * A(:,i);  % more commonly written A(:,i)*b(i)
end;

% Fast implementation of matrix-vector multiply
grad = A*b;
```

We recognize that Implementation 2 of our gradient descent calculation above is using the slow version with a for-loop, with
b(i) playing the role of (y(i) - sigmoid(theta'\*x(:,i))), and A playing the role of x. We can derive a fast implementation as follows:

```
% Implementation 3
grad = x * (y- sigmoid(theta'*x))';
```

Here, we assume that the Matlab/Octave sigmoid(z) takes as input a vector z, applies the sigmoid function component-wise to the input, and returns the result. The output of sigmoid(z) is therefore itself also a vector, of the same dimension as the input z

When the training set is large, this final implementation takes the greatest advantage of Matlab/Octave's highly optimized numerical linear algebra libraries to carry out the matrix-vector operations, and so this is far more efficient than the earlier implementations.

Coming up with vectorized implementations isn't always easy, and sometimes requires careful thought. But as you gain familiarity with vectorized operations, you'll find that there are design patterns (i.e., a small number of ways of vectorizing) that apply to many different pieces of code.

[Vectorization](Vectorization.md "Vectorization") | **Logistic Regression Vectorization Example** | [Neural Network Vectorization](Neural_Network_Vectorization.md "Neural Network Vectorization") | [Exercise:Vectorization](Exercise_Vectorization.md "Exercise:Vectorization")

---

> * Language: [中文](%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E7%9A%84%E5%90%91%E9%87%8F%E5%8C%96%E5%AE%9E%E7%8E%B0%E6%A0%B7%E4%BE%8B.md "逻辑回归的向量化实现样例")
> * This page was last modified on 7 April 2013, at 13:09.

