Sparse Coding: Autoencoder Interpretation
=========================================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->

|  |
| --- |
| Contents* [1 Sparse coding](#Sparse_coding)
* [2 Topographic sparse coding](#Topographic_sparse_coding)
* [3 Sparse coding in practice](#Sparse_coding_in_practice)
	+ [3.1 Batching examples into mini-batches](#Batching_examples_into_mini-batches)
	+ [3.2 Good initialization of s](#Good_initialization_of_s)
	+ [3.3 The practical algorithm](#The_practical_algorithm)
 |

  Sparse coding
---------------

In the sparse autoencoder, we tried to learn a set of weights *W* (and associated biases *b*) that would give us sparse features σ(*W**x* + *b*) useful in reconstructing the input *x*.

![STL SparseAE.png](images/thumb/f/ff/STL_SparseAE.png/240px-STL_SparseAE.png)

Sparse coding can be seen as a modification of the sparse autoencoder method in which we try to learn the set of features for some data "directly". Together with an associated basis for transforming the learned features from the feature space to the data space, we can then reconstruct the data from the learned features.

Formally, in sparse coding, we have some data *x* we would like to learn features on. In particular, we would like to learn *s*, a set of sparse features useful for representing the data, and *A*, a basis for transforming the features from the feature space to the data space. Our objective function is hence:

![
J(A, s) = \lVert As - x \rVert_2^2 + \lambda \lVert s \rVert_1
](images/math/2/0/7/2072898a1e2ee735eb51f8b527f21e2e.png)

(If you are unfamiliar with the notation, ![\lVert x \rVert_k](images/math/0/5/1/05140fa4a71b91000681ae96011488e9.png) refers to the L*k* norm of the *x* which is equal to ![\left( \sum{ \left| x_i^k \right| } \right) ^{\frac{1}{k}}](images/math/6/7/6/67605df92ae8c43173bbb80f7a93cb83.png). The L2 norm is the familiar Euclidean norm, while the L1 norm is the sum of absolute values of the elements of the vector)

The first term is the error in reconstructing the data from the features using the basis, and the second term is a sparsity penalty term to encourage the learned features to be sparse.

However, the objective function as it stands is not properly constrained - it is possible to reduce the sparsity cost (the second term) by scaling *A* by some constant and scaling *s* by the inverse of the same constant, without changing the error. Hence, we include the additional constraint that that for every column *A**j* of *A*, 
![A_j^TA_j \le 1](images/math/e/e/0/ee05eff183594aed415392b8104bfb1d.png). Our problem is thus:

![
\begin{array}{rcl}
     {\rm minimize} & \lVert As - x \rVert_2^2 + \lambda \lVert s \rVert_1 \\
     {\rm s.t.}     &    A_j^TA_j \le 1 \; \forall j \\
\end{array} 
](images/math/a/2/f/a2f57c5746669d09790f9d862352c89b.png)

Unfortunately, the objective function is non-convex, and hence impossible to optimize well using gradient-based methods. However, given *A*, the problem of finding *s* that minimizes *J*(*A*,*s*) is convex. Similarly, given *s*, the problem of finding *A* that minimizes *J*(*A*,*s*) is also convex. This suggests that we might try alternately optimizing for *A* for a fixed *s*, and then optimizing for *s* given a fixed *A*. It turns out that this works quite well in practice.

However, the form of our problem presents another difficulty - the constraint that ![A_j^TA_j \le 1 \; \forall j](images/math/4/c/1/4c19ae5304ebe923a3053ea8efbc7622.png) cannot be enforced using simple gradient-based methods. Hence, in practice, this constraint is weakened to a "weight decay" term designed to keep the entries of *A* small. This gives us a new objective function:

![
J(A, s) = \lVert As - x \rVert_2^2 + \lambda \lVert s \rVert_1 + \gamma \lVert A \rVert_2^2
](images/math/1/d/6/1d6a2cef1550cd6830cc45e56d120dd5.png)

(note that the third term, ![\lVert A \rVert_2^2](images/math/e/0/5/e05f2e1c0e4f84b54964b13b6d1aafe1.png) is simply the sum of squares of the entries of A, or ![\sum_r{\sum_c{A_{rc}^2}}](images/math/2/b/e/2be909f8e140d9bae7a5b5f2be0ed26c.png))

This objective function presents one last problem - the L1 norm is not differentiable at 0, and hence poses a problem for gradient-based methods. While the problem can be solved using other non-gradient descent-based methods, we will "smooth out" the L1 norm using an approximation which will allow us to use gradient descent. To "smooth out" the L1 norm, we use ![\sqrt{x^2 + \epsilon}](images/math/d/d/7/dd7d0966210455f769c5ed37c206c606.png) in place of ![\left| x \right|](images/math/6/a/3/6a37fe2d78bd89637e639ae2f90c1a1b.png), where ε is a "smoothing parameter" which can also be interpreted as a sort of "sparsity parameter" (to see this, observe that when ε is large compared to *x*, the *x* + ε is dominated by ε, and taking the square root yields approximately ![\sqrt{\epsilon}](images/math/a/b/6/ab6e222a1176d32e0a9ead3c70c69b02.png)). This "smoothing" will come in handy later when considering topographic sparse coding below.

Our final objective function is hence:

![
J(A, s) = \lVert As - x \rVert_2^2 + \lambda \sqrt{s^2 + \epsilon} + \gamma \lVert A \rVert_2^2
](images/math/f/5/a/f5a161dfa55cbc70b160e1e224134949.png)

(where ![\sqrt{s^2 + \epsilon}](images/math/7/2/3/72354bd3dca1a2904b08730bc124fd8a.png) is shorthand for ![\sum_k{\sqrt{s_k^2 + \epsilon}}](images/math/5/7/f/57fb2d3245ec566af9ec7c9de4b4f172.png))

This objective function can then be optimized iteratively, using the following procedure:

1. Initialize *A* randomly
- Repeat until convergence
	1. Find the *s* that minimizes *J*(*A*,*s*) for the *A* found in the previous step
	 - Solve for the *A* that minimizes *J*(*A*,*s*) for the *s* found in the previous step

Observe that with our modified objective function, the objective function *J*(*A*,*s*) given *s*, that is ![J(A; s) = \lVert As - x \rVert_2^2 + \gamma \lVert A \rVert_2^2](images/math/5/7/a/57a5b22ceffa2fbfcc2ca86bdc9372bd.png) (the L1 term in *s* can be omitted since it is not a function of *A*) is simply a quadratic term in *A*, and hence has an easily derivable analytic solution in *A*. A quick way to derive this solution would be to use matrix calculus - some pages about matrix calculus can be found in the  [useful links](Useful_Links.md "Useful Links") section. Unfortunately, the objective function given *A* does not have a similarly nice analytic solution, so that minimization step will have to be carried out using gradient descent or similar optimization methods.

In theory, optimizing for this objective function using the iterative method as above should (eventually) yield features (the basis vectors of *A*) similar to those learned using the sparse autoencoder. However, in practice, there are quite a few tricks required for better convergence of the algorithm, and these tricks are described in greater detail in the later section on  [sparse coding in practice](Sparse_Coding__Autoencoder_Interpretation#Sparse_coding_in_practice.md "Sparse Coding: Autoencoder Interpretation"). Deriving the gradients for the objective function may be slightly tricky as well, and using matrix calculus or  [using the backpropagation intuition](Deriving_gradients_using_the_backpropagation_idea.md "Deriving gradients using the backpropagation idea") can be helpful.

  Topographic sparse coding
---------------------------

With sparse coding, we can learn a set of features useful for representing the data. However, drawing inspiration from the brain, we would like to learn a set of features that are "orderly" in some manner. For instance, consider visual features. As suggested earlier, the V1 cortex of the brain contains neurons which detect edges at particular orientations. However, these neurons are also organized into hypercolumns in which adjacent neurons detect edges at similar orientations. One neuron could detect a horizontal edge, its neighbors edges oriented slightly off the horizontal, and moving further along the hypercolumn, the neurons detect edges oriented further off the horizontal.

Inspired by this example, we would like to learn features which are similarly "topographically ordered". What does this imply for our learned features? Intuitively, if "adjacent" features are "similar", we would expect that if one feature is activated, its neighbors will also be activated to a lesser extent.

Concretely, suppose we (arbitrarily) organized our features into a square matrix. We would then like adjacent features in the matrix to be similar. The way this is accomplished is to group these adjacent features together in the smoothed L1 penalty, so that instead of say ![\sqrt{s_{1,1}^2 + \epsilon}](images/math/3/3/9/3391a34bac754562a6e2d881627d324e.png), we use say ![\sqrt{s_{1,1}^2 + s_{1,2}^2 + s_{1,3}^2 + s_{2,1}^2 + s_{2,2}^2 + s_{3,2}^2 + s_{3,1}^2 + s_{3,2}^2 + s_{3,3}^2 + \epsilon}](images/math/0/d/9/0d9a543116996f237dcab61e5c78cbee.png) instead, if we group in 3x3 regions. The grouping is usually overlapping, so that the 3x3 region starting at the 1st row and 1st column is one group, the 3x3 region starting at the 1st row and 2nd column is another group, and so on. Further, the grouping is also usually done wrapping around, as if the matrix were a torus, so that every feature is counted an equal number of times.

Hence, in place of the smoothed L1 penalty, we use the sum of smoothed L1 penalties over all the groups, so our new objective function is:

![
J(A, s) = \lVert As - x \rVert_2^2 + \lambda \sum_{\text{all groups } g}{\sqrt{ \left( \sum_{\text{all } s \in g}{s^2} \right) + \epsilon} } + \gamma \lVert A \rVert_2^2
](images/math/2/5/b/25b426de9a1b46c94839f5f9dd4801a3.png)

In practice, the "grouping" can be accomplished using a "grouping matrix" *V*, such that the *r*th row of *V* indicates which features are grouped in the *r*th group, so *V**r*,*c* = 1 if group *r* contains feature *c*. Thinking of the grouping as being achieved by a grouping matrix makes the computation of the gradients more intuitive. Using this grouping matrix, the objective function can be rewritten as:

![
J(A, s) = \lVert As - x \rVert_2^2 + \lambda \sum{ \sqrt{Vss^T + \epsilon} } + \gamma \lVert A \rVert_2^2
](images/math/c/2/3/c23bf21d67df11fdbd7cc4ae9dc41c64.png)

(where ![\sum{ \sqrt{Vss^T + \epsilon} }](images/math/c/c/d/ccd5a0f991db6bdba852b147ee42d91b.png) is

|  |  |  |
| --- | --- | --- |
| ∑ | ∑ | *D**r*,*c* |
| *r* | *c* |  |

 if we let ![D = \sqrt{Vss^T + \epsilon}](images/math/9/8/4/9845dba65c5e5ff49ea4c134dc2c1bf0.png))

This objective function can be optimized using the iterated method described in the earlier section. Topographic sparse coding will learn features similar to those learned by sparse coding, except that the features will now be "ordered" in some way.

  Sparse coding in practice
---------------------------

As suggested in the earlier sections, while the theory behind sparse coding is quite simple, writing a good implementation that actually works and converges reasonably quickly to good optima requires a bit of finesse.

Recall the simple iterative algorithm proposed earlier:

1. Initialize *A* randomly
- Repeat until convergence
	1. Find the *s* that minimizes *J*(*A*,*s*) for the *A* found in the previous step
	 - Solve for the *A* that minimizes *J*(*A*,*s*) for the *s* found in the previous step

It turns out that running this algorithm out of the box will not produce very good results, if any results are produced at all. There are two main tricks to achieve faster and better convergence:

1. Batching examples into "mini-batches"
- Good initialization of *s*

###   Batching examples into mini-batches

If you try running the simple iterative algorithm on a large dataset of say 10 000 patches at one go, you will find that each iteration takes a long time, and the algorithm may hence take a long time to converge. To increase the rate of convergence, you can instead run the algorithm on mini-batches instead. To do this, instead of running the algorithm on all 10 000 patches, in each iteration, select a mini-batch - a (different) random subset of say 2000 patches from the 10 000 patches - and run the algorithm on that mini-batch for the iteration instead. This accomplishes two things - firstly, it speeds up each iteration, since now each iteration is operating on 2000 rather than 10 000 patches; secondly, and more importantly, it increases the rate of convergence (TODO: explain why).

###   Good initialization of *s*

Another important trick in obtaining faster and better convergence is good initialization of the feature matrix *s* before using gradient descent (or other methods) to optimize for the objective function for *s* given *A*. In practice, initializing *s* randomly at each iteration can result in poor convergence unless a good optima is found for *s* before moving on to optimize for *A*. A better way to initialize *s* is the following:

1. Set ![s \leftarrow W^Tx](images/math/f/0/b/f0b36b91f5e791ff8a59c1216da9af2d.png) (where *x* is the matrix of patches in the mini-batch)
- For each feature in *s* (i.e. each column of *s*), divide the feature by the norm of the corresponding basis vector in *A*. That is, if *s**r*,*c* is the *r*th feature for the *c*th example, and *A**c* is the *c*th basis vector in *A*, then set ![s_{r, c} \leftarrow \frac{ s_{r, c} } { \lVert A_c \rVert }.](images/math/2/0/7/20773e6ff4a4a9d48b6c3769e7b50780.png)

Very roughly and informally speaking, this initialization helps because the first step is an attempt to find a good *s* such that ![Ws \approx x](images/math/8/4/0/84002fcaba86b0ad04772d33a6aa556d.png), and the second step "normalizes" *s* in an attempt to keep the sparsity penalty small. It turns out that initializing *s* using only one but not both steps results in poor performance in practice. (TODO: a better explanation for why this initialization helps?)

###   The practical algorithm

With the above two tricks, the algorithm for sparse coding then becomes:

1. Initialize *A* randomly
- Repeat until convergence
	1. Select a random mini-batch of 2000 patches
	 - Initialize *s* as described above
	 - Find the *s* that minimizes *J*(*A*,*s*) for the *A* found in the previous step
	 - Solve for the *A* that minimizes *J*(*A*,*s*) for the *s* found in the previous step

With this method, you should be able to reach a good local optima relatively quickly.

---

> * Language: [中文](%E7%A8%80%E7%96%8F%E7%BC%96%E7%A0%81%E8%87%AA%E7%BC%96%E7%A0%81%E8%A1%A8%E8%BE%BE.md "稀疏编码自编码表达")
> * This page was last modified on 19 April 2013, at 02:49.

