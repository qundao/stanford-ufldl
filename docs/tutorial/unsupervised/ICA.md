

### ICA

---

### Introduction

If you recall, in [Sparse Coding](/wayback-mooc/stanford-ufldl/tutorial/unsupervised/SparseCoding), we wanted to learn an **over-complete** basis for the data. In particular, this implies that the basis vectors that we learn in sparse coding will not be linearly independent. While this may be desirable in certain situations, sometimes we want to learn a linearly independent basis for the data. In independent component analysis (ICA), this is exactly what we want to do. Further, in ICA, we want to learn not just any linearly independent basis, but an **orthonormal** basis for the data. (An orthonormal basis is a basis $(\phi_1, \ldots \phi_n)$ such that $\phi_i \cdot \phi_j = 0$ if $i \ne j$ and $1$ if $i = j$).

Like sparse coding, independent component analysis has a simple mathematical formulation. Given some data $x$, we would like to learn a set of basis vectors which we represent in the columns of a matrix $W$, such that, firstly, as in sparse coding, our features are **sparse**; and secondly, our basis is an **orthonormal** basis. (Note that while in sparse coding, our matrix $A$ was for mapping **features** $s$ to **raw data**, in independent component analysis, our matrix $W$ works in the opposite direction, mapping **raw data** $x$ to **features** instead). This gives us the following objective function:

$$
J(W) = \lVert Wx \rVert_1 
$$

This objective function is equivalent to the sparsity penalty on the features $s$ in sparse coding, since $Wx$ is precisely the features that represent the data. Adding in the orthonormality constraint gives us the full optimization problem for independent component analysis:

$$
\begin{array}{rcl}
{\rm minimize} & \lVert Wx \rVert_1 \\
{\rm s.t.} & WW^T = I \\
\end{array}
$$

As is usually the case in deep learning, this problem has no simple analytic solution, and to make matters worse, the orthonormality constraint makes it slightly more difficult to optimize for the objective using gradient descent - every iteration of gradient descent must be followed by a step that maps the new basis back to the space of orthonormal bases (hence enforcing the constraint).

In practice, optimizing for the objective function while enforcing the orthonormality constraint (as described in the section below) is feasible but slow. Hence, the use of orthonormal ICA is limited to situations where it is important to obtain an orthonormal basis.

### Orthonormal ICA

The orthonormal ICA objective is:

$$
\begin{array}{rcl}
{\rm minimize} & \lVert Wx \rVert_1 \\
{\rm s.t.} & WW^T = I
\end{array}
$$

Observe that the constraint $WW^T = I$ implies two other constraints.

Firstly, since we are learning an orthonormal basis, the number of basis vectors we learn must be less than the dimension of the input. In particular, this means that we cannot learn over-complete bases as we usually do in [[Sparse Coding: Autoencoder Interpretation | sparse coding]].

Secondly, the data must be [ZCA whitened](/wayback-mooc/stanford-ufldl/tutorial/unsupervised/PCAWhitening) with no regularization (that is, with $\epsilon$ set to 0).

Hence, before we even begin to optimize for the orthonormal ICA objective, we must ensure that our data has been **whitened**, and that we are learning an **under-complete** basis.

Following that, to optimize for the objective, we can use gradient descent, interspersing gradient descent steps with projection steps to enforce the orthonormality constraint. Hence, the procedure will be as follows:

Repeat until done:

1. $W \leftarrow W - \alpha \nabla_W \lVert Wx \rVert_1$
2. $W \leftarrow \operatorname{proj}_U W$ where $U$ is the space of matrices satisfying $WW^T = I$

In practice, the learning rate $\alpha$ is varied using a line-search algorithm to speed up the descent, and the projection step is achieved by setting $W \leftarrow (WW^T)^{-\frac{1}{2}} W$, which can actually be seen as ZCA whitening (`TODO`: explain how it is like ZCA whitening).
