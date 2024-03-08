Exercise:PCA in 2D
==================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->

|  |
| --- |
| Contents* [1 PCA, PCA whitening and ZCA whitening in 2D](#PCA.2C_PCA_whitening_and_ZCA_whitening_in_2D)
	+ [1.1 Step 0: Load data](#Step_0:_Load_data)
	+ [1.2 Step 1: Implement PCA](#Step_1:_Implement_PCA)
		- [1.2.1 Step 1a: Finding the PCA basis](#Step_1a:_Finding_the_PCA_basis)
		- [1.2.2 Step 1b: Check xRot](#Step_1b:_Check_xRot)
	+ [1.3 Step 2: Dimension reduce and replot](#Step_2:_Dimension_reduce_and_replot)
	+ [1.4 Step 3: PCA Whitening](#Step_3:_PCA_Whitening)
	+ [1.5 Step 4: ZCA Whitening](#Step_4:_ZCA_Whitening)
 |

  PCA, PCA whitening and ZCA whitening in 2D
--------------------------------------------

In this exercise you will implement PCA, PCA whitening and ZCA whitening, as described in the earlier sections of this tutorial, and generate the images shown in the earlier sections yourself. You will build on the starter code that has been provided at [pca\_2d.zip](http://ufldl.stanford.edu/wiki/resources/pca_2d.zip). You need only write code at the places indicated by "YOUR CODE HERE" in the files. The only file you need to modify is pca\_2d.m. Implementing this exercise will make the next exercise significantly easier to understand and complete.

###   Step 0: Load data

The starter code contains code to load 45 2D data points. When plotted using the scatter function, the results should look like the following:

![Raw images](images/thumb/f/f5/Raw_images_2d.png/400px-Raw_images_2d.png)

###   Step 1: Implement PCA

In this step, you will implement PCA to obtain *x**r**o**t*, the matrix in which the data is "rotated" to the basis comprising ![\textstyle u_1, \ldots, u_n](images/math/5/0/2/5028e168451f819195c63d9572f0233f.png) made up of the principal components. As mentioned in the implementation notes, you should make use of MATLAB's svd function here.

####   Step 1a: Finding the PCA basis

Find ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) and ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png), and draw two lines in your figure to show the resulting basis on top of the given data points. You may find it useful to use MATLAB's hold on and hold off functions. (After calling hold on, plotting functions such as plot will draw the new data on top of the previously existing figure rather than erasing and replacing it; and hold off turns this off.) You can use plot([x1,x2], [y1,y2], '-') to draw a line between (x1,y1) and (x2,y2). Your figure should look like this:

![Pca 2d basis.png](images/thumb/5/5b/Pca_2d_basis.png/400px-Pca_2d_basis.png)

If you are doing this in Matlab, you will probably get a plot that's identical to ours. However, eigenvectors are defined only up to a sign. I.e., instead of returning ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) as the first eigenvector, Matlab/Octave could just as easily have returned ![\textstyle -u_1](images/math/b/1/0/b10929071429909f3d52ebe5cd18a664.png), and similarly instead of ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png) Matlab/Octave could have returned ![\textstyle -u_2](images/math/f/b/0/fb04e59ea0095b98f06d254747837398.png). So if you wound up with one or both of the eigenvectors pointing in a direction opposite (180 degrees difference) from what's shown above, that's okay too.

####   Step 1b: Check xRot

Compute xRot, and use the scatter function to check that xRot looks as it should, which should be something like the following:

![Pca xrot 2d.png](images/thumb/0/0b/Pca_xrot_2d.png/360px-Pca_xrot_2d.png)

Because Matlab/Octave could have returned ![\textstyle -u_1](images/math/b/1/0/b10929071429909f3d52ebe5cd18a664.png) and/or ![\textstyle -u_2](images/math/f/b/0/fb04e59ea0095b98f06d254747837398.png) instead of ![\textstyle u_1](images/math/3/f/c/3fc01c8dc5d4c8c57cd758ec3a76283f.png) and ![\textstyle u_2](images/math/e/d/9/ed99a7fbd444e14555ad4f8eac78b94b.png), it's also possible that you might have gotten a figure which is "flipped" or "reflected" along the ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png)- and/or ![\textstyle y](images/math/c/8/1/c81e76c28ed991b22b8c1bb8fa392701.png)-axis; a flipped/reflected version of this figure is also a completely correct result.

###   Step 2: Dimension reduce and replot

In the next step, set *k*, the number of components to retain, to be 1 (we have already done this for you). Compute the resulting xHat and plot the results. You should get the following (this figure should **not** be flipped along the ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png)- or ![\textstyle y](images/math/c/8/1/c81e76c28ed991b22b8c1bb8fa392701.png)-axis):

![Pca xhat 2d.png](images/thumb/b/bb/Pca_xhat_2d.png/400px-Pca_xhat_2d.png)

###   Step 3: PCA Whitening

Implement PCA whitening using the formula from the notes. Plot xPCAWhite, and verify that it looks like the following (a figure that is flipped/reflected on either/both axes is also correct):

![Pca white 2d.png](images/thumb/c/c9/Pca_white_2d.png/400px-Pca_white_2d.png)

###   Step 4: ZCA Whitening

Implement ZCA whitening and plot the results. The results should look like the following (this should not be flipped/reflected along the ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png)- or ![\textstyle y](images/math/c/8/1/c81e76c28ed991b22b8c1bb8fa392701.png)-axis):

![Zca white 2d.png](images/thumb/9/9b/Zca_white_2d.png/400px-Zca_white_2d.png)

[PCA](PCA.md "PCA") | [Whitening](Whitening.md "Whitening") | [Implementing PCA/Whitening](/wayback-mooc/stanford-ufldl/wiki/Implementing_PCA/Whitening "Implementing PCA/Whitening") | **Exercise:PCA in 2D** | [Exercise:PCA and Whitening](Exercise_PCA_and_Whitening.md "Exercise:PCA and Whitening")

 Category: Exercises
> * This page was last modified on 26 May 2011, at 11:01.

