Self-Taught Learning
====================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->

|  |
| --- |
| Contents* [1 Overview](#Overview)
* [2 Learning features](#Learning_features)
* [3 On pre-processing the data](#On_pre-processing_the_data)
* [4 On the terminology of unsupervised feature learning](#On_the_terminology_of_unsupervised_feature_learning)
 |

  Overview
----------

Assuming that we have a sufficiently powerful learning algorithm, one of the most reliable 
ways to get better performance is to give the algorithm more data. This has led to the 
that aphorism that in
machine learning, "sometimes it's not who has the best algorithm that wins; it's 
who has the most data."

One can always try to get more labeled data, but this can be expensive. In
particular, researchers have already gone to extraordinary lengths to use tools
such as AMT (Amazon Mechanical Turk) to get large training sets. While having
large numbers of people hand-label lots of data is probably a step forward
compared to having large numbers of researchers hand-engineer features, it
would be nice to do better. In particular, the promise of **self-taught learning**
and **unsupervised feature learning** is that if we can get our algorithms to learn
from *unlabeled* data, then we can easily obtain and learn from massive
amounts of it. Even though a single unlabeled example is less informative than
a single labeled example, if we can get tons of the former---for example, by downloading
random unlabeled images/audio clips/text documents off the
internet---and if our algorithms can exploit this unlabeled data effectively,
then we might be able to achieve better performance than the massive
hand-engineering and massive hand-labeling approaches.

In Self-taught learning and Unsupervised feature learning, we will give our
algorithms a large amount of unlabeled data with which to learn a good feature
representation of the input. If we are trying to solve a specific
classification task, then we take this learned feature representation and
whatever (perhaps small amount of) labeled data we have for that classification task, and apply
supervised learning on that labeled data to solve the classification task.

These ideas probably have the most powerful effects in problems where we have a lot of
unlabeled data, and a smaller amount of labeled data. However,
they typically give good results even if we have only
labeled data (in which case we usually perform the feature learning step using
the labeled data, but ignoring the labels).

  Learning features
-------------------

We have already seen how an autoencoder can be used to learn features from
unlabeled data. Concretely, suppose we have an unlabeled
training set ![\textstyle \{ x_u^{(1)}, x_u^{(2)}, \ldots, x_u^{(m_u)}\}](images/math/3/a/3/3a330b29fcaa7c4fd1df8fcd4d19df92.png) 
with ![\textstyle m_u](images/math/7/c/7/7c7c3de05576d2f93d3522a032e2e9c4.png) unlabeled examples. (The subscript "u" stands for
"unlabeled.") We can then train a sparse autoencoder on this data 
(perhaps with appropriate whitening or other pre-processing):

![STL SparseAE.png](images/thumb/f/ff/STL_SparseAE.png/350px-STL_SparseAE.png)

Having trained the parameters ![\textstyle W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}](images/math/1/e/f/1efd10775b8d8b8dc59b9590661f3a2f.png) of this model,
given any new input ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png), we can now compute the corresponding vector of
activations ![\textstyle a](images/math/c/4/6/c469e9ab9efb42a55f860d809731dc77.png) of the hidden units. As we saw previously, this often gives a
better representation of the input than the original raw input ![\textstyle x](images/math/f/6/c/f6c0f8758a1eb9c99c0bbe309ff2c5a5.png). We can also
visualize the algorithm for computing the features/activations ![\textstyle a](images/math/c/4/6/c469e9ab9efb42a55f860d809731dc77.png) as the following
neural network:

![STL SparseAE Features.png](images/thumb/7/73/STL_SparseAE_Features.png/300px-STL_SparseAE_Features.png)

This is just the sparse autoencoder that we previously had, with with the final
layer removed.

Now, suppose we have a labeled training set ![\textstyle \{ (x_l^{(1)}, y^{(1)}),
(x_l^{(2)}, y^{(2)}), \ldots (x_l^{(m_l)}, y^{(m_l)}) \}](images/math/2/3/c/23cfc7b001a6a6b7a8df45a39d7ce812.png) of ![\textstyle m_l](images/math/6/c/2/6c270d29d4e7e24f2c756df33d564646.png) examples. 
(The subscript "l" stands for "labeled.") 
We can now find a better representation for the inputs. In particular, rather
than representing the first training example as ![\textstyle x_l^{(1)}](images/math/c/b/2/cb267edf5cbce3c54f09c7b975173d17.png), we can feed
![\textstyle x_l^{(1)}](images/math/c/b/2/cb267edf5cbce3c54f09c7b975173d17.png) as the input to our autoencoder, and obtain the corresponding
vector of activations ![\textstyle a_l^{(1)}](images/math/7/6/c/76c427b1075092b5b1f52f7681b6da30.png). To represent this example, we can either
just **replace** the original feature vector with ![\textstyle a_l^{(1)}](images/math/7/6/c/76c427b1075092b5b1f52f7681b6da30.png).
Alternatively, we can **concatenate** the two feature vectors together,
getting a representation ![\textstyle (x_l^{(1)}, a_l^{(1)})](images/math/6/2/5/62504902238e3007d8271a8def501a09.png).

Thus, our training set now becomes 
![\textstyle \{ (a_l^{(1)}, y^{(1)}), (a_l^{(2)}, y^{(2)}), \ldots (a_l^{(m_l)}, y^{(m_l)})
\}](images/math/0/d/2/0d2ccc3cd881f5dbb524aa3ed19e99be.png) (if we use the replacement representation, and use ![\textstyle a_l^{(i)}](images/math/b/c/b/bcb25bbe4cfb3179e029b7f85ed81399.png) to represent the 
![\textstyle i](images/math/0/b/3/0b36ee693126b34b58f77dba7ed23987.png)-th training example), or ![\textstyle \{
((x_l^{(1)}, a_l^{(1)}), y^{(1)}), ((x_l^{(2)}, a_l^{(1)}), y^{(2)}), \ldots, 
((x_l^{(m_l)}, a_l^{(1)}), y^{(m_l)}) \}](images/math/8/c/2/8c2f57fc671d4d7369a27db0b13eec14.png) (if we use the concatenated
representation). In practice, the concatenated representation often works
better; but for memory or computation representations, we will sometimes use
the replacement representation as well.

Finally, we can train a supervised learning algorithm such as an SVM, logistic
regression, etc. to obtain a function that makes predictions on the ![\textstyle y](images/math/c/8/1/c81e76c28ed991b22b8c1bb8fa392701.png) values. 
Given a test example ![\textstyle x_{\rm test}](images/math/d/f/7/df77f6f969ea9a1e99da9c100fe95a08.png), we would then follow the same procedure:
For feed it to the autoencoder to get ![\textstyle a_{\rm test}](images/math/d/f/a/dfa2797c22eb5c9e484c59f051d7ae68.png). Then, feed 
either ![\textstyle a_{\rm test}](images/math/d/f/a/dfa2797c22eb5c9e484c59f051d7ae68.png) or ![\textstyle (x_{\rm test}, a_{\rm test})](images/math/9/9/b/99b977ec4f2de28e15d9fa90fd60227f.png) to the trained classifier to get a prediction.

  On pre-processing the data
----------------------------

During the feature learning stage where we were learning from the unlabeled training set 
![\textstyle \{ x_u^{(1)}, x_u^{(2)}, \ldots, x_u^{(m_u)}\}](images/math/3/a/3/3a330b29fcaa7c4fd1df8fcd4d19df92.png), we may have computed
various pre-processing parameters. For example, one may have computed
a mean value of the data and subtracted off this mean to perform mean normalization,
or used PCA to compute a matrix ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) to represent the data as ![\textstyle U^Tx](images/math/e/0/a/e0aec5d033ea89dc9bd9c83bc2b4edec.png) (or used 
PCA 
whitening or ZCA whitening). If this is the case, then it is important to
save away these preprocessing parameters, and to use the *same* parameters
during the labeled training phase and the test phase, so as to make sure
we are always transforming the data the same way to feed into the autoencoder. 
In particular, if we have computed a matrix ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) using the unlabeled data and PCA,
we should keep the *same* matrix ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) and use it to preprocess the
labeled examples and the test data. We should **not** re-estimate a
different ![\textstyle U](images/math/6/a/5/6a55fb16b0464ccd6652a7f2a583217f.png) matrix (or data mean for mean normalization, etc.) using the
labeled training set, since that might result in a dramatically different
pre-processing transformation, which would make the input distribution to
the autoencoder very different from what it was actually trained on.

  On the terminology of unsupervised feature learning
-----------------------------------------------------

There are two common unsupervised feature learning settings, depending on what type of 
unlabeled data you have. The more general and powerful setting is the **self-taught learning**
setting, which does not assume that your unlabeled data *x**u* has to
be drawn from the same distribution as your labeled data *x**l*. The 
more restrictive setting where the unlabeled data comes from exactly the same 
distribution as the labeled data is sometimes called the **semi-supervised learning** 
setting. This distinctions is best explained with an example, which we now give.

Suppose your goal is a computer vision task where you'd like
to distinguish between images of cars and images of motorcycles; so, each labeled
example in your training set is either an image of a car or an image of a motorcycle. 
Where can we get lots of unlabeled data? The easiest way would be to obtain some
random collection of images, perhaps downloaded off the internet. We could then 
train the autoencoder on this large collection of images, and obtain useful features
from them. Because here the unlabeled data is drawn from a different distribution
than the labeled data (i.e., perhaps some of our unlabeled images may contain
cars/motorcycles, but not every image downloaded is either a car or a motorcycle), we
call this self-taught learning.

In contrast, if we happen to have lots of unlabeled images lying around
that are all images of *either* a car or a motorcycle, but where the data
is just missing its label (so you don't know which ones are cars, and which
ones are motorcycles), then we could use this form of unlabeled data to
learn the features. This setting---where each unlabeled example is drawn from the same
distribution as your labeled examples---is sometimes called the semi-supervised 
setting. In practice, we often do not have this sort of unlabeled data (where would you
get a database of images where every image is either a car or a motorcycle, but
just missing its label?), and so in the context of learning features from unlabeled
data, the self-taught learning setting is more broadly applicable.

**Self-Taught Learning** | [Exercise:Self-Taught Learning](Exercise_Self-Taught_Learning.md "Exercise:Self-Taught Learning")

---

> * Language: [中文](%E8%87%AA%E6%88%91%E5%AD%A6%E4%B9%A0.md "自我学习")
> * This page was last modified on 7 April 2013, at 13:26.

