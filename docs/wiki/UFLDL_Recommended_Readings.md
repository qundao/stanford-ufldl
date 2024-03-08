UFLDL Recommended Readings
==========================

<!-- Jump to: [navigation](#column-one), [search](#searchInput) -->
If you're learning about UFLDL (Unsupervised Feature Learning and Deep Learning), here is a list of papers to consider reading. We're assuming you're already familiar with basic machine learning at the level of [[CS229 (lecture notes available)](http://cs229.stanford.edu/)].

The basics:

* [[CS294A](http://cs294a.stanford.edu/)] Neural Networks/Sparse Autoencoder Tutorial. (Most of this is now in the [UFLDL Tutorial](UFLDL_Tutorial.md "UFLDL Tutorial"), but the exercise is still on the CS294A website.)
* [[1]](http://www.naturalimagestatistics.net/) Natural Image Statistics book, Hyvarinen et al.
	+ This is long, so just skim or skip the chapters that you already know.
	+ Important chapters: 5 (PCA and whitening; you'll probably already know the PCA stuff), 6 (sparse coding), 7 (ICA), 10 (ISA), 11 (TICA), 16 (temporal models).
* [[2]](http://redwood.psych.cornell.edu/papers/olshausen_field_nature_1996.pdf) Olshausen and Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images Nature 1996. (Sparse Coding)
* [[3]](http://www.cs.stanford.edu/~ang/papers/icml07-selftaughtlearning.pdf) Rajat Raina, Alexis Battle, Honglak Lee, Benjamin Packer and Andrew Y. Ng. Self-taught learning: Transfer learning from unlabeled data. ICML 2007

Autoencoders:

* [[4]](http://www.cs.toronto.edu/~hinton/science.pdf) Hinton, G. E. and Salakhutdinov, R. R. Reducing the dimensionality of data with neural networks. Science 2006.
	+ If you want to play with the code, you can also find it at [[5]](http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html).
* [[6]](http://books.nips.cc/papers/files/nips19/NIPS2006_0739.pdf) Bengio, Y., Lamblin, P., Popovici, P., Larochelle, H. Greedy Layer-Wise Training of Deep Networks. NIPS 2006
* [[7]](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) Pascal Vincent, Hugo Larochelle, Yoshua Bengio and Pierre-Antoine Manzagol. Extracting and Composing Robust Features with Denoising Autoencoders. ICML 2008.
	+ (They have a nice model, but then backwards rationalize it into a probabilistic model. Ignore the backwards rationalized probabilistic model [Section 4].)

Analyzing deep learning/why does deep learning work:

* [[8]](http://www.cs.toronto.edu/~larocheh/publications/deep-nets-icml-07.pdf) H. Larochelle, D. Erhan, A. Courville, J. Bergstra, and Y. Bengio. An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation. ICML 2007.
	+ (Someone read this and let us know if this is worth keeping,. [Most model related material already covered by other papers, it seems not many impactful conclusions can be made from results, but can serve as reading for reinforcement for deep models])
* [[9]](http://www.jmlr.org/papers/volume11/erhan10a/erhan10a.pdf) Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent, and Samy Bengio. Why Does Unsupervised Pre-training Help Deep Learning? JMLR 2010
* [[10]](http://cs.stanford.edu/~ang/papers/nips09-MeasuringInvariancesDeepNetworks.pdf) Ian J. Goodfellow, Quoc V. Le, Andrew M. Saxe, Honglak Lee and Andrew Y. Ng. Measuring invariances in deep networks. NIPS 2009.

RBMs:

* [[11]](http://deeplearning.net/tutorial/rbm.html) Tutorial on RBMs.
	+ But ignore the Theano code examples.
	+ (Someone tell us if this should be moved later. Useful for understanding some of DL literature, but not needed for many of the later papers? [Seems ok to leave in, useful introduction if reader had no idea about RBM's, and have to deal with Hinton's 06 Science paper or 3-way RBM's right away])

Convolution Networks:

* [[12]](http://deeplearning.net/tutorial/lenet.html) Tutorial on Convolution Neural Networks.
	+ But ignore the Theano code examples.

Applications:

* Computer Vision
	+ [[13]](http://www.ifp.illinois.edu/~jyang29/ScSPM.htm) Jianchao Yang, Kai Yu, Yihong Gong, Thomas Huang. Linear Spatial Pyramid Matching using Sparse Coding for Image Classification, CVPR 2009
	+ [[14]](http://people.csail.mit.edu/torralba/publications/cvpr2008.pdf) A. Torralba, R. Fergus and Y. Weiss. Small codes and large image databases for recognition. CVPR 2008.
* Audio Recognition
	+ [[15]](http://www.cs.stanford.edu/people/ang/papers/nips09-AudioConvolutionalDBN.pdf) Unsupervised feature learning for audio classification using convolutional deep belief networks, Honglak Lee, Yan Largman, Peter Pham and Andrew Y. Ng. In NIPS 2009.

Natural Language Processing:

* [[16]](http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/57) Yoshua Bengio, Réjean Ducharme, Pascal Vincent and Christian Jauvin, A Neural Probabilistic Language Model. JMLR 2003.
* [[17]](http://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf) R. Collobert and J. Weston. A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning. ICML 2008.
* [[18]](http://www.socher.org/uploads/Main/SocherPenningtonHuangNgManning_EMNLP2011.pdf) Richard Socher, Jeffrey Pennington, Eric Huang, Andrew Y. Ng, and Christopher D. Manning. Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions. EMNLP 2011
* [[19]](http://www.socher.org/uploads/Main/SocherHuangPenningtonNgManning_NIPS2011.pdf) Richard Socher, Eric Huang, Jeffrey Pennington, Andrew Y. Ng, and Christopher D. Manning. Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection. NIPS 2011
* [[20]](http://www.cs.toronto.edu/~hinton/absps/threenew.pdf) Mnih, A. and Hinton, G. E. Three New Graphical Models for Statistical Language Modelling. ICML 2007

Advanced stuff:

* Slow Feature Analysis:
	+ [[21]](http://itb.biologie.hu-berlin.de/~wiskott/Publications/BerkWisk2005c-SFAComplexCells-JoV.pdf) Slow feature analysis yields a rich repertoire of complex cell properties. Journal of Vision, 2005.
* Predictive Sparse Decomposition
	+ [[22]](http://cs.nyu.edu/~koray/publis/koray-psd-08.pdf) Koray Kavukcuoglu, Marc'Aurelio Ranzato, and Yann LeCun, "Fast Inference in Sparse Coding Algorithms with Applications to Object Recognition", Computational and Biological Learning Lab, Courant Institute, NYU, 2008.
	+ [[23]](http://cs.nyu.edu/~koray/publis/jarrett-iccv-09.pdf) Kevin Jarrett, Koray Kavukcuoglu, Marc'Aurelio Ranzato, and Yann LeCun, "What is the Best Multi-Stage Architecture for Object Recognition?", In ICCV 2009

Mean-Covariance models

* [[24]](http://www.cs.toronto.edu/~ranzato/publications/ranzato_aistats2010.pdf) M. Ranzato, A. Krizhevsky, G. Hinton. Factored 3-Way Restricted Boltzmann Machines for Modeling Natural Images. In AISTATS 2010.
* [[25]](http://www.cs.toronto.edu/~ranzato/publications/ranzato_cvpr2010.pdf) M. Ranzato, G. Hinton, Modeling Pixel Means and Covariances Using Factorized Third-Order Boltzmann Machines. CVPR 2010
	+ (someone and tell us if you need to read the 3-way RBM paper before the mcRBM one [I didn't find it necessary, in fact the CVPR paper seemed easier to understand.])
* [[26]](http://www.cs.toronto.edu/~hinton/absps/mcphone.pdf) Dahl, G., Ranzato, M., Mohamed, A. and Hinton, G. E. Phone Recognition with the Mean-Covariance Restricted Boltzmann Machine. NIPS 2010.
* [[27]](http://www.nature.com/nature/journal/v457/n7225/pdf/nature07481.pdf) Y. Karklin and M. S. Lewicki, Emergence of complex cell properties by learning to generalize in natural scenes, Nature, 2008.
	+ (someone tell us if this should be here. Interesting algorithm + nice visualizations, though maybe slightly hard to understand. [seems a good reminder there are other existing models])

Overview

* [[28]](http://www.iro.umontreal.ca/~bengioy/papers/ftml_book.pdf) Yoshua Bengio. Learning Deep Architectures for AI. FTML 2009.
	+ (Broad landscape description of the field, but technical details there are hard to follow so ignore that. This is also easier to read after you've gone over some of literature of the field.)

Practical guides:

* [[29]](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) Geoff Hinton. A practical guide to training restricted Boltzmann machines. UTML TR 2010–003.
	+ A practical guide (read if you're trying to implement and RBM; but otherwise skip since this is not really a tutorial).
* [[30]](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) Y. LeCun, L. Bottou, G. Orr and K. Muller. Efficient Backprop. Neural Networks: Tricks of the trade, Springer, 1998
	+ Read if you're trying to run backprop; but otherwise skip since very low-level engineering/hackery tricks and not that satisfying to read.

Also, for other lists of papers:

* [[31]](http://www.eecs.umich.edu/~honglak/teaching/eecs598/schedule.html) Honglak Lee's Course
* [[32]](http://www.cs.toronto.edu/~hinton/deeprefs.html) from Geoff's tutorial
> * This page was last modified on 18 February 2012, at 07:00.

