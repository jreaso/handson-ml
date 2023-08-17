# Support Vector Machines

- SVM works well for small to medium sized non-linear datasets. SVM regression especially does not scale well on large datasets.
- SVMs are sensitive to scaling.

## Linear SVM

Support vectors are the instances in the training set which lie inside of or on the margins.

The _decision function_ for SVM is the signed distance from the boundary. With a regular SVM classifier, we **cannot predict probabilities of an instance belonging to a class** like we can with logistic regression. However, if you are ok with training taking much longer, by using SciKit Learn's `SVC` class (not `LinearSVC`) an setting `probability=True`, it will train an additional model after training to map decision functions to probabilities.

When using a linear kernel, the `LinearSVC` class is much faster than `SVC(kernel="linear")`. `LinearSVC` has a training time complexity of roughly $\mathcal{O}(m \times n)$.

## Similarity Features

Similarity features use a similarity function, which measures how much each instance resembles a particular landmark. These landmarks could be chosen beforehand, or every instance could be taken as a landmark - this is what the Gaussian RBF Kernel does.

## Gaussian RBF Kernel

