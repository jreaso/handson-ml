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

Increase regukarizationn by decreasing $\gamma$ and $C$.

## Support Vector Regression

Intuition: fit a line which minimises the error beyond a certain distance $\epsilon$. The instances lying outside the $\epsilon$-insensitive region are called support vectors. $\epsilon$ is a hyperparameter.

Use `LinearSVR` for linear case and when using a kernel, use `SVR`, e.g. `SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)`.

## Under the Hood

**Decision Function** ($b$ is a bias term):
$$\boldsymbol{w}^\top \mathbf{x} + b = w_0 x_0+\cdots+w_n x_n$$

The classier boundary is when the decision function is 0 and the margins are when it is $\pm 1$.

If we define $t^{(i)} = +1$ for positive instances and $t^{(i)} = -1$ for negative instances then the hard margin SVM problem is:

**Hard Margin SVM Problem:**
$$\begin{array}{ll}
\underset{\mathbf{w}, b}{\operatorname{minimize}} & \frac{1}{2} \mathbf{w}^{\top} \mathbf{w} \\
\text { subject to } & t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \geq 1 \quad \text { for } i=1,2, \cdots, m
\end{array}$$

_Note: we minimize $\frac{1}{2} \mathbf{w}^{\top} \mathbf{w} = \frac{1}{2} \| \mathbf{w} \|^2$ instead of $\| \mathbf{w} \|$ since it is differentiable (with a simple derivative)_.

**Soft Marging SVM Problem:**

We add a slack variable $\xi^{(i)}$ for each instance which measures how much the instance violates the borders.

$$\begin{array}{ll}
\underset{\mathbf{w}, b, \zeta}{\operatorname{minimize}} & \frac{1}{2} \mathbf{w}^{\top} \mathbf{w}+C \sum_{i=1}^m \zeta^{(i)} \\
\text { subject to } & t^{(i)}\left(\mathbf{w}^{\top} \mathbf{x}^{(i)}+b\right) \geq 1-\zeta^{(i)} \quad \text { and } \quad \zeta^{(i)} \geq 0 \quad \text { for } i=1,2, \cdots, m
\end{array}$$


These are both **Quadratic Programming Problems** and can be trained by QP Solvers, Gradient Descent or by solving the Dual Problem. When using GD, you can use a hinge loss or a square hinge loss function. The latter tends to converge quicker but can be sensitive to outliers.


## Dual Problem

The solution to the dual problem typically gives a lower bound to the solution of the primal problem, but under some conditions (The objective function is convex, and the inequality constraints are continuously differentiable and convex functions) it can have the same solution as the primal problem. These conditions hold for SVM.

**Dual Problem:**
$$\underset{\boldsymbol{\alpha}}{\operatorname{minimize}} \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha^{(i)} \alpha^{(j)} t^{(i)} t^{(j)} \mathbf{x}^{(i)^{\top}} \mathbf{x}^{(j)}-\sum_{i=1}^m \alpha^{(i)} \text { subject to } \alpha^{(i)} \geq 0 \text { for all } i=1,2, \ldots, m \text { and } \sum_{i=1}^m \alpha^{(i)} t^{(i)}=0$$

Then you can use the $\alpha$ to compute $\mathbf{w}$ and $b$.

The dual problem **enables the kernel trick to be used**.

## Kernel Trick

In machine learning, a kernel is a function capable of computing the dot product $\phi(a)^\top \phi(b)$, based only on the original vectors $a$ and $b$, without having to compute (or even to know about) the transformation $\phi$.

**Common Kernels:**
$$
\begin{aligned}
\text { Linear: } & K(\mathbf{a}, \mathbf{b})=\mathbf{a}^{\top} \mathbf{b} \\
\text { Polynomial: } & K(\mathbf{a}, \mathbf{b})=\left(\gamma \mathbf{a}^{\top} \mathbf{b}+r\right)^d \\
\text { Gaussian RBF: } & K(\mathbf{a}, \mathbf{b})=\exp \left(-\gamma\|\mathbf{a}-\mathbf{b}\|^2\right) \\
\text { Sigmoid: } & K(\mathbf{a}, \mathbf{b})=\tanh \left(\gamma \mathbf{a}^{\top} \mathbf{b}+r\right)
\end{aligned}
$$


>**Mercer's Theorem:**
>
>If a function $K(a, b)$ respects a few mathematical conditions called Mercer’s conditions:
>- $K$ must be continuous
>- $K$ must be symmetric in its arguments
>
>then there exists a function $\phi$ that maps $a$ and $b$ into
another space (possibly with much higher dimensions) such that $K(a, b) = \phi(a)^{\top} \phi(b)$.

In the case of the RBF kernel, the function $\phi$ maps to an infinte dimensional space so it is impossible to do without the kernel trick.


_Note: not all popular kernels satisfy Mercer's conditions such as the Sigmoid kernel, but still work well in practice._

Making predictions with the kernalized SVM involves using an equation to write the decision function of a new instance in terms of the kernel rather than $\phi(\mathbf{w})$ (see book).




