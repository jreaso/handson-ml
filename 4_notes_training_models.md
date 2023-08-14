# Training Linear Models

Learning algorithms will often optimize a different loss function during training than the performance measure used to evaluate the final model. This is due to being easier to optimize or for regularization reasons. We should aim to get the loss function to be as similar as possible to the performance metric.

## Linear Regression

**MSE Cost Function**
$$
\operatorname{MSE}\left(\mathbf{X}, h_{\boldsymbol{\theta}}\right)=\frac{1}{m} \sum_{i=1}^m\left(\boldsymbol{\theta}^{\top} \mathbf{x}^{(i)}-y^{(i)}\right)^2
$$

There is a _closed-form solution_ for $\theta$ which minimises the MSE. This is called the normal equation.

**Normal Equation**
$$
\hat{\boldsymbol{\theta}}=\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \mathbf{y}
$$

In practice, the normal equation is not used directly as it is not as computationally efficient as other techniques. Scikit-Learn’s `LinearRegression` class has square $O(n^2)$ complexity for number of instances and linear $O(m)$ complexity for number of variables.

## Gradient Descent

Gradient descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems.