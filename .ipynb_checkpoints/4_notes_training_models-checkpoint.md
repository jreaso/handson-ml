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

General Considerations:
- Not all cost functions are convex
- Features should be on the same scale

For non-differentiable loss functions, you must use the subgradient (such as for lasso).

### Batch Gradient Descent

Batch GD uses the whole training set at every step of the algorithm.

**Gradient Vector of the Cost Cunction:**
$$
\nabla_{\boldsymbol{\theta}} \operatorname{MSE}(\boldsymbol{\theta})=\frac{2}{m} \mathbf{X}^{\top}(\mathbf{X} \boldsymbol{\theta}-\mathbf{y})
$$


This is extremely slow on large datasets.

**GD Step:**
$$
\boldsymbol{\theta}^{\text {(next step })}=\boldsymbol{\theta}-\eta \nabla_{\boldsymbol{\theta}} \operatorname{MSE}(\boldsymbol{\theta})
$$

### Stochastic Gradient Descent

To speed up gradient descent, SGD computes the gradient based on a random instance of the data. This means it is less stable and can bounce around but in general will converge much quicker. It can be implemented as an out of core algorithm.

If the learning rate is constant, the algorithm will not settle at a local minima so we use a _learning schedule_ to decrease the learning rate.

When using stochastic gradient descent, the training instances must be IID to ensure that you are optimizing for a global minimum. So if the training instances are sorted by label, they should be shuffled first.

### Mini-Batch Gradient Descent

Same as SGD but takes a random subset instead of a single random instance each iteration. Mini-batch GD computes the gradients on small random sets of instances called _mini-batches_.

The main benefit of mini-batch GD is that you can get a performance boost from the optimized matrix operations you can get from GPUs.

## Learning Curves

Learning curves can be used to tell if your model is over or underfitting to the data.

Learning Curves are plots of the model’s training error and validation error as a function of the training iteration: just evaluate the model at regular intervals during training on both the training set and the validation set, and plot the results.

> All Scikit-Learn estimators have a `fit()` method but some have a `partial_fit()` which runs a single round of training only. Other models have a `warm_start` hyperparameter instead (and some have both): if you set `warm_start=True`, calling the `fit()` method on a trained model will not reset the model; it will just continue training where it left off.

Scikit-Learn has a useful `learning_curve()` function to help with this: it trains and evaluates the model using cross-validation. By default it retrains the model on growing subsets of the training set, but if the model supports incremental learning you can set `exploit_incremental_learning=True` when calling `learning_curve()` and it will train the model incrementally instead.

### Biance-Variance Tradeoff

A model’s generalization error can be expressed as the sum of three very different errors:
- **Bias:** This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data
- **Variance:** This part is due to the model’s excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance and thus overfit the training data.
- **Irreducible Error:** This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers).

## Regularized Linear Models

There are several regularized linear models. 

### Ridge

Can be solved with closed form solutions (Cholesky) or GD.

$$\frac{\alpha}{m} \sum_{i=1}^n \theta_i{ }^2$$

#### Python Implementations

```python
Ridge(alpha=0.1, solver="cholesky")

SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None, max_iter=1000, eta0=0.01) #alpha different than for Ridge
```

There is also `RidgeCV()` which automatically tunes the hyperparameters. This is similar to using grid search but optimsed for ridge so runs a lot faster.

```python
RidgeCV(...)
```

### Lasso

Lasso tends to eliminate the weights of the least important features. Lasso automatically performs feature selection and outputs a sparse model.

$$2 \alpha \sum_{i=1}^n\left|\theta_i\right|$$

The $2 \alpha$ term in Lasso and the $\frac{\alpha}{m}$ in Ridge are chosen so that the optimal $\alpha$ value is independant of the training set size.

```python
LassoCV(...)
```

### Elastic Net

A compromise between Ridge and Lasso.

$$
r\left(2 \alpha \sum_{i=1}^n\left|\theta_i\right|\right)+(1-r)\left(\frac{\alpha}{m} \sum_{i=1}^n \theta_i^2\right)
$$

```python
ElasticNet(alpha=0.1, l1_ratio=0.5)

ElasticNetCV(...)
```


## Early Stopping

A very different way to regularize iterative learning algorithms such as gradient descent is to stop training as soon as the validation error reaches a minimum.

With early stopping you just stop training as soon as the validation error reaches the minimum.

_Note: With stochastic and mini-batch gradient descent, the curves are not so smooth, and it may be hard to know whether you have reached the minimum or not. One solution is to stop only after the validation error has been above the minimum for some time (when you are confident that the model will not do any better), then roll back the model parameters to the point where the validation error was at a minimum._

**Copying Models:** `copy.deepcopy()` copies models hyperparameters **and** learned parameters as opposed to `sklearn.base.clone()` which only copies the models hyperparameters.


## Logistic Regression

Logistic regression works similarly to linear regression but with a sigmoid applied to map the output space to $[0,1]$, estimating the probability of an outcome.
$$\widehat{p}=h_{\boldsymbol{\theta}}(\mathbf{x})=\sigma\left(\boldsymbol{\theta}^{\top} \mathbf{x}\right)$$

**Sigmoid Function:**$$\sigma(t)=\frac{1}{1+\exp (-t)}$$

The sigmoid is the logistic function. The inverse of the logistic function is called the_logit_.

**Cost Function** for a single instance:$$c(\boldsymbol{\theta})=\left\{\begin{array}{cc}
-\log (\widehat{p}) & \text { if } y=1 \\
-\log (1-\widehat{p}) & \text { if } y=0
\end{array}\right.$$

**Cost Function** for a whole training set:$$J(\boldsymbol{\theta})=-\frac{1}{m} \sum_{i=1}^m\left[y^{(i)} \log \left(\widehat{p}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\widehat{p}^{(i)}\right)\right]$$


There is no closed form expression to fin the parameters minimising this, but it is a convex function so GD can be used.

Logistic Regression can be regularized, Sci-Kit Learn adds an $\ell_2$ penalty by default.

### Softmax Regression




