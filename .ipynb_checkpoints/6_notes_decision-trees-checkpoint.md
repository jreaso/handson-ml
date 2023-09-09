# Decision Trees

Decision trees are non-parametric.


**SciKit Learn Classifier**
```python
tree_clf = DecisionTreeClassifier(max_depth=2)
```

- Decision trees don't need feature scaling or centering at all (very little data preperation).
- Decision trees are whiteblox models and an advantage is that they are interpretable.

Visualize tree:
```python
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file="images/iris_tree.dot",
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        filled=True
    )

from graphviz import Source

Source.from_file("images/iris_tree.dot")
```

### Gini Impurity

$G_i$ is the Gini impurity of thr $i$th node. $p_{i,k}{}^2$ is the ratio of class $k$ instances among the training instances in the $i$th node.

$$
G_i=1-\sum_{k=1}^n p_{i, k}{}^2
$$


### Entropy

Entropy is another impurity measure. A set’s entropy is zero when it contains instances of only one class.

The entropy of the $i$th node is $H_i$. $p_{i, k}$ is the ratio of class $k$ at node $i$.

$$
H_i=-\sum_{\substack{k=1 \\ p_{i, k} \neq 0}}^n p_{i, k} \log _2\left(p_{i, k}\right)
$$

### Estimating Class Probabilities

```python
tree_clf.predict_proba(...)
```

### CART Training Algorithm

Chooses a split on a class at a threshold that has lowest cost according to the cost function:
$$
J\left(k, t_k\right)=\frac{m_{\text {left }}}{m} G_{\text {left }}+\frac{m_{\text {right }}}{m} G_{\text {right }}
$$
where $\left\{\begin{array}{l}G_{\text {left/right}} \text { measures the impurity of the left/right subset } \\ m_{\text {left/right }} \text { is the number of instances in the left/right subset }\end{array}\right.$

This splitting proceeds until the max depth or another stopping condition.

**CART algorithms are greedy, computing all possibilities is intractable for even small datasets.**



## Regularization

- `max_depth` - Maximum depth of tree.
- `max_features` - Maximum number of features that are evaluated for splitting at each node.
- `max_leaf_nodes` - Maximum number of leaf nodes.
- `min_samples_split` - Minimum number of samples a node must have before it can be split.
- `min_samples_leaf` - Minimum number of samples a leaf node must have to be created.
- `min_weight_fraction_leaf` - Same as `min_samples_leaf` but expressed as a fraction of the total number of weighted instances.

# Regression

```python
tree_reg = DecisionTreeRegressor(max_depth=2)
```

CART training algorithm now tries to split tree to minimise MSE instead of impurity.

$$
J\left(k, t_k\right)=\frac{m_{\text {left }}}{m} \mathrm{MSE}_{\text {left }}+\frac{m_{\text {right }}}{m} \mathrm{MSE}_{\text {right }} \quad \text { where }\left\{\begin{array}{c}
\mathrm{MSE}_{\text {node }}=\frac{\sum_{i \in \text { node }}\left(\hat{y}_{\text {node }}-y^{(i)}\right)^2}{m_{\text {node }}} \\
\hat{y}_{\text {node }}=\frac{\sum_{i \in \text { node }} y^{(i)}}{m_{\text {node }}}
\end{array}\right.
$$

