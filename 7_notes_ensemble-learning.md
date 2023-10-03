# Ensemble Learning

## Voting Classifiers

An ensemble majority-vote classifier is called a hard voting classifier.

- Provided a sufficient number of diverse weak learners (only slightly better than random guessing), the ensemble can be a strong learner (high accuracy).

- If all classifiers in the ensemble were independent, the law of large numbers would give an increasing accuracy for higher number of predictors in the ensemble.

- The main problem for ensemble methods is that the predictors are not independant and their errors are correlated.

- Training the predictors with different algorithms is one way to make them more independent from each other.


Can use SciKit Learn's `VotingClassifier` class and just provide it with a list of name, estimator pairs:

```python
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
```

### Hard and Soft Voting

**Hard Voting:** Take most voted for class.

**Soft Voting:** Average class probabilities. _(Needs all classifiers to have a `predict_proba()` method)_.

Soft voting often performs better as it gives more weight to highly confident models

To do soft viting need to set `voting_clf.voting = "soft"`.


## Bagging and Pasting

Another way to get a diverse set of classifiers is to use the same training algorithm but train of different training sets. There are two main methods to get different training sets:

- **Bagging** (_Bootstrap Aggregating_): Random sampling _with_ replacement.

- **Pasting**: Random sampling _without_ replacement.

_Bagging is generally preferred_.

The aggregation function is typically the statistical mode for classification (i.e., the most frequent prediction, just like with a hard voting classifier), or the average for regression.


**These methods scale very well as different models can be trained on different cores or even servers.**


### SciKit Learn Bagging

Use `BaggingClassifier` or `BaggingRegressor`.

### Sampling features

Use the hyperparameters `max_features` and `bootstrap_features` to add more diversity to estimators. This is especially effective with high dimensional data.


## Out-of-Bag Evaluation

Taking a bootstrap sample the same size as the training set will only use approximately 63% of the training values. The remaining 37% of training instances are called _out-of-bag_ (**OOB**) instances.


With enough predictors, each training instance will be an OOB sample for several estimators and can use these estimators to give a ensemble prediction for the overall OOB evaluation.

To use with SciKit Learn, set `oob_score=True` in `BaggingClassifier` and then access the `.oob_score_`.


## Random Forests

A random forests is an ensemble of decision trees generally trained with bagging.

In SciKit Learn, use `RandomForestClassifier` (and `RandomForestRegressor`) instead of `BaggingClassifier`.

By default, the random forest algorithm builds trees on a random subset of roughly $\sqrt{n}$ features.

### Extra Trees

You can add even more diversity by using random thresholds for each feature rather than the optimal threshold. This can be done using `splitter="random"`.

A forest of such extremely random trees is called an extremely randomized trees⁠ (or extra-trees) ensemble. These are significantly faster to train.

Can use `ExtraTreesClassifier` and `ExtraTreesRegressor`.

### Feature Importance

Random forests make it easy to measure the relative importance of each feature. Scikit-Learn measures a feature’s importance by looking at how much the tree nodes that use that feature reduce impurity on average, across all trees in the forest. This can then be scaled.

Access this with `.feature_importances_`.


## Boosting

Boosting combines weak learners into a string learner by training them sequentially, each new learner trying to correct it's predecessor.

AdaBoost is the most popular.

### AdaBoost

Each new predictor pays more attention to the training instances which it's predecessor underfit.

_Note: a drawback of sequential learning is that the training cannot be parallelized so does not scale as well._

If the AdaBoost classifier is overfittng, you can reduce the number of estimators or try more strongly regularizing the base estimator.

In SciKit Learn, AdaBoost is implemented with `AdaBoostClassifier` from `sklearn.ensemble`. Example:
```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```




