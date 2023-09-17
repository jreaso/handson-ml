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


