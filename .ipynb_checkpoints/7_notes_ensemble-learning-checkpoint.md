# Ensemble Learning

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

## Hard and Soft Voting

**Hard Voting:** Take most voted for class.

**Soft Voting:** Average class probabilities. _(Needs all classifiers to have a `predict_proba()` method)_.

Soft voting often performs better as it gives more weight to highly confident models


To do soft viting need to set `voting_clf.voting = "soft"`.
