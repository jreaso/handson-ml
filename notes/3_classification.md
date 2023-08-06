# Classification

Classifiers:
- `SGDClassifier`
- `RandomForestClassifier`
- `LogisticRegression`
- `GaussianNB`
- `SVC`

## Binary Classifier

### Metrics

- **Accuracy:** Measures the ratio of correct predictions. Can be measured via cross validation using `cross_val_score`. This can be misleading if one class is not very common, if 90% of the instances are in class A, a dumy classifier always classifying as class A will have 90% accuracy.
- **Confusion Matrices:** Actual values are recorded in columns and predicted values are recorded in rows. Use `cross_val_predict` and `confusion_matrix`.
- **Precision and Recall**
    - **Precision** is the ratio of positive predictions that were correct. $$\text { precision }=\frac{T P}{T P+F P} $$
    - **Recall** is the ratio of positive instances that were predicted correctly. Also called **sensitivity**. $$\text { recall }=\frac{T P}{T P+F N} $$
    - **Precision Recall Tradeoff:** As you tune a classifier, increasing the precision will decrease the recall and vice versa. In some instances either precision or recall would be more important than the other.
- $F_1$ **Score:** This metric takes the harmonic mean of precision and recall to give the a metric that cares equally about both. This can be computed using `f1_score`.
- **The ROC Curve:** The receiver operating characteristic (ROC) curve plots TPR (true positive rate) against the FPR (false positive rate).
    - TPR is another name for **recall**, the ratio of positive instances correctly predicited as positive. This is also called **sensitivity**.
    - FPR is also called the **fall-out** and is the ratio of negative instances incorrectly classified as positive. This is $1- \text{TNR}$ and the TNR is also called specificity.
    - TPR and FPR also have a tradeoff where you want a high recall but low fall-out.
    - **The AUC Score** measures the area under the ROC curve. A perfect classifier will have an AUC score of 1, whereas a random classifier would have an AUC score of 0.5.
