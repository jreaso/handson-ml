# Machine Learning Workflow

## 1. Understand The Data

- Look at the types of variables and their units.
- Look at the distribution of different variables.
- Find out if any data has been capped, if outliers have been removed etc.
- Do not look at data too much, avoid **snooping bias**.

## 2. Create Train/Test Split

Want to set aside roughly 20% of the data to evaluate generalization error. In the case of large data, a smaller amoubt could be sufficient. When splitting the data you must make these considerations:

- Needs to be split in a reproducible way so that by re-running the program, you don't eventually see all the data.
- Needs to be done in a way that if more data is added, this doesn't cause the test set to mix with training set.
- For large datasets, taking a random 20% of the instances will be fine, for smaller datasets, you may need to stratified sampling.

## 3. Visualize and Explore Data

## 4. Prepare Data

1. **Clean Data**
2. **Deal with Missing Values**
    - Remove instances with missing values
    - Remove variables/attributes with missing data
    - **Imputation** - fill in the missing values
    
    *It is safest to apply an imputer to all numerical attributes in case some complete attributes get missing data once the system goes live.*

3. **Consider Outliers**
4. **Deal with Categorical Cariables**
    - Encode the categorical variable using a suitable encoder (e.g. ordinal encoder or a one-hot encoder)
    - For variables with a large number of categories, consider replacing the variable (e.g. replace country with GDP of country)

5. **Feature Scaling and Engineering**

    *Apply transformations to variables so they are on the same scale and of a desired distribution*

## 5. Make a Transformer Pipeline

Make a transformer which deals with all the data preparation.

## 6. Select and Train a Model

## 7. Fine Tune Model

Fine tune using a randomized or grid search to discover the optimal hyper-parameters. If a hyper-parameter is chosen that is on the boundary of choices, extend the search area. Use cross validation to estimate generalization error.

## 8. Evaluate Model

Evaluate the generalization error on the test set and **do no go back and tweak model after**.

## 9. Deploy Model

This step involves launching, monitoring and maintaining the system.