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

### Clean Data
### Deal with missing values
    - Remove instances with missing values
    - Remove variables/attributes with missing data
    - **Imputation** - fill in the missing values
    
    *It is safest to apply an imputer to all numerical attributes in case some complete attributes get missing data once the system goes live.*

### Consider outliers
### Deal with **categorical variables**
    - Encode the categorical variable using a suitable encoder (e.g. ordinal encoder or a one-hot encoder)
    - For variables with a large number of categories, consider replacing the variable (e.g. replace country with GDP of country)

## 5. 