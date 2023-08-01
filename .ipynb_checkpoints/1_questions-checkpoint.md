1. **How would you define Machine Learning?**

    A computer learning how to do a task without being explicitly programmed. Or in an engineering perspective, A system learns from experience E if more exposure to experience means a better performance P with respect to a task T.
  
2. **Can you name four types of problems where it shines?**

    Machine Learning is great for complex problems where we don't have any algorithmic solutions, to replace difficult to maintain complicated solutions, to adapt quickly to new data and to help humans learn.
  
3. **What is a labeled training set?**

    A subset of the total data used to train a supervised model, the desired outcomes have been labelled.
  
4. **What are the two most common supervised tasks?**

    Classification and regression.
  
5. **Can you name four common unsupervised tasks?**

    Clustering, dimensionality reduction, visualisation and anomaly detection
  
6. **What type of Machine Learning algorithm would you use to allow a robot to walk in various unknown terrains?**

    Reinforcement learning, we can let it experiment and reward/punish good/bad outcomes.
  
7. **What type of algorithm would you use to segment your customers into multiple groups?**

    In the unsupervised case, if you don't know the groups, you would use clustering. If you know the groups and have labelled examples then we are in the supervised case and can use a classification algorithm.
  
8. **Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?**

    Supervised since we would expect a training set of emails labelled as spam and ham (not spam).
  
9. **What is an online learning system?**

    A learning system which learns in small batches of data at a time and can learn from new data without retraining the entire model. This may be used when quick responsive learning is needed (such as a stock market model), an autonomous learning system is needed or when there is a huge amount of data.
  
10. **What is out-of-core learning?**

    Out of core learning is a form of online learning where a huge dataset (often too big to store locally) is split into smaller chunks of data and the model learns from one chunk at a time. Despite the name, this does not need to run online.
  
11. **What type of learning algorithm relies on a similarity measure to make predictions?**

    Instance based learning which learns results by heart then uses a similarity measure to compare new instances to learned instances. For example, k-mean classification.
  
12. **What is the difference between a model parameter and a learning algorithm’s hyperparameter?**

    A model parameter is tuned during the training process whereas a hyper parameter is specified before training.
  
13. **What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?**

    Optimal values of model parameters that will generalise well to new instances. This is usually done by minimising a cost function.
  
14. **Can you name four of the main challenges in Machine Learning?**

    Lack of data, poor quality data, excessively simple/complex models, non-representative data, uninformative features.
  
15. **If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?**

    The model is overfitting to the training set. Possible solutions are to choose a simpler model, include more data or reduce noise in the data.
  
16. **What is a test set and why would you want to use it?**

    Evaluates how well a model generalises to unseen data (estimates generalization error). Good to know how good a model is before it is deployed to production.
  
17. **What is the purpose of a validation set?**

    To evaluate and compare different models and tune hyper parameters.
  
18. **What can go wrong if you tune hyperparameters using the test set?**

    The test set can underestimate the generalization error as you can overfit.
  
19. **What is repeated cross-validation and why would you prefer it to using a single validation set?**

    To choose the best model, you can split the data into k validation sets and each model is evaluated on each validation set after being trained on the rest of the data, you can then average over the validation sets to get a more accurate estimate of the models performance but at the cost of more computation.


*Note: questions from second edition, not third.*
  


