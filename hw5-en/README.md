<!--- 
# Université de Montréal
# IFT-6758-B  -  A23  -  Data Science 
-->

# Homework 4

Assignment scoring:

| Section                                                     | Required Files   | Score |
|-------------------------------------------------------------|------------------|:-----:|
| Model Iterpretability                                       | `hw4.py`         |  25   |
| &emsp;+ figures, short answers                              | `hw4.ipynb`      |  25   |
| Outlier Detection & Removal, Feature Selection & Engineering| `hw4.py`         |  20   |
| &emsp;+ figures, short answers                              | `hw4.ipynb`      |  30   |

Model Explainability (Bonus: +5)

A portion of your homework will be autograded, i.e., you must **not modify the signature of the defined functions** (same inputs and outputs).


### Submitting

To submit the files, please submit **only the required files** (listed in the above table) that you completed to **gradescope**; do not include data or other miscellaneous files.

**Warning 1: You should be careful with the random states of many of the methods that we will implement, as a default value has been set in each function for them. Please, use this default values in your implementations**

**Warning 2: You will need the following version of scikit-learn==1.2.2, as there seems to be a conflict with the eli5 library**


## 1. Model Interpretability and Explainability


On this first part of the assignment, we will work with model interpretability and model explainability. We will be using a hospital readmission dataset (`hospital.csv`), that will help us to understand the importance of interpretating the interaction of our model with the features that are present in our dataset. Also, it is going to help us to learn how to explain the predictions of our models.


### 1.1 Load Data

- Complete `hw4.py:encode_target_column()`
- Complete the code executions in `hw4.ipynb`

We will start by loading our data and taking a quick look at it. In this section you won't need to do much, you will only have to complete the `encode_target_column()`. Use scikit-learn's `LabelEncoder` to complete this method.


### 1.2 Model Interpretability

- Complete `hw4.py:train_random_forest()`
- Complete `hw4.py:evaluate_model()`
- Complete the code executions in `hw4.ipynb`



In this section we will start by training a simple Random Forest Classifier and evaluating its performance on the evaluation set. You will need to create two functions `train_random_forest()` and `evaluate_model()`. The first function will require you to use scikit-learn's `RandomForestClassifier` and implement the training process within a single method. The `evaluate_model()` method will have to return the accuracy of our model and the classification report, here you will need to use two methods from `sklearn.metrics` that have already been imported into `hw4.py`.  

#### 1.2.1 Feature Importance

- Complete `hw4.py:calculate_permutation_importance()`
- Complete the code executions in `hw4.ipynb`
- Answer the question in `hw4.ipynb`


In this section you will learn to use a technique called `Permutation Importance`, that will help us to understand how important our dataset features are when they interact with a specific model. This technique benefits from being model agnostic and can be calculated many times with different permutations of the feature (Warning: Features that are deemed of low importance for a bad model could be very important for a good model). More information on this technique can be found [here](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html).  Here you will complete a method called `calculate_permutation_importance()` and we will use the method `PermutationImportance` from the `eli5` library to do so.

You will have to answer the question present in this section.

#### 1.2.2 Partial Dependence Plot

- Complete `hw4.py:plot_partial_dependence()`
- Complete `hw4.py:plot_mean_readmission_vs_time()`
- Complete the code executions in `hw4.ipynb`
- Complete the figures in `hw4.ipynb`

Also, we will use a type of visualization known as `Partial Dependence plot`. The Partial Dependence Plot (PDP) is a rather intuitive and easy-to-understand visualization of the features impact on the predicted outcome. If the assumptions for the PDP are met, it can show the way a feature impacts an outcome variable. More information on PD plots can be found [here](https://slds-lmu.github.io/iml_methods_limitations/pdp.html). Here you will complete a method called `plot_partial_dependence()` and to complete it you will use scikit-learn's `PartialDependenceDisplay` to create this visualization.

Finally, within the proposed hypothetical scenario, our stakeholders will ask you to verify that the data of a specific feature is correct. Therefore, they will ask us to produce a plot of the the "mean readmissions vs time". Here you will have to complete the `plot_mean_readmission_vs_time()` method, that will help us obtain this visualization.


### 1.3 Model Explainability (Bonus +5)


#### 1.3.1 SHAP values

- Complete `hw4.py:main_factors()`
- Complete the code executions in `hw4.ipynb`
- Complete the figure in `hw4.ipynb`
- Answer the question in `hw4.ipynb`

Now that we have learned a little bit of model interpretability, we will learn about model explainability. In this section we will learn how we can explain a model's prediction on a single example. To do so we will use SHAP values (an acronym from SHapley Additive exPlanations) to break down a prediction and show the impact of each feature. This is rather convenient, as they allow us to visually identify which features (and in what magnitude) are supporting the prediction and which ones are decreasing the prediction. To use SHAP Values, you will implement a method called `main_factors()` with the help of the `shap` library. You will need to read the library's documentation to learn how to produce the desired visualization for a single example. 

You will have to answer the question present in this section.


## Part 2. Anomaly Detection and Removal + Feature Selection and Engineering

### 2.1 Loading the Data

- Complete the code executions in `hw4.ipynb`

In this section you don't need to complete anything. We will just load the New York city taxi fare prediction dataset (`ny_taxi.csv`). This dataset will help us to link the `permutation importance` technique to the feature selection and feature engineering processes. You just have to run the code cells in the `hw4.ipynb`.


### 2.2 Outlier Handling and Feature Selection

- Complete `hw4.py:remove_outliers_iqr()`
- Complete the code executions in `hw4.ipynb`
- Answer the questions in `hw4.ipynb`

In this section we will explore an outlier detection method known as the IQR (Interquartile Range) Method. The IQR method defines outliers as data points that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR, where Q1 and Q3 are the 25th and 75th percentiles, respectively. Your task is to complete the `remove_outliers_iqr()` method that can be found in `hw4.py`. Considerations:

- Your function should implement the IQR method to detect and remove rows with outliers. 
- Given a list of the preselected features, your function should check each column for outliers and return a new dataset free of outliers. 

After this, we will train a Random Forest Regressor and use our `calculate_permutation_importance()` to identify the most relevant features. This will help us to select the features that we will use for the feature engineering exercise of the next section.

You will have to answer the questions present in this section.


### 2.3 Feature Engineering

- Complete `hw4.py:add_absolute_coordinate_changes()`
- Complete the code executions in `hw4.ipynb`
- Answer the questions in `hw4.ipynb`

We are close to the end, but we still need to learn a bit about feature engineering. In this assignment we will explore the creation of two new features called `abs_lon_change` and `abs_lat_change` that will represent the `absolute longitudinal distance` and the `absolute latitudinal distance`. To create this new features, you will have to complete the `add_absolute_coordinate_changes()` function. 

You will have to answer the questions present in this section.

**That's the end of this assignment. I hope that you found it useful.** 

# References

- **This assignment** is based on Kaggle's Machine Learning Explainability [course.](https://www.kaggle.com/learn/machine-learning-explainability)
