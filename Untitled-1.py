# %%
# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

# %%
df = pd.read_csv('HR_comma_sep.csv.txt')
df

# %%
# Examine the dataset
df.head()

# %%
# Rename Columns
# Renaming certain columns for better readability
df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })

# %%
round(df.turnover.value_counts(1), 2)

# %%
plt.figure(figsize=(12,8))
turnover = df.turnover.value_counts()
sns.barplot(y=turnover.values, x=turnover.index, alpha=0.6)
plt.title('Distribution of Employee Turnover')
plt.xlabel('Employee Turnover', fontsize=16)
plt.ylabel('Count', fontsize=16)

# %%
# Can you check to see if there are any missing values in our data set
df.isnull().any()

# %%
# Check the type of our features. Are there any data inconsistencies?
df.dtypes

# %%
# Display the statistical overview of the employees
round(df.describe(), 2)

# %%
# Create a correlation matrix. What features correlate the most with turnover? What other correlations did you find?
df1 = df.copy()
df1 = df1.drop(["department","salary"],axis=1)
corr = df1.corr()
corr

# %%
plt.figure(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True)
plt.title('Heatmap of Correlation Matrix')

# %%
# Plot the distribution of Employee Satisfaction, Evaluation, and Project Count. What story can you tell?

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=3, figsize=(16, 8))

# Graph Employee Satisfaction
sns.distplot(df.satisfaction, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')
axes[0].set_ylabel('Employee Count');

# Graph Employee Evaluation
sns.distplot(df.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')
axes[1].set_ylabel('Employee Count');

# Graph Employee Average Monthly Hours
sns.distplot(df.averageMonthlyHours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
axes[2].set_ylabel('Employee Count');

# %% [markdown]
# - More than half of the employees with **2, 6 and 7** projects left the company
# - Majority of the employees who did not leave the company had **3,4, and 5** projects
# - All of the employees with **7** projects left the company
# - There is an increase in employee turnover rate as project count increases

# %%
df['turnover'] = df['turnover'].astype(str)

# %%
plt.figure(figsize=(20,8))
ax = sns.barplot(x="projectCount", y="projectCount", hue="turnover", data=df, estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent");

# %% [markdown]
# <a id='pre_processing'></a>
# # Pre-processing
# ***

# %% [markdown]
# - Apply **get_dummies()** to the categorical variables.
# - Seperate categorical variables and numeric variables, then combine them.

# %%
cat_var = ['department','salary','turnover','promotion']
num_var = ['satisfaction','evaluation','projectCount','averageMonthlyHours','yearsAtCompany', 'workAccident']
categorical_df = pd.get_dummies(df[cat_var], drop_first=True, dummy_na=True)
numerical_df = df[num_var]

new_df = pd.concat([categorical_df,numerical_df], axis=1)
new_df.head()

# %% [markdown]
# <a id='train_test_split'></a>
# # Split Train/Test Set
# ***
# 

# %% [markdown]
# Let's split our data into a train and test set. We'll fit our model with the train set and leave our test set for our last evaluation.

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

# %%
# Create the X and y set
X = new_df.iloc[:,1:]
y = new_df.iloc[:,0]

# Define train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123, stratify=y)

# %%
print(X_train.shape)
print(X_test.shape)

# %% [markdown]
# <a id='class_imbalance'></a>
# # Class Imbalance

# %%
round(df.turnover.value_counts(1), 2)

# %% [markdown]
# ### Employee Turnover Rate: 24%

# %% [markdown]
# #Treat Imbalanced Datasets
# 

# %%
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=12, sampling_strategy = 1.0)
x_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# %% [markdown]
# # Model Training and Performance

# %% [markdown]
# <a id='lr'></a>
# ## Logistic Regression Classifier
# 

# %%
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


lr = LogisticRegression()
lr = lr.fit(x_train_sm, y_train_sm)
lr

# %%
print ("\n\n ---Logistic Regression Model---")
print(classification_report(y_test, lr.predict(X_test)))

# %%
from sklearn.tree import DecisionTreeClassifier
# Create a decision tree classifier
clf = DecisionTreeClassifier()
clf = clf.fit(x_train_sm,y_train_sm)
clf

# %%
print ("\n\n ---Decision Tree Model---")
print(classification_report(y_test, clf.predict(X_test)))

# %%
# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid for Decision Tree
param_grid_tree = {
    'max_depth': [3, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Grid Search for Decision Tree
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid_tree, refit=True, verbose=1, cv=5)
grid_tree.fit(x_train_sm, y_train_sm)

# Print the best parameters and accuracy for Decision Tree
print("Best parameters for Decision Tree:", grid_tree.best_params_)
print("Best Decision Tree accuracy on training data:", grid_tree.best_score_)

# Evaluate on test data
y_pred_tree_optimized = grid_tree.predict(X_test)



# %%
from sklearn.svm import SVC
from sklearn.metrics import classification_report

clf_svm = SVC(kernel='linear', C=1.0)  

clf_svm.fit(x_train_sm, y_train_sm)

y_pred_svm = clf_svm.predict(X_test)



# %%
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

# Define distributions for randomized search
param_distributions_svm = {
    'C': stats.uniform(0.1, 10),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.01, 0.1]
}

# Randomized Search for SVM
random_search_svm = RandomizedSearchCV(SVC(), param_distributions_svm, n_iter=10, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search_svm.fit(x_train_sm, y_train_sm)

# Print the best parameters and accuracy for SVM
print("Best parameters for SVM:", random_search_svm.best_params_)
print("Best SVM accuracy on training data:", random_search_svm.best_score_)


# %%



