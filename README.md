# CODTECH-TASK-1-DECISION-TREE-IMPLEMENTATION

COMPANY : CODTECH IT SOLUTION

NAME : SHAIK MUNWAR BASHA

INTERN ID : CT06DM431

DOMAIN : Machine Learning

MENTOR : Neela Santosh

DURATION : 6 weeks


#Building and Visualizing a Decision Tree Model Using Scikit-Learn
Introduction
A Decision Tree is a supervised machine learning algorithm used for classification and regression tasks. It works by splitting the data into subsets based on feature values, creating a tree structure where each node represents a decision rule, and each leaf represents an outcome. Scikit-learn provides an efficient implementation to build and visualize decision trees.

In this task, we are constructing a decision tree model using the GermanCredit.csv dataset to classify or predict credit risk. The dataset consists of 1,000 entries with various features related to personal finance and credit information.
Steps to Solve
Step 1: Importing Required Libraries
We begin by importing the necessary Python libraries:

python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus as pdot
from IPython.display import Image
These libraries help in data manipulation (pandas), splitting datasets (train_test_split), building a decision tree (DecisionTreeClassifier), exporting and visualizing the tree (export_graphviz, pydotplus), and displaying images (IPython.display).

Step 2: Loading and Exploring the Dataset
We read the dataset using pandas and check the first few rows:

python
df = pd.read_csv(r'/content/GermanCredit.csv')
df.head()
The dataset contains 21 columns, including features such as credit history, duration, amount, employment details, and others. The target variable (credit_risk) represents whether a person is considered a high-risk or low-risk borrower.

Step 3: Preparing the Data
We separate the target variable and preprocess categorical features:

python
y = df['credit_risk']
X = pd.get_dummies(df.drop(['credit_risk'], axis=1), drop_first=True)
Categorical variables are transformed into numerical representations using one-hot encoding (pd.get_dummies). We then split the dataset into training and testing sets:

python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
This ensures that 70% of the data is used for training and 30% for testing.

Step 4: Building the Decision Tree Model
We initialize and train a decision tree classifier:

python
clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=3).fit(X_train, y_train)
The criterion='gini' specifies that we are using the Gini impurity metric to evaluate splits, and max_depth=3 limits the tree's depth to prevent overfitting.

Step 5: Visualizing the Decision Tree
To visualize the tree, we first export it:

python
export_graphviz(clf_tree, out_file="chd_tree.odt", feature_names=X.columns, filled=True)
Next, we generate an image file:

python
chd_tree_graph = pdot.graphviz.graph_from_dot_file('chd_tree.odt')
chd_tree_graph.write_jpg('chd_tree.png')
Finally, we display the tree:

python
Image(filename='chd_tree.png')
Hyperparameter Tuning
We use GridSearchCV to optimize hyperparameters:

python
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'criterion': ['gini', 'entropy'], 'max_depth': range(2, 10)}]
clf_tree = DecisionTreeClassifier()
clf = GridSearchCV(clf_tree, tuned_parameters, cv=10, scoring='roc_auc').fit(X_train, y_train)
After training, we retrieve the best model parameters:

python
print(clf.best_score_)
print(clf.best_params_)
The best results indicate criterion='gini' and max_depth=6, providing an improved predictive performance.

Conclusion:
This workflow guides the construction and visualization of a Decision Tree model using Scikit-learn. The dataset is preprocessed, split into training/testing sets, and optimized using hyperparameter tuning. Decision Trees provide an intuitive representation of classification rules, making them valuable for interpretability in credit risk analysis.


#OUTPUT:
![Image](https://github.com/user-attachments/assets/a48b9d13-f797-4174-9fce-ecaa0b148538)


![Image](https://github.com/user-attachments/assets/aeff02f0-5f8d-4adc-b4b2-9da63b8df584)
