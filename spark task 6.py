import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Visualize the decision tree graphically
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=feature_names, class_names=target_names, filled=True)
plt.show()
