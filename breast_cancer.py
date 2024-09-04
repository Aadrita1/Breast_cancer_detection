from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
breast_cancer = pd.read_csv('C:\\Users\\Aadrita Nandy\\Downloads\\wisc_bc_data.csv')
del breast_cancer['id']
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.loc[:, breast_cancer.columns != 'diagnosis'],
                                breast_cancer['diagnosis'], stratify=breast_cancer['diagnosis'], random_state=42)
forest = RandomForestClassifier(max_depth=5, n_estimators=100)
forest.fit(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)

breast_cancer_features = [x for i, x in enumerate(breast_cancer.columns) if i != 30]

#
#n_features = 30
ind = np.arange(30)
width=0.4

#plt.barh(range(n_features), forest.feature_importances_, align='edge', color=['#FF1493'])
fig, ax = plt.subplots()

ax.barh(ind, forest.feature_importances_, 0.4, color='red', label='N')
ax.barh(ind+width, tree.feature_importances_, 0.4, color='green', label='M')
plt.yticks(ind+width, breast_cancer_features)
plt.title('Breast Cancer Random Forest Features Importance')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.ylim(-1,30)
plt.show()
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file = "model.dot", class_names = ["malignant", "benign"], feature_names = tree.feature_names_in_, impurity = True, filled = True)
import graphviz
with open("model.dot") as f:
  dot_graph = f.read()
graphviz.Source(dot_graph)
