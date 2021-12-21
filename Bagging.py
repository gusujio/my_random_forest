import numpy as np
import pandas as pd
from decision_tree.my_DecisionTree import DecisionTree
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


breast_cancer = load_breast_cancer()
data, target = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names), breast_cancer.target
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=21)

forest = RandomForestClassifier(random_state=21, n_estimators=50, max_depth=100)
forest.fit(train_data, train_target)
print("RandomForestClassifier ", accuracy_score(test_target, forest.predict(test_data)))

forest = BaggingClassifier(DecisionTreeClassifier(max_depth=100), random_state=21, n_estimators=50)
forest.fit(train_data, train_target)
print("BaggingClassifier ", accuracy_score(test_target, forest.predict(test_data)))




