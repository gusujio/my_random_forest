import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Tree:
    def __init__(self, father, left=None, right=None, index=None, porog=None):
        self.father = father
        self.left = left
        self.right = right
        self.index = index
        self.porog = porog

    def add_leaf(self, value):
        pass

    def depth(self):
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return max(left_depth, right_depth) + 1

    def leaf_value(self):
        left_value = self.left.leaf_value() if self.left else self.father[0][-1]
        right_value = self.right.leaf_value() if self.right else self.father[0][-1]
        return left_value, right_value




class Decision_tree:
    index = 0
    porog = 100
    def __init__(self, max_depth = None, min_samples = 1):
        self.max_depth = max_depth
        self.min_samples = min_samples


    def gini(self, rows):
        unik_target = set(rows)
        sums = 0
        lens_rows = len(rows)
        rows = np.array(rows)
        for target in unik_target:
            doli = sum(rows == target) / lens_rows
            sums += doli * (1 - doli)
        return sums

    def information(self, left, right):
        father = len(left) + len(right)
        return len(left) / father * self.gini(left[:,-1]) + len(right) / father * self.gini(right[:,-1])

    def check(self, value, index, porog):
        return value[index] >= porog

    def partition(self, data, index, porog):
        true_rows, fals_rows = [], []
        for row in data:
            if self.check(row, index, porog):
                true_rows.append(row)
            else:
                fals_rows.append(row)
        return np.array(true_rows), np.array(fals_rows)

    def best_split(self, data):
        n_feats = len(data[0]) - 1
        best_gini = float('inf')
        best_porog = best_index = None
        for feat in range(n_feats):
            values = set(data[:, feat])
            index = feat
            for val in values:
                porog = val
                true_rows, false_rows = self.partition(data, index, porog)
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue
                gini = self.information(true_rows, false_rows)
                if gini < best_gini:
                    best_porog, best_index, best_gini = porog, index, gini

        return best_porog, best_index

    def create_leaf(self, head):
        head.porog, head.index = self.best_split(head.father)
        left, right = self.partition(head.father, head.index, head.porog)
        head.left = Tree(left, None, None)
        head.right = Tree(right, None, None)
        # return Tree(head, left, right)

    def fit(self, data, target = None):
        if isinstance(data, pd.DataFrame):
            self.columns = data.columns
            tree = Tree(np.column_stack([data.values, target]))
            self.head_tree = tree
        else:
            tree = data

        while len(tree.father) != self.min_samples:
            self.create_leaf(tree)
            Decision_tree().fit(tree.left)
            Decision_tree().fit(tree.right)

    def leaf_target(self, data):
        np_data = data.values
        tree = self.head_tree
        mas_target = []
        for row in np_data:
            while tree.left or tree.right:
                if tree.right and self.check(row, tree.index, tree.porog):
                    tree = tree.right
                else:
                    tree = tree.left
            targets = []
            for target in tree.father:
                targets.append(target[-1])
            mas_target.append(targets)
        return mas_target

    def count_target(self, mas_target):
        count_target = dict()
        max_count_target = 0
        max_target = None
        for target in mas_target:
            if target not in count_target:
                count_target[target] = 1
            else:
                count_target[target] += 1
            if count_target[target] > max_count_target:
                max_count_target = count_target[target]
                max_target = target
        return max_target, count_target

    def predict(self, data):
        mas_targets = self.leaf_target(data)
        for index in range(len(mas_targets)):
            mas_targets[index], _ = self.count_target(mas_targets[index])
        return mas_targets

    def predict_proba(self, data):
        mas_targets = self.leaf_target(data)
        for index in range(len(mas_targets)):
            _, dict_target = self.count_target(mas_targets[index])
            sums = sum(list(dict_target.values()))
            mas_target = []
            for key in dict_target:
                mas_target.append(dict_target[key] / sums)
            mas_targets[index] = mas_target
        return mas_targets








breast_cancer = load_breast_cancer()
data, target = pd.DataFrame(breast_cancer.data), breast_cancer.target
data.columns = breast_cancer.feature_names

train_data, test_data, train_target, test_target= train_test_split(data, target, test_size=0.33)

sk_tree = DecisionTreeClassifier()
sk_tree.fit(train_data, train_target)

my_tree = Decision_tree()
my_tree.fit(train_data, train_target)

print(accuracy_score(test_target, sk_tree.predict(test_data)))
# my_tree.fit(data, target)

# print(a, b)
# worst radius 16.82
