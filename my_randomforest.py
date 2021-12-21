import pandas as pd
import numpy as np
import time
import collections
from decision_tree.my_DecisionTree import DecisionTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class RandomForest:
    def __init__(self, n_estimators=50, max_depth=100, random_state=21):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.list_estimators = None
        self.mas_bootstrap = None

    def create_list_estimators(self):
        self.list_estimators = []
        for estimator in range(self.n_estimators):
            self.list_estimators.append(DecisionTree(self.max_depth, splitter='2', type_sample='random'))

    def bootstrap(self, data_train, label_train):
        len_fi_part = int(len(data_train) * 0.63)
        len_se_part = len(data_train) - len_fi_part
        data_ = np.column_stack([data_train, label_train])
        self.mas_bootstrap = []
        for i in range(self.n_estimators):
            fi_part_in = np.random.choice(len(data_), size=len_fi_part, replace=False)
            se_part_in = np.random.choice(fi_part_in, size=len_se_part, replace=False)
            self.mas_bootstrap.append(np.concatenate([data_[fi_part_in], data_[se_part_in]]))

    def fit_estimators(self):
        for estimator, _data in zip(self.list_estimators, self.mas_bootstrap):
            estimator.fit(_data)

    def fit(self, data_train, label_train):
        if isinstance(data_train, pd.DataFrame):
            data_train = data_train.values
        self.create_list_estimators()
        self.bootstrap(data_train, label_train)
        self.fit_estimators()

    def predict_estimators(self, data_test):
        mas_ans_2d = []
        for estimator in self.list_estimators:
            mas_ans_2d.append(estimator.predict(data_test))
        return np.array(mas_ans_2d)

    def final_predict(self, mas_ans_2d, flag_prob):
        mas_ans = []
        mas_ans_2d_ = mas_ans_2d.T
        len_row = len(mas_ans_2d_[0])
        for row in mas_ans_2d_:
            if flag_prob:
                col = collections.Counter(row).most_common(1)
                mas_ans.append(col[0][1] / len_row) if col[0] else mas_ans.append(1 - col[0][1] / len_row)
            else:
                mas_ans.append(collections.Counter(row).most_common(1)[0][0])
        return mas_ans

    def predict(self, data_test):
        mas_ans_2d = self.predict_estimators(data_test)
        return self.final_predict(mas_ans_2d, False)

    def predict_proba(self, data_test):
        mas_ans_2d = self.predict_estimators(data_test)
        return self.final_predict(mas_ans_2d, True)


if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    data, target = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names), breast_cancer.target
    data.columns = breast_cancer.feature_names

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=21)
    start_time = time.time()
    sk_tree = RandomForestClassifier(max_depth=1, random_state=21)
    sk_tree.fit(train_data, train_target)
    print("--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    my_forest = RandomForest(n_estimators=10, max_depth=100)
    my_forest.fit(train_data, train_target)
    a = my_forest.predict(test_data)
    print("--- %s seconds ---" % (time.time() - start_time))
    #
    print(accuracy_score(test_target, sk_tree.predict(test_data)))
    print(accuracy_score(test_target, my_forest.predict(test_data)))
