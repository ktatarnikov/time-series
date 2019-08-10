import pandas as pd

import xgboost as xgb
from numpy.random import seed

from tensorflow import set_random_seed


class XGBoostModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(learning_rate = 0.01, max_depth = 5,\
                        min_child_weight = 5, objective = 'binary:logistic', seed = 42,\
                        gamma = 0.1, silent = True)

    def fit(self, X_train, y_train, X_valid, y_valid):
        eval_set = [(X_train, y_train), (X_valid, y_valid)]
        self.model.fit(X_train, y_train, eval_set = eval_set, \
            eval_metric="auc", verbose=False, early_stopping_rounds=1000)

    def predict(self, X_input):
        return self.model.predict(X_input,
                                  ntree_limit=self.model.best_ntree_limit)

    def evaluate(self, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
        print("class distribution in original training data.\n",
              Y_train.groupby(by=lambda v: Y_train.loc[v]).count())
        X_upsampled, Y_upsampled = feature_engineering.class_imbalance_fix(
            X_train, Y_train)
        elements, counts_elements = np.unique(Y_upsampled, return_counts=True)
        print("class distribution in upsampled training data.")
        print(elements)
        print(counts_elements)

        model.fit(X_upsampled, Y_upsampled, X_valid.as_matrix(),
                  Y_valid.as_matrix())

        Y_train_pred = model.predict(X_train.as_matrix())
        helper.print_results("Training",
                             helper.evaluate(Y_train, Y_train_pred))

        Y_valid_pred = model.predict(X_valid.as_matrix())
        helper.print_results("Validation",
                             helper.evaluate(Y_valid, Y_valid_pred))

        Y_test_pred = model.predict(X_test.as_matrix())
        helper.print_results("Test", helper.evaluate(Y_test, Y_test_pred))
