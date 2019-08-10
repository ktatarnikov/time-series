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
            eval_metric="auc", verbose=True, early_stopping_rounds=1000)

    def predict(self, X_input):
        return self.model.predict(X_input, ntree_limit = self.model.best_ntree_limit)
