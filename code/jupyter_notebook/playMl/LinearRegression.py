import numpy as np
from .metrics import R2_Score


class LinearReression:
    def __init__(self):
        """ 构建模型 """
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_nomal(self, X_train, y_train):
        """ 训练模型 """
        assert  X_train.shape[0]==y_train.shape[0]

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_=self._theta[1:]
        return self.interception_ is  not None and self.coef_ is not None

    def predict(self, X_predict):
        """ 给出测试数据X_predict,返回表示X_predict 的结果向量 """
        assert self.interception_ is  not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """ 确定当前模型 准确度 """
        y_predict=self.predict(X_test)
        return R2_Score(y_test, y_predict)

    def __repr__(self):
        return "LinearReression()"