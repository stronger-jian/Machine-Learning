import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt
from .metrics import accuracy_score


class kNN_classify(object):

    def __init__(self,k):
        """ 初始化kNN分类器 """
        assert k>=1, "k must be valid"
        
        self.k=k
        self._X_train=None
        self._y_train=None
	
    def fit(self,X_train,y_train):
        """ 根据数据训练集X_train和y_train 训练kNN分类器"""
        assert X_train.shape[0]==y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k<=X_train.shape[0],\
            "the size of X_train must ba at least k"
            
        self._X_train=X_train
        self._y_train=y_train
        return self
		
    def predict(self,X_predict):
        """ 给定测试数据集X_predict，返回表示X_predict的结果向量 """
        assert self._X_train is not None and self._y_train is not None, \
            " must fit before predict "
        assert X_predict.shape[1]==self._X_train.shape[1], \
            " the feature number of X_predict must be equal to X_train "
            
        y_predict=[self._predict(x) for x in X_predict]
        return np.array(y_predict)
		
    def _predict(self,x):
        """ 给定单个预测数据值x，返回x的预测结果 """
        assert x.shape[0]==self._X_train.shape[1], \
            " the feature number of X_predict must be equal to X_train "
        distances=[sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearest=np.argsort(distances)
        topk_y=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(topk_y)
        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """ 根据测试数据集X_test和y_test得到当前模型的准确度 """
        y_predict=self.predict(X_test)
        return accuracy_score(y_test, y_predict)
		
    def __repr__(self):
        return "kNN(k=%d)" %self.k