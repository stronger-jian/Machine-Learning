import numpy as np
from math import sqrt


def accuracy_score(y_true, y_predict):
    """ 计算预测结果集y_predict和原结果集y_true之间的准确率"""

    assert y_true.shape[0]==y_predict.shape[0], "the size of y_predict must be equal to the size of y_true"

    return sum(y_predict == y_true)/len(y_true)


def Mean_Squared_Error(y_ture, y_predict):
    """ 计算y_true 和y_preidct之间的MSE """
    assert len(y_ture)==len(y_predict), "the size of y_true mustbe equal to the size of y_predict"

    return np.sum((y_predict-y_ture)**2)/len(y_predict)


def Root_Mean_Squared_Error(y_true, y_predict):
    """ 计算y_true 和 y_predict 之间的RMSE """

    return sqrt(Mean_Squared_Error(y_true,y_predict))

def Mean_Absolute_Error(y_true, y_predict):
    """ 计算y_true 和 y_predict 之间的MAE """

    return np.sum(np.absolute(y_predict - y_true)) / len(y_predict)

def R2_Score(y_true, y_predict):
    """ 计算y_true 和 y_predict 之间的R Square """

    return 1-(Mean_Squared_Error(y_true, y_predict))/np.var(y_true)