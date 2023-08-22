import numpy as np
import time

class RidgeReg:
    def __init__(self,alpha):
        self.w = None
        self.base = None
        self.alpha = alpha

    # x,y: Dataframe
    def fit(self,x,y):
        alpha = self.alpha
        x_c = x.copy()
        x_c['1'] = 1
        
        transposed = x_c.transpose() # S^T
        sts = np.dot(transposed,x_c) # S^T.S
        alpha_I = alpha*np.identity(len(sts)) # AI
        alpha_I_sts = alpha_I+sts # AI+S^T.S
        inverted = np.linalg.inv(alpha_I_sts) # (AI+S^T.S)^-1
        w = inverted.dot(transposed).dot(y) # ((AI+S^T.S)^-1).S^T.y
        self.w0 = w[len(w)-1]
        self.w = w[:len(w)-1]
    
    # x: Dataframe
    def predict(self,x):
        return np.array(x.multiply(self.w).sum(axis = 1)+self.w0)

    def predict_old(self,x):
        predictions = []
        for _, row in x.iterrows():
            res = 0
            for i,x_i in enumerate(row):
                res += x_i*self.w[i]
            res += self.w0
            predictions.append(res)
        return predictions
    
    def r2_score(self, y_true, y_pred):
        y_values = y_true.values
        y_average = np.average(y_values)

        residual_sum_of_squares = 0
        total_sum_of_squares = 0

        for i in range(len(y_values)):
            residual_sum_of_squares += (y_values[i] - y_pred[i])**2
            total_sum_of_squares += (y_values[i] - y_average)**2

        return 1 - (residual_sum_of_squares/total_sum_of_squares)
    
    def mse(self,y_true,y_pred):
        y_values = y_true.values
        residual_sum_of_squares = 0
        for i in range(len(y_values)):
            residual_sum_of_squares += (y_values[i] - y_pred[i])**2
        return residual_sum_of_squares/len(y_true)
