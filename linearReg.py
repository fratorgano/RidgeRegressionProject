import numpy as np

class LinearReg:
    def __init__(self):
        self.w = None
        self.base = None

    def fit(self,x,y):
        x_c = x.copy()
        x_c['1'] = 1
        
        transposed = x_c.transpose()
        sts = np.dot(transposed,x_c)
        sts1 = np.linalg.inv(sts) # (S^T.S)^-1
        w = sts1.dot(transposed).dot(y) # ((S^T.S)^-1).S^T.y
        self.w0 = w[len(w)-1]
        self.w = w[:len(w)-1]
        #print(self.w)
        #print(self.w0)
    
    def predict(self,x):
        predictions = []
        for index, row in x.iterrows():
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
    
    def error(self,x,actual_y):
        y = self.predict(x)
        print(f"y:{y}, actual:{actual_y}")
        return (actual_y-y)**2