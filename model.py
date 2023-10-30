# model.py

from sklearn.linear_model import LinearRegression





class SalesPredictionModel:
    def __init__(self, X, y):
        self.model = LinearRegression().fit(X, y)

    def get_intercept(self):
        return self.model.intercept_[0]

    def get_coef(self):
        return self.model.coef_[0][0]

    def predict(self, X):
        return self.model.predict(X)




