# metrics.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class ModelEvaluator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def tahmini_satis(self,tahimini):
        return self.model.get_intercept() + self.model.get_coef() * tahimini
    def calculate_metrics(self):
        y_pred = self.model.predict(self.X)
        mse = mean_squared_error(self.y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y, y_pred)

        y_mean = self.y.mean()
        y_std = self.y.std()

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
        }
