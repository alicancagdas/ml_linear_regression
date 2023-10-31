import pandas as pd

class DataManager:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def get_data(self):
        return self.df

    def get_X_y(self):
        X = self.df[["TV"]]
        y = self.df[["sales"]]
        return X, y

    def getmulti_X_y(self):
        X = self.df.drop("sales",axis=1)
        y = self.df[["sales"]]
        return X, y

    def get_data_description(self):
        return self.df.describe().T