# visualizer.py

import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def visualize_data(self):
        g = sns.regplot(x=self.X, y=self.y, scatter_kws={'color': 'b', 's': 9}, ci=False, color="r")
        g.set_title(f"Model Denklemi: Sales = {round(self.model.get_intercept(), 2)} + TV*{round(self.model.get_coef(), 2)}")
        g.set_ylabel("Satış Sayısı")
        g.set_xlabel("TV Harcamaları")
        plt.xlim(-10, 310)
        plt.show()

