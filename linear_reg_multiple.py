import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_manager import DataManager


"""
Çoklu Doğrusal Regresyon - Satış Değeri Tahmini

Bu betik, çoklu doğrusal regresyon kullanarak satış değerini tahmin eder.
Aşağıda yer alan TV, radyo ve gazete harcamalarına göre tahmini satış miktarını hesaplar.

TV: 40
Radio: 15
Newspaper: 20

Çoklu regresyon denklemi: Sales = b + TV * w1 + Radio * w2 + Newspaper * w3

# b([2.90794702]), w([0.0468431  0.17854434 0.00258619)
# Sales = 2.90 + TV * 0.0468431 + radio * .17854434 +  newspaper * 0.00258619

# verilen gözlem değerlerine göre satışın beklenen değeri?
# çoklu regresyon deklemi?

"""


class SalesPredictionModel:
    def __init__(self, data_manager, test_size=0.20, random_state=1):
        # İnşa sınıfı ve gerekli parametreleri tanımlama
        self.data_manager = data_manager
        self.test_size = test_size
        self.random_state = random_state
        self.reg_model = None  # Modeli başlangıçta boş bırak

    def train_model(self):
        # Modelin eğitim işlemini gerçekleştirme
        df = pd.read_csv("datasets/advertising.csv", index_col=0)
        X, y = self.data_manager.getmulti_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.reg_model = LinearRegression().fit(X_train, y_train)

    def predict_sales(self, TV, radio, newspaper):
        # Satış tahminini hesaplama ve döndürme
        if self.reg_model is not None:
            ornek_veri = pd.DataFrame({
                "TV": [TV],
                "radio": [radio],
                "newspaper": [newspaper]
            })
            return self.reg_model.predict(ornek_veri)
        else:
            return "Model eğitilmemiş. Lütfen önce modeli eğitin."

# Kullanım
if __name__ == "__main__":
    data_manager = DataManager("datasets/advertising.csv")  # Veri yolu ile değiştirin
    model = SalesPredictionModel(data_manager)
    model.train_model()
    result = model.predict_sales(30, 10, 40)
    print("Tahmini Satış:", result)
