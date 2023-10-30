import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.float_format', lambda x: '%2f' % x)

# csvyi okuyoruz
df = pd.read_csv("datasets/advertising.csv")
df.shape

X = df[["TV"]]
y = df[["sales"]]

# model kuruyoruz

reg_model = LinearRegression().fit(X,y)

############
# Tahmin
############

# 150 birimlik TV harcaması olsa  ne kadar satış olması beklenir?
print(reg_model.intercept_[0] + reg_model.coef_[0][0]*150)

# 500 birimlik TV harcaması olsa  ne kadar satış olması beklenir?
print(reg_model.intercept_[0] + reg_model.coef_[0][0]*500)

df.describe().T

#Modelin Görselleştirilmesi
# ci guven araligi
# renkler
# title ekleme
# eksenleri belirleme
# eksenlere dağılımı set etme(dinamik bir şekilde) regresyon modelinin oluşturulması!


g= sns.regplot(x= X, y=y, scatter_kws={'color': 'b', 's': 9},ci=False, color="r")

g.set_title(f"Model Denklemi: Sales = {round (reg_model.intercept_[0], 2)}+ TV*{round (reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim (-10, 310)
plt.show()

#MSE
y_pred = reg_model.predict(X)

# y değerlerinin ortalamasi
# y değerlerinin standart sapmasi
yort = y.mean()
yssapma = y.std()

print("\nortalama: ", yort, "standart sapma: ", yssapma)
print("\nmodel basarisi MSE(ortalama hatamiz): ", mean_squared_error(y,y_pred))

#RMSE
print("\nmodel basarisi RMSE(ortalama hatamiz): ", np.sqrt(mean_squared_error(y,y_pred)))

#MAE
print("\nmodel basarisi MAE(ortalama hatamiz): ", (mean_absolute_error(y,y_pred)))
