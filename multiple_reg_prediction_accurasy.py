import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from data_manager import DataManager

if __name__ == "__main__":
    # Veri yönetimi işlemleri
    data_manager = DataManager("datasets/advertising.csv")
    df = data_manager.get_data()
    X, y = data_manager.getmulti_X_y()

    # Veri bölünmesi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # Regresyon modeli oluşturma ve eğitme
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Sabit (b-bias) ve Katsayılar (w-weights)
    print("Intercept (Bias):", reg_model.intercept_)
    print("Coefficients (Weights):", reg_model.coef_)

    # Örnek veri ile satış tahmini
    ornek_veri = pd.DataFrame({
        "TV": [40],
        "radio": [10],
        "newspaper": [40]
    })
    print("Predicted Sales for Example Data:", reg_model.predict(ornek_veri))

    # Eğitim verisi üzerinde RMSE hesaplama
    y_pred_train = reg_model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print("RMSE on Training Data:", rmse_train)
    # Hata sonucumuz 1.736902590147092

    # Eğitim verisi üzerinde R-kare hesaplama
    r_squared_train = reg_model.score(X_train, y_train)
    print("R-squared on Training Data:", r_squared_train)
    # Hata Yüzdemiz  0.8959372632325174

    # Test verisi üzerinde RMSE hesaplama
    y_pred_test = reg_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print("RMSE on Test Data:", rmse_test)
    # Test sonucumuz 1.4113417558581587

    # 10-Katlı çapraz doğrulama ile RMSE hesaplama
    cross_val_rmse = np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error"))
    print("10-Fold Cross-Validation RMSE:", cross_val_rmse)
    # Hata sonucumuz 1.69 burada en güvenilir hata sonucumuz 10 katlı çapraz doğrulamadır
