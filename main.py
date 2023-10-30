# main.py

# data_manager.py modülü, veriyi yükleme ve işleme işlemleri için kullanılır
from data_manager import DataManager

# model.py modülü, satış tahmini modelini oluşturma işlemleri için kullanılır
from model import SalesPredictionModel

# visualizer.py modülü, veriyi görselleştirme işlemleri için kullanılır
from visualizer import DataVisualizer

# metrics.py modülü, modelin değerlendirilmesi ve metriklerin hesaplanması için kullanılır
from metrics import ModelEvaluator

# Veri yönetimi işlemleri
data_manager = DataManager("datasets/advertising.csv")
df = data_manager.get_data()
X, y = data_manager.get_X_y()

# Satış tahmini modeli oluşturma
model = SalesPredictionModel(X, y)

# Veriyi görselleştirme
visualizer = DataVisualizer(X, y, model)
visualizer.visualize_data()

# Model değerlendirme işlemleri
evaluator = ModelEvaluator(model, X, y)
metrics = evaluator.calculate_metrics()

tahmini_Satis = evaluator.tahmini_satis(50)
print("Girilen deger icin bullunan tahmini satis degeri: " ,tahmini_Satis)
# Hesaplanan metrikleri bastırma
for key, value in metrics.items():
    print("\n"f"{key}: {value}")


