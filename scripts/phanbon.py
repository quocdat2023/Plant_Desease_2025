import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import numpy as np

# Giả định dữ liệu đã được đọc từ file CSV
data = pd.read_csv('data_core.csv', delimiter=',', quotechar='"', encoding='utf-8', on_bad_lines='warn')# Tiền xử lý: Mã hóa các biến mục tiêu (Soil Type, Crop Type, Fertilizer Name) thành số
# XGBoost yêu cầu nhãn là số nguyên (0, 1, 2, ...)
label_encoders = {}
y_encoded = pd.DataFrame()
y = data[['Soil Type', 'Crop Type', 'Fertilizer Name']]

for column in y.columns:
    le = LabelEncoder()
    y_encoded[column] = le.fit_transform(y[column])
    label_encoders[column] = le  # Lưu encoder để giải mã sau này

# Tách đặc trưng và các biến mục tiêu
X = data[['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']]

# Tạo pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', StandardScaler()),  # Chuẩn hóa các biến số
    ('classifier', MultiOutputClassifier(XGBClassifier(random_state=42, eval_metric='mlogloss')))
])

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Huấn luyện mô hình
pipeline.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = pipeline.predict(X_test)

# Đánh giá từng biến mục tiêu (giải mã nhãn về giá trị gốc)
for i, column in enumerate(y.columns):
    print(f"\nClassification Report for {column}:\n", 
          classification_report(y_test[column], y_pred[:, i], 
                              target_names=label_encoders[column].classes_))

# Lưu mô hình
joblib.dump(pipeline, 'recommender-models/multioutput_xgboost_fertilizer_model.pkl')
print("Model saved as 'multioutput_xgboost_fertilizer_model.pkl'")

# Lưu các label encoder để giải mã nhãn sau này
for column, le in label_encoders.items():
    joblib.dump(le, f'recommender-models/label_encoder_{column}.pkl')
print("Label encoders saved for Soil Type, Crop Type, and Fertilizer Name")

# Kiểm tra kết quả trên dữ liệu mới
new_data = pd.DataFrame({
    'Temperature': [25],
    'Humidity': [60],
    'Moisture': [30],
    'Nitrogen': [50],
    'Potassium': [40],
    'Phosphorous': [20]
})

# Tải mô hình
loaded_model = joblib.load('recommender-models\multioutput_xgboost_fertilizer_model.pkl')

# Dự đoán trên dữ liệu mới
predictions = loaded_model.predict(new_data)

# Giải mã kết quả dự đoán về giá trị gốc
print("\nPredictions for new data:")
for i, column in enumerate(y.columns):
    le = joblib.load(f'label_encoder_{column}.pkl')
    predicted_label = le.inverse_transform([predictions[0][i]])[0]
    print(f"{column}: {predicted_label}")