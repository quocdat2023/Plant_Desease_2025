import pandas as pd
import numpy as np
import pickle
import os

# Định nghĩa đường dẫn
MODELS = './recommender-models/'
filename = 'LogisticRegresion.pkl'

# Bước 1: Tải scaler, label encoder và mô hình
with open(MODELS + 'scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

with open(MODELS + 'label_encoder.pkl', 'rb') as f:
    loaded_label_encoder = pickle.load(f)

with open(MODELS + filename, 'rb') as f:
    loaded_model = pickle.load(f)

# Bước 2: Chuẩn bị dữ liệu đầu vào mới
# Ví dụ: [N, P, K, temperature, humidity, ph, rainfall]
new_data = np.array([[90, 42, 43, 20.879744, 82.002744, 6.502985, 202.935536]])  # Thay đổi giá trị nếu cần

# Chuẩn hóa dữ liệu đầu vào
new_data_scaled = loaded_scaler.transform(new_data)

# Bước 3: Dự đoán
prediction = loaded_model.predict(new_data_scaled)
predicted_crop = loaded_label_encoder.inverse_transform(prediction)[0]

print(f"Loại cây trồng được đề xuất: {predicted_crop}")