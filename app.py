import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model('weather_prediction_model.h5')

# Load dữ liệu đã xử lý
weather_df = pd.read_csv('processed_weather_data.csv')

# Khởi tạo StandardScaler cho dữ liệu chuẩn hóa
scaler = StandardScaler()
scaler.fit(weather_df.drop(columns=['temp', 'city', 'date']))

# Trang chính (hiển thị giao diện người dùng)
@app.route('/')
def index():
    return render_template('index.html')

# API dự báo thời tiết
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Lấy tên thành phố từ form
        city_name = request.form['city']
        
        # Lọc dữ liệu của thành phố
        city_data = weather_df[weather_df['city'] == city_name]
        
        # Kiểm tra nếu thành phố không có trong dữ liệu
        if city_data.empty:
            return render_template('index.html', prediction_text="City not found in data.")
        
        # Lấy các đặc trưng
        X_city = city_data.drop(columns=['temp', 'city', 'date'])
        
        # Dự đoán nhiệt độ
        y_pred = model.predict(X_city)
        
        # Trả về kết quả dự đoán
        return render_template('index.html', prediction_text=f"Predicted Temperature for {city_name}: {y_pred[-1][0]:.2f} °C")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
