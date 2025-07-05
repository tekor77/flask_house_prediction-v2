import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

# Pastikan direktori 'models' ada
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Muat set data
# Load data set (pastikan file 'DATA RUMAH 2.csv' ada di direktori yang sama dengan script ini)
try:
    df = pd.read_csv("DATA RUMAH 2.csv")
    print("File 'DATA RUMAH 2.csv' berhasil dimuat.")
except FileNotFoundError:
    print("Error: 'DATA RUMAH 2.csv' tidak ditemukan. Pastikan file CSV berada di direktori yang sama.")
    # Buat data dummy untuk demonstrasi jika CSV tidak ditemukan
    # Create dummy data for demonstration if CSV is not found
    data = {
        'LB': [100, 120, 80, 150, 110, 90, 130, 70, 140, 105, 500, 30, 1000, 60],
        'LT': [150, 180, 120, 200, 160, 130, 190, 110, 210, 155, 700, 40, 1200, 80],
        'KT': [3, 4, 2, 4, 3, 2, 3, 2, 4, 3, 5, 1, 6, 2],
        'KM': [2, 3, 1, 3, 2, 1, 2, 1, 3, 2, 4, 1, 5, 1],
        'GRS': [1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 3, 0, 4, 0],
        'HARGA': [1000000000, 1200000000, 800000000, 1500000000, 1100000000, 900000000, 1300000000, 750000000, 1450000000, 1080000000, 5000000000, 300000000, 10000000000, 600000000]
    }
    df = pd.DataFrame(data)
    print("Menggunakan data dummy karena 'DATA RUMAH 2.csv' tidak ditemukan. Prediksi mungkin tidak akurat.")

# Pra-pemrosesan data
# Data preprocessing
df = df.dropna()

# Hapus kolom 'NO' dan 'NAMA RUMAH' jika ada
# Drop 'NO' and 'NAMA RUMAH' columns if they exist
if 'NO' in df.columns:
    df = df.drop(columns=['NO'])
if 'NAMA RUMAH' in df.columns:
    df = df.drop(columns=['NAMA RUMAH'])

# Konversi harga ke juta TANPA PEMBULATAN AGAR PRESISI TERJAGA
df['HARGA'] = (df['HARGA'] / 1_000_000)

print(f"Jumlah baris setelah dropna dan drop kolom: {len(df)}")

# --- Penanganan Outlier pada HARGA ---
# Hitung Q1 (kuartil pertama) dan Q3 (kuartil ketiga)
Q1 = df['HARGA'].quantile(0.25)
Q3 = df['HARGA'].quantile(0.75)
IQR = Q3 - Q1 # Hitung Interquartile Range

# Tentukan batas bawah dan atas untuk outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Hapus outlier
df_cleaned = df[(df['HARGA'] >= lower_bound) & (df['HARGA'] <= upper_bound)]
print(f"Jumlah baris setelah penanganan outlier HARGA: {len(df_cleaned)}")
print(f"Outlier yang dihapus: {len(df) - len(df_cleaned)} baris.")

# Gunakan data yang sudah dibersihkan
df = df_cleaned

# Fitur dan target
# Features and target
X = df[["LB", "LT", "KT", "KM", "GRS"]]
y = df["HARGA"]

# Pisahkan data menjadi set pelatihan dan pengujian
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Pipeline Regresi Linear dengan penskalaan
# Linear Regression pipeline with scaling
linreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
linreg_pipeline.fit(X_train, y_train)

# Random Forest dengan penyetelan hyperparameter menggunakan GridSearchCV
# Memperluas param_grid dan menggunakan n_jobs=-1 untuk komputasi paralel
# Random Forest with hyperparameter tuning using GridSearchCV
# Expanding param_grid and using n_jobs=-1 for parallel computation
params = {
    'n_estimators': [100, 200, 300, 400], # Menambahkan 400 estimator
    'max_depth': [10, 20, 30, None],   # Menambahkan 30
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Menggunakan n_jobs=-1 untuk memanfaatkan semua core CPU
grid = GridSearchCV(RandomForestRegressor(random_state=42), params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid.fit(X_train, y_train)
rf_model = grid.best_estimator_

# Simpan model
# Save the models to the 'models' directory
joblib.dump(linreg_pipeline, os.path.join(models_dir, 'linreg_pipeline.pkl'))
joblib.dump(rf_model, os.path.join(models_dir, 'rf_model.pkl'))

print(f"Model Linear Regression berhasil disimpan di: {os.path.join(models_dir, 'linreg_pipeline.pkl')}")
print(f"Model Random Forest berhasil disimpan di: {os.path.join(models_dir, 'rf_model.pkl')}")

# Evaluasi model
# Model evaluation
linreg_pred = linreg_pipeline.predict(X_test)
rf_pred = rf_model.predict(X_test)

linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_r2 = r2_score(y_test, linreg_pred)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n=== Evaluasi Model ===")
print("--- Linear Regression ---")
print(f"MAE: {linreg_mae:.2f} Juta")
print(f"R²: {linreg_r2:.4f}") # Menampilkan R2 dengan 4 angka desimal

print("\n--- Random Forest ---")
print(f"MAE: {rf_mae:.2f} Juta")
print(f"R²: {rf_r2:.4f}") # Menampilkan R2 dengan 4 angka desimal
print(f"Best Params: {grid.best_params_}")

# Simpan R2 score ke file terpisah agar bisa dibaca oleh app.py (opsional, tapi disarankan)
# Save R2 scores to a separate file so app.py can read them (optional, but recommended)
with open(os.path.join(models_dir, 'model_accuracies.txt'), 'w') as f:
    f.write(f"linreg_r2_score={linreg_r2}\n")
    f.write(f"rf_r2_score={rf_r2}\n")
print(f"Akurasi model disimpan di: {os.path.join(models_dir, 'model_accuracies.txt')}")