from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Jalur ke direktori model
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Memuat model yang telah dilatih
try:
    linreg_pipeline = joblib.load(os.path.join(MODELS_DIR, 'linreg_pipeline.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
except FileNotFoundError:
    print("Error: Model files not found. Please ensure 'linreg_pipeline.pkl' and 'rf_model.pkl' are in the 'models' directory.")
    exit()

# --- Tambahan: Muat akurasi model (R-squared) dari file ---
linreg_r2_score = 0.0 # Default value
rf_r2_score = 0.0     # Default value
try:
    with open(os.path.join(MODELS_DIR, 'model_accuracies.txt'), 'r') as f:
        for line in f:
            if line.startswith('linreg_r2_score='):
                linreg_r2_score = float(line.split('=')[1])
            elif line.startswith('rf_r2_score='):
                rf_r2_score = float(line.split('=')[1])
    print("Akurasi model berhasil dimuat dari 'model_accuracies.txt'.")
except FileNotFoundError:
    print("Warning: 'model_accuracies.txt' not found. Using default R2 scores (0.0). Please run train_models.py.")
except Exception as e:
    print(f"Error loading model accuracies: {e}. Using default R2 scores (0.0).")
# --- Akhir Tambahan ---


@app.route('/')
def index():
    """
    Renders the main prediction form page.
    Merender halaman formulir prediksi utama.
    """

    # Mengirim akurasi model ke template
    return render_template('index.html', 
                           linreg_r2=f"{linreg_r2_score*100:.2f}%",
                           rf_r2=f"{rf_r2_score*100:.2f}%")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests.
    Menerima permintaan prediksi.
    """
    try:

        # Dapatkan data dari pengiriman formulir
        luas_bangunan = float(request.form['luas_bangunan'])
        luas_tanah = float(request.form['luas_tanah'])
        kamar_tidur = float(request.form['kamar_tidur'])
        kamar_mandi = float(request.form['kamar_mandi'])
        garasi = float(request.form['garasi'])

        # Buat DataFrame untuk prediksi
        input_data = pd.DataFrame([[luas_bangunan, luas_tanah, kamar_tidur, kamar_mandi, garasi]],
                                  columns=['LB', 'LT', 'KT', 'KM', 'GRS'])


        # Buat prediksi
        linreg_result_juta = linreg_pipeline.predict(input_data)[0]
        rf_result_juta = rf_model.predict(input_data)[0]


        # Konversi ke Rupiah penuh dan format dengan pemisah ribuan
        linreg_full_rupiah = int(linreg_result_juta * 1_000_000)
        rf_full_rupiah = int(rf_result_juta * 1_000_000)

        # Format hasil untuk tampilan prediksi harga
        linreg_output = f" Rp. {linreg_full_rupiah:,}".replace(',', '.')
        rf_output = f" Rp. {rf_full_rupiah:,}".replace(',', '.')

        return jsonify(
            linreg_prediction=linreg_output,
            rf_prediction=rf_output,
            linreg_raw_value=round(linreg_result_juta, 2), # Nilai mentah untuk grafik (dalam juta)
            rf_raw_value=round(rf_result_juta, 2)         # Nilai mentah untuk grafik (dalam juta)
        )

    except Exception as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':

    # Jalankan aplikasi Flask dalam mode debug
    app.run(debug=True)