    from flask import Flask, render_template, request, jsonify
    import joblib
    import pandas as pd
    import os

    # Dapatkan direktori absolut dari file index.py saat ini
    # Get the absolute directory of the current index.py file (which is inside 'api' folder)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # BASE_DIR sekarang adalah direktori satu tingkat di atas CURRENT_DIR (yaitu root proyek)
    # BASE_DIR is now one level up from CURRENT_DIR (i.e., the project root)
    BASE_DIR = os.path.dirname(CURRENT_DIR)


    # Tentukan jalur ke folder templates dan models relatif terhadap BASE_DIR
    # Define paths to templates and models folders relative to BASE_DIR
    TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # Inisialisasi aplikasi Flask, beritahu di mana folder templates berada
    # Initialize Flask app, tell it where the templates folder is
    app = Flask(__name__, template_folder=TEMPLATES_DIR)

    # Load the trained models
    # Muat model yang telah dilatih
    try:
        linreg_pipeline = joblib.load(os.path.join(MODELS_DIR, 'linreg_pipeline.pkl'))
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    except FileNotFoundError:
        print(f"Error: Model files not found. Expected in {MODELS_DIR}. Please ensure 'linreg_pipeline.pkl' and 'rf_model.pkl' are in the 'models' directory.")
        exit()

    # Load model accuracies from file
    # Muat akurasi model dari file
    linreg_r2_score = 0.0
    rf_r2_score = 0.0
    try:
        with open(os.path.join(MODELS_DIR, 'model_accuracies.txt'), 'r') as f:
            for line in f:
                if line.startswith('linreg_r2_score='):
                    linreg_r2_score = float(line.split('=')[1])
                elif line.startswith('rf_r2_score='):
                    rf_r2_score = float(line.split('=')[1])
        print("Model accuracies loaded from 'model_accuracies.txt'.")
    except FileNotFoundError:
        print(f"Warning: 'model_accuracies.txt' not found. Expected in {MODELS_DIR}. Using default R2 scores (0.0). Please run train_models.py.")
    except Exception as e:
        print(f"Error loading model accuracies: {e}. Using default R2 scores (0.0).")


    @app.route('/')
    def index():
        """
        Renders the main prediction form page.
        Merender halaman formulir prediksi utama.
        """
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
            luas_bangunan = float(request.form['luas_bangunan'])
            luas_tanah = float(request.form['luas_tanah'])
            kamar_tidur = float(request.form['kamar_tidur'])
            kamar_mandi = float(request.form['kamar_mandi'])
            garasi = float(request.form['garasi'])

            input_data = pd.DataFrame([[luas_bangunan, luas_tanah, kamar_tidur, kamar_mandi, garasi]],
                                      columns=['LB', 'LT', 'KT', 'KM', 'GRS'])

            linreg_result_juta = linreg_pipeline.predict(input_data)[0]
            rf_result_juta = rf_model.predict(input_data)[0]

            linreg_full_rupiah = int(linreg_result_juta * 1_000_000)
            rf_full_rupiah = int(rf_result_juta * 1_000_000)

            linreg_output = f"Linear Regression: Rp {linreg_full_rupiah:,}".replace(',', '.')
            rf_output = f"Random Forest: Rp {rf_full_rupiah:,}".replace(',', '.')

            return jsonify(
                linreg_prediction=linreg_output,
                rf_prediction=rf_output,
                linreg_raw_value=round(linreg_result_juta, 2),
                rf_raw_value=round(rf_result_juta, 2)
            )

        except Exception as e:
            return jsonify(error=str(e)), 400