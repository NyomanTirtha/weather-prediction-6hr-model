from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pymysql
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # Aktifkan CORS

# Konfigurasi database dari environment variables
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
    'database': os.getenv('DB_NAME'),
    'cursorclass': pymysql.cursors.DictCursor
}

# Load model prediksi 6 jam
def load_weather_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'weather_model.pkl')
        print(f"Loading model from: {model_path}")
        model_data = joblib.load(model_path)
        print("Model 6 jam loaded.")
        return model_data['model'], model_data['scaler'], model_data['features']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

model, scaler, features = load_weather_model()

def get_db_connection():
    try:
        connection = pymysql.connect(**db_config)
        return connection
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def fetch_latest_sensor_data(hours=3):
    connection = get_db_connection()
    if not connection:
        return None
    try:
        query = """
        SELECT temperature, humidity, pressure AS air_pressure, wind_speed, timestamp 
        FROM weather_data 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        with connection.cursor() as cursor:
            cursor.execute(query, (hours,))
            result = cursor.fetchall()
            df = pd.DataFrame(result)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception as e:
        print(f"Error fetching sensor data: {e}")
        return None
    finally:
        connection.close()

def preprocess_for_prediction(current_data, features, scaler):
    df = pd.DataFrame([current_data])

    # Time features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)

    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)

    df['temp_rolling_mean'] = df['temperature']
    df['humidity_rolling_mean'] = df['humidity']
    df['pressure_rolling_mean'] = df['air_pressure']
    
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['pressure_change'] = 0  # Placeholder

    X = df[features]
    X_scaled = scaler.transform(X)
    return X_scaled

def get_weather_prediction():
    if model is None or scaler is None:
        return None

    try:
        data = fetch_latest_sensor_data(3)
        if data is None or len(data) == 0:
            return None
        
        last_record = data.iloc[-1]
        prediction_timestamp = last_record['timestamp'] + timedelta(hours=6)

        current_data = {
            'temperature': last_record['temperature'],
            'humidity': last_record['humidity'],
            'air_pressure': last_record['air_pressure'],
            'wind_speed': last_record['wind_speed'],
            'timestamp': last_record['timestamp']
        }

        X_pred = preprocess_for_prediction(current_data, features, scaler)
        pred = model.predict(X_pred)

        weather_map = {
            0: 'Sunny', 
            1: 'Cloudy', 
            2: 'Partly Cloudy', 
            3: 'Rainy', 
            4: 'Overcast'
        }

        return {
            'prediction': weather_map.get(pred[0], 'Unknown'),
            'prediction_timestamp': prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'last_data_timestamp': last_record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'features': {
                'temperature': last_record['temperature'],
                'humidity': last_record['humidity'],
                'air_pressure': last_record['air_pressure'],
                'wind_speed': last_record['wind_speed']
            }
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/')
def index():
    prediction = get_weather_prediction()
    return render_template('index.html', prediction=prediction)

@app.route('/api/predict', methods=['GET'])
def api_predict():
    prediction = get_weather_prediction()
    if prediction:
        return jsonify(prediction)
    else:
        return jsonify({'error': 'Could not generate prediction'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        connection = get_db_connection()
        db_status = 'healthy' if connection else 'unhealthy'
        if connection: connection.close()

        model_status = 'healthy' if model is not None else 'unhealthy'
        return jsonify({
            'status': 'running',
            'database': db_status,
            'model': model_status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
