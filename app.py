from flask import Flask, jsonify, request, render_template
import pymysql
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import webbrowser
from threading import Timer

app = Flask(__name__)

# Database configuration
db_config = {
    'host': '153.92.13.207',
    'user': 'u346812618_kel2',
    'password': 'M0nitorcuaca65!',
    'database': 'u346812618_monitor_cuaca',
    'cursorclass': pymysql.cursors.DictCursor
}

# Load the trained model
def load_weather_model():
    try:
        model_data = joblib.load('weather_model.pkl')  # Changed to 6-hour model
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

def fetch_latest_sensor_data(hours=24):
    """Fetch latest sensor data with time range"""
    connection = get_db_connection()
    if not connection:
        return None
    
    try:
        query = """
        SELECT temperature, humidity, pressure as air_pressure, wind_speed, timestamp 
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
    """Prepare the current data for 6-hour prediction"""
    df = pd.DataFrame([current_data])
    
    # Time features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)
    
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    # Rolling features (use current values)
    df['temp_rolling_mean'] = df['temperature']
    df['humidity_rolling_mean'] = df['humidity']
    df['pressure_rolling_mean'] = df['air_pressure']
    
    # Interaction features
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['pressure_change'] = 0  # Can't calculate change with single data point
    
    X = df[features]
    X_scaled = scaler.transform(X)
    return X_scaled

def get_weather_prediction():
    """Get weather prediction for the next 6 hours"""
    if model is None or scaler is None:
        return None
    
    try:
        # Get the latest data (last 3 hours for rolling features)
        data = fetch_latest_sensor_data(3)
        if data is None or len(data) == 0:
            return None
        
        # Get the most recent record
        last_record = data.iloc[-1]
        prediction_timestamp = last_record['timestamp'] + timedelta(hours=6)
        
        # Prepare data for prediction
        current_data = {
            'temperature': last_record['temperature'],
            'humidity': last_record['humidity'],
            'air_pressure': last_record['air_pressure'],
            'wind_speed': last_record['wind_speed'],
            'timestamp': last_record['timestamp']  # Current timestamp
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
            'prediction': weather_map[pred[0]],
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
        if connection:
            connection.close()
            db_status = 'healthy'
        else:
            db_status = 'unhealthy'
        
        model_status = 'healthy' if model is not None else 'unhealthy'
        
        return jsonify({
            'status': 'running',
            'database': db_status,
            'model': model_status
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True)