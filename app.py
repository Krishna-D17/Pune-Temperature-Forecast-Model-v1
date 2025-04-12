from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

#####################################################
# 1. Define the same FeatureEngineer class used in the pipeline
#####################################################
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Nothing to fit; transformation is deterministic.
        return self

    def transform(self, X):
        # Create a copy so as not to alter the original DataFrame
        X = X.copy()
        
        # Determine the datetime source: use 'date_time' column if exists,
        # otherwise assume that the DataFrame index is a DatetimeIndex.
        if 'date_time' in X.columns:
            # Convert the date_time column to datetime (if not already)
            dt = pd.to_datetime(X['date_time'])
        elif isinstance(X.index, pd.DatetimeIndex):
            dt = X.index
        else:
            raise ValueError("No 'date_time' column and index is not a DatetimeIndex")
        
        # Extract month, day, and hour from the datetime source.
        X['month'] = dt.month
        X['day'] = dt.day
        X['hour'] = dt.hour

        # Create sine and cosine features for month, day, and hour.
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X['day_sin'] = np.sin(2 * np.pi * X['day'] / 31)
        X['day_cos'] = np.cos(2 * np.pi * X['day'] / 31)
        X['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)

        # Create a season feature based on month.
        def get_season(month):
            if month in [3, 4, 5]:
                return 'summer'
            elif month in [6, 7, 8, 9]:
                return 'monsoon'
            elif month in [10, 11]:
                return 'post-monsoon'
            else:
                return 'winter'
        X['season'] = X['month'].apply(get_season)

        # Create precipitation flag (1 if precipMM > 0, else 0)
        X['precip_flag'] = (X['precipMM'] > 0).astype(int)
        # Create precipitation amount field (can be transformed later if needed)
        X['precip_amount'] = X['precipMM']
        
        return X

#####################################################
# 2. Create the Flask app
#####################################################
app = Flask(__name__)

#####################################################
# 3. Load the trained pipeline (with the same FeatureEngineer defined)
#####################################################
model_pipeline = joblib.load("model_pipeline.pkl")

#####################################################
# 4. Routes
#####################################################
@app.route('/')
def index():
    # Render the input form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve user inputs
        # Note: if the user enters a date in dd/mm/yy format, we set dayfirst=True
        date_time_str = request.form.get("date_time")
        # Use dayfirst=True to convert from dd/mm/yy format properly
        dt = pd.to_datetime(date_time_str, dayfirst=True)
        
        dewpoint = int(request.form.get("DewPointC"))
        humidity = int(request.form.get("humidity"))
        cloudcover = int(request.form.get("cloudcover"))
        uvIndex = int(request.form.get("uvIndex"))
        sunHour = float(request.form.get("sunHour"))
        precipMM = float(request.form.get("precipMM"))
        pressure = int(request.form.get("pressure"))
        windspeedKmph = int(request.form.get("windspeedKmph"))

        # Build a DataFrame including date_time as a column.
        # We'll later set the index to this datetime value.
        input_data = pd.DataFrame({
            'date_time': [dt],  # dt is now a proper datetime object
            'DewPointC': [dewpoint],
            'humidity': [humidity],
            'cloudcover': [cloudcover],
            'uvIndex': [uvIndex],
            'sunHour': [sunHour],
            'precipMM': [precipMM],
            'pressure': [pressure],
            'windspeedKmph': [windspeedKmph]
        })

        # Set 'date_time' column as the DataFrame index (since the pipeline might use it)
        input_data.index = pd.to_datetime(input_data['date_time'])
        # Optionally, you can drop the 'date_time' column if your pipeline doesn't need it:
        input_data = input_data.drop(columns=['date_time'])

        # Predict using your loaded pipeline
        prediction = model_pipeline.predict(input_data)
        predicted_temp = round(prediction[0], 2)

        # Display the result
        return render_template('index.html', 
                               prediction_text=f"Predicted Temperature (°C): {predicted_temp}")
    except Exception as e:
        # In case of any error, print it on the page
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
