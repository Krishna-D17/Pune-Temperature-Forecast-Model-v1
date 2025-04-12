from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

#####################################################
# 1. Define the same FeatureEngineer class used in the pipeline
#####################################################
from feature_engineering import FeatureEngineer

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
