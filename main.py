import os
import joblib
from pandas import read_csv, DataFrame
from io import StringIO
from flask import Flask, render_template, request, send_from_directory
from google.cloud import storage
import tempfile
from joblib import load
import logging

app = Flask(__name__)

def download_model():
    model_file = os.path.join(temp_dir, MODEL_NAME)
    blob.download_to_filename(model_file)
    return model_file

def load_model(model_file):
    return load(model_file)

BUCKET_NAME = "wine_quality_1"
DATASET_NAME = "winequality-red.csv"
MODEL_NAME = "random_forest_regressor.joblib"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

# Load dataset from Google Cloud Storage
blob = storage.Blob(DATASET_NAME, bucket)
data = blob.download_as_text()
dataset = read_csv(StringIO(data), sep=";")

# Load trained model from Google Cloud Storage
blob = storage.Blob(MODEL_NAME, bucket)

temp_dir = tempfile.mkdtemp()
model_file = os.path.join(temp_dir, MODEL_NAME)
blob.download_to_filename(model_file)
model = joblib.load(model_file)

@app.route('/', methods=['GET', 'POST'])
def index():
    default_input_values = {
        'fixed_acidity': '',
        'volatile_acidity': '',
        'citric_acid': '',
        'residual_sugar': '',
        'chlorides': '',
        'free_sulfur_dioxide': '',
        'total_sulfur_dioxide': '',
        'density': '',
        'pH': '',
        'sulphates': '',
        'alcohol': ''
    }
    return render_template('index.html', input_values=default_input_values)

    if request.method == 'POST':
        # Get input values from the form
        input_values = {
            'fixed acidity': float(request.form['fixed_acidity']),
            'volatile acidity': float(request.form['volatile_acidity']),
            'citric acid': float(request.form['citric_acid']),
            'residual sugar': float(request.form['residual_sugar']),
            'chlorides': float(request.form['chlorides']),
            'free sulfur dioxide': float(request.form['free_sulfur_dioxide']),
            'total sulfur dioxide': float(request.form['total_sulfur_dioxide']),
            'density': float(request.form['density']),
            'pH': float(request.form['pH']),
            'sulphates': float(request.form['sulphates']),
            'alcohol': float(request.form['alcohol'])
        }

        # Create a DataFrame from the input values
        input_df = DataFrame(input_values, index=[0])

        # Make a prediction using the model
        prediction = model.predict(input_df)

        # Return the result as a string
        result = f'Predicted wine quality: {prediction[0]:.2f}'

        return render_template('index.html', result=result)
    else:
        return render_template('index.html')

    app.logger.info("Rendering template")
    result = render_template("index.html")
    app.logger.info("Template rendered")

    return result

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = [features]
    prediction = model.predict(input_data)
    return render_template("index.html", prediction_text=f"Predicted wine quality: {prediction[0]:.2f}", input_values=request.form)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

temp_dir = tempfile.mkdtemp()
model_file = download_model()
model = load_model(model_file)

if __name__ == "__main__":
    if os.environ.get("GAE_ENV", "").startswith("standard"):
        # Production mode
        app.logger.setLevel(logging.INFO)
    else:
        # Local development mode
        app.logger.setLevel(logging.DEBUG)

    app.run(debug=True)


