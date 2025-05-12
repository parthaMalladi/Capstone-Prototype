from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('StrokeModel.pkl')
encoders = joblib.load('StrokeEncoders.pkl')

# list of categories
categories = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# column names
colNames = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form.get('gender')
    age = np.float64(request.form.get('age'))
    hypertension = np.int64(request.form.get('hypertension'))
    heart_disease = np.int64(request.form.get('heart_disease'))
    ever_married = request.form.get('ever_married')
    work_type = request.form.get('work_type')
    residence_type = request.form.get('residence_type')
    avg_glucose_level = np.float64(request.form.get('avg_glucose_level'))
    bmi = np.float64(request.form.get('bmi'))
    smoking_status = request.form.get('smoking_status')

    userSample = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]], columns=colNames)

    for col in categories:
        le = encoders[col]
        userSample[col] = le.transform(userSample[col])
    
    prediction = model.predict(userSample)
    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)