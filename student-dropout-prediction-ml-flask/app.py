from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np

app = Flask(__name__)

MODEL_FILE = 'student_dropout_model.pkl'
DATA_FILE = 'data/student_data.csv'

# Step 1: Create sample data
def create_sample_data():
    data = {
        'attendance': [90, 60, 30, 95, 85, 40, 70, 55, 20, 75],
        'assignments_submitted': [10, 7, 2, 12, 9, 3, 8, 5, 1, 9],
        'absences': [1, 5, 12, 0, 2, 9, 3, 7, 15, 4],
        'avg_grade': [80, 65, 40, 90, 75, 50, 70, 60, 30, 72],
        'dropout': [0, 0, 1, 0, 0, 1, 0, 1, 1, 0]
    }
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print("✅ Sample data created.")

# Step 2: Train model
def train_model():
    df = pd.read_csv(DATA_FILE)
    X = df.drop('dropout', axis=1)
    y = df['dropout']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved.")

# Step 3: Ensure data & model exist
if not os.path.exists(DATA_FILE):
    create_sample_data()
if not os.path.exists(MODEL_FILE):
    train_model()

model = pickle.load(open(MODEL_FILE, 'rb'))

# Step 4: Flask Web App
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            data = [
                float(request.form['attendance']),
                float(request.form['assignments']),
                float(request.form['absences']),
                float(request.form['avg_grade'])
            ]
            prediction = model.predict([data])[0]
            prediction = "Dropout" if prediction == 1 else "Not Dropout"
        except:
            prediction = "Invalid Input!"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
