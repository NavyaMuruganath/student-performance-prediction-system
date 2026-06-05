from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/form')
def form():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    g1 = float(request.form['G1'])
    g2 = float(request.form['G2'])
    studytime = float(request.form['studytime'])
    failures = float(request.form['failures'])
    absences = float(request.form['absences'])

    new_student = pd.DataFrame({
        'G1': [g1],
        'G2': [g2],
        'studytime': [studytime],
        'failures': [failures],
        'absences': [absences]
    })

    prediction = model.predict(new_student)

    prediction = max(0, min(20, prediction[0]))

    return render_template(
        'index.html',
        prediction=round(prediction, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)